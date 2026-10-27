"""The Pendulum suite of ccil.experiments.runner (#112): the #24 study's task-specific half.

Inputs, subsets, label accuracy, rollouts and the audit are the #24 tooling's (tools/m3_pendulum); evaluation is
batched by tools/vector_env (every unit of a chunk and every noise level step together).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "tools" / "m3_pendulum", REPO / "tools" / "vector_env", REPO / "tools" / "legacy_conversion"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import audit as auditor  # noqa: E402
import env as canonical  # noqa: E402
import pendulum as vector  # noqa: E402  (tools/vector_env)
import study as s24  # noqa: E402

from ccil.config import resolve  # noqa: E402
from ccil.experiments.records import sha, unit_name  # noqa: E402

VERSION = "m3_experiments.pendulum_suite/1"


class PendulumSuite:
    name, version = "pendulum", VERSION

    def __init__(self, root, study, smoke: bool = False):
        self.root = Path(root).resolve()
        self.study, self.smoke = study, smoke
        self._inputs = s24.prepare(self.root)  # converts once; verifies the committed conversion's hashes

    def inputs(self, task: str) -> dict:
        return {"config_sha256": self._inputs[task]["config"], "data_sha256": self._inputs[task]["data"]}

    def config(self, phase, task, seed, settings, data_fraction, device, output_dir):
        inputs = self.inputs(task)
        data = self.root / "inputs" / "data" / f"{s24.DATASETS[task]}.npz"
        if data_fraction:
            data, inputs["data_sha256"] = s24.subset(self.root, task, seed, data_fraction)
        sets = [f"run.seed={seed}", f"compute.device={device}", f"data.path={data}", f"run.output_dir={output_dir}",
                *settings]
        if self.smoke:  # one dynamics epoch, one effective policy epoch, as the #24 smoke
            sets += ["transition_model.epochs=1", "policy.epochs=1"]
        cfg = resolve(config=self.root / "inputs" / "config" / s24.CONFIGS[task], sets=sets)
        if self.smoke and cfg["policy.balance_noisy_labels"] and cfg["policy.method"] == "ccil":
            cfg = resolve(config=self.root / "inputs" / "config" / s24.CONFIGS[task],
                          sets=[*sets, f"policy.epochs={cfg['generation.labels_per_transition']}"])
        return cfg, inputs

    def evaluate(self, phase, task, items):
        flat = [(key, predict if predict is not None else _zero, seed, n)
                for key, predict, seed, levels in items for n in levels]
        results = vector.rollouts([(p, seed, n) for _, p, seed, n in flat], task)
        out = {}
        for (key, _, seed, n), res in zip(flat, results):
            _, arm, _, value = key
            path = self.root / "rollouts" / phase / task / unit_name(arm, value) / f"s{seed}-noise{n}.npz"
            if path.exists():  # a rollout of a unit that stopped before its record: never overwritten
                path = _fresh(path)
            canonical.save(res, path)
            a = auditor.audit(path, replay=False)  # verify_phase re-audits every rollout with the replay
            out.setdefault(key, {})[str(n)] = {"R": float(res["R"]), "rollout": str(path.relative_to(self.root)),
                                               "rollout_sha256": sha(path), "failed_episodes": a["failed_episodes"],
                                               "audit_ok": a["ok"], "audit_issues": a["issues"]}
        return out

    def unit_report(self, task, arm, seed, cfg, stages):
        if cfg["policy.method"] != "ccil":
            return {}
        return {"label_accuracy": label_stats(stages["labels.generate"] / "labels" / "candidates.npz", task, seed)}

    def versions(self) -> dict:
        import scipy
        return {"suite": VERSION, "rollout": "vector_env.pendulum/1", "scipy": scipy.__version__}

    # records.verify_phase hooks
    @staticmethod
    def audit(path):
        """The #24 audit, with the replay through the batched integrator that produced the rollout: it agrees with
        the canonical integrator to a few ulps (tools/vector_env), so a bit-for-bit replay needs its own step()."""
        a = auditor.audit(path, replay=False)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ("states", "actions", "flipped")}
        bad = vector.replay_mismatches(arrays, a["task"])
        if bad:
            a["issues"].append(f"{bad} transitions do not replay through the batched integrator")
        return {**a, "ok": a["ok"] and not bad, "replayed": "vector_env.pendulum.step"}

    def expected_inputs(self, record):
        frac = self.study.arms[record["arm"]].data_fraction
        if frac:
            _, digest = s24.subset(self.root, record["task"], record["seed"], frac)
            return {**self.inputs(record["task"]), "data_sha256": digest}
        return self.inputs(record["task"])


def label_stats(cands_path, task, seed) -> dict:
    """The #24 label accuracy (tools/m3_pendulum/study.py _label_stats), with the true one-step error of the
    sampled labels computed by one batched step of the evaluation integrator (tools/vector_env) rather than one
    scipy call per label. The two integrators agree to a few ulps; the same rows are sampled."""
    from ccil.schema.io import read_candidates
    c = read_candidates(cands_path)
    acc = np.asarray(c.accepted, bool)
    q = (0.05, 0.25, 0.5, 0.75, 0.95)
    out = {"candidates": int(len(acc)), "accepted": int(acc.sum()), "acceptance_rate": float(acc.mean()),
           "rejections_by_reason": {code: int(np.asarray(c.reasons)[:, i].sum()) for i, code in enumerate(c.reason_codes)}}
    for key, label in (("generator.state_displacement_norm", "state_displacement"),
                       ("generator.action_noise_norm", "action_displacement")):
        if key in c.measurements:
            v = np.asarray(c.measurements[key], np.float64)[acc]
            out[label] = {"quantiles": dict(zip(map(str, q), np.quantile(v, q).tolist())) if v.size else None}
    rows = np.flatnonzero(acc)
    if rows.size:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
        rows = np.sort(rng.choice(rows, size=min(s24.LABEL_SAMPLE, rows.size), replace=False))
        z, a, y = c.observations["state"][rows], c.actions[rows], c.target_observations["state"][rows]
        nxt, _ = vector.step(np.asarray(z, np.float64), np.asarray(a[:, 0], np.float64), canonical.TASKS[task])
        err = np.linalg.norm(nxt - y, axis=1)
        out["true_label_error"] = {"sample": int(len(rows)), "sample_rule": "SeedSequence([seed, 10]) choice of accepted rows",
                                   "mean": float(err.mean()), "max": float(err.max()),
                                   "quantiles": dict(zip(map(str, q), np.quantile(err, q).tolist())),
                                   "definition": "||env.step(z_G, a_G) - y||_2 in raw state units, one step "
                                                 "(the batched evaluation integrator)"}
    return out


def _zero(obs):
    return np.zeros((len(obs), 1))


def _fresh(path: Path) -> Path:
    i = 2
    while path.with_name(f"{path.stem}-attempt{i}.npz").exists():
        i += 1
    return path.with_name(f"{path.stem}-attempt{i}.npz")
