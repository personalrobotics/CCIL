#!/usr/bin/env python3
"""Run drivers of the #24 study: tuning, the tuning choice, and the study runs.

    P="uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt python tools/m3_pendulum/study.py"
    $P prepare --root ROOT                     # convert the retained inputs into ROOT/inputs
    $P sweep --root ROOT --phase tuning --jobs N --device cuda
    $P choose --root ROOT                      # writes ROOT/tuning-choice.json, once
    $P sweep --root ROOT --phase study --jobs N --device cuda
    $P run --root ROOT --phase P --task T --arm A --seed S [--value V]   # one unit

Every unit is one (task, arm, seed[, value]): the full pipeline through
ccil policy train (data.prepare, transition.fit, labels.generate,
[labels.certify,] policy.train), then the evaluation of the trained policy
with env.rollout at every predeclared noise level, each rollout saved and
audited. Its record, ROOT/results/<phase>/<task>/<arm>/s<seed>.json, is
written once and never overwritten; a unit whose record exists is skipped.

Before any work, a unit passes the code-state gate (provenance.py), and the
record carries the gate's result, the resolved configuration, every stage
directory, the checkpoint SHA-256, the package versions and the GPU.

Failures (predeclaration §9):
- infrastructure: the training process was killed by a signal, or ran out
  of memory. The fault is recorded (s<seed>.fault<k>.json) and the unit is
  rerun once, from a fresh stage store; a second fault records the unit as
  infrastructure_fault, and it is excluded;
- algorithmic: any other non-zero exit of ccil policy train, a non-finite
  training loss, or a non-finite return. The record is the result.

An invalid configuration is a tooling defect: the unit stops without a
record.

--smoke runs the same code at 1 dynamics epoch and 1 policy epoch (20 for
discontinuous CCIL arms, whose balanced epochs are divided by 20) on seeds
9001 (tuning) and 9002-9003 (study), outside the predeclared seeds, to
exercise the drivers and the analysis; its records say smoke and the
analysis never mixes them with real runs.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "legacy_conversion"))
import audit as auditor  # noqa: E402
import env as pendulum  # noqa: E402
import provenance  # noqa: E402

VERSION = "m3_pendulum.study/1"
DATASETS = {"continuous": "pendulum_cont_100", "discontinuous": "pendulum_disc_500"}
CONFIGS = {"continuous": "pendulum_cont.corrected.yaml", "discontinuous": "pendulum_disc.corrected.yaml"}
PAPER_ACTION_NOISE = {"continuous": 1e-5, "discontinuous": 1e-4}
TUNING_SEEDS, STUDY_SEEDS = (1, 2, 3), tuple(range(101, 121))
SMOKE_TUNING_SEEDS, SMOKE_STUDY_SEEDS = (9001,), (9002, 9003)
NOISE_BC_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 3e-1)
CCIL_TUNED_GRID = {t: (PAPER_ACTION_NOISE[t], 1e-3, 1e-2, 1e-1, 3e-1) for t in DATASETS}
FRACTIONS = (0.1, 0.25, 0.5)
SELECTION_QUANTILE = 0.3
LABEL_SAMPLE = 10000  # accepted labels per CCIL unit whose true one-step error is measured
NOISY = (0.0, *pendulum.NOISE_LEVELS)

# The discontinuous paper configuration balances noisy labels (legacy aug.balance_data), which applies to CCIL only;
# legacy BC and NoiseBC trained on the demonstrations as they are.
NO_BALANCE = "policy.balance_noisy_labels=false"

# arm: (ccil policy train settings, evaluated noise levels, data fraction, value key)
ARMS = {
    "bc": (["policy.method=bc", NO_BALANCE], NOISY, None, None),
    "ccil": (["policy.method=ccil"], NOISY, None, None),
    "noise_bc": (["policy.method=noise_bc", "policy.noise_model=symmetric_uniform_half_batch", NO_BALANCE], NOISY, None,
                 "policy.noise_scale"),
    "ccil_filtered": (["policy.method=ccil", "certificate.profile=disturbed_action_paper_proxy",
                       f"certificate.selection_quantile={SELECTION_QUANTILE}"], (0.0,), None, None),
    "ccil_nolip": (["policy.method=ccil", "transition_model.lipschitz_type=none"], (0.0,), None, None),
    "ccil_tuned": (["policy.method=ccil"], NOISY, None, "generation.action_noise_std"),
    **{f"bc_frac{int(f * 100)}": (["policy.method=bc", NO_BALANCE], (0.0,), f, None) for f in FRACTIONS},
    **{f"ccil_frac{int(f * 100)}": (["policy.method=ccil"], (0.0,), f, None) for f in FRACTIONS},
    "zero": (None, NOISY, None, None),  # the zero-action reference (v3): no training
}
CCIL_ARMS = {a for a, (sets, *_) in ARMS.items() if sets and "policy.method=ccil" in sets}
# labels.certify of the paper-proxy profile scores every accepted label at once: about 60 GB of GPU memory for the
# discontinuous task's 2.9M labels (measured on the study host), so at most one such unit trains at a time.
EXCLUSIVE = {"ccil_filtered"}
TUNED = {"noise_bc": "policy.noise_scale", "ccil_tuned": "generation.action_noise_std"}


def grid(arm, task):
    return NOISE_BC_GRID if arm == "noise_bc" else CCIL_TUNED_GRID[task]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def unit_name(arm, value=None) -> str:
    return arm if value is None else f"{arm}@{value!r}"


def result_path(root, phase, task, arm, seed, value=None) -> Path:
    return Path(root) / "results" / phase / task / unit_name(arm, value) / f"s{seed}.json"


# --- inputs ---------------------------------------------------------------------------------------------------------
def prepare(root: Path) -> dict:
    """Convert the retained inputs (#23) into root/inputs and check them against the committed conversion."""
    from convert_pendulum import convert_pendulum
    inputs = root / "inputs"
    if (inputs / "conversion.json").exists():
        summary = json.loads((inputs / "conversion.json").read_text())
    else:
        summary = convert_pendulum(inputs)
    committed = json.loads((REPO / "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json").read_text())
    for name in ("pendulum_cont", "pendulum_disc"):
        got = sha(inputs / summary["retained"][name]["dataset"]["path"])
        if got != committed["retained"][name]["dataset"]["sha256"]:
            raise SystemExit(f"{name}: converted dataset {got} differs from the committed conversion")
    return summary


def subset(root: Path, task: str, seed: int, fraction: float) -> Path:
    """The nested episode subset of §7: the first round(fraction * N) episodes of SeedSequence([seed, 9])'s permutation."""
    from ccil.schema import read_trajectories, write_trajectories
    from ccil.schema.trajectories import TrajectoryDataset
    path = root / "inputs" / "subsets" / f"{DATASETS[task]}-s{seed}-f{fraction}.npz"
    if path.exists():
        return path
    ds = read_trajectories(root / "inputs" / "data" / f"{DATASETS[task]}.npz")
    order = np.random.default_rng(np.random.SeedSequence([seed, 9])).permutation(len(ds.episodes))
    keep = order[:round(fraction * len(ds.episodes))]
    sub = TrajectoryDataset([ds.episodes[i] for i in keep], {**ds.meta, "subset": {
        "of": f"{DATASETS[task]}.npz", "fraction": fraction, "seed": seed, "episodes": [int(i) for i in keep],
        "rule": "first round(fraction * N) of default_rng(SeedSequence([seed, 9])).permutation(N)"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    write_trajectories(sub, tmp)
    os.replace(tmp, path)
    return path


# --- one unit -------------------------------------------------------------------------------------------------------
def _gpu() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


def _versions() -> dict:
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "gpu": _gpu()}


def _label_stats(cands_path, task, seed) -> dict:
    """§7 label accuracy for the CCIL labels of one unit."""
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
        rows = np.sort(rng.choice(rows, size=min(LABEL_SAMPLE, rows.size), replace=False))
        z, a, y = c.observations["state"][rows], c.actions[rows], c.target_observations["state"][rows]
        err = np.array([np.linalg.norm(pendulum.step(np.asarray(z[i], np.float64), float(a[i, 0]), pendulum.TASKS[task])[0]
                                       - y[i]) for i in range(len(rows))])
        out["true_label_error"] = {"sample": int(len(rows)), "sample_rule": "SeedSequence([seed, 10]) choice of accepted rows",
                                   "mean": float(err.mean()), "max": float(err.max()),
                                   "quantiles": dict(zip(map(str, q), np.quantile(err, q).tolist())),
                                   "definition": "||env.step(z_G, a_G) - y||_2 in raw state units, one step"}
    return out


def _stages(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("wrote", "reused"):
            out[parts[0]] = parts[2]
    return out


@contextlib.contextmanager
def _exclusive(root):
    """A cross-process lock: one EXCLUSIVE unit's training at a time under root."""
    with open(Path(root) / "exclusive.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _classify(proc, stderr: str) -> str | None:
    if proc.returncode == 0:
        return None
    if stderr.startswith("invalid configuration"):  # a tooling defect, not a result: stop without a record
        raise SystemExit(f"ccil policy train refused the configuration:\n{stderr}")
    if proc.returncode < 0 or "out of memory" in stderr.lower() or "MemoryError" in stderr:
        return "infrastructure_fault"
    return "algorithmic_failure"


def run_unit(root, phase, task, arm, seed, value=None, *, device="cuda", smoke=False, attempt=1) -> dict:
    root = Path(root).resolve()
    final = result_path(root, phase, task, arm, seed, value)
    if final.exists():
        return json.loads(final.read_text())
    state = provenance.tree_state(REPO)
    if not state["ok"]:
        raise SystemExit(f"code-state gate refused: {state}")
    sets, noise_levels, fraction, value_key = ARMS[arm]
    if (value is None) != (value_key is None):
        raise SystemExit(f"{arm} takes {'a value' if value_key else 'no value'}")
    record = {"version": VERSION, "phase": phase, "task": task, "arm": arm, "seed": seed, "value": value,
              "value_key": value_key, "smoke": smoke, "attempt": attempt, "gate": state, "versions": _versions(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if phase == "study":
        choice = root / "tuning-choice.json"
        if not choice.exists():
            raise SystemExit("the study needs tuning-choice.json (study.py choose) before any study seed runs")
        record["tuning_choice_sha256"] = sha(choice)
    unit_dir = root / "units" / phase / task / unit_name(arm, value) / f"s{seed}"
    policy = None
    if sets is not None:
        data = subset(root, task, seed, fraction) if fraction else root / "inputs" / "data" / f"{DATASETS[task]}.npz"
        store = root / "stages" / phase / task / f"s{seed}" if attempt == 1 else unit_dir / f"stages-attempt{attempt}"
        overrides = [f"run.seed={seed}", f"compute.device={device}", f"data.path={data}", f"run.output_dir={store}",
                     "policy.exclude_validation_episodes=false", *sets]
        if value_key:
            overrides.append(f"{value_key}={value!r}")
        if smoke:  # one dynamics epoch, one effective policy epoch (discontinuous CCIL is balanced: epochs / 20)
            balanced = arm in CCIL_ARMS and task == "discontinuous"
            overrides += ["transition_model.epochs=1", f"policy.epochs={20 if balanced else 1}"]
        cmd = [sys.executable, "-m", "ccil", "policy", "train", "--config", str(root / "inputs" / "config" / CONFIGS[task])]
        for s in overrides:
            cmd += ["--set", s]
        unit_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with _exclusive(root) if arm in EXCLUSIVE else contextlib.nullcontext():
            proc = subprocess.run(cmd, cwd=root / "inputs", capture_output=True, text=True)
        (unit_dir / f"train-attempt{attempt}.log").write_text(f"$ {' '.join(cmd)}\n{proc.stdout}\n--- stderr\n{proc.stderr}")
        record.update(command=cmd, overrides=overrides, exit=proc.returncode, train_seconds=round(time.time() - t0, 1),
                      stages=_stages(proc.stdout), stderr_tail=proc.stderr[-2000:])
        fault = _classify(proc, proc.stderr)
        if fault == "infrastructure_fault":
            record["status"] = fault
            atomic_json(record, final.with_name(f"s{seed}.fault{attempt}.json"))
            if attempt == 1:
                return run_unit(root, phase, task, arm, seed, value, device=device, smoke=smoke, attempt=2)
            atomic_json(record, final)
            return record
        if fault:
            record["status"] = fault
            atomic_json(record, final)
            return record
        st = {k: Path(v) for k, v in record["stages"].items()}
        record["resolved_config"] = json.loads((st["policy.train"] / "config.json").read_text())
        ckpt = st["policy.train"] / "checkpoints" / "policy.npz"
        record["checkpoint_sha256"] = sha(ckpt)
        training = json.loads((st["policy.train"] / "metrics.json").read_text())["values"]["training"]
        losses = [e["mean_loss"] for e in training["epochs"]]
        record["training"] = {"epochs_run": training["epochs_run"], "rows": training["rows"],
                              "row_sources": training["row_sources"], "mean_loss": losses}
        record["dynamics_held_out"] = json.loads((st["transition.fit"] / "metrics.json").read_text())["values"]["held_out"]
        record["labels_report"] = json.loads((st["labels.generate"] / "metrics.json").read_text())["values"]
        if "labels.certify" in st:
            record["certify_report"] = json.loads((st["labels.certify"] / "metrics.json").read_text())["values"]
        record["supervision"] = json.loads((st["policy.train"] / "supervision.json").read_text())
        if arm in CCIL_ARMS:
            record["label_accuracy"] = _label_stats(st["labels.generate"] / "labels" / "candidates.npz", task, seed)
        if not all(x is not None and np.isfinite(x) for x in losses):
            record["status"] = "algorithmic_failure"
            record["failure"] = "non-finite training loss"
            atomic_json(record, final)
            return record
        from ccil.training.vector_bc import VectorBCPolicy
        net, _, _ = VectorBCPolicy.load(ckpt, device=device)
        policy = net.predict
    else:
        def policy(obs):
            return np.zeros((len(obs), 1))
    evaluations = {}
    for n in noise_levels:
        path = root / "rollouts" / phase / task / unit_name(arm, value) / f"s{seed}-noise{n}.npz"
        if attempt > 1:
            path = path.with_name(f"s{seed}-noise{n}-attempt{attempt}.npz")
        res = pendulum.rollout(policy, task, seed, n)
        pendulum.save(res, path)
        a = auditor.audit(path, replay=False)  # the analysis re-audits every rollout with the replay
        evaluations[str(n)] = {"R": float(res["R"]), "rollout": str(path.relative_to(root)), "rollout_sha256": sha(path),
                               "failed_episodes": a["failed_episodes"], "audit_ok": a["ok"], "audit_issues": a["issues"]}
    record["evaluations"] = evaluations
    if not all(e["audit_ok"] for e in evaluations.values()):
        raise SystemExit(f"audit refused a rollout of {final}: {evaluations}")
    finite = all(np.isfinite(e["R"]) for e in evaluations.values())
    record["status"] = "ok" if finite else "algorithmic_failure"
    if not finite:
        record["failure"] = "non-finite return"
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(record, final)
    return record


# --- phases ---------------------------------------------------------------------------------------------------------
def units(root, phase, smoke) -> list[tuple]:
    """Every (task, arm, seed, value) of a phase, in a fixed order."""
    out = []
    if phase == "tuning":
        for task in DATASETS:
            for seed in SMOKE_TUNING_SEEDS if smoke else TUNING_SEEDS:
                out += [(task, arm, seed, v) for arm in TUNED for v in grid(arm, task)]
        return out
    choice = json.loads((Path(root) / "tuning-choice.json").read_text())
    for task in DATASETS:
        for seed in SMOKE_STUDY_SEEDS if smoke else STUDY_SEEDS:
            for arm, (_, _, _, value_key) in ARMS.items():
                out.append((task, arm, seed, choice[task][arm]["value"] if value_key else None))
    return out


def _group(root, phase, items, device, smoke):
    return [(it, run_unit(root, phase, *it, device=device, smoke=smoke)["status"]) for it in items]


def sweep(root, phase, jobs, device, smoke) -> int:
    root = Path(root).resolve()
    meta = root / "smoke.json"
    if smoke != meta.exists() and any((root / "results").glob("*")):
        raise SystemExit("this root holds runs of the other kind (smoke or real); use a separate root")
    if smoke:
        atomic_json({"smoke": True}, meta)
    prepare(root)
    groups = {}
    for it in units(root, phase, smoke):
        groups.setdefault((it[0], it[2]), []).append(it)  # one (task, seed) per worker: stages are shared within it
    statuses = []
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:  # CUDA-safe workers
        futures = [pool.submit(_group, root, phase, items, device, smoke) for items in groups.values()]
        for f in as_completed(futures):
            for it, status in f.result():
                statuses.append(status)
                print(f"{status:22s} {phase} {it}", flush=True)
    print(f"{len(statuses)} units: " + ", ".join(f"{s} {statuses.count(s)}" for s in sorted(set(statuses))))
    return 0


def choose(root, smoke) -> dict:
    """§6 selection on the tuning seeds: the eligible value with the highest mean noise-free R; ties to the smaller."""
    root = Path(root).resolve()
    path = root / "tuning-choice.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the tuning choice is made once")
    seeds = SMOKE_TUNING_SEEDS if smoke else TUNING_SEEDS
    choice = {"version": VERSION, "seeds": list(seeds), "smoke": smoke, "gate": provenance.tree_state(REPO)}
    for task in DATASETS:
        choice[task] = {}
        for arm in TUNED:
            table, eligible = {}, {}
            for v in grid(arm, task):
                recs = [json.loads(result_path(root, "tuning", task, arm, s, v).read_text()) for s in seeds]
                ok = all(r["status"] == "ok" for r in recs)
                table[repr(v)] = {"R": {r["seed"]: r["evaluations"]["0.0"]["R"] if r["status"] == "ok" else None for r in recs},
                                  "status": {r["seed"]: r["status"] for r in recs}, "eligible": ok}
                if ok:
                    eligible[v] = float(np.mean([r["evaluations"]["0.0"]["R"] for r in recs]))
                    table[repr(v)]["mean_R"] = eligible[v]
            if eligible:
                best = max(eligible.values())
                value, rule = min(v for v, m in eligible.items() if m == best), "highest mean R; ties to the smaller value"
            else:
                value, rule = grid(arm, task)[0], "every value ineligible: the paper's value"
            choice[task][arm] = {"value": value, "rule": rule, "table": table}
    atomic_json(choice, path)
    return choice


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "sweep", "choose", "run"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--smoke", action="store_true")
        if name in ("sweep", "run"):
            p.add_argument("--phase", choices=("tuning", "study"), required=True)
            p.add_argument("--device", default="cuda")
        if name == "sweep":
            p.add_argument("--jobs", type=int, default=8)
        if name == "run":
            p.add_argument("--task", choices=tuple(DATASETS), required=True)
            p.add_argument("--arm", choices=tuple(ARMS), required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--value", type=float)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.cmd == "prepare":
        prepare(root)
        return 0
    if a.cmd == "sweep":
        return sweep(root, a.phase, a.jobs, a.device, a.smoke)
    if a.cmd == "choose":
        c = choose(root, a.smoke)
        for task in DATASETS:
            print(task, {arm: c[task][arm]["value"] for arm in TUNED})
        return 0
    prepare(root)
    r = run_unit(root, a.phase, a.task, a.arm, a.seed, a.value, device=a.device, smoke=a.smoke)
    print(r["status"], {k: v["R"] for k, v in r.get("evaluations", {}).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
