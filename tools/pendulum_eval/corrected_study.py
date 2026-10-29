#!/usr/bin/env python3
"""The corrected-demonstrator study (docs/experiments/pendulum-corrected-demonstrator-predeclaration.md).

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/corrected_study.py"
    $R preflight --root NEW_ROOT --data CANONICAL_DIR --v6 V6_ROOT   # stage identities before any training
    $R train     --root ROOT --data CANONICAL_DIR                    # BC and CCIL, v6's runner and stages
    $R audit     --root ROOT --v6 V6_ROOT                            # provenance after training
    $R evaluate  --root ROOT --out NEW_DIR                           # fresh starts, the declared analyses

The training is v6's: the same ccil source (the freeze's ccil commit), runner, stages, shared settings and BC and
CCIL arm settings (corrected_study.yaml). The only change is the dataset. CorrectedSuite places the canonical bytes
(docs/fixtures/evidence/pendulum-corrected-demonstrator/canonical-datasets.json, verified by sha256) where v6's suite
reads each task's dataset, so v6's configuration paths are unchanged; identity is by hash, not by file name. The
task configurations must still equal the committed conversion's.

Real runs require the corrected freeze (corrected_freeze.json): ccil paths equal to v6's ccil commit, and these
tooling paths equal to the commit it names, on a clean tree. Each report goes to the evidence folder, so it must be
committed before the next gated step (train) can pass the gate. Every step fails closed (exit 1) on a failed check,
and each report records the commit the process ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (HERE, REPO / "tools" / "m3_experiments", REPO / "tools" / "m3_pendulum", REPO / "tools" / "vector_env",
          REPO / "tools" / "legacy_conversion"):
    sys.path.insert(0, str(p))

EVIDENCE = REPO / "docs" / "fixtures" / "evidence" / "pendulum-corrected-demonstrator"
CANONICAL = EVIDENCE / "canonical-datasets.json"
PLAN = HERE / "corrected_study.yaml"
FREEZE = "tools/pendulum_eval/corrected_freeze.json"
V6_CCIL_COMMIT = "6fbcce543e8a1ebdf243063c179dd3af82d5db8a"
TOOLING_PATHS = ("tools/m3_experiments", "tools/m3_pendulum", "tools/vector_env", "tools/legacy_conversion",
                 "tools/pendulum_eval", "correct_il", "data", "config", "docs/artifacts/manifest.json",
                 "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json", str(CANONICAL.relative_to(REPO)))
C_PRIMARY = 1 - 0.05 / 4  # two tasks x two primary analyses


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def commit() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()}


def canonical() -> dict:
    return json.loads(CANONICAL.read_text())["datasets"]


def study_plan():
    import yaml

    from ccil.experiments import plan
    return plan.from_dict(yaml.safe_load(PLAN.read_text()))


def require_gate(real: bool = True) -> dict:
    from ccil.experiments import gate
    return gate.require(REPO, gate.Freeze(ccil_commit=V6_CCIL_COMMIT, tooling_paths=TOOLING_PATHS, manifest=FREEZE),
                        real=real)


def suite_class():
    import study as s24
    from pendulum_suite import PendulumSuite

    class CorrectedSuite(PendulumSuite):
        """v6's suite with each task's dataset replaced by the canonical corrected bytes."""

        def __init__(self, root, study, data: Path):
            self.root, self.study, self.smoke, self.preflight = Path(root).resolve(), study, False, False
            self._inputs = prepare_inputs(self.root, data, s24)

    return CorrectedSuite


def prepare_inputs(root: Path, data: Path, s24) -> dict:
    """root/inputs as v6's suite reads it: the committed conversion's configurations, and each task's dataset path
    holding the canonical corrected bytes. Verified on every call; refuses any other bytes."""
    from convert_pendulum import convert_pendulum
    inputs = root / "inputs"
    if not (inputs / "conversion.json").exists():
        convert_pendulum(inputs)
    committed = json.loads((REPO / "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json").read_text())
    out = {}
    for task, c in canonical().items():
        dest = inputs / "data" / f"{s24.DATASETS[task]}.npz"
        if sha(dest) != c["sha256"]:
            src = data / c["local_name"]
            if sha(src) != c["sha256"]:
                raise SystemExit(f"{src} is not the canonical {task} dataset {c['sha256']}")
            shutil.copyfile(src, dest)
        config = inputs / "config" / s24.CONFIGS[task]
        want_config = committed["retained"][s24.CONVERTED[task]]["configs"]["corrected"]["sha256"]
        got = {"data": sha(dest), "config": sha(config)}
        if got != {"data": c["sha256"], "config": want_config}:
            raise SystemExit(f"{task}: inputs {got} are not the canonical dataset and committed configuration")
        out[task] = got
    return out


# --- preflight and audit ----------------------------------------------------------------------------------------------
def v6_stages(v6: Path) -> dict:
    """Every v6 study stage directory name, and every v6 data.prepare dataset hash, per task and seed."""
    names, data = set(), {}
    for d in (v6 / "stages" / "study").glob("*/s*/*"):
        names.add(d.name)
        if d.name.startswith("data-prepare-"):
            data.setdefault(d.parts[-3:-1], set()).add(sha(d / "data" / "dataset.npz"))
    return {"names": names, "data": data}


def check_prepared(run_dir: Path, task: str, v6: dict) -> dict:
    got = json.loads((run_dir / "stage.json").read_text())["inputs"].get("dataset")
    art = sha(run_dir / "data" / "dataset.npz")
    return {"run_id": run_dir.name, "input_sha256": got,
            "input_is_canonical": got == canonical()[task]["sha256"],
            "run_id_not_in_v6": run_dir.name not in v6["names"],
            "artifact_not_in_v6": all(art not in s for s in v6["data"].values())}


def preflight(root: Path, data: Path, v6_root: Path, device: str) -> dict:
    from ccil.data.prepare import prepare_stage
    if root.exists():
        raise SystemExit(f"{root} exists: the corrected run root must be new and empty")
    state = require_gate()
    study = study_plan()
    suite = suite_class()(root, study, data)
    v6 = v6_stages(v6_root)
    units = {}
    for task, arm, seed, value in study.units("study"):
        cfg, _ = suite.config("study", task, seed, study.unit_settings(arm, value), None, device,
                              root / "stages" / "study" / task / f"s{seed}")
        units[f"{task}/{arm}/s{seed}"] = check_prepared(prepare_stage(cfg).run_dir, task, v6)
    ok = all(all(v for k, v in u.items() if isinstance(v, bool)) for u in units.values())
    return {"step": "preflight", "gate": state, "units": units, "all_pass": ok}


def audit(root: Path, v6_root: Path) -> dict:
    """After training: every record's dataset is canonical; every data.prepare stage consumed it; no corrected stage
    directory name (which hashes its identity, including every upstream artifact) appears among v6's."""
    v6 = v6_stages(v6_root)
    can = {t: c["sha256"] for t, c in canonical().items()}
    recs = [json.loads(f.read_text()) for f in sorted((root / "results" / "study").glob("*/*/s*.json"))]
    bad = [f"{r['task']}/{r['arm']}/s{r['seed']}" for r in recs
           if r["status"] == "ok" and r["inputs"]["data_sha256"] != can[r["task"]]]
    stages = sorted((root / "stages" / "study").glob("*/s*/*"))
    reused = [d.name for d in stages if d.name in v6["names"]]
    prepared = {f"{d.parts[-3]}/{d.parts[-2]}/{d.name}": check_prepared(d, d.parts[-3], v6)
                for d in stages if d.name.startswith("data-prepare-")}
    ok = (len(recs) == 80 and not bad and not reused
          and all(all(v for v in p.values() if isinstance(v, bool)) for p in prepared.values()))
    return {"step": "audit", "records": len(recs), "statuses": sorted({r["status"] for r in recs}),
            "records_not_canonical": bad, "stages": len(stages), "stage_names_in_v6": reused,
            "data_prepare": prepared, "all_pass": ok}


# --- training -------------------------------------------------------------------------------------------------------
def train(root: Path, data: Path, device: str, workers: int) -> dict:
    import time

    from ccil.experiments import gate, runner
    state = require_gate()
    study = study_plan()
    suite = suite_class()(root, study, data)

    def recheck():
        now = require_gate()
        if now["head"] != state["head"]:
            raise gate.GateRefused(f"code-state gate refused: HEAD moved from {state['head']} to {now['head']}")

    out = runner.run_phase(root, study, "study", suite, gate=state, smoke=False, device=device, workers=workers,
                           recheck=recheck, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    return {"step": "train", "gate": state, "summary": out, "all_pass": True}


# --- evaluation and analysis ------------------------------------------------------------------------------------------
def evaluate(root: Path, out: Path, device: str) -> dict:
    import env as canonical_env
    import expert
    import historical as h

    from ccil.experiments import analysis
    from ccil.training.vector_bc import VectorBCPolicy
    out.mkdir(parents=True)
    hist = json.loads((h.EVIDENCE / "part-b.json").read_text())["per_seed"]
    recs = {(r["task"], r["arm"], r["seed"]): r for r in
            (json.loads(f.read_text()) for f in sorted((root / "results" / "study").glob("*/*/s*.json")))}

    def demonstrator(obs):  # the corrected controller under its action contract: float64 clamp, cast to float32
        return expert.act(obs).astype(np.float32).astype(np.float64).reshape(-1, 1)

    scores, status, rollouts = {}, {}, {}
    for task in h.TASKS:
        items, keys = [], []
        for arm in ("bc", "ccil"):
            for s in h.SEEDS:
                r = recs[(task, arm, s)]
                ckpt = Path(r["stages"]["policy.train"]) / "checkpoints" / "policy.npz"
                if sha(ckpt) != r["checkpoint_sha256"]:
                    raise SystemExit(f"{task}/{arm}/s{s}: the checkpoint does not match its record")
                pol = VectorBCPolicy.load(ckpt, device=device)[0].predict
                conds = [("0.0", 0.0, "none")] + [(f"{ch}/{n}", n, ch) for ch in h.CHANNELS for n in h.LEVELS]
                for key, n, ch in conds:
                    items.append((pol, s + h.FRESH_OFFSET, n, ch))
                    keys.append((task, arm, key, s))
        for s in h.SEEDS:
            items.append((demonstrator, s + h.FRESH_OFFSET, 0.0, "none"))
            keys.append((task, "corrected_demonstrator", "0.0", s))
        for res, (t, arm, key, s) in zip(h.channel_rollouts(items, task), keys):
            path = out / t / arm / f"s{s}-{key.replace('/', '-')}.npz"
            canonical_env.save(res, path)
            rollouts[f"{t}/{arm}/s{s}/{key}"] = sha(path)
            st = status.setdefault(t, {}).setdefault(arm, {})
            if len(res["failed_episodes"]):
                if key == "0.0":
                    st[s] = "algorithmic_failure"
                continue
            if key == "0.0":
                st[s] = recs[(t, arm, s)]["status"] if arm in ("bc", "ccil") else "ok"
            scores[(t, arm, key, s)] = h.seed_scores(res)
    tables = h.tables_from(scores, status)

    def per_seed(t, arm, m, key="0.0"):
        return np.array([scores[(t, arm, key, s)][m] for s in h.SEEDS])

    def hist_seed(t, arm, m):
        return np.array([hist[f"{t}/{arm}/0.0/s{s}"][m] for s in h.SEEDS])

    def outcome(iv):
        if iv is None:
            return "inconclusive"
        return "increases" if iv[0] > 0 else "decreases" if iv[1] < 0 else "no_detectable_difference"

    report = {"label": "corrected-demonstrator study", "provenance": commit(), "device": device, "primary": {},
              "secondary_95": {}, "noise_95": {}}
    for t in h.TASKS:
        if any((t, a, "0.0", s) not in scores or status[t][a][s] != "ok" for a in ("bc", "ccil") for s in h.SEEDS):
            report["primary"][t] = {"outcome": "inconclusive", "reason": "a failed or missing primary unit"}
            continue
        d1 = per_seed(t, "ccil", "upright_occupancy") - per_seed(t, "bc", "upright_occupancy")
        d2 = d1 - (hist_seed(t, "ccil", "upright_occupancy") - hist_seed(t, "bc", "upright_occupancy"))
        r1, r2 = analysis.union_interval(d1, C_PRIMARY), analysis.union_interval(d2, C_PRIMARY)
        report["primary"][t] = {
            "corrected_ccil_minus_bc": {**r1, "outcome": outcome(r1["interval"]),
                                        "per_seed": dict(zip(map(str, h.SEEDS), map(float, d1)))},
            "change_in_ccil_minus_bc": {**r2, "outcome": outcome(r2["interval"]),
                                        "per_seed": dict(zip(map(str, h.SEEDS), map(float, d2))),
                                        "reading": "the effect of replacing the demonstrations on the whole CCIL-"
                                                   "versus-BC pipeline; not the isolated effect of the wrapping defect"}}
        for m in h.METRICS:
            report["secondary_95"][f"{t}/corrected_ccil_minus_bc/{m}"] = analysis.union_interval(
                per_seed(t, "ccil", m) - per_seed(t, "bc", m), 0.95)
            for arm in ("bc", "ccil"):
                report["secondary_95"][f"{t}/{arm}_corrected_minus_historical/{m}"] = analysis.union_interval(
                    per_seed(t, arm, m) - hist_seed(t, arm, m), 0.95)
        for ch in h.CHANNELS:
            for n in h.LEVELS:
                report["noise_95"][f"{t}/ccil-bc/{ch}/{n}"] = analysis.compare(
                    tables["upright_occupancy"], t, "ccil", "bc", 0.95, f"{ch}/{n}")
    report["arm_summaries"] = h.arm_summaries(scores)
    report["per_seed"] = {f"{t}/{a}/{n}/s{s}": {m: v[m] for m in h.METRICS} for (t, a, n, s), v in scores.items()}
    report["rollout_sha256"] = rollouts
    report["all_pass"] = True
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("preflight", "train", "audit", "evaluate"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--v6", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.step == "preflight":
        rep = preflight(root, a.data, a.v6, a.device)
    elif a.step == "train":
        rep = train(root, a.data, a.device, a.workers)
    elif a.step == "audit":
        rep = audit(root, a.v6)
    else:
        rep = evaluate(root, a.out, a.device)
    rep["process"] = commit()
    dest = EVIDENCE / f"study-{a.step}.json"
    dest.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"step": a.step, "pass": rep["all_pass"], "report": str(dest)}, indent=1))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
