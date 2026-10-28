#!/usr/bin/env python3
"""The Pendulum study driver (#112, #113): the runner, behind the code-state gate, step by step.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/m3_experiments/run_study.py STEP --root DIR [--plan PLAN] [--shard i/n] [--device cuda] [--smoke]

Steps, in order (each refuses to run out of order):
- tuning: the tuning phase (one process per shard; shards split by seed). Validation-tuned arms record no return.
- choose: verify every tuning record, then write the choice once (DIR/tuning-choice.json).
- returns: the deferred tuning returns of the validation-tuned arms, computed after the choice and naming it.
- study: the study phase at the chosen values (one process per shard).
- verify: verify the study records and the deferred returns; prints counts only, never a return.

Real runs need the frozen tooling (tools/m3_experiments/freeze.json, predeclaration section 11): a clean tree whose
ccil paths equal the frozen ccil commit and whose tooling paths equal the frozen tooling commit. --smoke runs
the same code at one epoch on smoke seeds before the freeze; its records say smoke and are never analyzed with
real ones. --preflight (#134) runs the exact schedule (the plan's seeds, the same shards) with every training
budget that is not compiled into a step cut to its minimum (one epoch; a step budget of the warm-up plus one), so
its records show every group's compilation as the real run will meet it; its records say smoke too.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

FREEZE = "tools/m3_experiments/freeze.json"
TOOLING_PATHS = ("tools/m3_experiments", "tools/m3_pendulum", "tools/vector_env", "tools/legacy_conversion",
                 "correct_il", "data", "config", "docs/artifacts/manifest.json",
                 "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json")
SMOKE_SEEDS = {"tuning": [9101], "study": [9102, 9103]}


def _freeze(smoke: bool):
    from ccil.experiments.gate import Freeze
    path = REPO / FREEZE
    if path.exists():
        commit = json.loads(path.read_text())["ccil_commit"]
    elif smoke:  # before the freeze: the smoke run records the current commit as its code state
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    else:
        raise SystemExit(f"{FREEZE} is missing: real runs need the frozen tooling (predeclaration section 11)")
    return Freeze(ccil_commit=commit, tooling_paths=TOOLING_PATHS, manifest=FREEZE)


def _study(plan_path: Path, smoke: bool):
    import yaml

    from ccil.experiments import plan
    spec = yaml.safe_load(plan_path.read_text())
    if smoke:  # the same design on smoke seeds
        spec["phases"]["tuning"]["seeds"] = SMOKE_SEEDS["tuning"]
        spec["phases"]["study"]["seeds"] = SMOKE_SEEDS["study"]
    return plan.from_dict(spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("tuning", "choose", "returns", "study", "verify"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=HERE / "pendulum_v6.yaml")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    smoke_seeds, a.smoke = a.smoke, a.smoke or a.preflight

    from ccil.experiments import gate, records, runner
    from pendulum_suite import PendulumSuite
    root = a.root.resolve()
    study = _study(a.plan, smoke_seeds)
    freeze = _freeze(a.smoke)
    state = gate.require(REPO, freeze, real=not a.smoke)
    suite = PendulumSuite(root, study, smoke=a.smoke, preflight=a.preflight)

    def recheck():  # the gate again before records are written: one clean commit at both boundaries (#127)
        now = gate.require(REPO, freeze, real=not a.smoke)
        if now["head"] != state["head"]:
            raise gate.GateRefused(f"code-state gate refused: HEAD moved from {state['head']} to {now['head']}")
    shard = tuple(map(int, a.shard.split("/")))

    def log(m):
        print(time.strftime("%H:%M:%S"), m, flush=True)

    def verify(phase):
        return records.verify_phase(root, study, phase, smoke=a.smoke, audit=PendulumSuite.audit,
                                    expected_inputs=suite.expected_inputs,
                                    choice=_choice(root) if phase != "tuning" else None)

    if a.step == "tuning":
        out = runner.run_phase(root, study, "tuning", suite, gate=state, smoke=a.smoke, device=a.device, shard=shard,
                               workers=a.workers, recheck=recheck, log=log)
    elif a.step == "choose":
        if (root / records.CHOICE).exists():
            raise SystemExit(f"{root / records.CHOICE} exists; the choice is written once")
        sha = records.record_choice(root, study, verify("tuning"), acceptance_arms=("ccil_tuned",))
        out = {"choice_sha256": sha, "values": {t: {arm: c["value"] for arm, c in v.items()}
                                                for t, v in _choice(root).items()}}
    elif a.step == "returns":
        n = runner.evaluate_deferred(root, study, "tuning", suite, gate=state, smoke=a.smoke, device=a.device,
                                     recheck=recheck, log=log)
        out = {"written": n, "verified": len(records.verify_deferred(root, study, "tuning", smoke=a.smoke,
                                                                     audit=PendulumSuite.audit))}
    elif a.step == "study":
        if not (root / records.CHOICE).exists():
            raise SystemExit("the study needs the recorded choice: run choose first")
        out = runner.run_phase(root, study, "study", suite, gate=state, smoke=a.smoke, device=a.device, shard=shard,
                               choice=_choice(root), choice_sha256=records.sha(root / records.CHOICE),
                               workers=a.workers, recheck=recheck, log=log)
    else:
        study_records = verify("study")
        deferred = records.verify_deferred(root, study, "tuning", smoke=a.smoke, audit=PendulumSuite.audit)
        statuses: dict = {}
        for r in study_records:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        out = {"study_records": len(study_records), "status": statuses, "deferred_returns": len(deferred)}
    print(json.dumps(out, indent=1, default=str))
    return 0


def _choice(root: Path) -> dict:
    from ccil.experiments.records import CHOICE
    return json.loads((root / CHOICE).read_text())["choice"]


if __name__ == "__main__":
    sys.exit(main())
