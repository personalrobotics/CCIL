#!/usr/bin/env python3
"""The version 6 analysis (predeclaration sections 8-9; #113, #25): decisions, intervals and the report tables.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/m3_experiments/analyze_v6.py --root DIR --out ANALYSIS.json [--smoke]

It refuses unless the code-state gate passes, both phases verify (every rollout re-audited), the recorded tuning
choice equals its recomputation from the verified tuning records, every study record names that choice, and the
deferred tuning returns verify. The tables use the audits' recomputed R, never a stored one. Then:
- primary, per task: CCIL - BC at c = 0.975 (two tasks, Bonferroni), the union of BCa and paired-t intervals,
  the positive control (BC - BC on 10%), the section 9 inconclusive rules, and the relative effect;
- secondary, descriptive, at c = 0.95: the version 4 comparisons; CCIL - BC per noise level and data size; the
  modern arms (CCIL-modern - BC-modern, BC-modern - BC, CCIL-modern - CCIL, noise-free and per noise level); and
  every arm against the zero-action reference;
- the tuning tables, label accuracy, dynamics diagnostics, failures and exclusions, and the paper comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ccil.experiments import analysis, gate, records  # noqa: E402
from run_study import REPO, _freeze, _study  # noqa: E402

VERSION = "m3_experiments.analysis_v6/1"
PRIMARY = analysis.Decision(treatment="ccil", baseline="bc", control=("bc", "bc_frac10"), confidence=0.975,
                            min_valid=16, max_excluded=2,
                            outcomes=("ccil_improves_on_bc", "ccil_is_worse_than_bc", "no_detectable_difference"))
SECONDARY_C = 0.95
SECONDARY = [("ccil", "noise_bc"), ("ccil_tuned", "bc"), ("ccil_tuned", "noise_bc"), ("ccil_filtered", "ccil"),
             ("ccil_nolip", "ccil"), ("ccil_modern", "bc_modern"), ("bc_modern", "bc"), ("ccil_modern", "ccil")]
NOISY_ARMS = ("bc", "noise_bc", "ccil", "ccil_tuned", "bc_modern", "ccil_modern")
PAPER = {"continuous": {"bc": -3244, "ccil": -2269}, "discontinuous": {"bc": -3204, "ccil": -2556}}
NOISE_FREE = "0.0"


def load(root: Path, study, smoke: bool, jobs: int) -> tuple[dict, dict]:
    state = gate.require(REPO, _freeze(smoke), real=not smoke)
    audit, suite_inputs = _hooks(root, study, smoke)
    tuning = records.verify_phase(root, study, "tuning", smoke=smoke, audit=audit, expected_inputs=suite_inputs,
                                  jobs=jobs)
    choice_path = root / records.CHOICE
    recorded = json.loads(choice_path.read_text())["choice"]
    recomputed = json.loads(json.dumps(records.compute_choice(study, tuning, acceptance_arms=("ccil_tuned",)),
                                       default=str))
    if recorded != recomputed:
        raise SystemExit("analysis refused: the tuning choice differs from its recomputation from the verified records")
    choice_sha = records.sha(choice_path)
    deferred = records.verify_deferred(root, study, "tuning", smoke=smoke, audit=audit)
    study_records = records.verify_phase(root, study, "study", smoke=smoke, audit=audit, expected_inputs=suite_inputs,
                                         choice=recorded, jobs=jobs)
    if {r.get("tuning_choice_sha256") for r in study_records} != {choice_sha}:
        raise SystemExit("analysis refused: a study record does not name the recorded tuning choice")
    tables = {"R": {}, "status": {}}
    for r in study_records:
        t, a, s = r["task"], r["arm"], r["seed"]
        tables["status"].setdefault(t, {}).setdefault(a, {})[s] = r["status"]
        for n in study.arms[a].noise_levels:
            tables["R"].setdefault(t, {}).setdefault(a, {}).setdefault(str(n), {})[s] = r["verified"].get(str(n))
    deferred_R = {f"{r['task']}/{r['arm']}@{r['value']!r}/s{r['seed']}": r["verified"] for r in deferred}
    extras = {"smoke": smoke, "gate": state, "code_commit": study_records[0]["gate"]["head"],
              "tooling_commit": study_records[0]["gate"]["tooling"].get("tooling_commit"),
              "rollouts_reaudited": sum(len(r["verified"]) for r in tuning + deferred + study_records),
              "tuning_choice": recorded, "tuning_choice_sha256": choice_sha,
              "modern_tuning_returns_after_the_choice": deferred_R,
              "label_accuracy": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["label_accuracy"]
                                 for r in study_records if "label_accuracy" in r},
              "dynamics_held_out": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["dynamics_held_out"]
                                    for r in study_records if r.get("dynamics_held_out")},
              "validation_selection": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["training"]["validation"]
                                       for r in study_records if (r.get("training") or {}).get("validation")},
              "failures": [{k: r.get(k) for k in ("task", "arm", "seed", "status", "failure", "failures")}
                           for r in study_records if r["status"] != "ok"],
              "versions": sorted({json.dumps(r["versions"], sort_keys=True) for r in study_records})}
    return tables, extras


def _hooks(root: Path, study, smoke: bool):
    from pendulum_suite import PendulumSuite
    suite = PendulumSuite(root, study, smoke=smoke)
    return PendulumSuite.audit, suite.expected_inputs


def analyze(tables: dict) -> dict:
    out = {"version": VERSION, "rng_seed": analysis.RNG_SEED, "resamples": analysis.RESAMPLES,
           "scipy": scipy.__version__, "numpy": np.__version__, "tasks": {}}
    for task, arms in tables["R"].items():
        compare = lambda a, b, n=NOISE_FREE: analysis.compare(tables, task, a, b, SECONDARY_C, n)  # noqa: E731
        t = {"primary": analysis.decide(tables, task, PRIMARY)}
        t["secondary"] = [compare(a, b) for a, b in SECONDARY]
        t["by_noise"] = [compare(a, b, n) for a, b in (("ccil", "bc"), ("ccil_modern", "bc_modern"))
                         for n in arms[a] if n != NOISE_FREE]
        t["by_data_size"] = [compare(f"ccil_frac{p}", f"bc_frac{p}") for p in (10, 25, 50)]
        t["zero_reference"] = ([compare(a, "zero") for a in arms if a != "zero"]
                               + [compare(a, "zero", n) for a in NOISY_ARMS for n in arms["zero"] if n != NOISE_FREE])
        t["arms"] = {a: {n: _summary([r for s, r in rs.items() if tables["status"][task][a][s] == "ok"], rs)
                         for n, rs in by_noise.items()} for a, by_noise in arms.items()}
        t["status"] = {a: analysis.counts(tables, task, a) for a in arms}
        t["paper_comparison"] = {a: {"paper": PAPER[task][a], "this_study": t["arms"][a][NOISE_FREE]}
                                 for a in ("bc", "ccil")}
        out["tasks"][task] = t
    return out


def _summary(ok: list, per_seed: dict) -> dict:
    return {"mean": float(np.mean(ok)) if ok else None, "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else None,
            "n": len(ok), "per_seed": per_seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=HERE / "pendulum_v6.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; analyses are never overwritten")
    root = a.root.resolve()
    tables, extras = load(root, _study(a.plan, a.smoke), a.smoke, a.jobs)
    result = {**analyze(tables), **extras}
    records.atomic_json(result, a.out)
    for task, t in result["tasks"].items():
        p = t["primary"]
        print(f"{task}: {p['outcome']} (CCIL - BC interval {p['comparison']['interval']}, n {p['comparison']['n']}); "
              f"positive control {'passes' if p['positive_control']['passes'] else 'fails'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
