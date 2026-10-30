#!/usr/bin/env python3
"""Read-only audit of the demonstration-count study's evaluation report (study-evaluate.json), from its saved rollouts.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/audit_count_evaluate.py --root ROOT --corrected CORRECTED_ROOT --rollouts EVAL_DIR

Nothing is trained or evaluated, and nothing under ROOT, CORRECTED_ROOT or EVAL_DIR is written. As for the corrected
study (audit_corrected_evaluate.py), it:
- checks that the report's rollout keys are exactly the 40 expected ones (bc_100 and ccil_100, noise-free, seeds
  1201-1220), that EVAL_DIR holds exactly those files, and that each file's sha256 matches the report;
- reopens every rollout and validates its metadata and arrays;
- checks the report's unit statuses against the study records in ROOT, and the pinned 500-demonstration report;
- recomputes every per-seed metric, arm summary, primary decision, secondary interval, the label-accuracy comparison
  (from the CCIL records of ROOT and CORRECTED_ROOT) and the continuous comparison, and compares each with the
  committed report exactly.
Writes study-evaluate-audit.json, naming the report's sha256; exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corrected_study as cs  # noqa: E402
import count_study as cnt  # noqa: E402
import historical as h  # noqa: E402
from audit_corrected_evaluate import git, validate  # noqa: E402

from ccil.experiments import analysis  # noqa: E402

REPORT = cnt.SPEC.evidence / "study-evaluate.json"
OUT = cnt.SPEC.evidence / "study-evaluate-audit.json"
T = cnt.TASK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--corrected", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    args = ap.parse_args()
    committed = json.loads(REPORT.read_text())
    seeds = list(cs.STUDY_SEEDS)
    checks = {}
    audit = {"report": REPORT.name, "report_sha256": cs.sha(REPORT),
             "auditor": {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()}}
    checks["report_is_study_run"] = committed["smoke"] is False and committed["gate"]["ok"] and committed["all_pass"]
    checks["pinned_500_report"] = cs.sha(cnt.CORRECTED_EVAL) == cnt.CORRECTED_EVAL_SHA256 == \
        committed["corrected_evaluation"]["sha256"]

    expected = {f"{T}/{a}/s{s}/0.0": (a, s) for a in cnt.ARMS for s in seeds}
    reported = committed["rollout_sha256"]
    files = sorted(str(f.relative_to(args.rollouts)) for f in args.rollouts.rglob("*") if f.is_file())
    expected_files = sorted(f"{T}/{a}/s{s}-0.0.npz" for a, s in expected.values())
    audit["keys"] = {"expected": len(expected), "reported": len(reported),
                     "missing": sorted(set(expected) - set(reported)), "unexpected": sorted(set(reported) - set(expected)),
                     "files": len(files), "files_missing": sorted(set(expected_files) - set(files)),
                     "files_unexpected": sorted(set(files) - set(expected_files)),
                     "distinct_hashes": len(set(reported.values()))}
    checks["keys"] = len(expected) == 40 and set(expected) == set(reported)
    checks["files"] = files == expected_files
    checks["distinct_hashes"] = audit["keys"]["distinct_hashes"] == len(reported)

    recs = {(r["arm"], r["seed"]): r for r in cs.records(args.root) if r["task"] == T}
    status = {a: {s: recs[(a, s)]["status"] if (a, s) in recs else None for s in seeds} for a in cnt.ARMS}
    checks["statuses"] = json.loads(json.dumps(status)) == committed["failures"]["status"] and \
        all(v == "ok" for arm in status.values() for v in arm.values())

    scores, bad_hash, problems, failed = {}, [], {}, []
    for name in sorted(expected):
        a, s = expected[name]
        path = args.rollouts / T / a / f"s{s}-0.0.npz"
        if not path.is_file() or cs.sha(path) != reported.get(name):
            bad_hash.append(name)
            continue
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            z = {k: f[k] for k in f.files if k != "meta"}
        p = validate(z, meta, T, s, 0.0, "none")
        if p:
            problems[name] = p
        if len(z["failed_episodes"]):
            failed.append(name)
            continue
        scores[(a, s)] = h.seed_scores(z)
    audit["rollouts"] = {"checked": len(expected), "hash_mismatches": bad_hash, "invalid": problems}
    checks["hashes"] = not bad_hash
    checks["rollouts_valid"] = not problems
    if bad_hash:
        checks["recomputation"] = False
        return finish(audit, checks, {"not_attempted": "rollouts failed the hash check"})

    v500 = json.loads(cnt.CORRECTED_EVAL.read_text())
    p500 = v500["per_seed"]

    def d100(m):
        return {s: scores[("ccil_100", s)][m] - scores[("bc_100", s)][m] for s in seeds}

    def d500(m):
        return {s: p500[f"{T}/ccil/0.0/s{s}"][m] - p500[f"{T}/bc/0.0/s{s}"][m] for s in seeds}

    occ, occ5 = d100("upright_occupancy"), d500("upright_occupancy")
    primary = {"ccil_minus_bc_100": cs.primary_decision(occ, status, cnt.C_PRIMARY),
               "change_100_minus_500": cs.primary_decision({s: occ[s] - occ5[s] for s in seeds}, status,
                                                           cnt.C_PRIMARY)}
    primary["change_100_minus_500"]["reading"] = committed["primary"]["change_100_minus_500"].get("reading")
    secondary = {}
    for m in h.METRICS:
        dm, dm5 = d100(m), d500(m)
        secondary[f"ccil_minus_bc_100/{m}"] = analysis.union_interval([dm[s] for s in seeds], 0.95)
        secondary[f"change_100_minus_500/{m}"] = analysis.union_interval([dm[s] - dm5[s] for s in seeds], 0.95)
        for a in cnt.ARMS:
            b = cnt.BASE[a]
            secondary[f"{a}_minus_{b}_500/{m}"] = analysis.union_interval(
                [scores[(a, s)][m] - p500[f"{T}/{b}/0.0/s{s}"][m] for s in seeds], 0.95)
    c500 = {r["seed"]: r for r in cs.records(args.corrected) if r["task"] == T and r["arm"] == "ccil"}
    per = {str(s): {"100": cnt.label_summary(recs[("ccil_100", s)]), "500": cnt.label_summary(c500[s])} for s in seeds}
    recomputed = {
        "primary": primary, "secondary_95": secondary,
        "failures": {"excluded_units": [], "failed_rollouts": failed, "status": status},
        "label_accuracy": {"per_seed": per, "difference_100_minus_500_95": {
            k: analysis.union_interval([v["100"][k] - v["500"][k] for v in per.values()], 0.95)
            for k in ("acceptance_rate", "true_label_error_mean")}},
        "continuous_beside": {
            "label": committed["continuous_beside"]["label"],
            "continuous_ccil_minus_bc_100_demonstrations":
                v500["secondary_95"]["continuous/corrected_ccil_minus_bc/upright_occupancy"],
            "discontinuous_ccil_minus_bc_100_demonstrations": secondary["ccil_minus_bc_100/upright_occupancy"]},
        "arm_summaries": h.arm_summaries({(T, a, "0.0", s): v for (a, s), v in scores.items()}),
        "per_seed": {f"{T}/{a}/0.0/s{s}": {m: v[m] for m in h.METRICS} for (a, s), v in scores.items()}}
    recomputed = json.loads(json.dumps(recomputed, sort_keys=True, default=str))
    same = {k: recomputed[k] == committed[k] for k in recomputed}
    checks["recomputation"] = all(same.values())
    return finish(audit, checks, {"sections_equal": same,
                                  "outcomes": {k: v["outcome"] for k, v in primary.items()}})


def finish(audit: dict, checks: dict, recomputation: dict) -> int:
    audit["recomputation"] = recomputation
    audit["checks"] = checks
    audit["all_pass"] = all(checks.values())
    OUT.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"report_sha256": audit["report_sha256"], "checks": checks,
                      "recomputation": recomputation.get("outcomes", recomputation)}, indent=1))
    return 0 if audit["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
