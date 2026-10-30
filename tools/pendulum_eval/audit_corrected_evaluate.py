#!/usr/bin/env python3
"""Read-only audit of the corrected study's evaluation report (study-evaluate.json), from its saved rollouts.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/audit_corrected_evaluate.py --root ROOT --rollouts EVAL_DIR

Nothing is trained or evaluated, and nothing under ROOT or EVAL_DIR is written. The evaluator scored each rollout from
memory after saving it; this audit reopens every saved file instead. It:
- checks that the report's rollout keys are exactly the 600 expected ones (2 tasks x {BC, CCIL} x 20 seeds x 7
  conditions, plus the corrected demonstrator noise-free x 20 seeds x 2 tasks), that EVAL_DIR holds exactly those
  files, and that each file's sha256 matches the report;
- reopens every rollout and validates its metadata (task, evaluation seed, noise level, channel) and arrays (shapes,
  finite values, no failed episodes, initial states equal to the seed's fresh starts, the noise channel touching only
  its side, no flips on the continuous task);
- checks the report's unit statuses against the study records in ROOT;
- recomputes every per-seed metric, arm summary, primary decision, secondary and noise interval, and the failure
  sections from the reopened rollouts, and compares each with the committed report exactly.
Writes study-evaluate-audit.json, naming the report's sha256; exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corrected_study as cs  # noqa: E402  (also puts tools/m3_pendulum on the path)
import env as canonical  # noqa: E402  (tools/m3_pendulum)
import historical as h  # noqa: E402

from ccil.experiments import analysis  # noqa: E402

REPORT = cs.EVIDENCE / "study-evaluate.json"
OUT = cs.EVIDENCE / "study-evaluate-audit.json"
DEMONSTRATOR = "corrected_demonstrator"
CONDITIONS = [("0.0", 0.0, "none")] + [(f"{ch}/{n}", n, ch) for ch in h.CHANNELS for n in h.LEVELS]


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(cs.REPO), *args], capture_output=True, text=True).stdout.strip()


def expected_keys() -> dict:
    out = {}
    for t in cs.TASKS:
        for a in cs.PRIMARY_ARMS:
            for s in cs.STUDY_SEEDS:
                for key, n, ch in CONDITIONS:
                    out[f"{t}/{a}/s{s}/{key}"] = (t, a, s, key, n, ch)
        for s in cs.STUDY_SEEDS:
            out[f"{t}/{DEMONSTRATOR}/s{s}/0.0"] = (t, DEMONSTRATOR, s, "0.0", 0.0, "none")
    return out


def validate(z: dict, meta: dict, t: str, s: int, n: float, ch: str) -> list[str]:
    """Problems with one reopened rollout; empty if it is the rollout its key names."""
    E, T = canonical.EPISODES, canonical.HORIZON
    p = []
    want = {"task": t, "eval_seed": s + h.FRESH_OFFSET, "noise_level": n, "channel": ch}
    for k, v in want.items():
        if meta.get(k) != v:
            p.append(f"meta {k}={meta.get(k)!r}, expected {v!r}")
    shapes = {"states": (E, T + 1, 3), "observations": (E, T, 3), "policy_actions": (E, T, 1), "actions": (E, T, 1),
              "rewards": (E, T), "flipped": (E, T), "returns": (E,)}
    for k, shape in shapes.items():
        if k not in z or z[k].shape != shape:
            p.append(f"{k} shape {None if k not in z else z[k].shape}, expected {shape}")
    if p:
        return p
    if len(z["failed_episodes"]):
        p.append(f"{len(z['failed_episodes'])} failed episodes")
    for k in ("states", "observations", "policy_actions", "actions", "rewards", "returns"):
        if not np.isfinite(z[k]).all():
            p.append(f"{k} has non-finite values")
    if not np.array_equal(z["states"][:, 0], canonical.initial_states(s + h.FRESH_OFFSET)):
        p.append("initial states differ from the seed's fresh starts")
    if ch in ("none", "action_only") and not np.array_equal(z["observations"], z["states"][:, :T]):
        p.append("observations differ from states without observation noise")
    if ch in ("none", "observation_only") and not np.array_equal(z["actions"], z["policy_actions"]):
        p.append("applied actions differ from commanded actions without action noise")
    if ch != "none" and np.array_equal(z["observations"], z["states"][:, :T]) \
            and np.array_equal(z["actions"], z["policy_actions"]):
        p.append("noise channel changed nothing")
    if not canonical.TASKS[t] and z["flipped"].any():
        p.append("flips on the continuous task")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    args = ap.parse_args()
    committed = json.loads(REPORT.read_text())
    seeds = list(cs.STUDY_SEEDS)
    checks = {}
    audit = {"report": REPORT.name, "report_sha256": cs.sha(REPORT),
             "auditor": {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()}}
    checks["report_is_study_run"] = committed["smoke"] is False and committed["gate"]["ok"] and committed["all_pass"]

    # Keys, files and hashes.
    expected = expected_keys()
    reported = committed["rollout_sha256"]
    files = sorted(str(f.relative_to(args.rollouts)) for f in args.rollouts.rglob("*") if f.is_file())
    expected_files = sorted(f"{t}/{a}/s{s}-{key.replace('/', '-')}.npz" for t, a, s, key, _, _ in expected.values())
    audit["keys"] = {"expected": len(expected), "reported": len(reported),
                     "missing": sorted(set(expected) - set(reported)), "unexpected": sorted(set(reported) - set(expected)),
                     "files": len(files), "files_missing": sorted(set(expected_files) - set(files)),
                     "files_unexpected": sorted(set(files) - set(expected_files)),
                     "distinct_hashes": len(set(reported.values()))}
    checks["keys"] = len(expected) == 600 and set(expected) == set(reported)
    checks["files"] = files == expected_files
    checks["distinct_hashes"] = audit["keys"]["distinct_hashes"] == len(reported)

    # Statuses from the study records.
    recs = {(r["task"], r["arm"], r["seed"]): r["status"] for r in cs.records(args.root)}
    status = {t: {a: {s: recs.get((t, a, s)) for s in seeds} for a in cs.PRIMARY_ARMS} for t in cs.TASKS}
    checks["statuses"] = json.loads(json.dumps(status)) == committed["failures"]["status"] and \
        all(v == "ok" for st in status.values() for arm in st.values() for v in arm.values())

    # Reopen, validate and score every rollout.
    scores, bad_hash, problems, failed = {}, [], {}, []
    for name in sorted(expected):
        t, a, s, key, n, ch = expected[name]
        path = args.rollouts / t / a / f"s{s}-{key.replace('/', '-')}.npz"
        if not path.is_file() or cs.sha(path) != reported.get(name):
            bad_hash.append(name)
            continue
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            z = {k: f[k] for k in f.files if k != "meta"}
        p = validate(z, meta, t, s, n, ch)
        if p:
            problems[name] = p
        if len(z["failed_episodes"]):
            failed.append(name)
            continue
        scores[(t, a, key, s)] = h.seed_scores(z)
    audit["rollouts"] = {"checked": len(expected), "hash_mismatches": bad_hash, "invalid": problems}
    checks["hashes"] = not bad_hash
    checks["rollouts_valid"] = not problems

    # Recompute the report's sections.
    def diff(t, m, key="0.0"):
        return {s: scores[(t, "ccil", key, s)][m] - scores[(t, "bc", key, s)][m]
                for s in seeds if (t, "ccil", key, s) in scores and (t, "bc", key, s) in scores}

    part_b = json.loads((h.EVIDENCE / "part-b.json").read_text())
    hist = part_b["per_seed"]

    def hist_diff(t, m):
        return {s: hist[f"{t}/ccil/0.0/s{s}"][m] - hist[f"{t}/bc/0.0/s{s}"][m] for s in seeds}

    primary, secondary, noise = {}, {}, {}
    for t in cs.TASKS:
        d1, hd = diff(t, "upright_occupancy"), hist_diff(t, "upright_occupancy")
        primary[t] = {"corrected_ccil_minus_bc": cs.primary_decision(d1, status[t], cs.C_PRIMARY),
                      "change_in_ccil_minus_bc": cs.primary_decision({s: d1[s] - hd[s] for s in d1}, status[t],
                                                                     cs.C_PRIMARY)}
        for m in h.METRICS:
            dm, hm = diff(t, m), hist_diff(t, m)
            secondary[f"{t}/corrected_ccil_minus_bc/{m}"] = analysis.union_interval([dm[s] for s in seeds], 0.95)
            secondary[f"{t}/change_in_ccil_minus_bc/{m}"] = analysis.union_interval(
                [dm[s] - hm[s] for s in seeds], 0.95)
            for a in cs.PRIMARY_ARMS:
                secondary[f"{t}/{a}_corrected_minus_historical/{m}"] = analysis.union_interval(
                    [scores[(t, a, "0.0", s)][m] - hist[f"{t}/{a}/0.0/s{s}"][m] for s in seeds], 0.95)
        for ch in h.CHANNELS:
            for n in h.LEVELS:
                dn = diff(t, "upright_occupancy", f"{ch}/{n}")
                noise[f"{t}/ccil-bc/{ch}/{n}"] = analysis.union_interval([dn[s] for s in seeds], 0.95)
    # The interpretation note is text, not a computed value; it is compared as committed.
    for t in cs.TASKS:
        primary[t]["change_in_ccil_minus_bc"]["reading"] = \
            committed["primary"][t]["change_in_ccil_minus_bc"].get("reading")
    recomputed = {
        "primary": primary, "secondary_95": secondary, "noise_95": noise,
        "failures": {"excluded_units": [], "failed_rollouts": failed, "status": status},
        "zero_action_reference": {"source": "historical Part B (part-b.json), unchanged",
                                  **{t: part_b["arm_summaries"][f"{t}/zero/0.0"] for t in cs.TASKS}},
        "arm_summaries": h.arm_summaries(scores),
        "per_seed": {f"{t}/{a}/{n}/s{s}": {m: v[m] for m in h.METRICS} for (t, a, n, s), v in scores.items()}}
    recomputed = json.loads(json.dumps(recomputed, sort_keys=True, default=str))
    same = {k: recomputed[k] == committed[k] for k in recomputed}
    audit["recomputation"] = {"sections_equal": same,
                              "outcomes": {t: {k: v["outcome"] for k, v in primary[t].items()} for t in cs.TASKS}}
    checks["recomputation"] = all(same.values())

    audit["checks"] = checks
    audit["all_pass"] = all(checks.values())
    OUT.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"report_sha256": audit["report_sha256"], "checks": checks,
                      "outcomes": audit["recomputation"]["outcomes"]}, indent=1))
    return 0 if audit["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
