#!/usr/bin/env python3
"""Audit of historical Part B (docs/fixtures/evidence/pendulum-eval-v2/part-b.json), from its saved rollouts.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/audit_part_b.py --root V6_ROOT --rollouts PART_B_DIR --log RUN_LOG

Nothing is re-evaluated. The audit:
- records the commit that ran (RUN_COMMIT), the commit the run log wrongly names (LOGGED_COMMIT) and why, and checks
  that the logged commit does not even contain the Part B script;
- records the run log's sha256 (the log is preserved unchanged);
- checks every reused checkpoint against its v6 record and every saved rollout against the report's sha256;
- recomputes every Part B result (per-seed scores, primary decisions, secondary and noise comparisons, arm summaries,
  failures) from the saved rollouts and compares each with the committed report exactly;
- measures how far commanded and applied actions exceed the declared range [-3, 3], per task, arm and condition.
The auditing process records its own commit and working-tree state. Writes part-b-audit.json; exits 1 if any check
fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "pendulum_eval"))
import historical as h  # noqa: E402
import profiles  # noqa: E402

from ccil.experiments import analysis  # noqa: E402

REPORT = h.EVIDENCE / "part-b.json"
OUT = h.EVIDENCE / "part-b-audit.json"
RUN_COMMIT = "70e25a461782cd0d7355f45fa80a7bec35b289bc"
LOGGED_COMMIT = "57faa1d52863824bbb4a1d3a9009a1701b33c413"
LOG_CAUSE = ("the launcher was written by an unquoted heredoc in ~/code/CCIL, so `git rev-parse HEAD` expanded there "
             "(the main checkout) when the script was written, not in the run's worktree; the run's worktree was "
             "checked out detached at RUN_COMMIT, and its only untracked file was the launcher itself")


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def has_path(commit: str, path: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{path}"],
                          capture_output=True).returncode == 0


def excess(x: np.ndarray) -> dict:
    a = np.abs(x[np.isfinite(x)])
    over = a[a > profiles.ACTION_BOUND] - profiles.ACTION_BOUND
    return {"steps": int(a.size), "fraction_over": float(over.size / a.size) if a.size else 0.0,
            "max_excess": float(over.max()) if over.size else 0.0,
            "p99_excess_among_over": float(np.quantile(over, 0.99)) if over.size else 0.0,
            "max_abs": float(a.max()) if a.size else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    args = ap.parse_args()
    committed = json.loads(REPORT.read_text())
    checks = {}

    audit = {"auditor": {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()},
             "run": {"run_commit": RUN_COMMIT, "logged_commit": LOGGED_COMMIT, "logged_commit_cause": LOG_CAUSE,
                     "run_commit_contains_part_b_script": has_path(RUN_COMMIT, "tools/pendulum_eval/historical.py"),
                     "logged_commit_contains_part_b_script": has_path(LOGGED_COMMIT,
                                                                      "tools/pendulum_eval/historical.py"),
                     "log": args.log.name, "log_sha256": h.sha(args.log),
                     "log_names_logged_commit": f"commit {LOGGED_COMMIT}" in args.log.read_text()}}
    checks["run_commit_has_script"] = audit["run"]["run_commit_contains_part_b_script"]
    checks["logged_commit_lacks_script"] = not audit["run"]["logged_commit_contains_part_b_script"]

    # Checkpoints against their v6 records.
    bad_ckpt, n_ckpt = [], 0
    for r in h.records(args.root):
        if r["arm"] in h.REUSED:
            n_ckpt += 1
            ckpt = Path(r["stages"]["policy.train"]) / "checkpoints" / "policy.npz"
            if h.sha(ckpt) != r["checkpoint_sha256"]:
                bad_ckpt.append(f"{r['task']}/{r['arm']}/s{r['seed']}")
    audit["checkpoints"] = {"checked": n_ckpt, "mismatches": bad_ckpt}
    checks["checkpoints"] = n_ckpt == len(h.TASKS) * len(h.REUSED) * len(h.SEEDS) and not bad_ckpt

    # Rollouts against the report, and the recomputation.
    scores, status, failed, bad_roll, bounds = {}, {}, [], [], {}
    for name, digest in sorted(committed["rollout_sha256"].items()):
        task, arm, s, key = name.split("/", 3)
        seed = int(s[1:])
        path = args.rollouts / task / arm / f"s{seed}-{key.replace('/', '-')}.npz"
        if h.sha(path) != digest:
            bad_roll.append(name)
            continue
        z = h.load_rollout(path)
        st = status.setdefault(task, {}).setdefault(arm, {})
        if len(z["failed_episodes"]):
            failed.append(name)
            if key == "0.0":
                st[seed] = "algorithmic_failure"
            continue
        if key == "0.0":
            st[seed] = "ok"
        scores[(task, arm, key, seed)] = h.seed_scores(z)
        b = bounds.setdefault(f"{task}/{arm}/{key}", {"commanded": [], "applied": []})
        b["commanded"].append(z["policy_actions"][..., 0])
        b["applied"].append(z["actions"][..., 0])
    audit["rollouts"] = {"checked": len(committed["rollout_sha256"]), "mismatches": bad_roll}
    checks["rollouts"] = not bad_roll

    tables = h.tables_from(scores, status)
    recomputed = {
        "primary": {t: analysis.decide(tables["upright_occupancy"], t, h.PRIMARY) for t in h.TASKS},
        "failures": {t: {a: analysis.counts(tables["upright_occupancy"], t, a) for a in status[t]} for t in h.TASKS},
        "failed_rollouts": failed,
        "secondary_95": h.describe(tables, [(a, "bc") for a in (*h.REUSED, "zero") if a != "bc"]
                                   + [("ccil_modern", "bc_modern"), *h.DATA_SIZE], ["0.0"]),
        "noise_95": h.describe(tables, [("ccil", "bc"), ("ccil_modern", "bc_modern")],
                               [f"{ch}/{n}" for ch in h.CHANNELS for n in h.LEVELS]),
        "arm_summaries": h.arm_summaries(scores),
        "per_seed": {f"{t}/{a}/{n}/s{s}": {m: v[m] for m in h.METRICS} for (t, a, n, s), v in scores.items()}}
    recomputed = json.loads(json.dumps(recomputed, sort_keys=True, default=str))
    same = {k: recomputed[k] == committed[k] for k in recomputed}
    audit["recomputation"] = {"sections_equal": same}
    checks["recomputation"] = all(same.values())

    # Action range.
    audit["action_bound"] = {"bound": profiles.ACTION_BOUND, "by_unit": {
        k: {kind: excess(np.stack(v[kind])) for kind in ("commanded", "applied")} for k, v in sorted(bounds.items())}}
    allc = [u["commanded"] for u in audit["action_bound"]["by_unit"].values()]
    audit["action_bound"]["overall_max_commanded_excess"] = max(u["max_excess"] for u in allc)
    nf = [u["commanded"] for k, u in audit["action_bound"]["by_unit"].items() if k.endswith("/0.0")]
    audit["action_bound"]["noise_free_max_commanded_excess"] = max(u["max_excess"] for u in nf)

    audit["checks"] = checks
    audit["all_pass"] = all(checks.values())
    OUT.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "overall_max_commanded_excess":
                      audit["action_bound"]["overall_max_commanded_excess"],
                      "noise_free_max_commanded_excess": audit["action_bound"]["noise_free_max_commanded_excess"]},
                     indent=1))
    return 0 if audit["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
