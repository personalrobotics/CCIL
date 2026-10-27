#!/usr/bin/env python3
"""Evidence for ccil.experiments (#112).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_experiments/check.py --out DIR

1. analysis: on synthetic R tables covering every outcome (improvement, deterioration, nulls with a passing
   and a failing positive control, a primary-arm algorithmic failure, too many exclusions, too few valid
   seeds, a degenerate BCa interval), ccil.experiments.analysis.decide with the #24 predeclaration gives
   exactly the #24 analysis's result: outcome, intervals, reasons, control and relative effect.
2. gate: on scratch clones of this commit (clean; an uncommitted src change; an untracked file; no freeze
   manifest; a manifest naming the current commit; a later committed tooling change; a manifest naming no
   commit), ccil.experiments.gate with the #24 freeze decides exactly as the #24 provenance module, for real
   and smoke runs.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
import analyze as m24  # noqa: E402  (the #24 analysis, the reference)
import check_study  # noqa: E402  (its synthetic tables)

import provenance as p24  # noqa: E402  (the #24 gate, the reference)

from ccil.experiments.analysis import Decision, decide  # noqa: E402
from ccil.experiments.gate import Freeze, gate  # noqa: E402

FREEZE_24 = Freeze(ccil_commit=p24.CCIL_FROZEN, tooling_paths=p24.TOOLING_PATHS, manifest=p24.FREEZE)

PREDECLARED_24 = Decision(treatment="ccil", baseline="bc", control=("bc", "bc_frac10"), confidence=0.975,
                          min_valid=16, max_excluded=2,
                          outcomes=("ccil_improves_on_bc", "ccil_is_worse_than_bc", "no_detectable_difference"))


def analysis_matches_24() -> dict:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 20, 20)
    sym = np.tile([15.0, -15.0], 10)
    syn = check_study.synthetic
    scenarios = {
        "improves": syn(100 + noise), "worse": syn(-100 + noise), "null_control_passes": syn(sym),
        "null_control_fails": syn(sym, control_gap=0.0),
        "algorithmic_failure": syn(100 + noise, status={("ccil", 105): "algorithmic_failure"}),
        "three_exclusions": syn(100 + noise, status={("bc", s): "infrastructure_fault" for s in (101, 102, 103)}),
        "fifteen_valid": syn(100 + noise, status={**{("bc", s): "infrastructure_fault" for s in (101, 102)},
                                                  **{("ccil", s): "infrastructure_fault" for s in (103, 104, 105)}}),
        "degenerate_bca": syn(np.full(20, 50.0))}
    rows = {}
    for name, tables in scenarios.items():
        ref = m24.primary(tables, "t")
        got = decide(tables, "t", PREDECLARED_24)
        keys = ("outcome", "inconclusive_reasons")
        same = (all(ref[k] == got[k] for k in keys) and ref["comparison"]["interval"] == got["comparison"]["interval"]
                and ref["positive_control"]["passes"] == got["positive_control"]["passes"]
                and ref["relative_effect"] == got["relative_effect"])
        rows[name] = {"identical": same, "outcome": got["outcome"]}
    return {"pass": all(r["identical"] for r in rows.values()), "scenarios": rows}


def gate_matches_24(out: Path) -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    clone = out / "clone"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(REPO), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", head], check=True)
    git = ["git", "-C", str(clone), "-c", "user.name=check", "-c", "user.email=check@invalid"]
    rows = {}

    def both(name):
        for real in (True, False):
            a, b = p24.gate(clone, real)["ok"], gate(clone, FREEZE_24, real)["ok"]
            rows[f"{name}/{'real' if real else 'smoke'}"] = {"p24": a, "experiments": b, "same": a == b}
    both("clean_no_manifest")
    init = clone / "src" / "ccil" / "__init__.py"
    data = init.read_bytes()
    init.write_bytes(data + b"\n")
    both("uncommitted_src_change")
    init.write_bytes(data)
    (clone / "tools" / "untracked.txt").write_text("x")
    both("untracked_file")
    (clone / "tools" / "untracked.txt").unlink()
    (clone / p24.FREEZE).parent.mkdir(parents=True, exist_ok=True)
    (clone / p24.FREEZE).write_text(json.dumps({"tooling_commit": head}) + "\n")
    subprocess.run([*git, "add", "-f", p24.FREEZE], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "freeze"], check=True)
    both("manifest_at_this_commit")
    readme = clone / "tools" / "m3_pendulum" / "README.md"
    readme.write_text(readme.read_text() + "\nchanged\n")
    subprocess.run([*git, "commit", "-q", "-am", "change the tooling"], check=True)
    both("committed_tooling_change")
    (clone / p24.FREEZE).write_text(json.dumps({"tooling_commit": "0" * 40}) + "\n")
    subprocess.run([*git, "commit", "-q", "-am", "name no commit"], check=True)
    both("manifest_names_no_commit")
    shutil.rmtree(clone)
    return {"pass": all(r["same"] for r in rows.values()), "scenarios": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)
    r = analysis_matches_24()
    record("analysis_matches_24", r.pop("pass"), **r)
    r = gate_matches_24(out)
    record("gate_matches_24", r.pop("pass"), **r)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "python": platform.python_version(), "platform": platform.platform()}}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
