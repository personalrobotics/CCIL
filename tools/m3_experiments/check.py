#!/usr/bin/env python3
"""Evidence for ccil.experiments (#112).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_experiments/check.py --out DIR

1. analysis: on synthetic R tables covering every outcome (improvement, deterioration, nulls with a passing
   and a failing positive control, a primary-arm algorithmic failure, too many exclusions, too few valid
   seeds, a degenerate BCa interval), ccil.experiments.analysis.decide with the #24 predeclaration gives
   exactly the #24 analysis's result: outcome, intervals, reasons, control and relative effect.
2. gate: on scratch clones of this commit (clean, with main's #24 v4 freeze manifest; an uncommitted src change; an untracked file;
   a manifest naming the current commit; a later committed tooling change; a manifest naming no
   commit), ccil.experiments.gate with the #24 freeze decides exactly as the #24 provenance module, for real
   and smoke runs.
3. plan: tools/m3_experiments/pendulum_v4.yaml (the #24 v4 design as a plan) expands to exactly the #24
   drivers' units for tuning and for the study (with a tuning choice), in the same order, including every
   arm's settings, noise levels and data fractions; malformed specs are refused with every problem listed.
4. records (with --smoke-root, a #24 check_study smoke root): ccil.experiments.records.verify_phase with the
   Pendulum hooks accepts the clean root with the same recomputed R as the #24 verify_phase, recomputes the
   same tuning choice, and refuses exactly when the #24 verifier refuses: an edited study-record R and an extra
   study unit (each restored afterwards).
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
from ccil.experiments import plan  # noqa: E402
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
    # Freeze this commit's ccil for both implementations, so the scenarios produce passes as well as refusals.
    saved, p24.CCIL_FROZEN = p24.CCIL_FROZEN, head
    freeze = Freeze(ccil_commit=head, tooling_paths=p24.TOOLING_PATHS, manifest=p24.FREEZE)

    def both(name):
        for real in (True, False):
            a, b = p24.gate(clone, real)["ok"], gate(clone, freeze, real)["ok"]
            rows[f"{name}/{'real' if real else 'smoke'}"] = {"p24": a, "experiments": b, "same": a == b}
    both("clean_with_main_v4_manifest")  # main carries the #24 v4 freeze manifest
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
    p24.CCIL_FROZEN = saved
    shutil.rmtree(clone)
    outcomes = {r["experiments"] for r in rows.values()}
    return {"pass": all(r["same"] for r in rows.values()) and outcomes == {True, False}, "scenarios": rows,
            "both_outcomes_exercised": outcomes == {True, False}}


def plan_matches_24(out: Path) -> dict:
    import study as s24  # the #24 drivers, the reference
    spec = plan.load(HERE / "pendulum_v4.yaml")
    root = out / "plan-root"
    root.mkdir()
    choice = {t: {"noise_bc": {"value": 1e-4}, "ccil_tuned": {"value": 0.3}} for t in s24.DATASETS}
    (root / "tuning-choice.json").write_text(json.dumps(choice))
    tuning_same = spec.units("tuning") == s24.units(root, "tuning", smoke=False)
    study_same = spec.units("study", choice) == s24.units(root, "study", smoke=False)
    arms_same = all(spec.arms[n].settings == tuple(a[0] or ()) and spec.arms[n].noise_levels == tuple(a[1])
                    and spec.arms[n].data_fraction == a[2] and (spec.arms[n].tuned or None) == a[3]
                    and spec.arms[n].train == (a[0] is not None) for n, a in s24.ARMS.items())
    bad = [{"study": "x"}, {**plan.yaml.safe_load((HERE / "pendulum_v4.yaml").read_text()), "phases": {
        "tuning": {"seeds": [1, 2], "arms": ["bc"], "tuning": True}, "study": {"seeds": [2, 3], "arms": ["nope"]}}}]
    refusals = []
    for b in bad:
        try:
            plan.from_dict(b)
            refusals.append(None)
        except plan.PlanError as e:
            refusals.append(e.args[0])
    refused = all(r for r in refusals) and len(refusals[1]) >= 3  # tuned-only, unknown arm, shared seed
    return {"pass": tuning_same and study_same and arms_same and refused, "tuning_units_same": tuning_same,
            "study_units_same": study_same, "arms_same": arms_same, "units": {"tuning": len(spec.units("tuning")),
                                                                           "study": len(spec.units("study", choice))},
            "refusals": refusals}


def records_match_24(root: Path) -> dict:
    import audit as a24
    import study as s24

    from ccil.experiments import records
    spec = plan.load(HERE / "pendulum_v4.yaml")
    smoke = plan.Study(spec.name, spec.version, spec.tasks,
                       {"tuning": {**spec.phases["tuning"], "seeds": s24.SMOKE_TUNING_SEEDS},
                        "study": {**spec.phases["study"], "seeds": s24.SMOKE_STUDY_SEEDS}}, spec.arms, spec.tuning)
    inputs = s24.prepare(root)

    def expected_inputs(r):  # the Pendulum suite's hook: the converted inputs, or the rebuilt subset
        want = dict(inputs[r["task"]])
        if spec.arms[r["arm"]].data_fraction:
            want["data"] = s24.subset(root, r["task"], r["seed"], spec.arms[r["arm"]].data_fraction)[1]
        return {"config_sha256": want["config"], "data_sha256": want["data"]}
    choice = json.loads((root / "tuning-choice.json").read_text())

    def both(phase):
        out = {}
        for name, fn in (("p24", lambda: s24.verify_phase(root, phase, True)),
                         ("experiments", lambda: records.verify_phase(root, smoke, phase, smoke=True, audit=a24.audit,
                                                                      expected_inputs=expected_inputs, choice=choice))):
            try:
                out[name] = {(r["task"], r["arm"], r["seed"], r["value"]): r["verified"] for r in fn()}
            except SystemExit:
                out[name] = "refused"
        return out
    clean = {ph: both(ph) for ph in ("tuning", "study")}
    same_clean = all(v["p24"] == v["experiments"] and v["p24"] != "refused" for v in clean.values())
    tuning = s24.verify_phase(root, "tuning", True)
    same_choice = all(json.loads(json.dumps(records.compute_choice(smoke, tuning)[t])) ==
                      json.loads(json.dumps(s24.compute_choice(tuning, True)[t])) for t in spec.tasks)
    rec = next((root / "results" / "study" / "continuous" / "bc").glob("s*.json"))
    tampered = {}
    data = rec.read_bytes()
    obj = json.loads(data)
    obj["evaluations"]["0.0"]["R"] += 1.0
    rec.write_text(json.dumps(obj))
    tampered["edited_R"] = both("study")
    rec.write_bytes(data)
    extra = rec.with_name("s9999.json")
    obj = json.loads(data)
    obj["seed"] = 9999
    extra.write_text(json.dumps(obj))
    tampered["extra_unit"] = both("study")
    extra.unlink()
    refusals = {k: (v["p24"] == "refused", v["experiments"] == "refused") for k, v in tampered.items()}
    return {"pass": same_clean and same_choice and all(a and b for a, b in refusals.values()),
            "clean_root_same": same_clean, "choice_same": same_choice, "refusals": refusals,
            "verified_records": {ph: len(v["experiments"]) if v["experiments"] != "refused" else 0
                                 for ph, v in clean.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke-root", type=Path, help="a #24 check_study smoke root, for the records case")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)
    r = analysis_matches_24()
    record("analysis_matches_24", r.pop("pass"), **r)
    r = gate_matches_24(out)
    record("gate_matches_24", r.pop("pass"), **r)
    r = plan_matches_24(out)
    record("plan_matches_24", r.pop("pass"), **r)
    if args.smoke_root:
        r = records_match_24(args.smoke_root.resolve())
        record("records_match_24", r.pop("pass"), **r)
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
