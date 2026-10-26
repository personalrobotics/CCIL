#!/usr/bin/env python3
"""Evidence for the #24 run drivers (study.py) and analysis (analyze.py), before the tooling freeze.

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_pendulum/check_study.py --out DIR [--device cuda] [--jobs N]

It produces no tuning or study outcome. The analysis is checked on synthetic
R tables with known answers; the drivers run end to end in smoke mode (1
dynamics epoch, 1 policy epoch, seeds 9001-9003), and the report records
statuses and decisions, never the smoke returns.
1. Known answers: improvement, deterioration, a null with a passing and with
   a failing positive control, a primary-arm algorithmic failure, too many
   infrastructure exclusions, too few valid seeds, and a degenerate BCa
   interval, each give the predeclared outcome.
2. Intervals: the t-interval matches its formula, and the union interval at
   c = 0.975 covers the true mean of normal differences at n = 20 in at least
   97% of 1,000 simulations.
3. Gates: a study unit refuses to run before the tuning choice exists.
4. Smoke, end to end: prepare, the tuning sweep, the choice (which refuses a
   second time), the study sweep (a second sweep changes no record), and the
   analysis. Every unit of both phases is ok, every rollout is re-audited
   with the replay, and every CCIL record has its label accuracy.
5. The analysis refuses, and each tampering is restored afterwards: a
   changed rollout; a study record's R and a tuning record's R changed in
   the JSON alone; a changed tuning choice; an extra and a duplicate study
   unit; a changed corrected configuration; a changed reused subset; and an
   existing output. A unit refuses a changed configuration too, and the
   restored root analyzes again.
6. The frozen-tooling gate, on scratch clones: a real run refuses without a
   freeze manifest; the gate passes when the manifest names the tooling's
   commit, refuses a later committed tooling change and a manifest naming no
   commit, and smoke runs do not require it.
Writes DIR/report.json; the smoke root is DIR/smoke.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import provenance  # noqa: E402
import study  # noqa: E402

SEEDS = list(range(101, 121))
CMD = [sys.executable, str(HERE / "study.py")]


def synthetic(d, *, control_gap=-500.0, status=None, seed=0):
    """One task's tables: BC around -3000, CCIL = BC + d, BC at 10% = BC + control_gap (+ noise)."""
    rng = np.random.default_rng(seed)
    bc = -3000 + rng.normal(0, 100, len(SEEDS))
    arms = {"bc": bc, "ccil": bc + np.asarray(d, float), "bc_frac10": bc + control_gap + rng.normal(0, 30, len(SEEDS))}
    R = {a: {"0.0": dict(zip(SEEDS, v.tolist()))} for a, v in arms.items()}
    st = {a: {s: "ok" for s in SEEDS} for a in arms}
    for (a, s), v in (status or {}).items():
        st[a][s] = v
    return {"R": {"t": R}, "status": {"t": st}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:900]}", flush=True)

    # 1. Known answers.
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 20, len(SEEDS))
    symmetric = np.tile([15.0, -15.0], len(SEEDS) // 2)
    scenarios = {
        "improves": (synthetic(100 + noise), "ccil_improves_on_bc"),
        "worse": (synthetic(-100 + noise), "ccil_is_worse_than_bc"),
        "null_control_passes": (synthetic(symmetric), "no_detectable_difference"),
        "null_control_fails": (synthetic(symmetric, control_gap=0.0), "inconclusive"),
        "algorithmic_failure_in_ccil": (synthetic(100 + noise, status={("ccil", 105): "algorithmic_failure"}),
                                        "inconclusive"),
        "three_infrastructure_exclusions": (synthetic(100 + noise, status={("bc", s): "infrastructure_fault"
                                                                            for s in (101, 102, 103)}), "inconclusive"),
        "fifteen_valid_seeds": (synthetic(100 + noise, status={**{("bc", s): "infrastructure_fault" for s in (101, 102)},
                                                               **{("ccil", s): "infrastructure_fault"
                                                                  for s in (103, 104, 105)}}), "inconclusive"),
        "degenerate_bca": (synthetic(np.full(len(SEEDS), 50.0)), "ccil_improves_on_bc"),
    }
    got = {}
    for name, (tables, expected) in scenarios.items():
        p = analyze.primary(tables, "t")
        got[name] = {"expected": expected, "outcome": p["outcome"], "reasons": p["inconclusive_reasons"],
                     "interval": p["comparison"]["interval"], "n": p["comparison"]["n"],
                     "note": p["comparison"].get("note"), "control_passes": p["positive_control"]["passes"]}
    record("known_answers", all(v["outcome"] == v["expected"] for v in got.values())
           and got["degenerate_bca"]["note"] == "BCa undefined; the t-interval alone", scenarios=got)

    # 2. Intervals.
    d = 100 + noise
    u = analyze.union_interval(d, 0.975)
    half = scipy.stats.t.ppf(0.9875, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    t_ok = np.allclose(u["t"], [d.mean() - half, d.mean() + half], rtol=0, atol=1e-9)
    sims, covered = 1000, 0
    sim_rng = np.random.default_rng(2)
    for _ in range(sims):
        x = sim_rng.normal(3.0, 10.0, 20)
        lo, hi = analyze.union_interval(x, 0.975)["interval"]
        covered += lo <= 3.0 <= hi
    record("intervals", t_ok and covered / sims >= 0.97, t_matches_formula=t_ok, coverage=covered / sims,
           simulations=sims, nominal=0.975, union=u["interval"], bca=u["bca"], t=u["t"])

    # 3. A study unit needs the tuning choice.
    smoke = out / "smoke"
    pre = subprocess.run([*CMD, "run", "--root", str(smoke / "gate"), "--phase", "study", "--task", "continuous",
                          "--arm", "bc", "--seed", "9002", "--smoke", "--device", args.device], capture_output=True, text=True)
    record("study_needs_the_tuning_choice", pre.returncode != 0 and "tuning-choice.json" in pre.stderr,
           exit=pre.returncode, stderr=pre.stderr.strip()[-200:])

    # 4. Smoke, end to end.
    root = smoke / "root"
    steps = {}

    def step(name, *a):
        p = subprocess.run([*CMD, *a, "--root", str(root), "--smoke"], capture_output=True, text=True)
        steps[name] = {"exit": p.returncode, "tail": (p.stdout + p.stderr).strip().splitlines()[-1:]}
        (smoke / f"{name}.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
        return p
    sweep = ["--jobs", str(args.jobs), "--device", args.device]
    step("tuning", "sweep", "--phase", "tuning", *sweep)
    step("choose", "choose")
    again = step("choose_again", "choose")
    step("study", "sweep", "--phase", "study", *sweep)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "results").rglob("*.json")}
    step("study_again", "sweep", "--phase", "study", *sweep)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "results").rglob("*.json")}
    recs = {ph: [json.loads(p.read_text()) for p in (root / "results" / ph).rglob("s*.json")] for ph in ("tuning", "study")}
    expected = {"tuning": len(study.units(root, "tuning", True)), "study": len(study.units(root, "study", True))}
    statuses = {ph: sorted({r["status"] for r in rs}) for ph, rs in recs.items()}
    ccil_labels = all("label_accuracy" in r for r in recs["study"] if r["arm"] in study.CCIL_ARMS)
    analysis_path = out / "smoke-analysis.json"
    an = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", str(root), "--out", str(analysis_path),
                         "--jobs", str(args.jobs)], capture_output=True, text=True)
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    record("smoke_end_to_end",
           all(steps[k]["exit"] == 0 for k in ("tuning", "choose", "study", "study_again")) and again.returncode != 0
           and before == after and {ph: len(rs) for ph, rs in recs.items()} == expected
           and statuses == {"tuning": ["ok"], "study": ["ok"]} and ccil_labels and an.returncode == 0
           and analysis.get("smoke") is True and analysis.get("rollouts_reaudited", 0) > 0,
           steps=steps, choose_again_refused=again.returncode != 0, sweep_again_changed_nothing=before == after,
           records={ph: len(rs) for ph, rs in recs.items()}, expected_records=expected, statuses=statuses,
           ccil_records_have_label_accuracy=ccil_labels, analysis_exit=an.returncode,
           analysis_stderr=an.stderr.strip()[-300:], rollouts_reaudited=analysis.get("rollouts_reaudited"),
           smoke_outcomes={t: v["primary"]["outcome"] for t, v in analysis.get("tasks", {}).items()},
           smoke_note="smoke outcomes are inconclusive by construction (2 seeds < 16); returns are not reported")

    # 5. The analysis refuses every tampering, each restored afterwards; and an existing output.
    def analysis_refuses(label, edit, expect):
        saved = edit()
        p = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", str(root), "--out",
                            str(out / f"refused-{label}.json"), "--jobs", str(args.jobs)], capture_output=True, text=True)
        for path, data in saved:
            if data is None:
                path.unlink()
            else:
                path.write_bytes(data)
        return {"refused": p.returncode != 0 and expect in p.stderr and not (out / f"refused-{label}.json").exists(),
                "expected_message": expect, "stderr": p.stderr.strip()[-400:]}

    def edit_json(path, change):
        data = path.read_bytes()
        obj = json.loads(data)
        change(obj)
        path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
        return [(path, data)]

    def add_file(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
        return [(path, None)]

    res = root / "results"
    bc_rec = res / "study" / "continuous" / "bc" / "s9002.json"
    tune_rec = next((res / "tuning" / "continuous").glob("noise_bc@*/s9001.json"))
    rollout = next((root / "rollouts" / "study").rglob("*.npz"))
    config = root / "inputs" / "config" / study.CONFIGS["continuous"]
    sub = next((root / "inputs" / "subsets").glob("*.npz"))
    other_sub = next(p for p in (root / "inputs" / "subsets").glob("*.npz") if p != sub)

    def bump_rollout():
        data = rollout.read_bytes()
        with np.load(rollout, allow_pickle=False) as z:
            r = {k: z[k] for k in z.files}
        r["rewards"][0, 0] += 1e-6
        np.savez(rollout, **r)
        return [(rollout, data)]

    def other_value(c):
        arm = c["continuous"]["noise_bc"]
        arm["value"] = next(v for v in study.NOISE_BC_GRID if v != arm["value"])

    def extra():
        obj = json.loads(bc_rec.read_text())
        obj["seed"] = 9004
        return add_file(bc_rec.with_name("s9004.json"), obj)

    def append(path, text):
        data = path.read_bytes()
        path.write_bytes(data + text)
        return [(path, data)]

    def swap_subset():
        data = sub.read_bytes()
        sub.write_bytes(other_sub.read_bytes())
        return [(sub, data)]

    refusals = {
        "changed_rollout": analysis_refuses("rollout", bump_rollout, "fails the re-audit"),
        "study_record_R": analysis_refuses("study-R", lambda: edit_json(
            bc_rec, lambda r: r["evaluations"]["0.0"].__setitem__("R", r["evaluations"]["0.0"]["R"] + 1.0)),
            "disagrees with its rollout's audit"),
        "tuning_record_R": analysis_refuses("tuning-R", lambda: edit_json(
            tune_rec, lambda r: r["evaluations"]["0.0"].__setitem__("R", r["evaluations"]["0.0"]["R"] + 1.0)),
            "disagrees with its rollout's audit"),
        "tuning_choice": analysis_refuses("choice", lambda: edit_json(root / "tuning-choice.json", other_value),
                                          "differs from its recomputation"),
        "extra_study_unit": analysis_refuses("extra", extra, "not exactly the study units"),
        "duplicate_study_unit": analysis_refuses("duplicate", lambda: add_file(
            res / "study" / "continuous" / "bc_copy" / "s9002.json", json.loads(bc_rec.read_text())),
            "not exactly the study units"),
        "changed_corrected_config": analysis_refuses("config", lambda: append(config, b"# edited\n"),
                                                     "differ from the committed conversion"),
        "changed_reused_subset": analysis_refuses("subset", swap_subset, "differs from its rebuild"),
    }
    exists = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", str(root), "--out", str(analysis_path)],
                            capture_output=True, text=True)
    saved = append(config, b"# edited\n")
    run_refused = subprocess.run([*CMD, "run", "--root", str(root), "--phase", "study", "--task", "continuous",
                                  "--arm", "bc", "--seed", "9004", "--smoke", "--device", args.device],
                                 capture_output=True, text=True)
    config.write_bytes(saved[0][1])
    refusals["unit_with_changed_config"] = {"refused": run_refused.returncode != 0
                                            and "differ from the committed conversion" in run_refused.stderr,
                                            "stderr": run_refused.stderr.strip()[-300:]}
    clean_after = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", str(root), "--out",
                                  str(out / "smoke-analysis-after.json"), "--jobs", str(args.jobs)], capture_output=True, text=True)
    record("analysis_refuses", all(v["refused"] for v in refusals.values()) and exists.returncode != 0
           and clean_after.returncode == 0, cases=refusals, existing_output=exists.stderr.strip()[-200:],
           restored_root_analyzes=clean_after.returncode == 0)

    # 6. The frozen-tooling gate of real runs, on scratch clones of this commit.
    head = provenance.tree_state(REPO)["head"]
    clone = out / "tooling-clone"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(REPO), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", head], check=True)
    git = ["git", "-C", str(clone), "-c", "user.name=check", "-c", "user.email=check@invalid"]
    tooling = {"no_manifest": provenance.gate(clone, real=True),
               "smoke_without_frozen_tooling": provenance.gate(clone, real=False)}
    cli = subprocess.run([sys.executable, str(clone / "tools/m3_pendulum/study.py"), "run", "--root", str(out / "real-root"),
                          "--phase", "tuning", "--task", "continuous", "--arm", "noise_bc", "--seed", "1",
                          "--value", "0.0001", "--device", args.device], capture_output=True, text=True)
    (clone / provenance.FREEZE).write_text(json.dumps({"tooling_commit": head}) + "\n")
    subprocess.run([*git, "add", provenance.FREEZE], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "freeze"], check=True)
    tooling["frozen_at_this_commit"] = provenance.gate(clone, real=True)
    readme = clone / "tools/m3_pendulum/README.md"
    readme.write_text(readme.read_text() + "\nchanged\n")
    subprocess.run([*git, "commit", "-q", "-am", "change the tooling"], check=True)
    tooling["committed_tooling_change"] = provenance.gate(clone, real=True)
    (clone / provenance.FREEZE).write_text(json.dumps({"tooling_commit": "0" * 40}) + "\n")
    subprocess.run([*git, "commit", "-q", "-am", "name no commit"], check=True)
    tooling["manifest_names_no_commit"] = provenance.gate(clone, real=True)
    oks = {k: v["ok"] for k, v in tooling.items()}
    record("frozen_tooling_gate", oks == {"no_manifest": False, "frozen_at_this_commit": True,
                                          "smoke_without_frozen_tooling": True, "committed_tooling_change": False,
                                          "manifest_names_no_commit": False}
           and cli.returncode != 0 and "freeze manifest" in cli.stderr and not (out / "real-root" / "results").exists(),
           gate_ok=oks, differing=tooling["committed_tooling_change"]["tooling"].get("differing_from_frozen_tooling"),
           real_run_without_manifest=cli.stderr.strip()[-300:])

    state = provenance.tree_state(REPO)
    record("clean_tree_at_frozen_ccil", state["ok"], **{k: v for k, v in state.items() if k != "ok"})
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": state["head"], "code_dirty": bool(state["dirty"]),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": scipy.__version__, "device": args.device}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
