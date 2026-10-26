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
5. The analysis refuses a changed rollout, and an existing output.
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

    # 5. The analysis refuses a changed rollout and an existing output.
    one = next((root / "rollouts" / "study").rglob("*.npz"))
    original = one.read_bytes()
    with np.load(one, allow_pickle=False) as z:
        r = {k: z[k] for k in z.files}
    r["rewards"][0, 0] += 1e-6
    np.savez(one, **r)
    tampered = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", str(root), "--out",
                               str(out / "tampered-analysis.json"), "--jobs", str(args.jobs)], capture_output=True, text=True)
    one.write_bytes(original)
    exists = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", str(root), "--out", str(analysis_path)],
                            capture_output=True, text=True)
    record("analysis_refuses", tampered.returncode != 0 and "re-audit" in tampered.stderr
           and not (out / "tampered-analysis.json").exists() and exists.returncode != 0,
           changed_rollout=tampered.stderr.strip()[-300:], existing_output=exists.stderr.strip()[-200:])

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
