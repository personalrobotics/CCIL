#!/usr/bin/env python3
"""The #24 analysis (predeclaration §§8-9): decisions, intervals, and the report tables.

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_pendulum/analyze.py --root ROOT --out ANALYSIS.json [--jobs N]

load() reads every unit record of ROOT, refuses a mix of smoke and real runs
or of code states, and re-audits every saved rollout with the replay
(audit.py), refusing any mismatch or changed file. analyze() then works on
the R tables alone:
- primary, per task: CCIL - BC with the union of the BCa and paired-t
  intervals at c = 0.975, the positive control (BC - BC at 10%), the §9
  inconclusive rules, and the relative effect with its paired BCa interval;
- secondary comparisons, per noise level and data size, and the zero-action
  reference, each with the union interval at c = 0.95, descriptive only;
- the tuning tables and choice, label accuracy, failures and exclusions, and
  the paper comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
import scipy.stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit as auditor  # noqa: E402
import study  # noqa: E402

VERSION = "m3_pendulum.analysis/1"
RNG_SEED, RESAMPLES = 20261026, 10000
PRIMARY_C, SECONDARY_C = 0.975, 0.95
MIN_VALID, MAX_EXCLUDED = 16, 2
PAPER = {"continuous": {"bc": -3244, "ccil": -2269}, "discontinuous": {"bc": -3204, "ccil": -2556}}
SECONDARY = [("ccil", "noise_bc"), ("ccil_tuned", "bc"), ("ccil_tuned", "noise_bc"), ("ccil_filtered", "ccil"),
             ("ccil_nolip", "ccil")]
NOISY_ARMS = ("bc", "noise_bc", "ccil", "ccil_tuned")
NOISE_FREE = "0.0"


# --- intervals ------------------------------------------------------------------------------------------------------
def union_interval(d, c) -> dict:
    """The §8 interval of the mean of d: the union of BCa bootstrap and paired t at level c."""
    d = np.asarray(d, np.float64)
    n = len(d)
    out = {"n": n, "mean": float(d.mean()) if n else None, "confidence": c}
    if n < 2:
        return {**out, "interval": None, "note": "fewer than 2 paired values"}
    sd = float(d.std(ddof=1))
    half = float(scipy.stats.t.ppf((1 + c) / 2, n - 1)) * sd / np.sqrt(n)
    t = [float(d.mean() - half), float(d.mean() + half)]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = scipy.stats.bootstrap((d,), np.mean, confidence_level=c, n_resamples=RESAMPLES, method="BCa",
                                       rng=np.random.default_rng(RNG_SEED)).confidence_interval
    bca = [float(ci.low), float(ci.high)]
    if np.isfinite(bca).all():
        union, note = [min(bca[0], t[0]), max(bca[1], t[1])], None
    else:
        bca, union, note = None, t, "BCa undefined; the t-interval alone"
    return {**out, "bca": bca, "t": t, "interval": union, "note": note}


def relative_effect(d, r_bc, c=PRIMARY_C) -> dict:
    d, b = np.asarray(d, np.float64), np.asarray(r_bc, np.float64)
    value = float(d.mean() / abs(b.mean()))
    if len(d) < 2:
        return {"value": value, "bca": None}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = scipy.stats.bootstrap((d, b), lambda x, y: x.mean() / abs(y.mean()), paired=True, vectorized=False,
                                   confidence_level=c, n_resamples=RESAMPLES, method="BCa",
                                   rng=np.random.default_rng(RNG_SEED)).confidence_interval
    bca = [float(ci.low), float(ci.high)]
    return {"value": value, "bca": bca if np.isfinite(bca).all() else None}


# --- tables ---------------------------------------------------------------------------------------------------------
def paired(tables, task, a, b, noise=NOISE_FREE):
    """Seeds valid for both arms (status ok), with R(s, a) and R(s, b)."""
    ra, rb = tables["R"][task].get(a, {}).get(noise, {}), tables["R"][task].get(b, {}).get(noise, {})
    seeds = sorted(s for s in ra if s in rb and tables["status"][task][a][s] == "ok" and tables["status"][task][b][s] == "ok")
    return seeds, np.array([ra[s] for s in seeds]), np.array([rb[s] for s in seeds])


def counts(tables, task, arm) -> dict:
    st = tables["status"][task].get(arm, {})
    return {k: sorted(s for s, v in st.items() if v == k) for k in ("ok", "algorithmic_failure", "infrastructure_fault")}


def compare(tables, task, a, b, c, noise=NOISE_FREE) -> dict:
    seeds, ra, rb = paired(tables, task, a, b, noise)
    return {"A": a, "B": b, "noise": noise, "seeds": seeds, **union_interval(ra - rb, c),
            "per_seed": {s: float(x - y) for s, x, y in zip(seeds, ra, rb)}}


def primary(tables, task) -> dict:
    c_bc, c_ccil = counts(tables, task, "bc"), counts(tables, task, "ccil")
    res = compare(tables, task, "ccil", "bc", PRIMARY_C)
    reasons = []
    if c_bc["algorithmic_failure"] or c_ccil["algorithmic_failure"]:
        reasons.append(f"algorithmic failure in a primary arm (bc {c_bc['algorithmic_failure']}, "
                       f"ccil {c_ccil['algorithmic_failure']})")
    if max(len(c_bc["infrastructure_fault"]), len(c_ccil["infrastructure_fault"])) > MAX_EXCLUDED:
        reasons.append("more than 2 infrastructure exclusions in a primary arm")
    if res["n"] < MIN_VALID:
        reasons.append(f"{res['n']} seeds valid for both primary arms (< {MIN_VALID})")
    control = positive_control(tables, task)
    lo_hi = res.get("interval")
    if reasons or lo_hi is None:
        outcome = "inconclusive"
    elif lo_hi[0] > 0:
        outcome = "ccil_improves_on_bc"
    elif lo_hi[1] < 0:
        outcome = "ccil_is_worse_than_bc"
    elif control["passes"]:
        outcome = "no_detectable_difference"
    else:
        outcome, reasons = "inconclusive", ["the interval contains 0 and the positive control fails"]
    seeds, _, rb = paired(tables, task, "ccil", "bc")
    rel = relative_effect(np.array(list(res["per_seed"].values())), rb) if seeds else None
    return {"comparison": res, "outcome": outcome, "inconclusive_reasons": reasons, "positive_control": control,
            "relative_effect": rel, "status": {"bc": c_bc, "ccil": c_ccil}}


def positive_control(tables, task) -> dict:
    res = compare(tables, task, "bc", "bc_frac10", PRIMARY_C)
    fails = []
    for arm in ("bc", "bc_frac10"):
        if counts(tables, task, arm)["algorithmic_failure"]:
            fails.append(f"algorithmic failure in {arm}")
    if res["n"] < MIN_VALID:
        fails.append(f"{res['n']} seeds valid for both (< {MIN_VALID})")
    if not fails and not (res["interval"] is not None and res["interval"][0] > 0):
        fails.append("the lower endpoint is not above 0")
    return {"comparison": res, "passes": not fails, "failures": fails}


def analyze(tables) -> dict:
    out = {"version": VERSION, "rng_seed": RNG_SEED, "resamples": RESAMPLES, "scipy": scipy.__version__,
           "numpy": np.__version__, "tasks": {}}
    for task in tables["R"]:
        arms = tables["R"][task]
        t = {"primary": primary(tables, task)}
        t["secondary"] = [compare(tables, task, a, b, SECONDARY_C) for a, b in SECONDARY]
        t["by_noise"] = [compare(tables, task, "ccil", "bc", SECONDARY_C, n) for n in arms["bc"] if n != NOISE_FREE]
        t["by_data_size"] = [compare(tables, task, f"ccil_frac{p}", f"bc_frac{p}", SECONDARY_C) for p in (10, 25, 50)]
        t["zero_reference"] = (
            [compare(tables, task, a, "zero", SECONDARY_C) for a in arms if a != "zero"]
            + [compare(tables, task, a, "zero", SECONDARY_C, n) for a in NOISY_ARMS for n in arms["zero"]
               if n != NOISE_FREE])
        t["arms"] = {a: {n: {"mean": float(np.mean(v_ok)) if (v_ok := [r for s, r in rs.items()
                                                                         if tables["status"][task][a][s] == "ok"]) else None,
                             "std": float(np.std(v_ok, ddof=1)) if len(v_ok) > 1 else None, "n": len(v_ok),
                             "per_seed": rs} for n, rs in by_noise.items()}
                     for a, by_noise in arms.items()}
        t["status"] = {a: counts(tables, task, a) for a in arms}
        t["paper_comparison"] = {a: {"paper": PAPER[task][a], "this_study": t["arms"][a][NOISE_FREE]}
                                 for a in ("bc", "ccil")}
        out["tasks"][task] = t
    return out


# --- loading --------------------------------------------------------------------------------------------------------
def _audit_one(args):
    root, rel, expected = args
    path = Path(root) / rel
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    a = auditor.audit(path)
    return rel, got == expected, a["ok"], a["issues"]


def load(root, jobs=8) -> tuple[dict, dict]:
    root = Path(root).resolve()
    records = [json.loads(p.read_text()) for p in sorted((root / "results" / "study").rglob("s*.json"))
               if ".fault" not in p.name]
    if not records:
        raise SystemExit(f"no study records under {root}")
    smoke = {r["smoke"] for r in records}
    heads = {r["gate"]["head"] for r in records}
    choice = json.loads((root / "tuning-choice.json").read_text())
    choice_sha = hashlib.sha256((root / "tuning-choice.json").read_bytes()).hexdigest()
    problems = []
    if len(smoke) != 1:
        problems.append("smoke and real records are mixed")
    if len(heads) != 1:
        problems.append(f"records come from {len(heads)} code states: {sorted(heads)}")
    if not all(r["gate"]["ok"] for r in records):
        problems.append("a record was made without a passing code-state gate")
    if {r.get("tuning_choice_sha256") for r in records} != {choice_sha}:
        problems.append("a study record does not carry the tuning choice's SHA-256")
    expected = set(study.units(root, "study", smoke == {True}))
    have = {(r["task"], r["arm"], r["seed"], r["value"]) for r in records}
    if expected - have:
        problems.append(f"{len(expected - have)} study units have no record, e.g. {sorted(expected - have)[:3]}")
    jobs_list = [(str(root), e["rollout"], e["rollout_sha256"]) for r in records for e in r.get("evaluations", {}).values()]
    with ProcessPoolExecutor(jobs) as pool:
        audits = list(pool.map(_audit_one, jobs_list, chunksize=4))
    bad = [(rel, same, ok, issues) for rel, same, ok, issues in audits if not (same and ok)]
    if bad:
        problems.append(f"{len(bad)} rollouts fail the re-audit, e.g. {bad[:3]}")
    if problems:
        raise SystemExit("analysis refused: " + "; ".join(problems))
    tables = {"R": {}, "status": {}}
    for r in records:
        t, a, s = r["task"], r["arm"], r["seed"]
        tables["status"].setdefault(t, {}).setdefault(a, {})[s] = r["status"]
        for n in study.ARMS[a][1]:
            ev = r.get("evaluations", {}).get(str(n))
            tables["R"].setdefault(t, {}).setdefault(a, {}).setdefault(str(n), {})[s] = ev["R"] if ev else None
    extras = {"smoke": smoke == {True}, "code_commit": heads.pop(), "rollouts_reaudited": len(audits),
              "tuning_choice": choice, "tuning_choice_sha256": choice_sha,
              "label_accuracy": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["label_accuracy"]
                                 for r in records if "label_accuracy" in r},
              "dynamics_held_out": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["dynamics_held_out"]
                                    for r in records if "dynamics_held_out" in r},
              "failures": [{k: r.get(k) for k in ("task", "arm", "seed", "status", "failure", "exit", "stderr_tail")}
                           for r in records if r["status"] != "ok"],
              "versions": sorted({json.dumps(r["versions"], sort_keys=True) for r in records})}
    return tables, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; analyses are never overwritten")
    tables, extras = load(a.root, a.jobs)
    result = {**analyze(tables), **extras}
    study.atomic_json(result, a.out)
    for task, t in result["tasks"].items():
        p = t["primary"]
        print(f"{task}: {p['outcome']} (CCIL - BC interval {p['comparison']['interval']}, n {p['comparison']['n']}); "
              f"positive control {'passes' if p['positive_control']['passes'] else 'fails'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
