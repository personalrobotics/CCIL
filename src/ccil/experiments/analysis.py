"""Predeclared paired comparisons and decisions (#112), extracted from the #24 analysis (tools/m3_pendulum/analyze.py).

Tables are {"R": {task: {arm: {noise: {seed: R}}}}, "status": {task: {arm: {seed: status}}}}, with R taken from
audited rollouts. Every rule is a parameter of a Decision (the predeclaration), so another suite reuses the same
statistics with its own arms and thresholds:
- union_interval: the union of a BCa bootstrap interval and a paired t-interval at level c (the t-interval alone
  when BCa is undefined);
- relative_effect: mean(d) / |mean(R_baseline)| with its paired BCa interval;
- compare: the paired difference A - B over the seeds valid for both arms;
- decide: the primary outcome of A - B, with the positive control and the failure and validity rules.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.stats

RNG_SEED, RESAMPLES = 20261026, 10000
STATUSES = ("ok", "algorithmic_failure", "infrastructure_fault")


def union_interval(d, c: float, rng_seed: int = RNG_SEED, resamples: int = RESAMPLES) -> dict:
    """The interval of the mean of d: the union of BCa bootstrap and paired t at level c."""
    d = np.asarray(d, np.float64)
    n = len(d)
    out = {"n": n, "mean": float(d.mean()) if n else None, "confidence": c}
    if n < 2:
        return {**out, "interval": None, "note": "fewer than 2 paired values"}
    half = float(scipy.stats.t.ppf((1 + c) / 2, n - 1)) * float(d.std(ddof=1)) / np.sqrt(n)
    t = [float(d.mean() - half), float(d.mean() + half)]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        ci = scipy.stats.bootstrap((d,), np.mean, confidence_level=c, n_resamples=resamples, method="BCa",
                                   rng=np.random.default_rng(rng_seed)).confidence_interval
    bca = [float(ci.low), float(ci.high)]
    if np.isfinite(bca).all():
        return {**out, "bca": bca, "t": t, "interval": [min(bca[0], t[0]), max(bca[1], t[1])], "note": None}
    return {**out, "bca": None, "t": t, "interval": t, "note": "BCa undefined; the t-interval alone"}


def relative_effect(d, baseline, c: float, rng_seed: int = RNG_SEED, resamples: int = RESAMPLES) -> dict:
    """mean(d) / |mean(baseline)| with its paired BCa interval."""
    d, b = np.asarray(d, np.float64), np.asarray(baseline, np.float64)
    value = float(d.mean() / abs(b.mean()))
    if len(d) < 2:
        return {"value": value, "bca": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = scipy.stats.bootstrap((d, b), lambda x, y: x.mean() / abs(y.mean()), paired=True, vectorized=False,
                                   confidence_level=c, n_resamples=resamples, method="BCa",
                                   rng=np.random.default_rng(rng_seed)).confidence_interval
    bca = [float(ci.low), float(ci.high)]
    return {"value": value, "bca": bca if np.isfinite(bca).all() else None}


def paired(tables, task, a, b, noise):
    """Seeds valid for both arms (status ok), with R(s, a) and R(s, b)."""
    ra, rb = tables["R"][task].get(a, {}).get(noise, {}), tables["R"][task].get(b, {}).get(noise, {})
    st = tables["status"][task]
    seeds = sorted(s for s in ra if s in rb and st[a][s] == "ok" and st[b][s] == "ok")
    return seeds, np.array([ra[s] for s in seeds]), np.array([rb[s] for s in seeds])


def counts(tables, task, arm) -> dict:
    st = tables["status"][task].get(arm, {})
    return {k: sorted(s for s, v in st.items() if v == k) for k in STATUSES}


def compare(tables, task, a, b, c, noise) -> dict:
    seeds, ra, rb = paired(tables, task, a, b, noise)
    return {"A": a, "B": b, "noise": noise, "seeds": seeds, **union_interval(ra - rb, c),
            "per_seed": {s: float(x - y) for s, x, y in zip(seeds, ra, rb)}}


@dataclass(frozen=True)
class Decision:
    """A predeclared primary decision: treatment - baseline, with a positive control and validity rules."""

    treatment: str
    baseline: str
    control: tuple[str, str]       # (stronger, weaker): stronger - weaker must be detectable
    confidence: float              # e.g. 0.975 for two Bonferroni-adjusted tasks
    min_valid: int                 # fewer seeds valid for both primary arms is inconclusive
    max_excluded: int              # more infrastructure exclusions in a primary arm is inconclusive
    noise: str = "0.0"
    outcomes: tuple[str, str, str] = ("treatment_improves", "treatment_is_worse", "no_detectable_difference")


def positive_control(tables, task, d: Decision) -> dict:
    res = compare(tables, task, d.control[0], d.control[1], d.confidence, d.noise)
    fails = [f"algorithmic failure in {arm}" for arm in d.control if counts(tables, task, arm)["algorithmic_failure"]]
    if res["n"] < d.min_valid:
        fails.append(f"{res['n']} seeds valid for both (< {d.min_valid})")
    if not fails and not (res["interval"] is not None and res["interval"][0] > 0):
        fails.append("the lower endpoint is not above 0")
    return {"comparison": res, "passes": not fails, "failures": fails}


def decide(tables, task, d: Decision) -> dict:
    """The primary outcome: an algorithmic failure in a primary arm, too many exclusions or too few valid seeds is
    inconclusive; otherwise the interval decides, and a null stands only when the positive control passes."""
    ct, cb = counts(tables, task, d.treatment), counts(tables, task, d.baseline)
    res = compare(tables, task, d.treatment, d.baseline, d.confidence, d.noise)
    reasons = []
    if ct["algorithmic_failure"] or cb["algorithmic_failure"]:
        reasons.append(f"algorithmic failure in a primary arm ({d.baseline} {cb['algorithmic_failure']}, "
                       f"{d.treatment} {ct['algorithmic_failure']})")
    if max(len(ct["infrastructure_fault"]), len(cb["infrastructure_fault"])) > d.max_excluded:
        reasons.append(f"more than {d.max_excluded} infrastructure exclusions in a primary arm")
    if res["n"] < d.min_valid:
        reasons.append(f"{res['n']} seeds valid for both primary arms (< {d.min_valid})")
    control = positive_control(tables, task, d)
    iv = res.get("interval")
    if reasons or iv is None:
        outcome = "inconclusive"
    elif iv[0] > 0:
        outcome = d.outcomes[0]
    elif iv[1] < 0:
        outcome = d.outcomes[1]
    elif control["passes"]:
        outcome = d.outcomes[2]
    else:
        outcome, reasons = "inconclusive", ["the interval contains 0 and the positive control fails"]
    seeds, _, rb = paired(tables, task, d.treatment, d.baseline, d.noise)
    rel = relative_effect(np.array(list(res["per_seed"].values())), rb, d.confidence) if seeds else None
    return {"comparison": res, "outcome": outcome, "inconclusive_reasons": reasons, "positive_control": control,
            "relative_effect": rel, "status": {d.baseline: cb, d.treatment: ct}}
