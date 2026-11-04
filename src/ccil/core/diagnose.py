"""Diagnose: the data-only recovery-length decision made before generation (#167).

    demonstrations + dynamics -> probe -> gates -> admitted M (or none, with a reason)

Only the demonstrations, their split, the transition model, and a model that
never saw the held-out episodes enter it; no environment, reward, policy, or
reference dynamics. The ladder is {1} until recovery segments (#171) exist.

The probe is a fixed, evenly spaced subset of the training transitions: the
labels the configured generator and validators would produce there, measured
in the demonstration metric (core/quality.py) and then discarded. Its gates:
- yield: accepted / produced probe labels >= diagnose.min_yield;
- reach: eta_reach = r_reach / tau_demo >= diagnose.min_eta, r_reach the
  diagnose.reach_quantile of accepted d_source;
- trust: the diagnose.trust_quantile of the held-out one-step model error,
  divided by the median accepted d_source, <= diagnose.trust_ratio: the model's
  error must be small next to the correction it supports.
The document it returns is a ccil.diagnosis (schema/diagnosis.py).
"""
from __future__ import annotations

import numpy as np

from ..schema.diagnosis import SCHEMA, STATEMENT, VERSION, check
from .labels import generate_labels
from .quality import DEFINITION, SourceDisplacement, tau_demo

LADDER = (1,)
BAND_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, float("inf"))
NOISY = ("noisy_action", "noisy_action_preimage")
BOUND_PROFILES = ("continuity", "backtrack", "disturbed_action")


def probe_rows(batch_episode_ids, train_episodes, n: int) -> np.ndarray:
    """Evenly spaced indices of up to n training transitions, in transitions() order."""
    train = np.flatnonzero(np.isin(np.asarray(batch_episode_ids).astype(str), list(train_episodes)))
    if len(train) <= n:
        return train
    return train[np.unique(np.round(np.linspace(0, len(train) - 1, n)).astype(np.int64))]


def timing(dataset) -> dict:
    """dt_ctrl from the episodes' timestamps when every episode has them and they are uniform."""
    if any(ep.timestamps is None for ep in dataset.episodes):
        return {"dt_ctrl": None, "source": None,
                "note": "no timestamps: recovery lengths are in dataset action intervals and the physical horizon is unknown"}
    dts = np.concatenate([np.diff(np.asarray(ep.timestamps, np.float64)) for ep in dataset.episodes])
    if not len(dts) or not (dts > 0).all():
        return {"dt_ctrl": None, "source": "timestamps", "note": "timestamps do not increase: no action period"}
    dt = float(np.median(dts))
    if not np.allclose(dts, dt, rtol=1e-6, atol=0.0):
        return {"dt_ctrl": None, "source": "timestamps",
                "note": "the action period varies: no single dt_ctrl, and the physical horizon is not claimed"}
    return {"dt_ctrl": dt, "source": "timestamps", "note": "the median timestamp difference; every difference agrees"}


def evidence_class(profile: str | None, reference: bool) -> dict:
    """The strongest class the configured certificate can give from data alone."""
    if profile is None:
        cls = "uncertified"
    elif profile not in BOUND_PROFILES:
        cls = "empirical"
    elif reference:
        cls = "none from data alone: its constants come from reference dynamics"
    else:
        cls = "conditional"
    return {"profile": profile, "data_only_class": cls}


def _q(x, q):
    return None if not len(x) else float(np.quantile(x, q))


def diagnose(dataset, split, representation, model, evaluation_model, held_out, generator, validators, metric, *,
             requested, params: dict, seed: int, inputs: dict, profile=None, reference=False,
             labels_per_transition=None) -> dict:
    """The ccil.diagnosis of one configuration; see the module docstring.

    held_out is (z, a, z_next) of the held-out transitions in the model's direction, or None; evaluation_model never
    saw them. requested is "auto" or an int. params holds the diagnose.* thresholds.
    """
    tau = tau_demo(metric, params["tau_quantile"])
    batch_ids = np.concatenate([[ep.episode_id] * (ep.length - 1) for ep in dataset.episodes])
    rows = probe_rows(batch_ids, metric.train_episodes, params["probe_transitions"])
    probe = generate_labels(dataset, representation, model, generator, list(validators) + [SourceDisplacement(metric)],
                            run_id="diagnose-probe", seed=seed, rows=rows, inputs={"transition_model": "0" * 64})
    accepted = np.asarray(probe.accepted, bool)
    d = np.asarray(probe.measurements["quality.d_source"])[accepted]
    timing_ = timing(dataset)
    yield_ = float(accepted.mean()) if len(accepted) else 0.0
    r_reach = _q(d, params["reach_quantile"])
    median_reach = _q(d, 0.5)
    eta = None if r_reach is None or not tau else r_reach / tau
    eps = None
    if held_out is not None and evaluation_model is not None and len(held_out[0]):
        z, a, z_next = (np.asarray(x, np.float64) for x in held_out)
        pred = z + np.asarray(evaluation_model.delta(held_out[0], held_out[1]), np.float64)
        eps = _q(np.linalg.norm((pred - z_next) / metric.scale, axis=1), params["trust_quantile"])
    ratio = None if eps is None or not median_reach else eps / median_reach
    gate = {
        "M": 1,
        "horizon_s": None if timing_["dt_ctrl"] is None else timing_["dt_ctrl"],
        "yield": {"value": yield_, "accepted": int(accepted.sum()), "candidates": int(len(accepted)),
                  "threshold": params["min_yield"], "pass": bool(len(accepted)) and yield_ >= params["min_yield"]},
        "reach": {"eta_reach": eta, "r_reach": r_reach, "quantile": params["reach_quantile"], "tau_demo": tau,
                  "threshold": params["min_eta"], "pass": eta is not None and eta >= params["min_eta"]},
        "trust": {"ratio": ratio, "epsilon": eps, "quantile": params["trust_quantile"], "median_reach": median_reach,
                  "threshold": params["trust_ratio"], "pass": ratio is not None and ratio <= params["trust_ratio"],
                  "model": "the evaluation model, fitted on the training episodes, scored on the held-out transitions"},
    }
    gate["pass"] = gate["yield"]["pass"] and gate["reach"]["pass"] and gate["trust"]["pass"]
    if requested == "auto":
        admitted, by = (1, "gates") if gate["pass"] else (None, None)
    else:
        admitted, by = int(requested), "declared"
    reason = next_step = None
    if admitted is None:
        reason, next_step = _why(gate, tau, eps)
    bands = None
    if tau:
        counts, _ = np.histogram(d / tau, bins=np.array(BAND_EDGES))
        bands = {"edges": list(BAND_EDGES[:-1]) + ["inf"], "counts": [int(c) for c in counts],
                 "of": "accepted probe labels, by d_source / tau_demo"}
    nested = None
    if generator.name in NOISY and labels_per_transition:
        k = int(labels_per_transition)
        nested = sorted({1, max(1, k // 2), k})
    doc = {
        "schema": SCHEMA, "version": f"{VERSION[0]}.{VERSION[1]}", "evidence_mode": "data_only",
        "recovery_steps": {"requested": requested if requested == "auto" else int(requested), "ladder": list(LADDER),
                           "cap": max(LADDER)},
        "metric": metric.spec(), "timing": timing_,
        "tau_demo": {"value": tau, "quantile": params["tau_quantile"],
                     "definition": "quantile of the distance from each held-out demonstration state to the nearest "
                                   "training state: " + DEFINITION},
        "probe": {"transitions": int(len(rows)), "rule": "evenly spaced training transitions, in transitions() order",
                  "max_transitions": params["probe_transitions"], "candidates": int(len(accepted))},
        "gates": [gate], "admitted": admitted, "admitted_by": by, "reason": reason, "next_step": next_step,
        "bands": bands, "nested_views": nested, "evidence_class": evidence_class(profile, reference),
        "statement": STATEMENT, "inputs": inputs,
    }
    return check(_finite(doc))


def _why(gate, tau, eps) -> tuple[str, str]:
    """The first failing gate of M = 1, and what could change it."""
    if tau is None:
        return ("tau_demo needs held-out demonstration episodes, and there are none",
                "hold out episodes (data.validation_fraction > 0) so the demonstrations' spacing and the model's "
                "held-out error can be measured")
    if not gate["yield"]["pass"]:
        return (f"only {gate['yield']['value']:.3g} of the probe labels are accepted "
                f"(threshold {gate['yield']['threshold']})",
                "check the generator and validity settings (support radius, action bounds, model misfit) and the "
                "dynamics model's fit")
    if not gate["reach"]["pass"]:
        eta = gate["reach"]["eta_reach"]
        return (f"eta_reach = {eta:.3g} < {gate['reach']['threshold']}: the labels stay within the demonstrations' own "
                "spacing, so they add no coverage the demonstrations lack",
                "collect recovery demonstrations where the policy fails, or use a generator that reaches further "
                "where the model is trusted")
    if eps is None:
        return ("the model's held-out error cannot be measured without held-out transitions",
                "hold out episodes (data.validation_fraction > 0)")
    return (f"the model's held-out error is {gate['trust']['ratio']:.3g} of the median correction "
            f"(threshold {gate['trust']['threshold']})",
            "improve the dynamics model or collect more demonstrations before trusting its corrections")


def _finite(x):
    """JSON-safe: non-finite floats become null."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


__all__ = ["diagnose", "probe_rows", "timing", "evidence_class", "LADDER"]
