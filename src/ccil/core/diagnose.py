"""Diagnose: the data-only recovery-length decision made before generation (#167).

    demonstrations + dynamics -> probe -> gates -> admitted M (or none, with a reason)

Only the demonstrations, their split, the transition model, a model that never
saw the held-out episodes, and the configured certificate's declared
assumptions enter it; no environment, reward, policy, or reference dynamics.
The ladder is {1} until recovery segments (#171) exist.

The probe is a fixed sample of the training transitions, stratified by
episode under diagnose.probe_transitions (it covers min(episodes, budget)
episodes): the labels the configured generator and validators would produce
there, measured in the demonstration metric (core/quality.py) and then
discarded. Its gates are distinct questions, and M is admitted only when all
pass:
- yield: did generation work? accepted / produced >= diagnose.min_yield.
- displacement: how far did labels move from their sources?
  q(d_source) / tau_demo >= diagnose.min_displacement.
- novelty: do they reach states the dataset does not already cover?
  q(d_demo) / tau_demo >= diagnose.min_novelty, d_demo the exact distance to
  the nearest training state.
- fit: does the model fit where it has data? The held-out one-step error
  (diagnose.fit_quantile), over the median d_source, <= diagnose.fit_ratio.
  Necessary, never sufficient: it is measured at demonstration states.
- trust: is the model credible at the proposed states? Data cannot show
  that off the data, so this gate passes only with off-data evidence: a bound
  certificate profile whose constants are declared assumptions or computed
  from the model, never from reference dynamics. Its class is recorded. The
  distance of each proposed query (z_G, a_G) from the nearest training (z, a)
  pair is reported beside it, as the measure of how far the model is asked to
  extrapolate.
tau_demo uses a deterministic sample of at most diagnose.tau_sample held-out
states, so its cost is linear in the training set. The document returned is
a ccil.diagnosis (schema/diagnosis.py).
"""
from __future__ import annotations

import numpy as np

from ..schema.diagnosis import SCHEMA, STATEMENT, VERSION, check
from .labels import generate_labels
from .quality import DEFINITION, SourceDisplacement, nearest_distance
from .representation import observation, transitions

LADDER = (1,)
BAND_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, float("inf"))
NOISY = ("noisy_action", "noisy_action_preimage")
BOUND_PROFILES = ("continuity", "backtrack", "disturbed_action")


def spread(n: int, k: int) -> np.ndarray:
    """k evenly spaced indices of range(n) (all of them when k >= n)."""
    if k >= n:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(np.int64))


def probe_rows(batch_episode_ids, train_episodes, budget: int) -> tuple[np.ndarray, dict]:
    """Training transitions stratified by episode under budget, in transitions() order, and their coverage.

    With E training episodes and budget B: when E <= B every episode gets B // E (+1 for the first B mod E)
    evenly spaced transitions; when E > B, B evenly spaced episodes get one each, at their middle.
    """
    ids = np.asarray(batch_episode_ids).astype(str)
    order = [e for e in dict.fromkeys(ids) if e in set(train_episodes)]
    rows = {e: np.flatnonzero(ids == e) for e in order}
    E = len(order)
    if E <= budget:
        quota = {e: budget // E + (i < budget % E) for i, e in enumerate(order)} if E else {}
        chosen = [rows[e][spread(len(rows[e]), quota[e])] for e in order]
    else:
        chosen = [rows[order[i]][[len(rows[order[i]]) // 2]] for i in spread(E, budget)]
    picked = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)
    covered = len({ids[i] for i in picked})
    return picked, {"training_episodes": E, "episodes_probed": covered, "transitions": int(len(picked)),
                    "rule": "stratified by training episode under the budget: every episode when there are at most "
                            "as many as the budget, otherwise budget evenly spaced episodes, one transition each"}


def sampled_tau(metric, quantile: float, sample: int) -> tuple[float | None, dict]:
    """tau_demo from at most sample evenly spaced held-out states, against every training state."""
    held = metric.held_out_states
    if not len(held) or not len(metric.train_states):
        return None, {"held_out_states": int(len(held)), "sampled": 0}
    idx = spread(len(held), sample)
    d = nearest_distance(held[idx], metric.train_states, metric.scale)
    return float(np.quantile(d, quantile)), {"held_out_states": int(len(held)), "sampled": int(len(idx))}


def timing(dataset) -> dict:
    """dt_ctrl from the episodes' timestamps when every episode has them and they are uniform.

    Uniform means every difference agrees with the median to within the timestamps' own storage precision: four
    machine epsilons of their dtype at the largest timestamp (float32 LeRobot timestamps differ by about 2e-5
    relative at 0.1 s), and never tighter than 1e-6 relative."""
    if any(ep.timestamps is None for ep in dataset.episodes):
        return {"dt_ctrl": None, "source": None,
                "note": "no timestamps: recovery lengths are in dataset action intervals and the physical horizon is unknown"}
    dts = np.concatenate([np.diff(np.asarray(ep.timestamps, np.float64)) for ep in dataset.episodes])
    if not len(dts) or not (dts > 0).all():
        return {"dt_ctrl": None, "source": "timestamps", "note": "timestamps do not increase: no action period"}
    dt = float(np.median(dts))
    stored = [np.asarray(ep.timestamps) for ep in dataset.episodes]
    eps = max(np.finfo(t.dtype).eps if np.issubdtype(t.dtype, np.floating) else 0.0 for t in stored)
    tol = 4 * eps * max(float(np.abs(t).max()) for t in stored if len(t))
    if not np.allclose(dts, dt, rtol=1e-6, atol=tol):
        return {"dt_ctrl": None, "source": "timestamps",
                "note": f"the action period varies beyond the timestamps' precision ({tol:.3g} s): no single dt_ctrl, "
                        "and the physical horizon is not claimed"}
    return {"dt_ctrl": dt, "source": "timestamps",
            "note": f"the median timestamp difference; every difference agrees within {tol:.3g} s, the timestamps' "
                    "storage precision"}


def evidence_class(profile: str | None, reference: bool) -> dict:
    """The strongest class the configured certificate can give from data and declared assumptions alone."""
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
    tau, tau_sample = sampled_tau(metric, params["tau_quantile"], params["tau_sample"])
    batch = transitions(dataset, representation)
    rows, coverage = probe_rows(batch.episode_id, metric.train_episodes, params["probe_transitions"])
    probe = generate_labels(dataset, representation, model, generator, list(validators) + [SourceDisplacement(metric)],
                            run_id="diagnose-probe", seed=seed, rows=rows, inputs={"transition_model": "0" * 64})
    accepted = np.asarray(probe.accepted, bool)
    d_source = np.asarray(probe.measurements["quality.d_source"])[accepted]
    label_z = np.asarray(observation(probe.observations, representation.output_modality), np.float64)[accepted]
    label_a = np.asarray(probe.actions, np.float64)[accepted]
    d_demo = nearest_distance(label_z, metric.train_states, metric.scale)
    # How far the model is queried from its data: the proposed (z_G, a_G) against the training (z, a) pairs.
    train = np.isin(batch.episode_id.astype(str), list(metric.train_episodes))
    za = np.hstack([np.asarray(batch.z, np.float64)[train], np.asarray(batch.a, np.float64)[train]])
    a_std = np.asarray(batch.a, np.float64)[train].std(axis=0) if train.any() else np.ones(batch.a.shape[1])
    joint_scale = np.concatenate([metric.scale, np.where(a_std > 0, a_std, 1.0)])
    d_query = nearest_distance(np.hstack([label_z, label_a]), za, joint_scale) if len(za) else np.full(len(label_z), np.inf)
    timing_ = timing(dataset)
    yield_ = float(accepted.mean()) if len(accepted) else 0.0
    q = params["label_quantile"]
    disp, novel, median_disp = _q(d_source, q), _q(d_demo, q), _q(d_source, 0.5)
    eta_disp = None if disp is None or not tau else disp / tau
    eta_novel = None if novel is None or not tau else novel / tau
    eps = None
    if held_out is not None and evaluation_model is not None and len(held_out[0]):
        z, a, z_next = (np.asarray(x, np.float64) for x in held_out)
        pred = z + np.asarray(evaluation_model.delta(held_out[0], held_out[1]), np.float64)
        eps = _q(np.linalg.norm((pred - z_next) / metric.scale, axis=1), params["fit_quantile"])
    fit_ratio = None if eps is None or not median_disp else eps / median_disp
    cls = evidence_class(profile, reference)
    off_data = cls["data_only_class"] == "conditional"
    gate = {
        "M": 1,
        "horizon_s": None if timing_["dt_ctrl"] is None else timing_["dt_ctrl"],
        "yield": {"value": yield_, "accepted": int(accepted.sum()), "candidates": int(len(accepted)),
                  "threshold": params["min_yield"], "pass": bool(len(accepted)) and yield_ >= params["min_yield"]},
        "displacement": {"eta": eta_disp, "d_source": disp, "quantile": q, "tau_demo": tau,
                         "threshold": params["min_displacement"],
                         "pass": eta_disp is not None and eta_disp >= params["min_displacement"]},
        "novelty": {"eta": eta_novel, "d_demo": novel, "quantile": q, "tau_demo": tau, "threshold": params["min_novelty"],
                    "pass": eta_novel is not None and eta_novel >= params["min_novelty"]},
        "fit": {"ratio": fit_ratio, "epsilon": eps, "quantile": params["fit_quantile"], "median_d_source": median_disp,
                "threshold": params["fit_ratio"], "pass": fit_ratio is not None and fit_ratio <= params["fit_ratio"],
                "model": "the evaluation model, fitted on the training episodes, scored on held-out transitions",
                "scope": "measured at demonstration states; necessary, never sufficient"},
        "trust": {"evidence": cls["data_only_class"], "profile": profile, "pass": off_data,
                  "query_extrapolation": {"q95": _q(d_query, 0.95), "median": _q(d_query, 0.5),
                                          "definition": "distance of each proposed (z_G, a_G) to the nearest training "
                                                        "(z, a) pair, scaled by the training states' and actions' std"},
                  "rule": "passes only with off-data evidence: a bound certificate profile whose constants are declared "
                          "or model-computed, never reference dynamics"},
    }
    gate["pass"] = all(gate[k]["pass"] for k in ("yield", "displacement", "novelty", "fit", "trust"))
    if requested == "auto":
        admitted, by = (1, "gates") if gate["pass"] else (None, None)
    else:
        admitted, by = int(requested), "declared"
    reason = next_step = None
    if admitted is None:
        reason, next_step = _why(gate, tau, eps)
    bands = None
    if tau:
        counts, _ = np.histogram(d_demo / tau, bins=np.array(BAND_EDGES))
        bands = {"edges": list(BAND_EDGES[:-1]) + ["inf"], "counts": [int(c) for c in counts],
                 "of": "accepted probe labels, by d_demo / tau_demo"}
    nested = None
    if generator.name in NOISY and labels_per_transition:
        k = int(labels_per_transition)
        nested = sorted({1, max(1, k // 2), k})
    doc = {
        "schema": SCHEMA, "version": f"{VERSION[0]}.{VERSION[1]}", "evidence_mode": "data_only",
        "recovery_steps": {"requested": requested if requested == "auto" else int(requested), "ladder": list(LADDER),
                           "cap": max(LADDER)},
        "metric": metric.spec(), "timing": timing_,
        "tau_demo": {"value": tau, "quantile": params["tau_quantile"], "sample": tau_sample,
                     "definition": "quantile of the distance from evenly spaced held-out demonstration states to the "
                                   "nearest training state: " + DEFINITION},
        "probe": {**coverage, "max_transitions": params["probe_transitions"], "candidates": int(len(accepted))},
        "gates": [gate], "admitted": admitted, "admitted_by": by, "reason": reason, "next_step": next_step,
        "bands": bands, "nested_views": nested, "evidence_class": cls, "statement": STATEMENT, "inputs": inputs,
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
    if not gate["displacement"]["pass"]:
        return (f"labels move {gate['displacement']['eta']:.3g} tau_demo (the demonstrations' own spacing) from their sources "
                f"(threshold {gate['displacement']['threshold']}): they barely leave the demonstration they correct",
                "use a generator or settings that move labels further, where the model can be trusted")
    if not gate["novelty"]["pass"]:
        return (f"labels lie {gate['novelty']['eta']:.3g} tau_demo (the demonstrations' own spacing) from the nearest demonstration state "
                f"(threshold {gate['novelty']['threshold']}): the dataset already covers the states they reach",
                "move labels further where the model can be trusted (for noisy_action, a larger "
                "generation.action_noise_std), or collect recovery demonstrations where the policy fails")
    if eps is None or not gate["fit"]["pass"]:
        return ("the model's held-out error is " + ("unmeasured" if eps is None else
                f"{gate['fit']['ratio']:.3g} of the median displacement (threshold {gate['fit']['threshold']})"),
                "improve the dynamics model or collect more demonstrations; it does not fit where it has data")
    return ("no off-data evidence that the model is credible at the proposed states (data alone cannot show it); "
            f"queries lie a median {gate['trust']['query_extrapolation']['median']:.3g} from the training (z, a) pairs",
            "declare a bound certificate profile (certificate.profile with declared constants) to admit under its "
            "assumptions, or set generation.recovery_steps 1 to generate without admission")


def _finite(x):
    """JSON-safe: non-finite floats become null."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


__all__ = ["diagnose", "probe_rows", "sampled_tau", "timing", "evidence_class", "LADDER"]
