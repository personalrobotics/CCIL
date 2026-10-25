"""Validators decide which candidates are valid labels.

Contract
- check(batch, cands, model) returns (reject, measurements): reject is (n,) bool,
  measurements maps names to (n,) float. It does not modify its inputs.
- reason is the ccil.candidates reason code a rejection records, or None for a
  validator that only measures.
- Validators run independently and all of them run; a candidate is accepted only
  if none rejects it, so one candidate can carry several reasons.
- measurement_specs maps each measurement name a validator returns to its
  definition, norm, and units ({} for a validator that measures nothing);
  generate_labels raises InterfaceError when the names do not match exactly.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .batch import CandidateBatch, InterfaceError, TransitionBatch


@runtime_checkable
class Validator(Protocol):
    reason: str | None

    def check(self, batch: TransitionBatch, cands: CandidateBatch, model) -> tuple[np.ndarray, dict]: ...


class Finite:
    reason = "non_finite"
    measurement_specs: dict = {}

    def check(self, batch, cands, model):
        return ~(np.isfinite(cands.z).all(axis=1) & np.isfinite(cands.a).all(axis=1)), {}


class GenerationFailed:
    reason = "generation_failed"
    measurement_specs: dict = {}

    def check(self, batch, cands, model):
        return cands.failed.copy(), {}


class SupportRadius:
    """Reject unless ||z_G - z_source|| < radius (the legacy aug.epsilon rule)."""

    reason = "out_of_support"
    measurement_specs = {"distance": {"definition": "||z_G - z_t||: distance of the label state from the demonstration "
                                                    "state of its source transition, accepted when < radius",
                                      "norm": "l2", "units": "representation units (state units for "
                                                             "representation.kind=identity)"}}

    def __init__(self, radius: float):
        self.radius = float(radius)

    def check(self, batch, cands, model):
        dist = np.linalg.norm(cands.z.astype(np.float64) - batch.z[cands.source], axis=1)
        with np.errstate(invalid="ignore"):
            return ~(dist < self.radius), {"distance": dist}


class ActionBounds:
    reason = "action_out_of_bounds"
    measurement_specs: dict = {}

    def __init__(self, low, high):
        self.low, self.high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    def check(self, batch, cands, model):
        with np.errstate(invalid="ignore"):
            return ((cands.a < self.low) | (cands.a > self.high)).any(axis=1), {}


class ModelResidual:
    """Measure ||z_G + f(z_G, a) - target|| under the model; never rejects."""

    reason = None
    measurement_specs = {"residual": {"definition": "||z_G + f(z_G, a_G) - target||: one-step error of the label under "
                                                    "the model", "norm": "l2",
                                      "units": "representation units (state units for representation.kind=identity)"}}

    def check(self, batch, cands, model):
        resid = np.linalg.norm(cands.z + model.delta(cands.z, cands.a) - cands.target, axis=1)
        return np.zeros(len(cands), bool), {"residual": resid}



_REP = "representation units (state units for representation.kind=identity)"
# Extension reasons (#78) of CCIL's own label filters; ccil.candidates 2.1+ readers accept them.
MODEL_MISFIT, LABEL_ERROR = "label_filter.model_misfit", "label_filter.label_error"


def _anchor_model_error(batch, model, source):
    return np.linalg.norm(batch.z + model.delta(batch.z, batch.a) - batch.z_next, axis=1)[source]


class ModelMisfit:
    """Measure each candidate's anchor model error; with a threshold, reject misfit anchors.

    anchor_model_error is ||z_t + f(z_t, a_t) - z_{t+1}|| at the candidate's
    source transition: the epsilon the CCIL label-error bounds assume small
    (Ke et al., Thm. 4.2, Def. 4.5). With a threshold (legacy
    aug.model_err_thresh), candidates whose anchor error is not below it are
    rejected as label_filter.model_misfit. Legacy generated nothing for such a
    transition; here its candidates are generated and recorded as rejected.
    """

    measurement_specs = {"anchor_model_error": {
        "definition": "||z_t + f(z_t, a_t) - z_{t+1}||: one-step model error at the candidate's source transition",
        "norm": "l2", "units": _REP}}

    def __init__(self, threshold: float | None = None):
        self.threshold = None if threshold is None else float(threshold)
        self.reason = None if threshold is None else MODEL_MISFIT

    def check(self, batch, cands, model):
        err = _anchor_model_error(batch, model, cands.source)
        if self.threshold is None:
            return np.zeros(len(cands), bool), {"anchor_model_error": err}
        with np.errstate(invalid="ignore"):
            return ~(err < self.threshold), {"anchor_model_error": err}


PROFILES = {
    "backtrack": "BackTrack label error (Ke et al., Def. 4.5, Eq. 11): ||J_S f(z_t, a_t)|| * ||z_G - z_t||",
    "disturbed_action": "DisturbedAction label error (Ke et al., Def. 4.6, Eq. 12): ||J_A f(z_t, a_t)|| * ||a_G - a_t|| "
                        "+ (1 + ||J_S f(z_t, a_t)||) * ||z_G - z_t||",
}


class LabelError:
    """The CCIL label-error estimate of the generator's profile, optionally filtering on it.

    profile is "backtrack" (backward Euler and exact-preimage labels) or
    "disturbed_action" (noisy-action labels); generate_labels checks it against
    the generator's label_error_profile. J_S and J_A are the model's exact
    Jacobians with respect to state and action at the source transition
    (z_t, a_t); a model without jacobian_za is refused rather than
    approximated by finite differences, which fail at float32 for large inputs.
    The norms are spectral; the Jacobians are evaluated at the anchor only, so
    the estimate is pointwise and empirical, not a bound over the neighborhood.

    Both definitions assume a small anchor model error (epsilon ~ 0), so every
    row records label_error_precondition: 1 when its anchor model error is
    below model_error_threshold, else 0. With max_error and/or quantile set, a
    row is rejected (label_filter.label_error) when its estimate exceeds
    max_error or the quantile threshold; the quantile is taken over the finite
    estimates of candidates that generation produced and whose anchors meet the
    precondition (legacy skipped misfit anchors before computing it), and the
    applied threshold is recorded per row as label_error_threshold.
    """

    def __init__(self, profile: str, model_error_threshold: float, max_error: float | None = None,
                 quantile: float | None = None):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
        if quantile is not None and not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile!r}")
        self.profile, self.model_error_threshold = profile, float(model_error_threshold)
        self.max_error = None if max_error is None else float(max_error)
        self.quantile = None if quantile is None else float(quantile)
        filtering = self.max_error is not None or self.quantile is not None
        self.reason = LABEL_ERROR if filtering else None
        specs = {
            "label_error_estimate": {"definition": PROFILES[profile] + "; exact model Jacobians at the anchor, pointwise "
                                                                      "(empirical)", "norm": "l2", "units": _REP},
            "label_error_state_jacobian_norm": {"definition": "||J_S f(z_t, a_t)||: spectral norm of the exact state "
                                                              "Jacobian at the source transition",
                                                "norm": "spectral (l2 operator)", "units": "dimensionless"},
            "label_error_state_displacement": {"definition": "||z_G - z_t||", "norm": "l2", "units": _REP},
            "label_error_precondition": {"definition": "1 when the anchor model error is below model_error_threshold "
                                                       f"({self.model_error_threshold}), else 0: the small-model-error "
                                                       "assumption of the definition", "norm": None, "units": "indicator"},
        }
        if profile == "disturbed_action":
            specs["label_error_action_jacobian_norm"] = {
                "definition": "||J_A f(z_t, a_t)||: spectral norm of the exact action Jacobian at the source transition",
                "norm": "spectral (l2 operator)", "units": "representation units per action unit"}
            specs["label_error_action_perturbation"] = {"definition": "||a_G - a_t||", "norm": "l2",
                                                        "units": "action units"}
        if filtering:
            specs["label_error_threshold"] = {
                "definition": "the limit each label_error_estimate was compared with: max_error, the "
                              f"{self.quantile}-quantile of the eligible estimates, or the smaller",
                "norm": None, "units": _REP}
        self.measurement_specs = specs

    def check(self, batch, cands, model):
        if not hasattr(model, "jacobian_za"):
            raise InterfaceError(f"LabelError needs the exact model Jacobian: {type(model).__name__} has no jacobian_za")
        D = batch.z.shape[1]
        jac = np.asarray(model.jacobian_za(batch.z, batch.a), np.float64)
        js = np.linalg.norm(jac[:, :, :D], ord=2, axis=(1, 2))[cands.source]
        ds = np.linalg.norm(cands.z.astype(np.float64) - batch.z[cands.source].astype(np.float64), axis=1)
        meas = {"label_error_state_jacobian_norm": js, "label_error_state_displacement": ds}
        if self.profile == "backtrack":
            est = js * ds
        else:
            ja = np.linalg.norm(jac[:, :, D:], ord=2, axis=(1, 2))[cands.source]
            da = np.linalg.norm(cands.a.astype(np.float64) - batch.a[cands.source].astype(np.float64), axis=1)
            est = ja * da + (1.0 + js) * ds
            meas.update({"label_error_action_jacobian_norm": ja, "label_error_action_perturbation": da})
        with np.errstate(invalid="ignore"):
            met = _anchor_model_error(batch, model, cands.source) < self.model_error_threshold
        meas.update({"label_error_estimate": est, "label_error_precondition": met.astype(np.float64)})
        if self.reason is None:
            return np.zeros(len(cands), bool), meas
        limit = np.inf if self.max_error is None else self.max_error
        if self.quantile is not None:
            population = est[~cands.failed & met & np.isfinite(est)]
            if len(population):
                limit = min(limit, float(np.quantile(population, self.quantile)))
        meas["label_error_threshold"] = np.full(len(cands), limit)
        with np.errstate(invalid="ignore"):
            return ~(est <= limit), meas
