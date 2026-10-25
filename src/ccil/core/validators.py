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

from .batch import CandidateBatch, TransitionBatch
from .transition import residual_jacobian_za


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


class ModelMisfit:
    """Reject candidates whose source transition the model does not fit: the legacy aug.model_err_thresh.

    Legacy (label-filtering) generated nothing for such a transition; here its
    candidates are generated and rejected, so the decision is recorded.
    """

    reason = "model_misfit"
    measurement_specs = {"anchor_model_error": {
        "definition": "||z_t + f(z_t, a_t) - z_{t+1}||: one-step model error at the candidate's source transition, "
                      "rejected when not below the threshold", "norm": "l2", "units": _REP}}

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def check(self, batch, cands, model):
        err = np.linalg.norm(batch.z + model.delta(batch.z, batch.a) - batch.z_next, axis=1)[cands.source]
        with np.errstate(invalid="ignore"):
            return ~(err < self.threshold), {"anchor_model_error": err}


class LabelError:
    """Estimate each label's error from the model's local continuity at its anchor, optionally filtering on it.

    estimate = L * ||(z_G, a_G) - (z_t, a_t)||, with L the spectral norm of the
    residual Jacobian df/d(z, a) at the source transition (z_t, a_t). The
    label-filtering branch used the next-state map z + f, whose Jacobian is
    I + df/dz and has norm near 1 for any small residual, so its estimate was
    close to the bare distance (docs/migration/label-methods.md). L is a
    pointwise value at the anchor, not a bound over the neighborhood.

    With max_error and/or quantile set, a label is rejected (label_error) when
    its estimate exceeds max_error or the quantile threshold. The quantile is
    taken over the finite estimates of candidates that generation produced
    (not failed), as in legacy, and recorded per row as label_error_threshold.
    Without either, it only measures.
    """

    def __init__(self, max_error: float | None = None, quantile: float | None = None):
        if quantile is not None and not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile!r}")
        self.max_error = None if max_error is None else float(max_error)
        self.quantile = None if quantile is None else float(quantile)
        filtering = self.max_error is not None or self.quantile is not None
        self.reason = "label_error" if filtering else None
        specs = {
            "label_error_estimate": {"definition": "L * ||(z_G, a_G) - (z_t, a_t)||, L = anchor_local_lipschitz: "
                                                   "first-order estimate of the label's error (empirical, pointwise)",
                                     "norm": "l2", "units": _REP},
            "anchor_local_lipschitz": {"definition": "spectral norm of the residual Jacobian df/d(z, a) at the source "
                                                     "transition (z_t, a_t)", "norm": "spectral (l2 operator)",
                                       "units": "representation units per (representation or action) unit"},
            "state_action_distance": {"definition": "||(z_G, a_G) - (z_t, a_t)||: distance of the label from its anchor "
                                                    "in the joint state-action space", "norm": "l2",
                                      "units": "mixed representation and action units"},
        }
        if filtering:
            specs["label_error_threshold"] = {
                "definition": "the effective limit each label_error_estimate was compared with: max_error, the "
                              f"{self.quantile}-quantile of produced candidates' finite estimates, or the smaller",
                "norm": None, "units": _REP}
        self.measurement_specs = specs

    def check(self, batch, cands, model):
        jac = residual_jacobian_za(model, batch.z, batch.a)
        lips = np.linalg.norm(jac, ord=2, axis=(1, 2))[cands.source]
        dist = np.linalg.norm(np.hstack([cands.z.astype(np.float64) - batch.z[cands.source],
                                         cands.a.astype(np.float64) - batch.a[cands.source]]), axis=1)
        est = lips * dist
        meas = {"label_error_estimate": est, "anchor_local_lipschitz": lips, "state_action_distance": dist}
        if self.reason is None:
            return np.zeros(len(cands), bool), meas
        limit = np.inf if self.max_error is None else self.max_error
        if self.quantile is not None:
            population = est[~cands.failed & np.isfinite(est)]
            if len(population):
                limit = min(limit, float(np.quantile(population, self.quantile)))
        meas["label_error_threshold"] = np.full(len(cands), limit)
        with np.errstate(invalid="ignore"):
            return ~(est <= limit), meas
