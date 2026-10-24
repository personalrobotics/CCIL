"""Validators decide which candidates are valid labels.

Contract
- check(batch, cands, model) returns (reject, measurements): reject is (n,) bool,
  measurements maps names to (n,) float. It does not modify its inputs.
- reason is the ccil.candidates reason code a rejection records, or None for a
  validator that only measures.
- Validators run independently and all of them run; a candidate is accepted only
  if none rejects it, so one candidate can carry several reasons.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .batch import CandidateBatch, TransitionBatch


@runtime_checkable
class Validator(Protocol):
    reason: str | None

    def check(self, batch: TransitionBatch, cands: CandidateBatch, model) -> tuple[np.ndarray, dict]: ...


class Finite:
    reason = "non_finite"

    def check(self, batch, cands, model):
        return ~(np.isfinite(cands.z).all(axis=1) & np.isfinite(cands.a).all(axis=1)), {}


class GenerationFailed:
    reason = "generation_failed"

    def check(self, batch, cands, model):
        return cands.failed.copy(), {}


class SupportRadius:
    """Reject unless ||z_G - z_source|| < radius (the legacy aug.epsilon rule)."""

    reason = "out_of_support"

    def __init__(self, radius: float):
        self.radius = float(radius)

    def check(self, batch, cands, model):
        dist = np.linalg.norm(cands.z.astype(np.float64) - batch.z[cands.source], axis=1)
        with np.errstate(invalid="ignore"):
            return ~(dist < self.radius), {"distance": dist}


class ActionBounds:
    reason = "action_out_of_bounds"

    def __init__(self, low, high):
        self.low, self.high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    def check(self, batch, cands, model):
        with np.errstate(invalid="ignore"):
            return ((cands.a < self.low) | (cands.a > self.high)).any(axis=1), {}


class ModelResidual:
    """Measure ||z_G + f(z_G, a) - target|| under the model; never rejects."""

    reason = None

    def check(self, batch, cands, model):
        resid = np.linalg.norm(cands.z + model.delta(cands.z, cands.a) - cands.target, axis=1)
        return np.zeros(len(cands), bool), {"residual": resid}
