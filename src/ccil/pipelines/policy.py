"""Downstream policy trainers for the integrated pipeline.

Contract
- fit(z, a, weights) trains on representation-space inputs (N, D), actions
  (N, A), and per-row weights (N,) >= 0, and returns self. The trainer owns its
  parameters, optimizer, and device; numpy crosses the interface.
- predict(z) returns (N, A) actions without changing state.
- save(path) persists the trained policy; load(path) restores a policy saved
  by the same trainer kind into this object and returns it. Evaluation always
  loads the policy.train checkpoint, so a reused training stage evaluates the
  same policy a fresh one would.
Any implementation can replace the reference trainers here without changing the
CCIL stages (representation, transition model, labels).
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np

from ..schema.io import _atomic_savez


@runtime_checkable
class PolicyTrainer(Protocol):
    name: str

    def fit(self, z: np.ndarray, a: np.ndarray, weights: np.ndarray) -> "PolicyTrainer": ...

    def predict(self, z: np.ndarray) -> np.ndarray: ...

    def save(self, path) -> None: ...

    def load(self, path) -> "PolicyTrainer": ...


def _load(path, name) -> dict:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    saved = json.loads(str(arrays.pop("meta")))["trainer"]
    if saved != name:
        raise ValueError(f"{path} holds a {saved!r} policy, not {name!r}")
    return arrays


class LinearPolicyTrainer:
    """Weighted least-squares behavior cloning: a = W z + b."""

    name = "linear_least_squares"

    def fit(self, z, a, weights):
        X = np.hstack([z, np.ones((len(z), 1))]) * np.sqrt(weights)[:, None]
        sol, *_ = np.linalg.lstsq(X, a * np.sqrt(weights)[:, None], rcond=None)
        self.W, self.b = sol[:-1].T, sol[-1]
        return self

    def predict(self, z):
        return z @ self.W.T + self.b

    def save(self, path):
        _atomic_savez(path, {"W": self.W, "b": self.b, "meta": np.array(json.dumps({"trainer": self.name}))})

    def load(self, path):
        arrays = _load(path, self.name)
        self.W, self.b = arrays["W"], arrays["b"]
        return self


class MeanActionTrainer:
    """Predict the weighted mean action everywhere; a minimal alternative trainer."""

    name = "mean_action"

    def fit(self, z, a, weights):
        self.mean = (a * weights[:, None]).sum(axis=0) / weights.sum()
        return self

    def predict(self, z):
        return np.broadcast_to(self.mean, (len(z), len(self.mean))).copy()

    def save(self, path):
        _atomic_savez(path, {"mean": self.mean, "meta": np.array(json.dumps({"trainer": self.name}))})

    def load(self, path):
        self.mean = _load(path, self.name)["mean"]
        return self


TRAINERS = {t.name: t for t in (LinearPolicyTrainer, MeanActionTrainer)}
