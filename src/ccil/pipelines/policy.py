"""Downstream policy trainers for the integrated pipeline.

Contract
- name and version identify the trainer (name is the policy.trainer value);
  checkpoints record both, and load refuses a checkpoint of another name or
  version with CheckpointError.
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

from ..core.transition import CheckpointError, checkpoint_meta
from ..schema.io import _atomic_savez


@runtime_checkable
class PolicyTrainer(Protocol):
    name: str
    version: str

    def fit(self, z: np.ndarray, a: np.ndarray, weights: np.ndarray) -> "PolicyTrainer": ...

    def predict(self, z: np.ndarray) -> np.ndarray: ...

    def save(self, path) -> None: ...

    def load(self, path) -> "PolicyTrainer": ...


def _save(path, trainer, arrays: dict) -> None:
    meta = {"component": "policy", "trainer": trainer.name, "version": trainer.version}
    _atomic_savez(path, {**arrays, "meta": np.array(json.dumps(meta, sort_keys=True))})


def _load(path, trainer) -> dict:
    meta = checkpoint_meta(path, "policy")
    if (meta.get("trainer"), meta.get("version")) != (trainer.name, trainer.version):
        raise CheckpointError(f"{path} holds a {meta.get('trainer')!r} version {meta.get('version')!r} policy, "
                              f"not {trainer.name!r} version {trainer.version!r}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta"}


class LinearPolicyTrainer:
    """Weighted least-squares behavior cloning: a = W z + b."""

    name, version = "linear_least_squares", "1"

    def fit(self, z, a, weights):
        X = np.hstack([z, np.ones((len(z), 1))]) * np.sqrt(weights)[:, None]
        sol, *_ = np.linalg.lstsq(X, a * np.sqrt(weights)[:, None], rcond=None)
        self.W, self.b = sol[:-1].T, sol[-1]
        return self

    def predict(self, z):
        return z @ self.W.T + self.b

    def save(self, path):
        _save(path, self, {"W": self.W, "b": self.b})

    def load(self, path):
        arrays = _load(path, self)
        self.W, self.b = arrays["W"], arrays["b"]
        return self


class MeanActionTrainer:
    """Predict the weighted mean action everywhere; a minimal alternative trainer."""

    name, version = "mean_action", "1"

    def fit(self, z, a, weights):
        self.mean = (a * weights[:, None]).sum(axis=0) / weights.sum()
        return self

    def predict(self, z):
        return np.broadcast_to(self.mean, (len(z), len(self.mean))).copy()

    def save(self, path):
        _save(path, self, {"mean": self.mean})

    def load(self, path):
        self.mean = _load(path, self)["mean"]
        return self


TRAINERS = {t.name: t for t in (LinearPolicyTrainer, MeanActionTrainer)}
