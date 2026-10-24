"""Transition models: residual dynamics z_next = z + f(z, a) in a representation space.

Contract
- fit(z, a, z_next, seed) fits in place and returns metrics; the model owns its
  parameters, optimizer, and device. Fitting is the only state-changing call.
- delta(z, a) returns f(z, a) as (N, D) float for (N, D) and (N, A) inputs; it is
  batched, deterministic, and has no side effects (no training-mode updates).
- jacobian_z(z, a) is optional and returns df/dz as (N, D, D); generators fall back
  to central finite differences (jacobian_fd) when it is absent.
- evaluate(z, a, z_next) returns one-step error metrics without changing state.
- save(path) / load(path, device) persist float32-equivalent parameters; numpy
  arrays cross the interface, so torch models convert internally.
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np

from ..schema.io import _atomic_savez


@runtime_checkable
class TransitionModel(Protocol):
    def fit(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray, seed: int) -> dict: ...

    def delta(self, z: np.ndarray, a: np.ndarray) -> np.ndarray: ...

    def evaluate(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray) -> dict: ...

    def save(self, path) -> None: ...


def jacobian_fd(model, z: np.ndarray, a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference df/dz, (N, D, D)."""
    N, D = z.shape
    J = np.empty((N, D, D))
    for j in range(D):
        step = np.zeros(D)
        step[j] = eps
        J[:, :, j] = (model.delta(z + step, a) - model.delta(z - step, a)) / (2 * eps)
    return J


def one_step_metrics(model, z, a, z_next) -> dict:
    err = np.linalg.norm(z + model.delta(z, a) - z_next, axis=1)
    return {"transitions": int(len(err)), "mean_l2": float(np.mean(err)), "max_l2": float(np.max(err))}


class LinearTransitionModel:
    """f(z, a) = A z + B a + c. Exact for linear systems; fitted by least squares."""

    FORMAT = "ccil.linear-transition/1"

    def __init__(self, A=None, B=None, c=None):
        self.A = None if A is None else np.asarray(A, dtype=np.float64)
        self.B = None if B is None else np.asarray(B, dtype=np.float64)
        self.c = None if c is None else np.asarray(c, dtype=np.float64)
        if self.A is not None and self.c is None:
            self.c = np.zeros(len(self.A))

    def fit(self, z, a, z_next, seed: int = 0) -> dict:
        """Deterministic least squares; seed is accepted for interface uniformity."""
        X = np.hstack([z, a, np.ones((len(z), 1))]).astype(np.float64)
        W, *_ = np.linalg.lstsq(X, (z_next - z).astype(np.float64), rcond=None)
        D, A = z.shape[1], a.shape[1]
        self.A, self.B, self.c = W[:D].T, W[D:D + A].T, W[-1]
        return {"method": "least_squares", **one_step_metrics(self, z, a, z_next)}

    def delta(self, z, a) -> np.ndarray:
        return z @ self.A.T + a @ self.B.T + self.c

    def jacobian_z(self, z, a) -> np.ndarray:
        return np.broadcast_to(self.A, (len(z),) + self.A.shape)

    def evaluate(self, z, a, z_next) -> dict:
        return one_step_metrics(self, z, a, z_next)

    def save(self, path) -> None:
        _atomic_savez(path, {"A": self.A, "B": self.B, "c": self.c, "meta": np.array(json.dumps({"format": self.FORMAT}))})

    @classmethod
    def load(cls, path, device: str = "cpu") -> "LinearTransitionModel":
        if device != "cpu":
            raise ValueError("LinearTransitionModel runs on cpu only")
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["meta"])).get("format") != cls.FORMAT:
                raise ValueError(f"{path} is not a {cls.FORMAT} file")
            return cls(z["A"], z["B"], z["c"])
