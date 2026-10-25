"""Transition models: residual dynamics z_next = z + f(z, a) in a representation space.

Contract
- fit(z, a, z_next, seed) fits in place and returns metrics; the model owns its
  parameters, optimizer, and device. Fitting is the only state-changing call.
- delta(z, a) returns f(z, a) as (N, D) float for (N, D) and (N, A) inputs; it is
  batched, deterministic, and has no side effects (no training-mode updates).
- jacobian_z(z, a) is optional and returns df/dz as (N, D, D); generators fall back
  to central finite differences (jacobian_fd) when it is absent. jacobian_za is
  optional and returns df/d(z, a) as (N, D, D + A); label-error estimates
  require it and have no finite-difference fallback.
- evaluate(z, a, z_next) returns one-step error metrics without changing state.
- kind and version identify the implementation; kind is the
  transition_model.kind value that selects it, and version changes whenever
  saved checkpoints or fitted results would differ. Pipelines record both in
  provenance and stage identity.
- save(path) writes a checkpoint whose meta records kind and version;
  load(path, device) is a classmethod that restores it. load_transition_model
  reads the kind from the checkpoint and dispatches through TRANSITION_MODELS.
  A file that is not a checkpoint of a known kind and version raises
  CheckpointError; model-specific failures (an unavailable device, for example)
  propagate unchanged. numpy arrays cross the interface, so torch models
  convert internally.
"""
from __future__ import annotations

import importlib
import json
from typing import Protocol, runtime_checkable

import numpy as np

from ..schema.io import _atomic_savez
from .verified import gamma, is_spectral_upper_bound, up, verified_spectral_norm


class CheckpointError(ValueError):
    """A file is not a checkpoint of a known component kind and version."""


@runtime_checkable
class TransitionModel(Protocol):
    kind: str
    version: str

    def fit(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray, seed: int) -> dict: ...

    def delta(self, z: np.ndarray, a: np.ndarray) -> np.ndarray: ...

    def evaluate(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray) -> dict: ...

    def save(self, path) -> None: ...

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TransitionModel": ...


def checkpoint_meta(path, component: str) -> dict:
    """The meta of a component checkpoint, or CheckpointError."""
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError, EOFError) as e:
        raise CheckpointError(f"{path} is not a {component} checkpoint: {e}") from None
    if not isinstance(meta, dict) or meta.get("component") != component:
        raise CheckpointError(f"{path} is not a {component} checkpoint")
    return meta


def check_identity(path, meta: dict, cls) -> None:
    if (meta.get("kind"), meta.get("version")) != (cls.kind, cls.version):
        raise CheckpointError(f"{path} holds {meta.get('kind')!r} version {meta.get('version')!r}; "
                              f"{cls.__name__} reads {cls.kind!r} version {cls.version!r}")


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

    kind, version = "linear", "1"
    evaluation_dtype = "float64"

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

    def jacobian_za(self, z, a) -> np.ndarray:
        return np.broadcast_to(np.hstack([self.A, self.B]), (len(z), self.A.shape[0], self.A.shape[1] + self.B.shape[1]))

    def evaluate(self, z, a, z_next) -> dict:
        return one_step_metrics(self, z, a, z_next)

    def lipschitz_bound(self, maps: str, state_scale, action_scale) -> dict:
        """The Lipschitz constant of f_hat in state or action, in scaled coordinates, as a verified upper bound (#88)."""
        sz, sa = np.asarray(state_scale, np.float64), np.asarray(action_scale, np.float64)
        M, cols = (self.A, sz) if maps == "state" else (self.B, sa)
        value = verified_spectral_norm(M, row_div=sz, col_mul=cols)
        return {"value": value, "factors": [value], "activation": None, "activation_lipschitz": None, "verified": True}

    def verify_lipschitz_bound(self, maps: str, value: float, state_scale, action_scale) -> bool:
        sz, sa = np.asarray(state_scale, np.float64), np.asarray(action_scale, np.float64)
        return is_spectral_upper_bound(self.A if maps == "state" else self.B, value, row_div=sz,
                                       col_mul=sz if maps == "state" else sa)

    def delta_error_bound(self, z, a) -> np.ndarray:
        """Componentwise bound on |delta(z, a) - (A z + B a + c)| for this float64 evaluation (Higham, Sec. 3.1)."""
        n = self.A.shape[1] + self.B.shape[1] + 1
        mag = np.abs(np.asarray(z, np.float64)) @ np.abs(self.A).T + np.abs(np.asarray(a, np.float64)) @ np.abs(self.B).T + np.abs(self.c)
        return up(gamma(n) * mag * (1 + gamma(n + 2)))

    @classmethod
    def from_config(cls, v) -> "LinearTransitionModel":
        return cls()

    def save(self, path) -> None:
        meta = {"component": "transition_model", "kind": self.kind, "version": self.version}
        _atomic_savez(path, {"A": self.A, "B": self.B, "c": self.c, "meta": np.array(json.dumps(meta, sort_keys=True))})

    @classmethod
    def load(cls, path, device: str = "cpu") -> "LinearTransitionModel":
        check_identity(path, checkpoint_meta(path, "transition_model"), cls)
        if device != "cpu":
            raise ValueError("LinearTransitionModel runs on cpu only")
        with np.load(path, allow_pickle=False) as z:
            return cls(z["A"], z["B"], z["c"])


# transition_model.kind -> implementation, or "module:Class" for one that needs an
# optional extra, imported only when that kind is used.
TRANSITION_MODELS = {LinearTransitionModel.kind: LinearTransitionModel, "mlp": "ccil.models.mlp:MLPTransitionModel"}
EXTRAS = {"mlp": "the ccil[torch] extra (Python 3.11+): pip install 'ccil[torch]'"}


def transition_model_class(kind: str):
    """The class implementing kind; ModuleNotFoundError with install guidance if its extra is missing."""
    impl = TRANSITION_MODELS[kind]
    if isinstance(impl, str):
        module, name = impl.split(":")
        try:
            impl = getattr(importlib.import_module(module), name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"transition_model.kind {kind!r} needs {EXTRAS[kind]} ({e})") from e
    return impl


def load_transition_model(path, device: str = "cpu"):
    """Restore any registered transition model from its checkpoint."""
    meta = checkpoint_meta(path, "transition_model")
    if meta.get("kind") not in TRANSITION_MODELS:
        raise CheckpointError(f"{path} holds transition model kind {meta.get('kind')!r}; "
                              f"this ccil reads {sorted(TRANSITION_MODELS)}")
    return transition_model_class(meta["kind"]).load(path, device)
