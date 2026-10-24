"""Generators propose corrective-label candidates from transitions and a model.

Contract
- generate(batch, model, seed) returns a CandidateBatch for every transition; it
  never accepts or rejects (validators do) but marks failed candidates.
- Generators are stateless: everything they use is in params, and params fully
  determine the output given (batch, model, seed).
- Randomness is drawn per transition from default_rng((seed, crc32(episode_id),
  frame)), so splitting a batch into chunks never changes any candidate.
- name/version/params identify the generator in the candidate provenance.
"""
from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable

import numpy as np

from .batch import CandidateBatch, TransitionBatch
from .transition import jacobian_fd


@runtime_checkable
class Generator(Protocol):
    name: str
    version: str
    params: dict

    def generate(self, batch: TransitionBatch, model, seed: int) -> CandidateBatch: ...


def transition_rng(seed: int, episode_id: str, frame: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(str(episode_id).encode()), int(frame)])


class BackwardEuler:
    """z_G = z - f(z, a): applying a at z_G approximately reaches z."""

    name, version = "backward_euler", "1"

    def __init__(self):
        self.params = {}

    def generate(self, batch, model, seed):
        n = len(batch)
        return CandidateBatch(source=np.arange(n), z=batch.z - model.delta(batch.z, batch.a), a=batch.a.copy(),
                              target=batch.z.copy(), failed=np.zeros(n, bool))


class ExactPreimage:
    """Solve z_G + f(z_G, a) = z by Newton's method, row by row.

    A row stops iterating once its own residual is within tolerance, so its
    result never depends on the other rows in the batch. A row whose Newton
    system is singular, whose iterate becomes non-finite, or which does not
    converge is marked failed; the other rows are unaffected.
    """

    name, version = "exact_preimage", "1"

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-4):
        self.params = {"max_iterations": int(max_iterations), "tolerance": float(tolerance)}

    def generate(self, batch, model, seed):
        z, a = batch.z.astype(np.float64), batch.a.astype(np.float64)
        tol = self.params["tolerance"]
        x = z.copy()
        eye = np.eye(z.shape[1])
        broken = np.zeros(len(batch), bool)
        resid = np.linalg.norm(x + model.delta(x, a) - z, axis=1)
        for _ in range(self.params["max_iterations"]):
            idx = np.flatnonzero(~broken & ~(resid <= tol))
            if not len(idx):
                break
            xs, as_ = x[idx], a[idx]
            r = xs + model.delta(xs, as_) - z[idx]
            J = eye + (model.jacobian_z(xs, as_) if hasattr(model, "jacobian_z") else jacobian_fd(model, xs, as_))
            step, ok = _solve_rows(J, r)
            ok &= np.isfinite(xs - step).all(axis=1)
            x[idx[ok]] = xs[ok] - step[ok]
            broken[idx[~ok]] = True
            moved = idx[ok]
            resid[moved] = np.linalg.norm(x[moved] + model.delta(x[moved], a[moved]) - z[moved], axis=1)
        # Convergence is judged on the residual of the returned iterate, not a stale one.
        failed = broken | ~(resid <= tol)
        return CandidateBatch(source=np.arange(len(batch)), z=x.astype(batch.z.dtype), a=batch.a.copy(),
                              target=batch.z.copy(), failed=failed, measurements={"solver_residual": resid})


def _solve_rows(J, r):
    """Solve J[i] s[i] = r[i]; rows with a singular or non-finite system get ok=False."""
    ok = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(r).all(axis=1)
    step = np.zeros_like(r)
    for i in np.flatnonzero(ok):
        try:
            step[i] = np.linalg.solve(J[i], r[i])
        except np.linalg.LinAlgError:
            ok[i] = False
    return step, ok & np.isfinite(step).all(axis=1)


class NoisyAction:
    """For k draws a~ = a + N(0, sigma^2): z_G = z_next - f(z_next, a~), targeting z_next."""

    name, version = "noisy_action", "1"

    def __init__(self, labels_per_transition: int, action_noise_std: float):
        self.params = {"labels_per_transition": int(labels_per_transition), "action_noise_std": float(action_noise_std)}

    def generate(self, batch, model, seed):
        k, sigma = self.params["labels_per_transition"], self.params["action_noise_std"]
        A = batch.a.shape[1]
        noise = np.concatenate([transition_rng(seed, e, f).normal(0.0, sigma, size=(k, A))
                                for e, f in zip(batch.episode_id, batch.frame)]) if len(batch) else np.zeros((0, A))
        source = np.repeat(np.arange(len(batch)), k)
        a_tilde = (batch.a[source] + noise).astype(batch.a.dtype)
        target = batch.z_next[source]
        return CandidateBatch(source=source, z=target - model.delta(target, a_tilde), a=a_tilde, target=target.copy(),
                              failed=np.zeros(len(source), bool), measurements={"action_noise_norm": np.linalg.norm(noise, axis=1)})
