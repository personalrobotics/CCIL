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
    """Solve z_G + f(z_G, a) = z by Newton's method; failed where it does not converge."""

    name, version = "exact_preimage", "1"

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-4):
        self.params = {"max_iterations": int(max_iterations), "tolerance": float(tolerance)}

    def generate(self, batch, model, seed):
        z, a = batch.z.astype(np.float64), batch.a.astype(np.float64)
        x = z.copy()
        eye = np.eye(z.shape[1])
        for _ in range(self.params["max_iterations"]):
            r = x + model.delta(x, a) - z
            if np.all(np.linalg.norm(r, axis=1) <= self.params["tolerance"]):
                break
            J = eye + (model.jacobian_z(x, a) if hasattr(model, "jacobian_z") else jacobian_fd(model, x, a))
            with np.errstate(invalid="ignore"):
                x = x - np.linalg.solve(J, r[..., None])[..., 0]
        # Convergence is judged on the residual of the returned iterate, not a stale one.
        resid = np.linalg.norm(x + model.delta(x, a) - z, axis=1)
        failed = ~(resid <= self.params["tolerance"])
        return CandidateBatch(source=np.arange(len(batch)), z=x.astype(batch.z.dtype), a=batch.a.copy(),
                              target=batch.z.copy(), failed=failed, measurements={"solver_residual": resid})


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
