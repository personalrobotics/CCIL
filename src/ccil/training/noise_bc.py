"""NoiseBC: online observation noise for behavior-cloning training inputs (#22, used by #89).

Legacy CustomBC.compute_loss_weight resampled, on every loss computation, a
half-batch mask (rand > 0.5) and noise noise_cov * U[0, 1), which is always
positive, and wrote it into the caller's batch in place. This component keeps
the online, per-draw resampling and the masking probability, and offers two
modes:

- symmetric_uniform_half_batch (canonical): s~ = s + scale * (2U - 1), in
  [-scale, scale): zero-centered, with scale the maximum magnitude as before;
- legacy_uniform_half_batch (historical, for parity): s~ = s + scale * U, in
  [0, scale).

A sample is perturbed when its mask uniform exceeds 1 - mask_probability,
which is legacy rand > 0.5 at the default 0.5. Every uniform is a pure
function of (seed, optimizer step, sample identity, draw index, component)
through a splitmix64 counter hash, with the sample identity (episode_id, frame)
entering as its 128-bit digest (ccil.core.identity), so a draw is reproducible
from its key, fresh at every new step or draw, independent of batch order and
composition, and never shared by distinct samples. perturb validates its inputs
and never modifies them.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..core.identity import identity_words64

MODES = ("symmetric_uniform_half_batch", "legacy_uniform_half_batch")
RNG = "splitmix64-counter/1 over ccil.transition-identity/1 (BLAKE2b-128)"
_M64 = (1 << 64) - 1
_GAMMA, _M1, _M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer of x + golden gamma, on uint64 arrays (wrapping arithmetic)."""
    z = x + _GAMMA
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def counter_uniforms(seed: int, step: int, episode_ids, frames, draw: int, components: int) -> np.ndarray:
    """Uniforms in [0, 1), shape (B, components), each a pure function of its full key."""
    words = np.array([identity_words64(e, f) for e, f in zip(episode_ids, frames)], dtype=np.uint64).reshape(-1, 2)
    with np.errstate(over="ignore"):
        h = _mix(np.full(len(words), np.uint64(seed & _M64)))
        h = _mix(h ^ np.uint64(step & _M64))
        h = _mix(h ^ words[:, 0])
        h = _mix(h ^ words[:, 1])
        h = _mix(h ^ np.uint64(draw & _M64))
        h = _mix(h[:, None] ^ np.arange(components, dtype=np.uint64)[None, :])
    return (h >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


class NoiseBC:
    """Per-draw observation noise for NoiseBC training batches."""

    version = "1"

    def __init__(self, mode: str, scale: float, seed: int, mask_probability: float = 0.5,
                 units: str = "observation units"):
        if mode not in MODES:
            raise ValueError(f"NoiseBC mode {mode!r} is not one of {MODES}")
        if not (scale >= 0 and np.isfinite(scale)) or not 0 <= mask_probability <= 1:
            raise ValueError("NoiseBC needs a finite scale >= 0 and a mask probability in [0, 1]")
        self.mode, self.scale, self.seed = mode, float(scale), int(seed)
        self.mask_probability, self.units = float(mask_probability), units

    def perturb(self, obs: np.ndarray, episode_ids, frames, step: int, draw: int = 0):
        """Return (perturbed copy of obs, {"mask": (B,), "noise": noise added, zero where unmasked})."""
        obs, episode_ids, frames = _check_inputs(obs, episode_ids, frames, step, draw)
        flat = obs.reshape(len(obs), -1)
        u = counter_uniforms(self.seed, step, episode_ids, frames, draw, 1 + flat.shape[1])
        mask = u[:, 0] > 1.0 - self.mask_probability
        unit = 2.0 * u[:, 1:] - 1.0 if self.mode == "symmetric_uniform_half_batch" else u[:, 1:]
        noise = np.where(mask[:, None], self.scale * unit, 0.0).astype(obs.dtype)
        return (flat + noise).reshape(obs.shape), {"mask": mask, "noise": noise.reshape(obs.shape)}

    def descriptor(self) -> dict:
        """What a run records about its NoiseBC draws."""
        support = "[-scale, scale)" if self.mode == "symmetric_uniform_half_batch" else "[0, scale)"
        return {"component": "noise_bc", "version": self.version, "mode": self.mode,
                "distribution": f"uniform on {support} per observation dimension, applied to masked samples",
                "scale": self.scale, "units": self.units, "truncation": "none (bounded by construction)",
                "mask_probability": self.mask_probability, "seed": self.seed, "rng": RNG,
                "draw_schedule": "fresh draw per (optimizer step, draw index, sample identity)"}


def _check_inputs(obs, episode_ids, frames, step, draw):
    """Reject inputs for which a result would not implement the configured distribution."""
    if not isinstance(obs, np.ndarray) or not np.issubdtype(obs.dtype, np.floating) or obs.ndim < 2:
        raise ValueError("NoiseBC observations must be a floating-point array of shape (B, ...); integer arrays "
                         "would truncate the noise")
    ids = np.asarray(episode_ids)
    fr = np.asarray(frames)
    if ids.ndim != 1 or len(ids) != len(obs) or not all(isinstance(e, str) and e for e in ids.tolist()):
        raise ValueError(f"episode_ids must be {len(obs)} non-empty strings, one per observation row")
    if fr.ndim != 1 or len(fr) != len(obs) or not np.issubdtype(fr.dtype, np.integer) or (fr < 0).any():
        raise ValueError(f"frames must be {len(obs)} non-negative integers, one per observation row")
    for name, value in (("step", step), ("draw", draw)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return obs, ids, fr


class NoiseTrace:
    """Running SHA-256 over every draw, for a reproducibility summary without storing the draws."""

    def __init__(self):
        self._h, self.draws = hashlib.sha256(), 0

    def update(self, step: int, draw: int, episode_ids, frames, record: dict) -> None:
        self._h.update(np.array([step, draw], np.int64).tobytes())
        self._h.update("\0".join(map(str, episode_ids)).encode())
        self._h.update(np.asarray(frames, np.int64).tobytes())
        self._h.update(np.asarray(record["mask"], np.bool_).tobytes())
        self._h.update(np.ascontiguousarray(record["noise"]).tobytes())
        self.draws += 1

    def summary(self) -> dict:
        return {"draws": self.draws, "sha256": self._h.hexdigest()}
