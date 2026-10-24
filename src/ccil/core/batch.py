"""Batches exchanged between stages. Arrays are numpy at every interface boundary."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class InterfaceError(RuntimeError):
    """A stage returned output that violates its interface contract (a programming
    defect in that implementation, not a rejected candidate)."""


@dataclass(frozen=True)
class TransitionBatch:
    """Transitions in a representation space: z_next = z + f(z, a).

    z, z_next: (N, D) float. a: (N, A) float. episode_id: (N,) str. frame: (N,) int.
    Built per episode, so no transition crosses an episode boundary.
    """

    z: np.ndarray
    a: np.ndarray
    z_next: np.ndarray
    episode_id: np.ndarray
    frame: np.ndarray

    def __len__(self) -> int:
        return len(self.z)

    def take(self, idx) -> "TransitionBatch":
        return TransitionBatch(self.z[idx], self.a[idx], self.z_next[idx], self.episode_id[idx], self.frame[idx])


@dataclass
class CandidateBatch:
    """Candidates from one generator call, before validation.

    source: (n,) int index into the TransitionBatch the candidate came from.
    z: (n, D) candidate state. a: (n, A) candidate action. target: (n, D) the state
    the candidate action should reach. failed: (n,) True where the generator could
    not produce a candidate. measurements: generator-side (n,) float diagnostics.
    """

    source: np.ndarray
    z: np.ndarray
    a: np.ndarray
    target: np.ndarray
    failed: np.ndarray
    measurements: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.source)


def check_transitions(batch: TransitionBatch) -> None:
    n = len(batch.z)
    ok = (batch.z.ndim == 2 and batch.z_next.shape == batch.z.shape and batch.a.ndim == 2 and len(batch.a) == n
          and batch.episode_id.shape == (n,) and batch.frame.shape == (n,))
    if not ok:
        raise InterfaceError(f"malformed TransitionBatch: z {batch.z.shape}, a {batch.a.shape}, "
                             f"z_next {batch.z_next.shape}, episode_id {batch.episode_id.shape}, frame {batch.frame.shape}")


def check_candidates(batch: TransitionBatch, cands: CandidateBatch, producer: str) -> None:
    """Raise InterfaceError unless cands has the shapes its interface promises."""
    n, D, A = len(cands.source), batch.z.shape[1], batch.a.shape[1]
    problems = []
    if cands.source.shape != (n,) or not np.issubdtype(cands.source.dtype, np.integer):
        problems.append(f"source {cands.source.shape} {cands.source.dtype}")
    elif n and (cands.source.min() < 0 or cands.source.max() >= len(batch)):
        problems.append("source indexes outside the transition batch")
    for name, x, shape in (("z", cands.z, (n, D)), ("a", cands.a, (n, A)), ("target", cands.target, (n, D))):
        if not isinstance(x, np.ndarray) or x.shape != shape or not np.issubdtype(x.dtype, np.floating):
            problems.append(f"{name} {getattr(x, 'shape', None)}, expected float {shape}")
    if not isinstance(cands.failed, np.ndarray) or cands.failed.shape != (n,) or cands.failed.dtype != np.bool_:
        problems.append(f"failed must be a bool ({n},) array")
    for k, v in cands.measurements.items():
        if not isinstance(v, np.ndarray) or v.shape != (n,):
            problems.append(f"measurement {k!r} must be a ({n},) array")
    if problems:
        raise InterfaceError(f"{producer} returned a malformed CandidateBatch: " + "; ".join(problems))


def concat(batches: list[CandidateBatch], offsets: list[int]) -> CandidateBatch:
    """Join per-chunk candidates, shifting each chunk's source indices by its offset."""
    keys = set().union(*(b.measurements for b in batches)) if batches else set()
    return CandidateBatch(
        source=np.concatenate([b.source + off for b, off in zip(batches, offsets)]),
        z=np.concatenate([b.z for b in batches]), a=np.concatenate([b.a for b in batches]),
        target=np.concatenate([b.target for b in batches]), failed=np.concatenate([b.failed for b in batches]),
        measurements={k: np.concatenate([b.measurements[k] for b in batches]) for k in keys})
