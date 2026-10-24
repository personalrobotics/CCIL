"""Representations map observations to the space where dynamics and labels live.

Contract
- encode(observations) takes a dict of modality -> (N, ...) arrays from one
  trajectory and returns (N, D) float. It is deterministic and batch-invariant:
  encoding frames together or separately gives the same rows.
- output_modality names the label space: a trajectory modality such as "state"
  (labels are real observations) or "latent.<name>" (labels in a learned space).
- trainable representations own their parameters; fitting them is the job of an
  integrated pipeline (#19, M5), never of label generation. Gradients do not
  cross this interface.
- kind and version identify the implementation (kind is the
  representation.kind value); a persisted representation records both in its
  checkpoint, like TransitionModel.
- Failures: a wrong output shape or dtype is an InterfaceError raised by the caller.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..schema import TrajectoryDataset
from .batch import InterfaceError, TransitionBatch, check_transitions


@runtime_checkable
class Representation(Protocol):
    kind: str
    version: str
    output_modality: str
    trainable: bool

    def encode(self, observations: dict) -> np.ndarray: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


class IdentityRepresentation:
    """Use one stored modality unchanged; the vector CCIL representation."""

    kind, version = "identity", "1"
    trainable = False

    def __init__(self, modality: str = "state"):
        self.output_modality = modality

    def encode(self, observations: dict) -> np.ndarray:
        return observations[self.output_modality]

    def state_dict(self) -> dict:
        return {"modality": self.output_modality}

    def load_state_dict(self, state: dict) -> None:
        self.output_modality = state["modality"]


def transitions(dataset: TrajectoryDataset, representation) -> TransitionBatch:
    """Encode every episode and pair consecutive frames; never crosses an episode boundary."""
    z, a, zn, ids, frames = [], [], [], [], []
    for ep in dataset.episodes:
        x = representation.encode(ep.observations)
        if not isinstance(x, np.ndarray) or x.ndim != 2 or len(x) != ep.length or not np.issubdtype(x.dtype, np.floating):
            raise InterfaceError(f"{type(representation).__name__}.encode returned {getattr(x, 'shape', type(x))} "
                                 f"{getattr(x, 'dtype', '')} for episode {ep.episode_id!r}; expected float ({ep.length}, D)")
        z.append(x[:-1]); zn.append(x[1:]); a.append(ep.actions[:-1])
        ids.append(np.full(ep.length - 1, ep.episode_id)); frames.append(np.arange(ep.length - 1))
    batch = TransitionBatch(*(np.concatenate(v) for v in (z, a, zn, ids, frames)))
    check_transitions(batch)
    return batch
