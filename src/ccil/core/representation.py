"""Representations map observations to the space where dynamics and labels live.

Contract
- encode(observations) takes a dict of modality -> (N, ...) arrays from one
  trajectory and returns (N, D) float. It is deterministic and batch-invariant:
  encoding frames together or separately gives the same rows.
- output_modality names the label space: a trajectory modality such as "state"
  (labels are real observations), an ordered "+"-joined list of trajectory
  modalities, concatenated in that order (labels keep each part under its own
  name), or "latent.<name>" (labels in a learned space).
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


JOIN = "+"


def parts(modality: str) -> list[str]:
    """The stored modalities a label-space name reads, in order: one, or the parts of a "+"-joined name."""
    return modality.split(JOIN)


def observation(observations: dict, modality: str) -> np.ndarray:
    """The label-space array of a modality name: the stored array, or its parts concatenated in order."""
    names = parts(modality)
    if len(names) == 1:
        return observations[modality]
    return np.concatenate([np.asarray(observations[m]).reshape(len(observations[m]), -1) for m in names], axis=1)


def split(modality: str, z: np.ndarray, widths: dict) -> dict:
    """Label-space rows back into their stored modalities (the inverse of observation for a joined name)."""
    names = parts(modality)
    if len(names) == 1:
        return {modality: z}
    edges = np.cumsum([0] + [widths[m] for m in names])
    return {m: z[:, edges[i]:edges[i + 1]] for i, m in enumerate(names)}


def widths(dataset: TrajectoryDataset, modality: str) -> dict:
    """The width of each stored part of a joined modality name, from the dataset ({} for a single name, which needs none:
    it may name a learned space, such as latent.pca, that the dataset does not store)."""
    names = parts(modality)
    if len(names) == 1:
        return {}
    first = dataset.episodes[0].observations
    return {m: int(np.prod(first[m].shape[1:])) for m in names}


class IdentityRepresentation:
    """Use stored modalities unchanged: one, or an ordered "+"-joined list concatenated; the vector CCIL representation."""

    kind, version = "identity", "1"
    trainable = False

    def __init__(self, modality: str = "state"):
        self.output_modality = modality

    def encode(self, observations: dict) -> np.ndarray:
        return observation(observations, self.output_modality)

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
