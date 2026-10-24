"""Compose the stages into a validated ccil.candidates set.

dataset -> representation -> transitions -> generator -> validators -> weighting.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from ..schema import REASON_CODES, CandidateSet, TrajectoryDataset
from .batch import InterfaceError, check_candidates, concat
from .representation import transitions


class Uniform:
    """Weight 1 for accepted candidates and 0 for rejected ones."""

    name, version, params = "uniform", "1", {}

    def weights(self, cands, accepted: np.ndarray) -> np.ndarray:
        return accepted.astype(np.float64)


def params_hash(generator, validators) -> str:
    desc = {"generator": [generator.name, generator.version, generator.params],
            "validators": [[type(v).__name__, v.reason, {k: (x.tolist() if isinstance(x, np.ndarray) else x)
                                                          for k, x in sorted(vars(v).items())}] for v in validators]}
    return hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()[:16]


def checkpoint_sha256(model) -> str:
    """SHA-256 of the checkpoint model.save() writes; checkpoints are byte-deterministic."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.npz"
        model.save(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_labels(dataset: TrajectoryDataset, representation, model, generator, validators, run_id: str, seed: int,
                    weighting=None, batch_size: int | None = None, inputs: dict | None = None) -> CandidateSet:
    """Generate, validate, and weight candidates; return a schema-valid CandidateSet.

    batch_size splits generation into chunks; generators are batch-invariant, so it
    changes memory use, not results. weighting defaults to Uniform. inputs maps
    input names to SHA-256 digests for the candidate provenance; a missing
    transition_model entry is the hash of the checkpoint model.save() writes.
    """
    dataset.validate()
    batch = transitions(dataset, representation)
    size = batch_size or max(len(batch), 1)
    chunks, offsets = [], []
    for start in range(0, len(batch), size):
        part = batch.take(slice(start, start + size))
        out = generator.generate(part, model, seed)
        check_candidates(part, out, f"{type(generator).__name__}.generate")
        chunks.append(out)
        offsets.append(start)
    cands = concat(chunks, offsets)

    reasons = np.zeros((len(cands), len(REASON_CODES)), bool)
    measurements = {f"generator.{k}": v.astype(np.float64) for k, v in cands.measurements.items()}
    for v in validators:
        if v.reason is not None and v.reason not in REASON_CODES:
            raise InterfaceError(f"{type(v).__name__}.reason {v.reason!r} is not one of {list(REASON_CODES)}")
        reject, meas = v.check(batch, cands, model)
        if not isinstance(reject, np.ndarray) or reject.shape != (len(cands),) or reject.dtype != np.bool_:
            raise InterfaceError(f"{type(v).__name__}.check must return a bool ({len(cands)},) reject mask")
        if v.reason is None and reject.any():
            raise InterfaceError(f"{type(v).__name__} has no reason code but rejected candidates")
        if v.reason is not None:
            reasons[:, REASON_CODES.index(v.reason)] |= reject
        for k, x in meas.items():
            if k in measurements or not isinstance(x, np.ndarray) or x.shape != (len(cands),):
                raise InterfaceError(f"{type(v).__name__} measurement {k!r} is duplicated or not a ({len(cands)},) array")
            measurements[k] = x.astype(np.float64)
    accepted = ~reasons.any(axis=1)
    weighting = weighting or Uniform()
    weight = np.asarray(weighting.weights(cands, accepted), dtype=np.float64)
    inputs = dict(inputs or {})
    if "transition_model" not in inputs:
        if not hasattr(model, "save"):
            raise InterfaceError("pass inputs['transition_model'] or give the transition model a save() so labels "
                                 "record its checkpoint hash")
        inputs["transition_model"] = checkpoint_sha256(model)

    modality = representation.output_modality
    return CandidateSet(
        run_id=run_id,
        generator={"name": generator.name, "version": generator.version, "params_hash": params_hash(generator, validators)},
        source_episode_id=batch.episode_id[cands.source], source_frame=batch.frame[cands.source].astype(np.int64),
        observations={modality: cands.z}, actions=cands.a, target_observations={modality: cands.target},
        accepted=accepted, reasons=reasons, weight=weight, seed=int(seed), inputs=inputs,
        weighting={"name": weighting.name, "version": weighting.version, "params": dict(weighting.params)},
        reason_codes=REASON_CODES, measurements=measurements,
    ).validate(dataset)
