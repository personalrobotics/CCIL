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
from ..schema.labels import canonical_reason_order, reason_code_problem
from .batch import InterfaceError, check_candidates, concat
from .chunks import SPEC as CHUNK_SPEC
from .chunks import TARGET_FRAME_SPEC as CHUNK_TARGET_FRAME_SPEC
from .chunks import action_chunks
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


def _declared_specs(producer, measured: set, where: str) -> dict:
    """The producer's measurement_specs, which must name exactly the measurements it returned."""
    specs = getattr(producer, "measurement_specs", None)
    if not isinstance(specs, dict):
        raise InterfaceError(f"{type(producer).__name__} declares no measurement_specs; every generator and validator "
                             "must declare the definition, norm, and units of what it measures ({} if nothing)")
    if set(specs) != measured:
        raise InterfaceError(f"{where} returned measurements {sorted(measured)} but its measurement_specs name "
                             f"{sorted(specs)}")
    return specs


# Generators whose k draws per transition come from that transition's own stream (generators._noisy_actions), so
# the first k of k_max draws are the k draws a direct generation at k makes; every built-in validator and the
# uniform weighting act row by row, so a row filter of a k_max file is the file a direct generation at k writes.
# The view selects by each row's recorded draw ordinal, never by its position in the file.
NESTED_GENERATORS = ("noisy_action", "noisy_action_preimage")
DRAW_INDEX = "generator.draw_index"
ROW_LOCAL_REASONS = (*REASON_CODES, "label_filter.model_misfit")


def nested_view(cands: CandidateSet, k: int, parent_sha256: str) -> CandidateSet:
    """The candidates a generation at labels_per_transition=k makes, as a row filter of a k_max candidate set.

    Rows are selected by their recorded draw ordinal (generator.draw_index < k). Refuses a set without draw
    ordinals, one whose transitions do not each carry the ordinals 0..k_max-1 exactly once in order, a generator
    without nested draws, a validator or weighting that is not known to act row by row (a population rule such as a
    quantile would change under a view), and k outside [1, k_max]. Certificates are not views: certify the view
    itself. parent_sha256 is the k_max file's hash, recorded as the view's input.
    """
    g = cands.generator
    k_max = (g.get("params") or {}).get("labels_per_transition")
    n = len(cands.actions)
    problems = []
    if g.get("name") not in NESTED_GENERATORS:
        problems.append(f"generator {g.get('name')!r} has no nested draws (allowed: {list(NESTED_GENERATORS)})")
    if isinstance(k, bool) or not isinstance(k, int) or not isinstance(k_max, int) or not 1 <= k <= k_max:
        problems.append(f"k must be an integer in [1, labels_per_transition={k_max!r}], got {k!r}")
    if cands.weighting.get("name") != Uniform.name:
        problems.append(f"weighting {cands.weighting.get('name')!r} is not known to be row-local")
    if not set(cands.reason_codes) <= set(ROW_LOCAL_REASONS):
        problems.append(f"reason codes {sorted(set(cands.reason_codes) - set(ROW_LOCAL_REASONS))} are not known to be "
                        "row-local")
    if DRAW_INDEX not in cands.measurements:
        problems.append(f"no per-row draw ordinal (measurement {DRAW_INDEX!r}); regenerate with a generator that "
                        "records it")
    if not problems:
        if n % k_max:
            problems.append(f"{n} rows are not a whole number of {k_max}-draw transitions")
        else:
            ep = np.asarray(cands.source_episode_id).astype(str).reshape(-1, k_max)
            fr = np.asarray(cands.source_frame).reshape(-1, k_max)
            draw = np.asarray(cands.measurements[DRAW_INDEX]).reshape(-1, k_max)
            keys = list(zip(ep[:, 0], fr[:, 0]))
            if not ((ep == ep[:, :1]).all() and (fr == fr[:, :1]).all()) or len(set(keys)) != len(keys):
                problems.append(f"rows are not {k_max} consecutive draws of one transition each")
            elif not (draw == np.arange(k_max)).all():
                problems.append(f"draw ordinals are not 0..{k_max - 1} in order within every transition")
    if problems:
        raise InterfaceError("nested_view: " + "; ".join(problems))
    keep = np.asarray(cands.measurements[DRAW_INDEX]) < k

    def rows(x):
        return None if x is None else x[keep]
    view_hash = hashlib.sha256(f"{g['params_hash']}|nested_view|k={k}".encode()).hexdigest()[:16]
    return CandidateSet(
        run_id=f"{cands.run_id}.k{k}",
        generator={**g, "params": {**g["params"], "labels_per_transition": k}, "params_hash": view_hash},
        source_episode_id=cands.source_episode_id[keep], source_frame=cands.source_frame[keep],
        observations={m: v[keep] for m, v in cands.observations.items()}, actions=cands.actions[keep],
        target_observations={m: v[keep] for m, v in cands.target_observations.items()},
        accepted=cands.accepted[keep], reasons=cands.reasons[keep], weight=cands.weight[keep], seed=cands.seed,
        inputs={**cands.inputs, "nested_view_of": parent_sha256}, weighting=dict(cands.weighting),
        reason_codes=tuple(cands.reason_codes), measurements={m: v[keep] for m, v in cands.measurements.items()},
        measurement_specs=cands.measurement_specs, action_chunks=rows(cands.action_chunks),
        action_chunk_mask=rows(cands.action_chunk_mask), action_chunk_spec=cands.action_chunk_spec,
    ).validate()


def checkpoint_sha256(model) -> str:
    """SHA-256 of the checkpoint model.save() writes; checkpoints are byte-deterministic."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.npz"
        model.save(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_labels(dataset: TrajectoryDataset, representation, model, generator, validators, run_id: str, seed: int,
                    weighting=None, batch_size: int | None = None, inputs: dict | None = None,
                    action_chunk_size: int | None = None) -> CandidateSet:
    """Generate, validate, and weight candidates; return a schema-valid CandidateSet.

    batch_size splits generation into chunks; generators are batch-invariant, so it
    changes memory use, not results. weighting defaults to Uniform. inputs maps
    input names to SHA-256 digests for the candidate provenance; a missing
    transition_model entry is the hash of the checkpoint model.save() writes.
    action_chunk_size adds action chunks of that size (ccil.core.chunks).
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

    for v in validators:
        why = None if v.reason is None else reason_code_problem(v.reason)
        if why:
            raise InterfaceError(f"{type(v).__name__}.reason: {why}")
    codes = canonical_reason_order(set(REASON_CODES) | {v.reason for v in validators if v.reason is not None})
    reasons = np.zeros((len(cands), len(codes)), bool)
    measurements = {f"generator.{k}": v.astype(np.float64) for k, v in cands.measurements.items()}
    # Every measurement's meaning travels with the file; a producer must declare exactly what it measures.
    specs = {f"generator.{k}": dict(v)
             for k, v in _declared_specs(generator, set(cands.measurements), f"{type(generator).__name__}.generate").items()}
    for v in validators:
        reject, meas = v.check(batch, cands, model)
        if not isinstance(reject, np.ndarray) or reject.shape != (len(cands),) or reject.dtype != np.bool_:
            raise InterfaceError(f"{type(v).__name__}.check must return a bool ({len(cands)},) reject mask")
        if v.reason is None and reject.any():
            raise InterfaceError(f"{type(v).__name__} has no reason code but rejected candidates")
        if v.reason is not None:
            reasons[:, codes.index(v.reason)] |= reject
        for k, x in meas.items():
            if k in measurements or not isinstance(x, np.ndarray) or x.shape != (len(cands),):
                raise InterfaceError(f"{type(v).__name__} measurement {k!r} is duplicated or not a ({len(cands)},) array")
            measurements[k] = x.astype(np.float64)
        specs.update({k: dict(x) for k, x in _declared_specs(v, set(meas), f"{type(v).__name__}.check").items()})
    accepted = ~reasons.any(axis=1)
    weighting = weighting or Uniform()
    weight = np.asarray(weighting.weights(cands, accepted), dtype=np.float64)
    inputs = dict(inputs or {})
    if "transition_model" not in inputs:
        if not hasattr(model, "save"):
            raise InterfaceError("pass inputs['transition_model'] or give the transition model a save() so labels "
                                 "record its checkpoint hash")
        inputs["transition_model"] = checkpoint_sha256(model)

    chunk_fields = {}
    if action_chunk_size is not None:
        chunks, mask, tau, offset = action_chunks(dataset, generator, batch.episode_id[cands.source],
                                                  batch.frame[cands.source], cands.a, action_chunk_size)
        measurements["chunk_target_frame"] = tau
        specs["chunk_target_frame"] = dict(CHUNK_TARGET_FRAME_SPEC)
        chunk_fields = {"action_chunks": chunks, "action_chunk_mask": mask,
                        "action_chunk_spec": {"chunk_size": int(action_chunk_size), "target_frame_offset": offset,
                                              **CHUNK_SPEC}}

    modality = representation.output_modality
    return CandidateSet(
        run_id=run_id,
        generator={"name": generator.name, "version": generator.version, "params_hash": params_hash(generator, validators),
                   "params": json.loads(json.dumps(generator.params))},
        source_episode_id=batch.episode_id[cands.source], source_frame=batch.frame[cands.source].astype(np.int64),
        observations={modality: cands.z}, actions=cands.a, target_observations={modality: cands.target},
        accepted=accepted, reasons=reasons, weight=weight, seed=int(seed), inputs=inputs,
        weighting={"name": weighting.name, "version": weighting.version, "params": dict(weighting.params)},
        reason_codes=codes, measurements=measurements, measurement_specs=specs, **chunk_fields,
    ).validate(dataset)
