"""Non-executable .npz storage for ccil schemas.

Files are written atomically and read with allow_pickle=False. `meta` is a JSON
string with schema, version "MAJOR.MINOR", and optional_fields. A file that
cannot be decoded (not an archive, truncated, pickled arrays, non-JSON or
mistyped meta, inconsistent array layout) raises SchemaError, never a raw
decoding error. Compatibility:
a minor version only adds optional fields; a reader accepts its own major
version, ignores ext/<name> keys, ignores unknown keys only when the file
declares them optional (written by a newer minor), and rejects anything else.
"""
from __future__ import annotations

import io as io_
import json
import os
import zipfile

import numpy as np

from . import labels, trajectories
from .fields import SchemaError
from .labels import CandidateSet
from .trajectories import Episode, TrajectoryDataset


def _atomic_savez(path, arrays: dict) -> None:
    """Write an .npz byte-for-byte deterministically: np.savez stamps each member with
    the current time, so members are written in sorted order with a fixed timestamp."""
    tmp = f"{path}.tmp-{os.getpid()}.npz"
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name in sorted(arrays):
                buf = io_.BytesIO()
                np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _meta(schema, version, optional_fields, **extra) -> np.ndarray:
    return np.array(json.dumps({"schema": schema, "version": f"{version[0]}.{version[1]}",
                                "optional_fields": sorted(optional_fields), **extra}, sort_keys=True))


# What a malformed or truncated file can raise while being decoded; anything else is a defect.
_DECODE_ERRORS = (OSError, ValueError, EOFError, zipfile.BadZipFile)


def _format_error(schema, path, details) -> SchemaError:
    return SchemaError(schema, [{"rule": "format", "where": str(path), "detail": d} for d in details],
                       {"format": len(details)})


def _load_npz(path, schema) -> tuple[dict, dict]:
    """(arrays, meta) of an .npz whose meta is a JSON object; SchemaError if it cannot be decoded."""
    try:
        z = np.load(path, allow_pickle=False)
    except _DECODE_ERRORS as e:
        raise _format_error(schema, path, [f"not a readable .npz file: {e}"]) from None
    if not hasattr(z, "files"):
        raise _format_error(schema, path, ["not an .npz archive"])
    try:
        with z:
            arrays = {k: z[k] for k in z.files}
    except _DECODE_ERRORS as e:
        raise _format_error(schema, path, [f"an array cannot be read: {e}"]) from None
    raw = arrays.pop("meta", None)
    if raw is None:
        raise _format_error(schema, path, ["missing meta"])
    if raw.ndim != 0 or raw.dtype.kind != "U":
        raise _format_error(schema, path, ["meta must be a JSON string"])
    try:
        meta = json.loads(str(raw))
    except ValueError as e:
        raise _format_error(schema, path, [f"meta is not valid JSON: {e}"]) from None
    if not isinstance(meta, dict):
        raise _format_error(schema, path, [f"meta must be a JSON object, got {type(meta).__name__}"])
    return arrays, meta


def read_meta(path, schema="ccil") -> dict:
    """The meta object of any ccil .npz, for format detection; SchemaError if it cannot be decoded."""
    return _load_npz(path, schema)[1]


def _open(path, schema, version, fixed_keys, prefixes) -> tuple[dict, dict]:
    arrays, meta = _load_npz(path, schema)
    problems = []
    if meta.get("schema") != schema:
        problems.append(f"schema is {meta.get('schema')!r}, expected {schema!r}")
    try:
        major, minor = (int(p) for p in str(meta.get("version")).split("."))
    except ValueError:
        major, minor = None, None
        problems.append(f"version {meta.get('version')!r} is not MAJOR.MINOR")
    if major is not None and major != version[0]:
        problems.append(f"major version {major} is not supported (this reader supports {version[0]}.x)")
    declared = meta.get("optional_fields", [])
    if not isinstance(declared, list) or not all(isinstance(k, str) for k in declared):
        problems.append("optional_fields must be a list of field names")
        declared = []
    for key in arrays:
        if key in fixed_keys or key.startswith("ext/") or any(key.startswith(p) for p in prefixes):
            continue
        if key in declared and minor is not None and minor > version[1]:
            continue  # an optional field from a newer minor version
        problems.append(f"unknown field {key!r}")
    if problems:
        raise _format_error(schema, path, problems)
    return arrays, meta


def _layout(schema, path, arrays, specs) -> None:
    """Check rank and dtype kind of the arrays a reader slices before building objects."""
    problems = []
    for key, (ndim, kinds) in specs.items():
        if key in arrays and (arrays[key].ndim != ndim or arrays[key].dtype.kind not in kinds):
            problems.append(f"{key} has shape {arrays[key].shape} and dtype {arrays[key].dtype}; "
                            f"expected rank {ndim} and kind {kinds}")
    if problems:
        raise _format_error(schema, path, problems)


# --- trajectories ---------------------------------------------------------
TRAJ_REQUIRED = {"episode_ids", "episode_lengths", "terminated", "actions"}
TRAJ_OPTIONAL = {"timestamps", "task", "task_present", "action_chunks", "action_chunk_mask"}


def write_trajectories(dataset: TrajectoryDataset, path, validate: bool = True) -> None:
    """Write a dataset; validate=False exists only to build malformed fixtures for tests."""
    if validate:
        dataset.validate()
    eps = dataset.episodes
    arrays = {
        "episode_ids": np.array([ep.episode_id for ep in eps]),
        "episode_lengths": np.array([ep.length for ep in eps], dtype=np.int64),
        "terminated": np.array([ep.terminated for ep in eps], dtype=bool),
        "actions": np.concatenate([ep.actions for ep in eps]),
    }
    for name in eps[0].observations:
        arrays[f"obs/{name}"] = np.concatenate([ep.observations[name] for ep in eps])
    optional = set()
    if any(ep.timestamps is not None for ep in eps):
        arrays["timestamps"] = np.concatenate([ep.timestamps for ep in eps])
    if any(ep.task is not None for ep in eps):
        arrays["task"] = np.array([ep.task or "" for ep in eps])
        arrays["task_present"] = np.array([ep.task is not None for ep in eps])
    if any(ep.action_chunks is not None for ep in eps):
        arrays["action_chunks"] = np.concatenate([ep.action_chunks for ep in eps])
        arrays["action_chunk_mask"] = np.concatenate([ep.action_chunk_mask for ep in eps])
    arrays["meta"] = _meta(trajectories.SCHEMA, trajectories.VERSION, optional, dataset=dataset.meta)
    _atomic_savez(path, arrays)


def read_trajectories(path) -> TrajectoryDataset:
    """Read and validate a ccil.trajectories file."""
    z, meta = _open(path, trajectories.SCHEMA, trajectories.VERSION, TRAJ_REQUIRED | TRAJ_OPTIONAL, ("obs/",))
    missing = sorted(TRAJ_REQUIRED - set(z))
    if missing or not any(k.startswith("obs/") for k in z):
        raise _format_error(trajectories.SCHEMA, path, [f"missing required fields {missing or ['obs/<modality>']}"])
    _layout(trajectories.SCHEMA, path, z, {"episode_ids": (1, "U"), "episode_lengths": (1, "iu"), "terminated": (1, "b"),
                                           "task": (1, "U"), "task_present": (1, "b")})
    problems = []
    if ("task" in z) != ("task_present" in z):
        problems.append("task and task_present must be given together")
    if ("action_chunks" in z) != ("action_chunk_mask" in z):
        problems.append("action_chunks and action_chunk_mask must be given together")
    if not isinstance(meta.get("dataset", {}), dict):
        problems.append("meta.dataset must be a JSON object")
    lengths = z["episode_lengths"]
    total = int(lengths.sum())
    obs = {k[4:]: z[k] for k in z if k.startswith("obs/")}
    per_frame = [z["actions"], *obs.values()] + [z[k] for k in ("timestamps", "action_chunks", "action_chunk_mask") if k in z]
    per_episode = [z[k] for k in ("episode_ids", "terminated", "task", "task_present") if k in z]
    if (lengths < 0).any() or any(x.ndim == 0 or len(x) != total for x in per_frame) \
            or any(len(x) != len(lengths) for x in per_episode):
        problems.append("episode_lengths do not match the per-frame and per-episode arrays")
    if problems:
        raise _format_error(trajectories.SCHEMA, path, problems)
    cut = np.concatenate([[0], np.cumsum(lengths)])
    episodes = []
    for e, (b, f) in enumerate(zip(cut[:-1], cut[1:])):
        episodes.append(Episode(
            episode_id=str(z["episode_ids"][e]),
            observations={k: v[b:f] for k, v in obs.items()},
            actions=z["actions"][b:f],
            terminated=bool(z["terminated"][e]),
            timestamps=z["timestamps"][b:f] if "timestamps" in z else None,
            task=str(z["task"][e]) if "task" in z and bool(z["task_present"][e]) else None,
            action_chunks=z["action_chunks"][b:f] if "action_chunks" in z else None,
            action_chunk_mask=z["action_chunk_mask"][b:f] if "action_chunk_mask" in z else None))
    return TrajectoryDataset(episodes, meta.get("dataset", {})).validate()


# --- candidates -----------------------------------------------------------
CAND_REQUIRED = {"source_episode_id", "source_frame", "actions", "accepted", "reasons", "weight"}
CAND_OPTIONAL: set = set()
CAND_META = ("run_id", "generator", "reason_codes", "seed", "inputs", "weighting")  # CandidateSet attributes in meta


def write_candidates(cands: CandidateSet, path, validate: bool = True) -> None:
    if validate:
        cands.validate()
    arrays = {"source_episode_id": cands.source_episode_id, "source_frame": cands.source_frame,
              "actions": cands.actions, "accepted": cands.accepted, "reasons": cands.reasons, "weight": cands.weight}
    arrays.update({f"obs/{k}": v for k, v in cands.observations.items()})
    arrays.update({f"target/{k}": v for k, v in cands.target_observations.items()})
    arrays.update({f"meas/{k}": v for k, v in cands.measurements.items()})
    meta = {k: getattr(cands, k) for k in CAND_META}
    meta["reason_codes"] = list(meta["reason_codes"])
    arrays["meta"] = _meta(labels.SCHEMA, labels.file_version(meta["reason_codes"]), CAND_OPTIONAL, **meta)
    _atomic_savez(path, arrays)


def read_candidates(path, dataset: TrajectoryDataset | None = None) -> CandidateSet:
    """Read and validate a ccil.candidates file; with dataset, also check source identity."""
    z, meta = _open(path, labels.SCHEMA, labels.VERSION, CAND_REQUIRED | CAND_OPTIONAL, ("obs/", "target/", "meas/"))
    missing = sorted(CAND_REQUIRED - set(z))
    if missing:
        raise _format_error(labels.SCHEMA, path, [f"missing required fields {missing}"])
    codes = meta.get("reason_codes", [])
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        raise _format_error(labels.SCHEMA, path, ["meta.reason_codes must be a list of strings"])
    cands = CandidateSet(
        run_id=meta.get("run_id"), generator=meta.get("generator"),
        source_episode_id=z["source_episode_id"], source_frame=z["source_frame"],
        observations={k[4:]: z[k] for k in z if k.startswith("obs/")},
        actions=z["actions"],
        target_observations={k[7:]: z[k] for k in z if k.startswith("target/")},
        accepted=z["accepted"], reasons=z["reasons"], weight=z["weight"], seed=meta.get("seed"),
        inputs=meta.get("inputs"), weighting=meta.get("weighting"), reason_codes=tuple(codes),
        measurements={k[5:]: z[k] for k in z if k.startswith("meas/")})
    return cands.validate(dataset)
