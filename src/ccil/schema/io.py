"""Non-executable .npz storage for ccil schemas.

Files are written atomically and read with allow_pickle=False. `meta` is a JSON
string with schema, version "MAJOR.MINOR", and optional_fields. Compatibility:
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


def _open(path, schema, version, fixed_keys, prefixes):
    z = np.load(path, allow_pickle=False)
    try:
        if "meta" not in z.files:
            raise SchemaError(schema, [{"rule": "meta", "where": str(path), "detail": "missing meta"}], {"meta": 1})
        meta = json.loads(str(z["meta"]))
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
        declared = set(meta.get("optional_fields", []))
        for key in z.files:
            if key == "meta" or key in fixed_keys or key.startswith("ext/") or any(key.startswith(p) for p in prefixes):
                continue
            if key in declared and minor is not None and minor > version[1]:
                continue  # an optional field from a newer minor version
            problems.append(f"unknown field {key!r}")
        if problems:
            raise SchemaError(schema, [{"rule": "format", "where": str(path), "detail": p} for p in problems],
                              {"format": len(problems)})
        return z, meta
    except Exception:
        z.close()
        raise


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
    with z:
        missing = sorted(TRAJ_REQUIRED - set(z.files))
        if missing or not any(k.startswith("obs/") for k in z.files):
            raise SchemaError(trajectories.SCHEMA, [{"rule": "format", "where": str(path),
                                                    "detail": f"missing required fields {missing or ['obs/<modality>']}"}], {"format": 1})
        lengths = z["episode_lengths"]
        total = int(lengths.sum())
        obs = {k[4:]: z[k] for k in z.files if k.startswith("obs/")}
        per_frame = [z["actions"], *obs.values()] + [z[k] for k in ("timestamps", "action_chunks", "action_chunk_mask") if k in z.files]
        if lengths.ndim != 1 or (lengths < 0).any() or any(len(x) != total for x in per_frame) \
                or len(z["episode_ids"]) != len(lengths) or len(z["terminated"]) != len(lengths):
            raise SchemaError(trajectories.SCHEMA, [{"rule": "format", "where": str(path),
                                                    "detail": "episode_lengths do not match the per-frame and per-episode arrays"}], {"format": 1})
        cut = np.concatenate([[0], np.cumsum(lengths)])
        episodes = []
        for e, (b, f) in enumerate(zip(cut[:-1], cut[1:])):
            episodes.append(Episode(
                episode_id=str(z["episode_ids"][e]),
                observations={k: v[b:f] for k, v in obs.items()},
                actions=z["actions"][b:f],
                terminated=bool(z["terminated"][e]),
                timestamps=z["timestamps"][b:f] if "timestamps" in z.files else None,
                task=str(z["task"][e]) if "task" in z.files and bool(z["task_present"][e]) else None,
                action_chunks=z["action_chunks"][b:f] if "action_chunks" in z.files else None,
                action_chunk_mask=z["action_chunk_mask"][b:f] if "action_chunk_mask" in z.files else None))
    return TrajectoryDataset(episodes, meta.get("dataset", {})).validate()


# --- candidates -----------------------------------------------------------
CAND_REQUIRED = {"source_episode_id", "source_frame", "actions", "accepted", "reasons"}
CAND_OPTIONAL = {"weight"}


def write_candidates(cands: CandidateSet, path, validate: bool = True) -> None:
    if validate:
        cands.validate()
    arrays = {"source_episode_id": cands.source_episode_id, "source_frame": cands.source_frame,
              "actions": cands.actions, "accepted": cands.accepted, "reasons": cands.reasons}
    arrays.update({f"obs/{k}": v for k, v in cands.observations.items()})
    arrays.update({f"target/{k}": v for k, v in cands.target_observations.items()})
    arrays.update({f"meas/{k}": v for k, v in cands.measurements.items()})
    if cands.weight is not None:
        arrays["weight"] = cands.weight
    arrays["meta"] = _meta(labels.SCHEMA, labels.VERSION, set(), run_id=cands.run_id, generator=cands.generator,
                           reason_codes=list(cands.reason_codes))
    _atomic_savez(path, arrays)


def read_candidates(path, dataset: TrajectoryDataset | None = None) -> CandidateSet:
    """Read and validate a ccil.candidates file; with dataset, also check source identity."""
    z, meta = _open(path, labels.SCHEMA, labels.VERSION, CAND_REQUIRED | CAND_OPTIONAL, ("obs/", "target/", "meas/"))
    with z:
        missing = sorted(CAND_REQUIRED - set(z.files))
        if missing:
            raise SchemaError(labels.SCHEMA, [{"rule": "format", "where": str(path),
                                              "detail": f"missing required fields {missing}"}], {"format": 1})
        cands = CandidateSet(
            run_id=meta.get("run_id"), generator=meta.get("generator"),
            source_episode_id=z["source_episode_id"], source_frame=z["source_frame"],
            observations={k[4:]: z[k] for k in z.files if k.startswith("obs/")},
            actions=z["actions"],
            target_observations={k[7:]: z[k] for k in z.files if k.startswith("target/")},
            accepted=z["accepted"], reasons=z["reasons"], reason_codes=tuple(meta.get("reason_codes", ())),
            measurements={k[5:]: z[k] for k in z.files if k.startswith("meas/")},
            weight=z["weight"] if "weight" in z.files else None)
    return cands.validate(dataset)
