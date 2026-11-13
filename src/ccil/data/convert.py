"""Convert a trusted ``ccil-trajectories/1`` intermediate to the current schema.

CCIL never opens the original pickle. The conversion preserves its recorded
hash and trust metadata, verifies an exact read-back, and writes a manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..schema import read_trajectories, write_trajectories
from ..schema.legacy import CONVERTER, LEGACY_FORMAT, from_m1_npz

FIELD_MAP = {"observations": "obs/state", "actions": "actions",
             "episode_lengths": "episode_lengths (episode i becomes episode_id 'i')",
             "(none)": "terminated = False for every episode (legacy recorded no termination); no timestamps"}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_m1(src, dst, manifest=None) -> dict:
    """Convert, verify, and atomically record a legacy-intermediate dataset."""
    src, dst = Path(src), Path(dst)
    dataset = from_m1_npz(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_trajectories(dataset, dst)
    back = read_trajectories(dst)
    exact = len(back.episodes) == len(dataset.episodes) and all(
        a.episode_id == b.episode_id and np.array_equal(a.actions, b.actions) and a.actions.dtype == b.actions.dtype
        and np.array_equal(a.observations["state"], b.observations["state"])
        and a.observations["state"].dtype == b.observations["state"].dtype
        for a, b in zip(dataset.episodes, back.episodes))
    if not exact:
        os.remove(dst)
        raise RuntimeError(f"{dst} does not read back exactly as converted; removed")
    m1 = dataset.meta["converted_from"]["meta"]
    record = {
        "converter": CONVERTER, "source_format": LEGACY_FORMAT,
        "source": {"path": src.name, "sha256": _sha(src)},
        "legacy_pickle": m1.get("source"), "trust": m1.get("trust"), "m1_converter_version": m1.get("converter_version"),
        "output": {"path": dst.name, "sha256": _sha(dst), "schema": "ccil.trajectories 1.0"},
        "field_map": FIELD_MAP,
        "counts": {"episodes": len(back.episodes), "frames": int(sum(ep.length for ep in back.episodes)),
                   "transitions": int(sum(ep.length - 1 for ep in back.episodes))},
        "dtypes": {"obs/state": str(back.episodes[0].observations["state"].dtype),
                   "actions": str(back.episodes[0].actions.dtype)},
        "shapes": {"state_dim": int(back.episodes[0].observations["state"].shape[1]),
                   "action_dim": int(back.episodes[0].actions.shape[1])},
        "read_back_exact": True,
    }
    if manifest is not None:
        manifest = Path(manifest)
        tmp = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, manifest)
    return record
