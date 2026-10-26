"""Explicit, one-way conversion of an M1 dataset into a canonical ccil.trajectories file (#23).

    ccil data convert M1.npz OUT.npz --manifest MANIFEST.json

Legacy pickles cross the trust boundary only through
correct_il/convert_legacy_dataset.py, which verifies their SHA-256 (or takes
an explicit acknowledgement) before unpickling and writes the M1
ccil-trajectories/1 file; ccil itself never unpickles. This step turns that
file into the canonical schema, which data.prepare also reads on the fly,
as an artifact that can be inspected before any model is trained. Its
meta.dataset keeps the provenance chain: the M1 file's SHA-256, and from the M1
meta the legacy pickle's SHA-256, trust mode, and converter version.
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
    """Convert src (M1) to dst (ccil.trajectories); write manifest atomically; return it. SchemaError if src is not M1."""
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
