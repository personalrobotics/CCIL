"""One-way conversion from the M1 `ccil-trajectories/1` .npz into `ccil.trajectories` 1.0.

Legacy data has a single state modality, integer episode order, and no record
of termination or timing, so episodes become state-only, are identified by
their index, and are marked truncated (terminated=False).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .fields import SchemaError
from .trajectories import Episode, TrajectoryDataset

LEGACY_FORMAT = "ccil-trajectories/1"
CONVERTER = "ccil.schema.legacy/1"


def from_m1_npz(path) -> TrajectoryDataset:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if meta.get("format") != LEGACY_FORMAT or not {"observations", "actions", "episode_lengths"} <= set(z.files):
            raise SchemaError("ccil.schema.legacy", [{"rule": "format", "where": str(path),
                                                      "detail": f"not a {LEGACY_FORMAT} file"}], {"format": 1})
        obs, act, lengths = z["observations"], z["actions"], z["episode_lengths"]
    cut = np.concatenate([[0], np.cumsum(lengths)])
    episodes = [Episode(episode_id=str(e), observations={"state": obs[b:f]}, actions=act[b:f], terminated=False)
                for e, (b, f) in enumerate(zip(cut[:-1], cut[1:]))]
    return TrajectoryDataset(episodes, {"converted_from": {"format": LEGACY_FORMAT, "sha256": digest, "meta": meta},
                                        "converter": CONVERTER}).validate()
