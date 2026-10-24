"""One-way conversion from the M1 `ccil-trajectories/1` .npz into `ccil.trajectories` 1.0.

Legacy data has a single state modality, integer episode order, and no record
of termination or timing, so episodes become state-only, are identified by
their index, and are marked truncated (terminated=False).
"""
from __future__ import annotations

import hashlib

import numpy as np

from .io import _format_error, _load_npz
from .trajectories import Episode, TrajectoryDataset

LEGACY_FORMAT = "ccil-trajectories/1"
CONVERTER = "ccil.schema.legacy/1"
SCHEMA = "ccil.schema.legacy"


def from_m1_npz(path) -> TrajectoryDataset:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    z, meta = _load_npz(path, SCHEMA)
    if meta.get("format") != LEGACY_FORMAT or not {"observations", "actions", "episode_lengths"} <= set(z):
        raise _format_error(SCHEMA, path, [f"not a {LEGACY_FORMAT} file"])
    obs, act, lengths = z["observations"], z["actions"], z["episode_lengths"]
    if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or (lengths < 0).any() \
            or obs.ndim == 0 or act.ndim == 0 or not len(obs) == len(act) == int(lengths.sum()):
        raise _format_error(SCHEMA, path, ["episode_lengths do not match observations and actions"])
    cut = np.concatenate([[0], np.cumsum(lengths)])
    episodes = [Episode(episode_id=str(e), observations={"state": obs[b:f]}, actions=act[b:f], terminated=False)
                for e, (b, f) in enumerate(zip(cut[:-1], cut[1:]))]
    return TrajectoryDataset(episodes, {"converted_from": {"format": LEGACY_FORMAT, "sha256": digest, "meta": meta},
                                        "converter": CONVERTER}).validate()
