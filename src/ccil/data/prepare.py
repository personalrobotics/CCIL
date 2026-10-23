"""`data prepare`: validate a dataset, split it by episode, and write a run directory.

The run (see docs/runs.md) holds data/dataset.npz (ccil.trajectories),
data/split.json (when data.validation_fraction > 0), config.json, and
manifest.json. The run ID is derived from the configuration and the source
data, so identical inputs produce identical run directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import ConfigError, Resolved
from ..runs import RunWriter, layout
from ..runs.provenance import sha256
from ..runs.writer import dump_json
from ..schema import read_trajectories, write_trajectories
from ..schema.legacy import LEGACY_FORMAT, from_m1_npz
from .split import make_split


def load_dataset(path):
    """Read a ccil.trajectories file or an M1 ccil-trajectories/1 file, by its declared format."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if meta.get("schema") == "ccil.trajectories":
        return read_trajectories(path)
    if meta.get("format") == LEGACY_FORMAT:
        return from_m1_npz(path)
    raise ConfigError([f"data.path {path}: neither ccil.trajectories nor {LEGACY_FORMAT}"])


def prepare(cfg: Resolved) -> Path:
    v = cfg.values
    if v["data.path"] is None:
        raise ConfigError(["data.path is required for data prepare"])
    source = Path(v["data.path"])
    if not source.exists():
        raise ConfigError([f"data.path {source} does not exist"])
    dataset = load_dataset(source)
    dataset.validate(action_low=v["data.action_low"], action_high=v["data.action_high"])
    split = None
    if v["data.validation_fraction"] > 0:
        split = make_split([ep.episode_id for ep in dataset.episodes], v["data.validation_fraction"], v["run.seed"])

    run_id = "prepare-" + hashlib.sha256(f"{cfg.config_hash()}:{sha256(source)}".encode()).hexdigest()[:12]
    with RunWriter(v["run.output_dir"], run_id, "data.prepare", cfg, overwrite=v["run.overwrite"]) as run:
        run.add_input("dataset", source)
        write_trajectories(dataset, run.path(layout.DATASET))
        if split is not None:
            dump_json(split, run.path(layout.SPLIT))
        run.metrics({"episodes": len(dataset.episodes),
                     "transitions": sum(ep.length - 1 for ep in dataset.episodes),
                     "train_episodes": None if split is None else len(split["train"]),
                     "validation_episodes": None if split is None else len(split["validation"])})
        run.log(f"validated {len(dataset.episodes)} episodes from {source}")
    return run.run_dir
