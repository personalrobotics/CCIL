"""`data prepare`: validate a dataset, split it by episode, and write a run directory.

The run directory holds dataset.npz (ccil.trajectories), split.json (when
data.validation_fraction > 0), config.json (the resolved configuration with
every value's source), and manifest.json (SHA-256 of every file). It is built
in a temporary directory and renamed into place, so an existing run directory
is never left half-written. The run ID is derived from the configuration and
the source data, so identical inputs produce identical run directories.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import ConfigError, Resolved
from ..schema import read_trajectories, write_trajectories
from ..schema.legacy import LEGACY_FORMAT, from_m1_npz
from .split import make_split

MANIFEST_SCHEMA = "ccil.run-manifest"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path):
    """Read a ccil.trajectories file or an M1 ccil-trajectories/1 file, by its declared format."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if meta.get("schema") == "ccil.trajectories":
        return read_trajectories(path)
    if meta.get("format") == LEGACY_FORMAT:
        return from_m1_npz(path)
    raise ConfigError([f"data.path {path}: neither ccil.trajectories nor {LEGACY_FORMAT}"])


def _dump(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


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

    source_sha = _sha256(source)
    run_id = "prepare-" + hashlib.sha256(f"{cfg.config_hash()}:{source_sha}".encode()).hexdigest()[:12]
    run_dir = Path(v["run.output_dir"]) / run_id
    if run_dir.exists() and not v["run.overwrite"]:
        raise FileExistsError(f"{run_dir} exists; set run.overwrite=true to replace it")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = run_dir.parent / f".{run_id}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        write_trajectories(dataset, tmp / "dataset.npz")
        if split is not None:
            _dump(split, tmp / "split.json")
        _dump(cfg.to_dict(), tmp / "config.json")
        files = sorted(p.name for p in tmp.iterdir())
        _dump({"schema": MANIFEST_SCHEMA, "version": "1.0", "run_id": run_id, "workflow": "data.prepare",
               "ccil_version": __version__, "seed": v["run.seed"], "config_hash": cfg.config_hash(),
               "source": {"path": str(source), "sha256": source_sha},
               "episodes": len(dataset.episodes),
               "split": None if split is None else {"train": len(split["train"]), "validation": len(split["validation"])},
               "files": {name: _sha256(tmp / name) for name in files}}, tmp / "manifest.json")
        if run_dir.exists():
            old = run_dir.parent / f".{run_id}.old-{os.getpid()}"
            os.replace(run_dir, old)
            os.replace(tmp, run_dir)
            shutil.rmtree(old)
        else:
            os.replace(tmp, run_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return run_dir
