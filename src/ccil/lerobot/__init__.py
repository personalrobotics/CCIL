"""CCIL's LeRobot Dataset v3.0 format bridge (#26). It does not install or import the LeRobot runtime.

CCIL reads and writes the tabular subset of the LeRobotDataset v3.0 storage format (Parquet shards described by
meta/info.json) with PyArrow directly: states, actions, timestamps, task indices, episode metadata, statistics and
schema metadata. Images, depth, audio and video are not supported until M5 and are refused, never dropped.
Unknown format versions and unsupported feature encodings fail closed. Hub access is an I/O convenience; the
bridge works entirely on local directories.

Install with the extra: pip install 'ccil[lerobot]' (pyarrow and huggingface_hub; not lerobot, torch, datasets
or video decoders). Which LeRobot releases consume and produce what this bridge reads and writes is an
evidence-backed matrix (docs/lerobot.md), not a version promise.
"""
from __future__ import annotations

from .v3 import (CODEBASE_VERSION, LeRobotFormatError, fetch, iter_episodes, read_dataset, read_info,
                 read_stats, write_dataset)

__all__ = ["CODEBASE_VERSION", "LeRobotFormatError", "fetch", "iter_episodes", "read_dataset", "read_info",
           "read_stats", "write_dataset"]
