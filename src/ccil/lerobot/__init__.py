"""LeRobot Dataset v3.0 tabular-format bridge.

The bridge uses PyArrow directly and never imports the LeRobot runtime.
Unsupported modalities and format versions are refused, never dropped. See
``docs/lerobot.md`` for the supported feature and release matrix.
"""
from __future__ import annotations

from .v3 import (CODEBASE_VERSION, LeRobotFormatError, convert, fetch, iter_episodes, read_dataset, read_info,
                 read_stats, write_dataset)

__all__ = ["CODEBASE_VERSION", "LeRobotFormatError", "convert", "fetch", "iter_episodes", "read_dataset", "read_info",
           "read_stats", "write_dataset"]
