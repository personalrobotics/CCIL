"""Trajectory-level validation splits, `ccil.split` 1.0.

Whole episodes are assigned, identified by episode_id, so no episode contributes
frames to both splits. Ported from correct_il/validation.py with its strict
schema checks (#13, #66).
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA, VERSION = "ccil.split", "1.0"
FIELDS = {"schema", "version", "seed", "validation_fraction", "episode_ids", "train", "validation"}


class SplitError(ValueError):
    """A valid split cannot be formed, or a split record is malformed."""


def make_split(episode_ids: list[str], validation_fraction: float, seed: int) -> dict:
    n = len(episode_ids)
    n_val = int(round(n * validation_fraction))
    if not 1 <= n_val <= n - 1:
        raise SplitError(
            f"dataset has {n} episodes; data.validation_fraction={validation_fraction} gives {n_val} validation "
            f"episodes, but each split needs at least one. Use more episodes, another fraction, or 0 to skip "
            f"held-out evaluation.")
    perm = np.random.default_rng(seed).permutation(n)
    val = sorted(episode_ids[i] for i in perm[:n_val])
    train = sorted(episode_ids[i] for i in perm[n_val:])
    split = {"schema": SCHEMA, "version": VERSION, "seed": int(seed), "validation_fraction": float(validation_fraction),
             "episode_ids": list(episode_ids), "train": train, "validation": val}
    check_split(split)
    return split


def check_split(split: dict) -> None:
    """Validate the complete split record and that it partitions its episodes."""
    if not isinstance(split, dict):
        raise SplitError(f"split must be a mapping, got {type(split).__name__}")
    missing = sorted(FIELDS - set(split))
    if missing:
        raise SplitError(f"split is missing fields {missing}")
    if split["schema"] != SCHEMA or str(split["version"]).split(".")[0] != VERSION.split(".")[0]:
        raise SplitError(f"unsupported split {split['schema']!r} {split['version']!r}")
    if type(split["seed"]) is not int:
        raise SplitError(f"seed must be an integer, got {split['seed']!r}")
    f = split["validation_fraction"]
    if type(f) not in (int, float) or not math.isfinite(f) or not 0 < f < 1:
        raise SplitError(f"validation_fraction must be in (0, 1), got {f!r}")
    ids = split["episode_ids"]
    if not isinstance(ids, list) or any(type(i) is not str for i in ids) or len(set(ids)) != len(ids):
        raise SplitError("episode_ids must be a list of distinct strings")
    parts = {}
    for name in ("train", "validation"):
        part = split[name]
        if not isinstance(part, list) or not part or any(type(i) is not str for i in part):
            raise SplitError(f"{name} must be a non-empty list of episode_id strings")
        if part != sorted(part) or len(set(part)) != len(part):
            raise SplitError(f"{name} must be sorted and free of duplicates")
        parts[name] = set(part)
    if parts["train"] & parts["validation"] or parts["train"] | parts["validation"] != set(ids):
        raise SplitError(f"train and validation do not partition the {len(ids)} episodes: "
                         f"overlap {sorted(parts['train'] & parts['validation'])}, "
                         f"missing {sorted(set(ids) - parts['train'] - parts['validation'])}")
    if len(parts["validation"]) != int(round(len(ids) * f)):
        raise SplitError(f"validation has {len(parts['validation'])} episodes; {len(ids)} x {f} requires "
                         f"{int(round(len(ids) * f))}")
