"""Declarative array specifications and violation collection.

Validation reports every violation with where it occurred and never coerces,
broadcasts, truncates, clamps, or mutates the value it checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOAT32_MAX = float(np.finfo(np.float32).max)
MAX_EXAMPLES = 20  # per rule; counts stay exact


class SchemaError(ValueError):
    """A value violates a ccil schema; .violations lists each violation."""

    def __init__(self, schema: str, violations: list[dict], counts: dict[str, int]):
        self.schema, self.violations, self.counts = schema, violations, counts
        summary = ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
        lines = [f"{v['rule']} at {v['where']}: {v['detail']}" for v in violations]
        super().__init__(f"{schema} failed validation ({summary}):\n  " + "\n  ".join(lines))


class Violations:
    def __init__(self, schema: str):
        self.schema, self.items, self.counts = schema, [], {}

    def add(self, rule: str, where: str, detail: str) -> None:
        self.counts[rule] = self.counts.get(rule, 0) + 1
        if self.counts[rule] <= MAX_EXAMPLES:
            self.items.append({"rule": rule, "where": where, "detail": detail})

    def raise_if_any(self) -> None:
        if self.items:
            raise SchemaError(self.schema, self.items, self.counts)


_KINDS = {
    "float": lambda dt: np.issubdtype(dt, np.floating),
    "uint8": lambda dt: dt == np.uint8,
    "bool": lambda dt: dt == np.bool_,
    "int": lambda dt: np.issubdtype(dt, np.integer) and dt != np.bool_,
    "str": lambda dt: dt.kind == "U",
}


@dataclass(frozen=True)
class ArraySpec:
    """One array field. shape entries are symbolic dimension names or fixed ints."""

    name: str
    kind: str
    shape: tuple
    units: str
    description: str
    required: bool = True

    def check(self, x, where: str, dims: dict, out: Violations, finite: bool = True) -> bool:
        """Check type, dtype kind, rank, and shape; bind symbolic dims. Returns False on failure."""
        if not isinstance(x, np.ndarray):
            out.add("type", where, f"{self.name} must be a numpy array, got {type(x).__name__}")
            return False
        if not _KINDS[self.kind](x.dtype):
            out.add("dtype", where, f"{self.name} dtype {x.dtype} is not {self.kind}")
            return False
        if x.ndim != len(self.shape):
            out.add("rank", where, f"{self.name} has shape {x.shape}; expected rank {len(self.shape)} {self.shape}")
            return False
        for axis, (dim, size) in enumerate(zip(self.shape, x.shape)):
            if isinstance(dim, int):
                if size != dim:
                    out.add("shape", where, f"{self.name} axis {axis} is {size}, expected {dim}")
                    return False
            elif dim in dims and dims[dim] != size:
                out.add("shape", where, f"{self.name} axis {axis} ({dim}) is {size}, expected {dims[dim]}")
                return False
            else:
                dims[dim] = size
        if finite and self.kind == "float":
            check_finite(x, self.name, where, out)
        return True


def check_finite(x: np.ndarray, name: str, where: str, out: Violations, rows=None) -> None:
    """Report frames (axis 0) with non-finite or float32-overflowing values."""
    flat = x.reshape(len(x), -1)
    if rows is not None:
        flat = flat[rows]
    idx = np.arange(len(x)) if rows is None else np.flatnonzero(rows) if rows.dtype == bool else rows
    bad = ~np.isfinite(flat).all(axis=1)
    for i in np.flatnonzero(bad):
        out.add("non_finite", f"{where}, frame {int(idx[i])}", f"{name} has non-finite values")
    big = ~bad & (np.abs(flat) > FLOAT32_MAX).any(axis=1)
    for i in np.flatnonzero(big):
        out.add("float32_overflow", f"{where}, frame {int(idx[i])}", f"{name} exceeds the float32 range")
