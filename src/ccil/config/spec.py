"""Parameter specifications: the one canonical source for runtime checks and docs."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

STAGES = ("run", "compute", "data", "representation", "transition_model", "generation", "certificate", "validation",
          "policy", "evaluation", "environment")
TYPES = ("int", "float", "bool", "str", "path", "floats", "ints", "choice")
STATUS = ("implemented", "declared")  # declared: validated and documented, consumed by a later milestone


class ConfigError(ValueError):
    """Configuration failed validation; .errors lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid configuration:\n  " + "\n  ".join(errors))


@dataclass(frozen=True)
class Param:
    key: str
    type: str
    default: object
    description: str
    rationale: str
    stage: str
    units: str = "-"
    low: float | None = None          # inclusive bounds for numbers and number lists
    high: float | None = None
    low_open: bool = False            # True: the lower bound is exclusive
    high_open: bool = False
    choices: tuple = ()
    nullable: bool = False
    kind: str = "tunable"             # tunable or derived
    status: str = "implemented"
    milestone: str = ""               # owning issue or milestone when declared
    legacy: tuple = field(default=())  # legacy correct_il keys this replaces

    def __post_init__(self):
        assert self.stage in STAGES and self.type in TYPES and self.status in STATUS, self.key
        assert self.key.split(".")[0] == self.stage, self.key


_UNIT = re.compile(r"^\s*([-+0-9.eE]+)\s+(\S.*?)\s*$")


def parse_text(p: Param, text: str):
    """Parse a command-line string strictly by the parameter's type (no truthiness casts)."""
    t = text.strip()
    if p.nullable and t.lower() in ("null", "none"):
        return None
    if p.type == "bool":
        if t.lower() not in ("true", "false"):
            raise ValueError(f"expected true or false, got {text!r}")
        return t.lower() == "true"
    if p.type in ("int", "float"):
        return t  # numbers (optionally with units) are checked by coerce()
    if p.type in ("floats", "ints"):
        inner = t[1:-1] if t.startswith("[") and t.endswith("]") else t
        return [s.strip() for s in inner.split(",") if s.strip()] if inner.strip() else []
    return t


def _number(p: Param, v, where: str):
    if isinstance(v, bool):
        raise ValueError(f"{where}: expected a number, got a boolean")
    if isinstance(v, str):
        m = _UNIT.match(v)
        if m:
            num, unit = m.groups()
            if unit != p.units:
                raise ValueError(f"{where}: unit {unit!r} does not match the declared unit {p.units!r} (no conversion is applied)")
            v = num
        try:
            v = int(v) if p.type == "int" or (p.type == "ints") else float(v)
        except ValueError:
            raise ValueError(f"{where}: {v!r} is not a valid {'integer' if p.type in ('int', 'ints') else 'number'}")
    if p.type in ("int", "ints"):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"{where}: expected an integer, got {v!r}")
    elif not isinstance(v, (int, float)):
        raise ValueError(f"{where}: expected a number, got {type(v).__name__}")
    else:
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"{where}: must be finite, got {v!r}")
    lo_ok = p.low is None or (v > p.low if p.low_open else v >= p.low)
    hi_ok = p.high is None or (v < p.high if p.high_open else v <= p.high)
    if not (lo_ok and hi_ok):
        lo = "-inf" if p.low is None else p.low
        hi = "inf" if p.high is None else p.high
        raise ValueError(f"{where}: {v!r} is outside {'(' if p.low_open else '['}{lo}, {hi}{')' if p.high_open else ']'}")
    return v


def coerce(p: Param, v):
    """Validate a value for p without lossy conversion; ints are accepted for floats."""
    if v is None:
        if p.nullable:
            return None
        raise ValueError(f"{p.key}: may not be null")
    if p.type == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{p.key}: expected true or false, got {v!r}")
        return v
    if p.type in ("int", "float"):
        return _number(p, v, p.key)
    if p.type in ("floats", "ints"):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = [v]
        if not isinstance(v, list):
            raise ValueError(f"{p.key}: expected a list, got {type(v).__name__}")
        return [_number(p, x, f"{p.key}[{i}]") for i, x in enumerate(v)]
    if p.type == "choice":
        if v not in p.choices:
            raise ValueError(f"{p.key}: {v!r} is not one of {list(p.choices)}")
        return v
    if not isinstance(v, str):
        raise ValueError(f"{p.key}: expected a string, got {type(v).__name__}")
    return v
