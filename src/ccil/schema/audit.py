"""Reference audit schema `ccil.reference-audit` 1.0: reference dynamics annotating a decided selection (#167).

A JSON document that compares one selection with the label error observed
under reference dynamics, ||(T(z_G, a_G) - y) / state_scale||. It records the
observed error of the selected and of the accepted but unselected rows; where
the selection's error is a bound, how many selected rows exceed their bound;
with a tolerance, how many selected rows are within it; and on calibrated
rows, the observed coverage of their calibrated bound, by row and by episode.
That coverage is an observation of one draw of the calibration episodes, not
evidence for the calibration's guarantee, which is over the draw.

An audit is an annotation, written after the selection: it is never an input
to admission, generation, quality evidence, or selection, so a data_only
result stays data_only when it is audited (docs/contract.md).
"""
from __future__ import annotations

import math
import re

from .calibration import read_json, write_json
from .fields import SchemaError

SCHEMA = "ccil.reference-audit"
VERSION = (1, 0)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATEMENT = ("an annotation of a decided result: it is never an input to admission, generation, quality evidence, "
             "or selection")
KEYS = {"schema", "version", "selection", "error", "groups", "bounds", "tolerance", "calibrated", "statement", "inputs"}
GROUPS = ("selected", "accepted_unselected")
FIELDS = [
    ("selection", "dict", "The audited selection's policy, evidence_mode, and evidence_allowed, as recorded."),
    ("error", "dict", "definition and units of the observed error."),
    ("groups", "dict", f"{list(GROUPS)}: rows, and the observed error's median, q90, and max (null without rows)."),
    ("bounds", "dict | null", "Where the selection's error is a bound: selected rows, violations (observed error "
                              "above the recorded bound), and the largest observed-to-bound ratio; null otherwise."),
    ("tolerance", "dict | null", "With a tolerance: its value, and the selected rows whose observed error is within it."),
    ("calibrated", "dict | null", "With a calibration: its level; the calibrated rows (selected or not) and how many "
                                  "the bound covers; their episodes and how many have every calibrated row covered."),
    ("statement", "str", "The audit's role."),
    ("inputs", "dict[str, str | null]", "SHA-256 of the selection, candidates, certificates (null without), "
                                        "calibration (null without), and reference."),
]


def _count(x) -> bool:
    return type(x) is int and x >= 0


def _summary(g) -> bool:
    if not (isinstance(g, dict) and set(g) == {"rows", "median", "q90", "max"} and _count(g["rows"])):
        return False
    values = [g["median"], g["q90"], g["max"]]
    if g["rows"] == 0:
        return all(v is None for v in values)
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values)


def check(d) -> dict:
    """Raise SchemaError listing every structural problem of an audit document; return it."""
    problems = []

    def bad(rule, detail):
        problems.append({"rule": rule, "where": "audit", "detail": detail})

    if not isinstance(d, dict) or set(d) != KEYS:
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise SchemaError(SCHEMA, [{"rule": "format", "where": "audit",
                                    "detail": f"an audit is a JSON object with exactly {sorted(KEYS)}; got {got}"}],
                          {"format": 1})
    if d["schema"] != SCHEMA or d["version"] != f"{VERSION[0]}.{VERSION[1]}":
        bad("format", f"schema and version must be {SCHEMA} {VERSION[0]}.{VERSION[1]}")
    s = d["selection"]
    if not (isinstance(s, dict) and set(s) == {"policy", "evidence_mode", "evidence_allowed"}):
        bad("selection", "selection must be {policy, evidence_mode, evidence_allowed}")
    if not (isinstance(d["error"], dict) and set(d["error"]) == {"definition", "units"}):
        bad("error", "error must be {definition, units}")
    g = d["groups"]
    if not (isinstance(g, dict) and set(g) == set(GROUPS) and all(_summary(g[k]) for k in GROUPS)):
        bad("groups", f"groups must hold {list(GROUPS)}, each {{rows, median, q90, max}}")
    b = d["bounds"]
    if b is not None and not (isinstance(b, dict) and set(b) == {"rows", "violations", "max_ratio"} and _count(b["rows"])
                              and _count(b["violations"]) and b["violations"] <= b["rows"]):
        bad("bounds", "bounds must be null or {rows, violations <= rows, max_ratio}")
    t = d["tolerance"]
    if t is not None and not (isinstance(t, dict) and set(t) == {"value", "rows", "within"} and _count(t["rows"])
                              and _count(t["within"]) and t["within"] <= t["rows"]):
        bad("tolerance", "tolerance must be null or {value, rows, within <= rows}")
    c = d["calibrated"]
    if c is not None and not (isinstance(c, dict) and set(c) == {"level", "rows", "covered", "episodes", "episodes_covered"}
                              and all(_count(c[k]) for k in ("rows", "covered", "episodes", "episodes_covered"))
                              and c["covered"] <= c["rows"] and c["episodes_covered"] <= c["episodes"] <= c["rows"]):
        bad("calibrated", "calibrated must be null or {level, rows, covered <= rows, episodes, episodes_covered <= episodes}")
    if d["statement"] != STATEMENT:
        bad("statement", f"statement must be {STATEMENT!r}")
    inputs = d["inputs"]
    if not (isinstance(inputs, dict) and set(inputs) == {"selection", "candidates", "certificates", "calibration", "reference"}
            and all(v is None or (isinstance(v, str) and SHA256.match(v)) for v in inputs.values())
            and all(inputs[k] for k in ("selection", "candidates", "reference"))):
        bad("inputs", "inputs must give the SHA-256 of selection, candidates, and reference, and of certificates and "
                      "calibration or null")
    if problems:
        raise SchemaError(SCHEMA, problems, {p["rule"]: sum(q["rule"] == p["rule"] for q in problems) for p in problems})
    return d


def write_audit(d: dict, path) -> None:
    write_json(d, path, check)


def read_audit(path) -> dict:
    """Read and validate a ccil.reference-audit file."""
    return read_json(path, SCHEMA, check)
