"""Calibration schema `ccil.calibration` 1.0: an empirical score mapped to observed label error (#167).

A JSON document that maps one certificate file's empirical scores to the label
error observed under reference dynamics, on a declared population of its rows
(the held-out episodes' accepted, supported rows). The scores are split into
bins of equal calibration count; each bin records its score range, its count,
the observed-error quantiles, and a split-conformal bound: the k-th smallest
observed error, k = ceil((count + 1) * level). For a row exchangeable with a
bin's calibration rows, the bound covers the row's true error with
probability at least level. It is a coverage statement, not a worst-case
bound, and the bin edges come from the same scores, so per-bin coverage is
approximate. A bin with too few rows (k > count) gives no bound.

A row of that certificate file is calibrated exactly when it is outside the
population, empirical, and its score falls in a bin that has a bound: the
regime is the bins' score range under the certificate's profile, neighborhood,
and inputs. Every other row keeps its class: a calibration never makes a row
stronger outside its regime, and never applies to another certificate file.

The observed errors come from reference dynamics, so a calibration is always
reference_assisted evidence. It does not establish that the calibration
transfers to other tasks or model families.
"""
from __future__ import annotations

import json
import math
import os
import re

from .fields import SchemaError

SCHEMA = "ccil.calibration"
VERSION = (1, 0)
METHOD = {"name": "split_conformal_by_score_bin", "version": "1"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATEMENT = ("a calibrated bound covers the true label error with probability at least the level, for rows "
             "exchangeable with the calibration rows of its bin; it is not a worst-case bound, and it does not "
             "transfer to other tasks or model families")
KEYS = {"schema", "version", "evidence_mode", "method", "level", "score", "error", "population", "regime", "bins",
        "statement", "inputs"}
INPUTS = ("candidates", "certificates", "reference")
FIELDS = [
    ("evidence_mode", "str", "Always reference_assisted: the observed errors come from reference dynamics."),
    ("method", "dict", f"name and version of the calibration rule ({METHOD['name']} {METHOD['version']}), and the "
                       "requested bin count; tied score quantiles merge bins."),
    ("level", "float", "The declared coverage level, in (0, 1)."),
    ("score", "dict", "The calibrated score: the certificate's profile, version, definition, and units."),
    ("error", "dict", "The observed error: definition (||(T(z_G, a_G) - y) / state_scale|| under the reference "
                      "dynamics) and units, the certificate's."),
    ("population", "dict", "rule, the episodes whose rows calibrate, and their row count."),
    ("regime", "dict", "score_min and score_max over the bins, and the certificate's neighborhood: where the "
                       "calibration applies."),
    ("bins", "list", "Per bin: lower, upper (score), count, k, bound (null when k > count), and observed {median, "
                     "at_level, max}; contiguous and increasing."),
    ("statement", "str", "What a calibrated bound means and does not establish."),
    ("inputs", "dict[str, str]", f"SHA-256 of {list(INPUTS)}; a calibration applies to that certificate file only."),
]


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check(d) -> dict:
    """Raise SchemaError listing every structural problem of a calibration document; return it."""
    problems = []

    def bad(rule, detail):
        problems.append({"rule": rule, "where": "calibration", "detail": detail})

    if not isinstance(d, dict) or set(d) != KEYS:
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise SchemaError(SCHEMA, [{"rule": "format", "where": "calibration",
                                    "detail": f"a calibration is a JSON object with exactly {sorted(KEYS)}; got {got}"}],
                          {"format": 1})
    if d["schema"] != SCHEMA or d["version"] != f"{VERSION[0]}.{VERSION[1]}":
        bad("format", f"schema and version must be {SCHEMA} {VERSION[0]}.{VERSION[1]}")
    if d["evidence_mode"] != "reference_assisted":
        bad("evidence_mode", "a calibration is reference_assisted")
    m = d["method"]
    if not (isinstance(m, dict) and set(m) == {*METHOD, "bins"} and all(m[k] == v for k, v in METHOD.items())
            and type(m["bins"]) is int and m["bins"] >= 1):
        bad("method", f"method must be {METHOD} with bins >= 1")
    elif isinstance(d["bins"], list) and len(d["bins"]) > m["bins"]:
        bad("bins", f"{len(d['bins'])} bins exceed the requested {m['bins']}")
    level = d["level"]
    if not (_number(level) and 0 < level < 1):
        bad("level", "level must be in (0, 1)")
        level = None
    for key, need in (("score", {"profile", "version", "definition", "units"}), ("error", {"definition", "units"})):
        v = d[key]
        if not (isinstance(v, dict) and set(v) == need and all(isinstance(x, str) and x for x in v.values())):
            bad(key, f"{key} must be {{{', '.join(sorted(need))}}}, non-empty strings")
    p = d["population"]
    if not (isinstance(p, dict) and set(p) == {"rule", "episodes", "rows"} and isinstance(p["rule"], str)
            and isinstance(p["episodes"], list) and p["episodes"] and all(isinstance(e, str) for e in p["episodes"])
            and type(p["rows"]) is int and p["rows"] >= 1):
        bad("population", "population must be {rule, episodes: non-empty list of ids, rows >= 1}")
        p = None
    bins = d["bins"]
    if not (isinstance(bins, list) and bins):
        bad("bins", "bins must be a non-empty list")
    else:
        prev, total = None, 0
        for i, b in enumerate(bins):
            where = f"bin {i}"
            if not (isinstance(b, dict) and set(b) == {"lower", "upper", "count", "k", "bound", "observed"}
                    and _number(b["lower"]) and _number(b["upper"]) and 0 <= b["lower"] <= b["upper"]
                    and type(b["count"]) is int and b["count"] >= 1 and type(b["k"]) is int and b["k"] >= 1
                    and isinstance(b["observed"], dict) and set(b["observed"]) == {"median", "at_level", "max"}
                    and all(_number(x) and x >= 0 for x in b["observed"].values())):
                bad("bins", f"{where} must be {{lower <= upper, count >= 1, k >= 1, bound, observed {{median, "
                            "at_level, max}}}}")
                continue
            if prev is not None and b["lower"] != prev:
                bad("bins", f"{where} must start where bin {i - 1} ends ({prev})")
            prev, total = b["upper"], total + b["count"]
            if level is not None and b["k"] != math.ceil((b["count"] + 1) * level):
                bad("bins", f"{where}: k must be ceil((count + 1) * level)")
            if (b["bound"] is None) != (b["k"] > b["count"]):
                bad("bins", f"{where}: bound is null exactly when k > count")
            elif b["bound"] is not None and not (_number(b["bound"]) and 0 <= b["bound"] <= b["observed"]["max"]):
                bad("bins", f"{where}: bound must be one of the observed errors")
        r = d["regime"]
        if not (isinstance(r, dict) and set(r) == {"score_min", "score_max", "neighborhood"}
                and r["score_min"] == bins[0].get("lower") and r["score_max"] == prev):
            bad("regime", "regime must be {score_min, score_max, neighborhood}, the bins' score range")
        if p is not None and total != p["rows"]:
            bad("bins", f"the bin counts sum to {total}, not the population's {p['rows']} rows")
    if d["statement"] != STATEMENT:
        bad("statement", f"statement must be {STATEMENT!r}")
    inputs = d["inputs"]
    if not (isinstance(inputs, dict) and set(inputs) == set(INPUTS)
            and all(isinstance(v, str) and SHA256.match(v) for v in inputs.values())):
        bad("inputs", f"inputs must map exactly {list(INPUTS)} to SHA-256")
    if problems:
        raise SchemaError(SCHEMA, problems, {p["rule"]: sum(q["rule"] == p["rule"] for q in problems) for p in problems})
    return d


def write_json(d: dict, path, check_fn) -> None:
    """Validate with check_fn and write a JSON document atomically."""
    check_fn(d)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(d, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path, schema, check_fn) -> dict:
    try:
        d = json.loads(open(path).read())
    except (OSError, ValueError) as e:
        raise SchemaError(schema, [{"rule": "format", "where": str(path), "detail": f"not a readable JSON file: {e}"}],
                          {"format": 1}) from None
    return check_fn(d)


def write_calibration(d: dict, path) -> None:
    write_json(d, path, check)


def read_calibration(path) -> dict:
    """Read and validate a ccil.calibration file."""
    return read_json(path, SCHEMA, check)
