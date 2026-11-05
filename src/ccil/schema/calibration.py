"""Calibration schema `ccil.calibration` 1.0: one episode-level conformal bound on observed label error (#167).

A JSON document that bounds the label error, observed under reference
dynamics, of one certificate file's eligible rows: accepted rows the
certificate classifies empirical. The sampling unit is the episode, since rows
of one trajectory, and the labels of one transition, are not independent.
Each calibration episode with an eligible row is scored by the largest
observed error over its eligible rows; the bound is the k-th smallest of
those n scores, k = ceil((n + 1) * level), the split-conformal rule, or null
when k > n.

What it guarantees: when the calibration episodes are a random subset of the
dataset's episodes, drawn independently of the labels, and the labels and the
certificate do not depend on which episodes were drawn (the label model is
fitted on every episode: transition_model.fit_episodes all), the expected
fraction of the other episodes whose every eligible row has observed error at
most the bound is at least level. The probability is over the draw of the
calibration episodes; it is not conditional on one draw, a row, or a score.
Because it covers every eligible row of an episode at once, it holds for any
subset of those rows, including the ones curation selects. It is not a
worst-case bound, and it does not transfer to other datasets, tasks, model
families, or certificate files.

A row of that certificate file is calibrated exactly when it is eligible, its
episode is not a calibration episode, and the bound is not null. Every other
row keeps its class.

The observed errors come from reference dynamics, so a calibration is always
reference_assisted evidence.
"""
from __future__ import annotations

import json
import math
import os
import re

from .fields import SchemaError

SCHEMA = "ccil.calibration"
VERSION = (1, 0)
METHOD = {"name": "split_conformal_by_episode", "version": "1"}
UNIT = {"name": "episode", "score": "the largest observed error over the episode's eligible rows"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATEMENT = ("when the calibration episodes are drawn at random, independently of the labels, and the labels and "
             "certificate do not depend on the draw (transition_model.fit_episodes all), the expected fraction of the "
             "dataset's other episodes whose every eligible row has true label error at most the bound is at least "
             "the level; the probability is over the draw, not conditional on one draw, a row, or a score, and it "
             "holds for any subset of an episode's eligible rows; it is not a worst-case bound, and it does not "
             "transfer to other datasets, tasks, model families, or certificate files")
KEYS = {"schema", "version", "evidence_mode", "method", "level", "unit", "certificate", "error", "population", "bound",
        "statement", "inputs"}
INPUTS = ("candidates", "certificates", "reference")
FIELDS = [
    ("evidence_mode", "str", "Always reference_assisted: the observed errors come from reference dynamics."),
    ("method", "dict", f"name and version of the calibration rule ({METHOD['name']} {METHOD['version']})."),
    ("level", "float", "The declared coverage level, in (0, 1)."),
    ("unit", "dict", "The exchangeable unit (episode) and its score (the largest observed error over the "
                     "episode's eligible rows)."),
    ("certificate", "dict", "The calibrated certificate's profile, version, and neighborhood; its eligible rows are "
                            "its accepted, empirical rows."),
    ("error", "dict", "The observed error: definition (||(T(z_G, a_G) - y) / state_scale|| under the reference "
                      "dynamics) and units, the certificate's."),
    ("population", "dict", "rule, the calibration episodes, and their eligible row count."),
    ("bound", "dict", "n (calibration episodes with an eligible row), k = ceil((n + 1) * level), their scores in "
                      "increasing order, and value: the k-th score, or null when k > n."),
    ("statement", "str", "What the bound guarantees and does not establish."),
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
    if d["method"] != METHOD:
        bad("method", f"method must be {METHOD}")
    if d["unit"] != UNIT:
        bad("unit", f"unit must be {UNIT}")
    level = d["level"]
    if not (_number(level) and 0 < level < 1):
        bad("level", "level must be in (0, 1)")
        level = None
    c = d["certificate"]
    if not (isinstance(c, dict) and set(c) == {"profile", "version", "neighborhood"} and isinstance(c["profile"], str)
            and c["profile"] and isinstance(c["version"], str) and c["version"] and isinstance(c["neighborhood"], dict)):
        bad("certificate", "certificate must be {profile, version: non-empty strings, neighborhood}")
    e = d["error"]
    if not (isinstance(e, dict) and set(e) == {"definition", "units"} and all(isinstance(x, str) and x for x in e.values())):
        bad("error", "error must be {definition, units}, non-empty strings")
    p = d["population"]
    if not (isinstance(p, dict) and set(p) == {"rule", "episodes", "rows"} and isinstance(p["rule"], str)
            and isinstance(p["episodes"], list) and p["episodes"] and all(isinstance(x, str) for x in p["episodes"])
            and p["episodes"] == sorted(set(p["episodes"])) and type(p["rows"]) is int and p["rows"] >= 1):
        bad("population", "population must be {rule, episodes: sorted distinct ids, rows >= 1}")
        p = None
    b = d["bound"]
    if not (isinstance(b, dict) and set(b) == {"n", "k", "scores", "value"} and type(b["n"]) is int and b["n"] >= 1
            and type(b["k"]) is int and isinstance(b["scores"], list) and len(b["scores"]) == b["n"]
            and all(_number(x) and x >= 0 for x in b["scores"]) and b["scores"] == sorted(b["scores"])):
        bad("bound", "bound must be {n >= 1, k, scores: n increasing errors >= 0, value}")
    else:
        if p is not None and not b["n"] <= min(len(p["episodes"]), p["rows"]):
            bad("bound", "n counts calibration episodes with an eligible row: at most the episodes and the rows")
        if level is not None and b["k"] != math.ceil((b["n"] + 1) * level):
            bad("bound", "k must be ceil((n + 1) * level)")
        if b["k"] > b["n"]:
            if b["value"] is not None:
                bad("bound", "value is null when k > n")
        elif b["value"] != b["scores"][b["k"] - 1]:
            bad("bound", "value must be the k-th smallest score")
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
