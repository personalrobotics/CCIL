"""Diagnosis schema `ccil.diagnosis` 1.0: which recovery length, if any, the data support (#167).

A JSON document written before generation from the demonstrations and the
dynamics evidence alone. For each recovery length M on the ladder it records
three gates, every quantity in the demonstration metric (core/quality.py) with
its raw value, quantile, and threshold:
- yield: the fraction of a fixed probe's labels the validators accept;
- reach: eta_reach = r_reach / tau_demo, r_reach a quantile of the accepted
  probe labels' d_source; eta_reach < 1 means the labels stay within the
  demonstrations' own spacing;
- trust: a quantile of the held-out M-step model error over the median
  accepted d_source, from a model that never saw the held-out episodes.
The admitted M is the smallest whose gates all pass (recovery_steps auto), or
the declared one (an integer, which skips the ladder but not this report).
When none is admitted, the reason names the first failing gate and an
actionable next step, and generation writes a valid empty synthetic view.

The diagnosis is data-only evidence: it supports curation and does not
establish that the labels improve a downstream policy.
"""
from __future__ import annotations

import json
import math
import os
import re

from .fields import SchemaError

SCHEMA = "ccil.diagnosis"
VERSION = (1, 0)
GATES = ("yield", "reach", "trust")
ADMITTED_BY = ("gates", "declared")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATEMENT = "data-only evidence supports curation; it does not establish downstream policy improvement"
KEYS = {"schema", "version", "evidence_mode", "recovery_steps", "metric", "timing", "tau_demo", "probe", "gates",
        "admitted", "admitted_by", "reason", "next_step", "bands", "nested_views", "evidence_class", "statement", "inputs"}
FIELDS = [
    ("evidence_mode", "str", "Always data_only: no reference dynamics enter the diagnosis."),
    ("recovery_steps", "dict", "requested (auto or an integer), ladder (the lengths probed), and cap."),
    ("metric", "dict", "The demonstration metric: definition, units, per-coordinate scale, and the train and held-out "
                       "episode and state counts."),
    ("timing", "dict", "dt_ctrl (seconds, or null when the data carry no consistent timing), its source, and a note."),
    ("tau_demo", "dict", "value (null without held-out episodes), quantile, and definition."),
    ("probe", "dict", "The fixed probe: transitions, the rule choosing them, and the candidates it produced."),
    ("gates", "list", "Per recovery length: M, horizon_s (M * dt_ctrl, or null), and each of yield, reach, trust "
                      "with value, raw quantities, threshold, and pass; pass is all three."),
    ("admitted", "int | null", "The admitted recovery length, or null."),
    ("admitted_by", "str | null", f"{list(ADMITTED_BY)}, or null when nothing is admitted."),
    ("reason", "str | null", "Why nothing is admitted (the first failing gate); null when M is admitted."),
    ("next_step", "str | null", "An actionable next step when nothing is admitted."),
    ("bands", "dict | null", "Accepted probe labels counted by d_source / tau_demo in the band edges; null without tau_demo."),
    ("nested_views", "list | null", "Recommended labels_per_transition views for a downstream pilot (noisy generators)."),
    ("evidence_class", "dict", "The strongest evidence class the configured certificate can give, from data alone."),
    ("statement", "str", "What the diagnosis does not establish."),
    ("inputs", "dict[str, str]", "SHA-256 of the dataset, split, transition model, and evaluation model."),
]


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check(d) -> dict:
    """Raise SchemaError listing every structural problem of a diagnosis document; return it."""
    problems = []

    def bad(where, detail):
        problems.append({"rule": where, "where": "diagnosis", "detail": detail})

    if not isinstance(d, dict):
        raise SchemaError(SCHEMA, [{"rule": "type", "where": "diagnosis", "detail": "a diagnosis is a JSON object"}],
                          {"type": 1})
    if d.get("schema") != SCHEMA or d.get("version") != f"{VERSION[0]}.{VERSION[1]}":
        bad("format", f"schema and version must be {SCHEMA} {VERSION[0]}.{VERSION[1]}")
    if set(d) != KEYS:
        bad("format", f"fields must be exactly {sorted(KEYS)}; got extra {sorted(set(d) - KEYS)}, "
                      f"missing {sorted(KEYS - set(d))}")
        raise SchemaError(SCHEMA, problems, {p["rule"]: 1 for p in problems})
    if d["evidence_mode"] != "data_only":
        bad("evidence_mode", "a diagnosis is data_only")
    rs = d["recovery_steps"]
    ladder = rs.get("ladder") if isinstance(rs, dict) else None
    if not (isinstance(rs, dict) and set(rs) == {"requested", "ladder", "cap"}
            and (rs["requested"] == "auto" or (type(rs["requested"]) is int and rs["requested"] >= 1))
            and isinstance(ladder, list) and ladder and all(type(m) is int and m >= 1 for m in ladder)
            and ladder == sorted(set(ladder)) and type(rs["cap"]) is int and max(ladder) <= rs["cap"]):
        bad("recovery_steps", "recovery_steps must be {requested: auto or int >= 1, ladder: increasing ints, cap}")
        ladder = None
    tau = d["tau_demo"]
    if not (isinstance(tau, dict) and set(tau) == {"value", "quantile", "definition"}
            and (tau["value"] is None or (_number(tau["value"]) and tau["value"] >= 0))
            and _number(tau["quantile"]) and 0 < tau["quantile"] <= 1 and isinstance(tau["definition"], str)):
        bad("tau_demo", "tau_demo must be {value: null or finite >= 0, quantile in (0, 1], definition}")
    timing = d["timing"]
    if not (isinstance(timing, dict) and set(timing) == {"dt_ctrl", "source", "note"}
            and (timing["dt_ctrl"] is None or (_number(timing["dt_ctrl"]) and timing["dt_ctrl"] > 0))):
        bad("timing", "timing must be {dt_ctrl: null or finite > 0, source, note}")
    gates = d["gates"]
    passed = []
    if not isinstance(gates, list) or (ladder is not None and [g.get("M") if isinstance(g, dict) else None
                                                               for g in gates] != ladder):
        bad("gates", "gates must list one entry per ladder length, in order")
    else:
        for g in gates:
            ok = isinstance(g, dict) and set(g) == {"M", "horizon_s", *GATES, "pass"} and all(
                isinstance(g[k], dict) and isinstance(g[k].get("pass"), bool) for k in GATES)
            if not ok:
                bad("gates", f"gate entry {g!r:.80} must hold M, horizon_s, {list(GATES)} (each with pass), and pass")
                continue
            if g["pass"] != all(g[k]["pass"] for k in GATES):
                bad("gates", f"M = {g['M']}: pass must be the conjunction of its gates")
            if g["pass"]:
                passed.append(g["M"])
    admitted, by = d["admitted"], d["admitted_by"]
    if admitted is None:
        if by is not None or not (isinstance(d["reason"], str) and d["reason"] and isinstance(d["next_step"], str)
                                  and d["next_step"]):
            bad("admitted", "with nothing admitted, admitted_by is null and reason and next_step are given")
        elif isinstance(rs, dict) and rs.get("requested") == "auto" and passed:
            bad("admitted", f"auto admits the smallest passing length, but {passed} passed")
    else:
        if by not in ADMITTED_BY or d["reason"] is not None or d["next_step"] is not None:
            bad("admitted", f"an admitted length has admitted_by in {list(ADMITTED_BY)} and no reason or next_step")
        elif by == "gates" and (not passed or admitted != min(passed) or rs.get("requested") != "auto"):
            bad("admitted", "admitted by the gates is the smallest passing length, under auto")
        elif by == "declared" and admitted != rs.get("requested"):
            bad("admitted", "a declared length is the requested one")
    if d["statement"] != STATEMENT:
        bad("statement", f"statement must be {STATEMENT!r}")
    inputs = d["inputs"]
    if not (isinstance(inputs, dict) and all(isinstance(v, str) and SHA256.match(v) for v in inputs.values())
            and {"dataset", "transition_model"} <= set(inputs)):
        bad("inputs", "inputs must map names (including dataset and transition_model) to SHA-256")
    for key in ("metric", "probe", "evidence_class"):
        if not isinstance(d[key], dict):
            bad(key, f"{key} must be an object")
    if problems:
        raise SchemaError(SCHEMA, problems, {p["rule"]: sum(q["rule"] == p["rule"] for q in problems) for p in problems})
    return d


def write_diagnosis(d: dict, path) -> None:
    """Validate and write a diagnosis atomically."""
    check(d)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(d, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_diagnosis(path) -> dict:
    """Read and validate a ccil.diagnosis file."""
    try:
        d = json.loads(open(path).read())
    except (OSError, ValueError) as e:
        raise SchemaError(SCHEMA, [{"rule": "format", "where": str(path), "detail": f"not a readable JSON file: {e}"}],
                          {"format": 1}) from None
    return check(d)
