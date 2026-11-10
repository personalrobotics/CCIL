"""Result schemas `ccil.generate-result` 1.0 and `ccil.curate-result` 1.0: what a Generate or a Curate wrote (#170).

A JSON document written whole at the end of `ccil generate` (to
run.output_dir/generate-result.json) or `ccil curate` (beside it, one per
selection). It records the resolved configuration with each value's source,
every stage's run directory and whether it was reused, and each artifact's
path and SHA-256. Paths are relative to the document's directory, so a moved
output directory stays readable. `ccil curate` reads a generate result and
refuses it, before writing anything, unless every recorded artifact still has
its recorded hash.
"""
from __future__ import annotations

import json
import os
import re

from .fields import SchemaError

GENERATE, CURATE = "ccil.generate-result", "ccil.curate-result"
VERSION = (1, 0)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GENERATE_ARTIFACTS = ("dataset", "split", "transition_model", "diagnosis", "candidates", "certificates")
CURATE_ARTIFACTS = ("calibration", "selection", "audit")
REQUIRED = {GENERATE: ("dataset", "transition_model", "diagnosis", "candidates"), CURATE: ("selection",)}
KEYS = {GENERATE: {"schema", "version", "config", "stages", "artifacts", "counts", "next_step"},
        CURATE: {"schema", "version", "generated", "config", "stages", "artifacts", "counts", "next_step"}}
FIELDS = [
    ("generated", "dict", "Curate only: path and sha256 of the generate result it curated."),
    ("config", "dict", "The resolved configuration (ccil.config): each value with its source and units."),
    ("stages", "list", "Each stage: stage name, run_dir (relative), and reused."),
    ("artifacts", "dict", f"By name, each with path (relative) and sha256. Generate: {list(GENERATE_ARTIFACTS)} "
                          f"(split without held-out episodes and certificates without a certificate.profile are "
                          f"absent); Curate: {list(CURATE_ARTIFACTS)} (calibration and audit only when configured)."),
    ("counts", "dict", "Generate: candidates, accepted, rejected by reason, certificate classifications (or null), and "
                       "the admitted recovery length (or null). Curate: rows, selected, the policy, its threshold, "
                       "and the rows not selected by reason."),
    ("next_step", "str", "The command or action to take next."),
]


def check(d) -> dict:
    """Raise SchemaError listing every structural problem of a generate or curate result; return it."""
    if not isinstance(d, dict) or d.get("schema") not in KEYS:
        raise SchemaError("ccil result", [{"rule": "format", "where": "result", "detail":
                                           f"a result is a JSON object with schema {GENERATE} or {CURATE}"}],
                          {"format": 1})
    schema, problems = d["schema"], []

    def bad(rule, detail):
        problems.append({"rule": rule, "where": schema, "detail": detail})

    if d.get("version") != f"{VERSION[0]}.{VERSION[1]}":
        bad("version", f"version must be {VERSION[0]}.{VERSION[1]}, got {d.get('version')!r}")
    if set(d) != KEYS[schema]:
        bad("fields", f"fields must be exactly {sorted(KEYS[schema])}, got {sorted(d)}")
    if not isinstance(d.get("config"), dict) or d["config"].get("schema") != "ccil.config":
        bad("config", "config must be a ccil.config record")
    stages = d.get("stages")
    if not isinstance(stages, list) or not all(
            isinstance(s, dict) and set(s) == {"stage", "run_dir", "reused"} and isinstance(s["run_dir"], str)
            and isinstance(s["reused"], bool) for s in stages):
        bad("stages", "stages must be a list of {stage, run_dir, reused}")
    arts = d.get("artifacts")
    allowed = GENERATE_ARTIFACTS if schema == GENERATE else CURATE_ARTIFACTS
    if not isinstance(arts, dict):
        bad("artifacts", "artifacts must be an object")
    else:
        for name, a in arts.items():
            if name not in allowed:
                bad("artifacts", f"unknown artifact {name!r}; allowed: {list(allowed)}")
            elif not (isinstance(a, dict) and set(a) == {"path", "sha256"} and isinstance(a["path"], str)
                      and not os.path.isabs(a["path"]) and isinstance(a["sha256"], str) and SHA256.match(a["sha256"])):
                bad("artifacts", f"{name} must be {{path (relative), sha256}}")
        missing = [n for n in REQUIRED[schema] if n not in arts]
        if missing:
            bad("artifacts", f"missing required artifacts {missing}")
    if schema == CURATE:
        g = d.get("generated")
        if not (isinstance(g, dict) and set(g) == {"path", "sha256"} and isinstance(g.get("sha256"), str)
                and SHA256.match(g["sha256"])):
            bad("generated", "generated must be {path, sha256}")
    if not isinstance(d.get("counts"), dict):
        bad("counts", "counts must be an object")
    if not isinstance(d.get("next_step"), str):
        bad("next_step", "next_step must be a string")
    if problems:
        counts = {}
        for p in problems:
            counts[p["rule"]] = counts.get(p["rule"], 0) + 1
        raise SchemaError(schema, problems, counts)
    return d


def write_result(d: dict, path) -> None:
    """Validate and write a result atomically: to a temporary file beside it, then renamed into place."""
    check(d)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(d, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_result(path) -> dict:
    """Read and validate a ccil.generate-result or ccil.curate-result file."""
    try:
        d = json.loads(open(path).read())
    except (OSError, ValueError) as e:
        raise SchemaError("ccil result", [{"rule": "format", "where": str(path), "detail":
                                           f"not a readable JSON file: {e}"}], {"format": 1}) from None
    return check(d)
