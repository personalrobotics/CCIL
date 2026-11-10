"""Result schemas `ccil.generate-result` 1.0 and `ccil.curate-result` 1.0: what a Generate or a Curate wrote (#170).

A JSON document written whole at the end of `ccil generate` or `ccil curate`,
in run.output_dir. Each is immutable and named by the SHA-256 of its bytes
(generate-<id>.json, curate-<id>.json); generate-result.json is a copy of the
latest Generate there. It records the resolved configuration with each value's
source, every stage's run directory and whether it was reused, and each
artifact's path and SHA-256. Paths are relative to the document's directory and
must resolve, through any symlink, inside it: a run directory is a direct child
of it. `ccil curate` and `ccil inspect` refuse a result, before writing
anything, unless every recorded artifact still has its recorded hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .fields import SchemaError

GENERATE, CURATE = "ccil.generate-result", "ccil.curate-result"
VERSION = (1, 0)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GENERATE_ARTIFACTS = ("dataset", "split", "transition_model", "diagnosis", "candidates", "certificates")
CURATE_ARTIFACTS = ("calibration", "selection", "audit")
REQUIRED = {GENERATE: ("dataset", "transition_model", "diagnosis", "candidates"), CURATE: ("selection",)}
COUNTS = {GENERATE: {"candidates", "accepted", "rejected_by_reason", "by_classification", "admitted"},
          CURATE: {"rows", "selected", "policy", "threshold", "not_selected_by_reason"}}
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


def _relative(p) -> bool:
    return isinstance(p, str) and bool(p) and not os.path.isabs(p) and "\\" not in p and "\x00" not in p


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
    config = d.get("config")
    if not (isinstance(config, dict) and config.get("schema") == "ccil.config" and isinstance(config.get("values"), dict)
            and all(isinstance(e, dict) and "value" in e and isinstance(e.get("source"), str)
                    for e in config["values"].values())):
        bad("config", "config must be a ccil.config record: values, each with its value and source")
    stages = d.get("stages")
    if not isinstance(stages, list) or not all(
            isinstance(s, dict) and set(s) == {"stage", "run_dir", "reused"} and isinstance(s["stage"], str)
            and _relative(s["run_dir"]) and isinstance(s["reused"], bool) for s in stages):
        bad("stages", "stages must be a list of {stage, run_dir (relative), reused}")
    arts = d.get("artifacts")
    allowed = GENERATE_ARTIFACTS if schema == GENERATE else CURATE_ARTIFACTS
    if not isinstance(arts, dict):
        bad("artifacts", "artifacts must be an object")
    else:
        for name, a in arts.items():
            if name not in allowed:
                bad("artifacts", f"unknown artifact {name!r}; allowed: {list(allowed)}")
            elif not (isinstance(a, dict) and set(a) == {"path", "sha256"} and _relative(a["path"])
                      and isinstance(a["sha256"], str) and SHA256.match(a["sha256"])):
                bad("artifacts", f"{name} must be {{path (relative), sha256}}")
        missing = [n for n in REQUIRED[schema] if n not in arts]
        if missing:
            bad("artifacts", f"missing required artifacts {missing}")
    if schema == CURATE:
        g = d.get("generated")
        if not (isinstance(g, dict) and set(g) == {"path", "sha256"} and _relative(g["path"])
                and isinstance(g["sha256"], str) and SHA256.match(g["sha256"])):
            bad("generated", "generated must be {path (relative), sha256}")
    counts = d.get("counts")
    if not (isinstance(counts, dict) and set(counts) == COUNTS[schema]):
        bad("counts", f"counts must be an object with exactly {sorted(COUNTS[schema])}")
    elif schema == GENERATE and not (all(isinstance(counts[k], int) for k in ("candidates", "accepted"))
                                     and isinstance(counts["rejected_by_reason"], dict)):
        bad("counts", "candidates and accepted must be integers, rejected_by_reason an object")
    elif schema == CURATE and not (all(isinstance(counts[k], int) for k in ("rows", "selected"))
                                   and isinstance(counts["policy"], dict)
                                   and isinstance(counts["not_selected_by_reason"], dict)):
        bad("counts", "rows and selected must be integers, policy and not_selected_by_reason objects")
    if not isinstance(d.get("next_step"), str):
        bad("next_step", "next_step must be a string")
    if problems:
        counts = {}
        for p in problems:
            counts[p["rule"]] = counts.get(p["rule"], 0) + 1
        raise SchemaError(schema, problems, counts)
    return d


def _text(d: dict) -> str:
    return json.dumps(d, indent=1, sort_keys=True) + "\n"


def write_result(d: dict, path) -> None:
    """Validate and write a result atomically: to a temporary file beside it, then renamed into place."""
    check(d)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(_text(d))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def immutable_name(text: bytes, prefix: str) -> str:
    """The immutable file name of a result's bytes: prefix-<first 12 hex digits of their SHA-256>.json."""
    return f"{prefix}-{hashlib.sha256(text).hexdigest()[:12]}.json"


def write_once(d: dict, directory, prefix: str) -> Path:
    """Write a result, validated, to its content-addressed name in directory; never replace a file there.

    An existing file of that name with the same bytes is kept; one with other bytes was changed after it was
    written, and is refused (FileExistsError)."""
    check(d)
    text = _text(d).encode()
    path = Path(directory) / immutable_name(text, prefix)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(text)
        try:
            os.link(tmp, path)  # fails, and replaces nothing, when path exists
        except FileExistsError:
            if path.read_bytes() != text:
                raise FileExistsError(f"{path} exists with other content: it was changed after it was written") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _inside(base: Path, rel: str, where: str, child: bool = False) -> str | None:
    """Why rel, resolved through symlinks from base, is not inside base (a direct child, with child), or None."""
    p = (base / rel).resolve()
    if p == base or not p.is_relative_to(base) or (child and p.parent != base):
        return f"{where} {rel!r} resolves to {p}, outside {base}" + (" (a run directory is a direct child)" if child else "")
    return None


def read_result(path) -> dict:
    """Read and validate a ccil.generate-result or ccil.curate-result file, and that every path it records stays in
    its directory (resolved, through symlinks)."""
    try:
        d = json.loads(open(path).read())
    except (OSError, ValueError) as e:
        raise SchemaError("ccil result", [{"rule": "format", "where": str(path), "detail":
                                           f"not a readable JSON file: {e}"}], {"format": 1}) from None
    check(d)
    base = Path(path).resolve().parent
    problems = [_inside(base, s["run_dir"], f"stage {s['stage']} run_dir", child=True) for s in d["stages"]]
    problems += [_inside(base, a["path"], f"artifact {n}") for n, a in sorted(d["artifacts"].items())]
    if d["schema"] == CURATE:
        problems.append(_inside(base, d["generated"]["path"], "generated"))
    problems = [{"rule": "containment", "where": str(path), "detail": p} for p in problems if p]
    if problems:
        raise SchemaError(d["schema"], problems, {"containment": len(problems)})
    return d
