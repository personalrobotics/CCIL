"""Classify a run directory as complete, corrupt, incompatible, or incomplete."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .layout import CANDIDATES, MANIFEST, MANIFEST_SCHEMA, MANIFEST_VERSION, READABLE, STAGING_MARKER
from .provenance import component_schema, sha256

ORDER = ("incompatible", "corrupt", "incomplete")


def _major(v) -> int | None:
    try:
        return int(str(v).split(".")[0])
    except ValueError:
        return None


def validate_run(path, staging: bool = False) -> tuple[str, list[dict]]:
    """Return (status, problems). staging=True validates a writer's own staging directory."""
    root = Path(path)
    problems = []

    def bad(kind, detail):
        problems.append({"status": kind, "detail": detail})

    if not root.is_dir():
        return "incomplete", [{"status": "incomplete", "detail": f"{root} is not a directory"}]
    if not staging and STAGING_MARKER in root.name:
        bad("incomplete", "this is the staging directory of an interrupted run")
    mf = root / MANIFEST
    if not mf.exists():
        bad("incomplete", "manifest.json is missing")
        return "incomplete", problems
    try:
        manifest = json.loads(mf.read_text())
        files = manifest["files"]
    except (ValueError, KeyError, TypeError) as e:
        bad("corrupt", f"manifest.json is unreadable: {e}")
        return "corrupt", problems
    if manifest.get("schema") != MANIFEST_SCHEMA or _major(manifest.get("version")) != MANIFEST_VERSION[0]:
        bad("incompatible", f"manifest is {manifest.get('schema')!r} {manifest.get('version')!r}; this ccil reads "
                            f"{MANIFEST_SCHEMA} {MANIFEST_VERSION[0]}.x")
        return "incompatible", problems
    if manifest.get("status") != "complete":
        bad("incomplete", f"manifest status is {manifest.get('status')!r}")

    for rel, info in sorted(files.items()):
        p = root / rel
        if not p.is_file():
            bad("incomplete", f"{rel} is listed but missing")
            continue
        if p.stat().st_size != info.get("size") or sha256(p) != info.get("sha256"):
            bad("corrupt", f"{rel} does not match its recorded size and SHA-256")
            continue
        schema, version = component_schema(p)
        if schema != info.get("schema") or (schema is not None and version != info.get("version")):
            bad("corrupt", f"{rel} declares {schema!r} {version!r} but the manifest records "
                           f"{info.get('schema')!r} {info.get('version')!r}")
        elif schema in READABLE and (_major(version) or 0) > READABLE[schema]:
            bad("incompatible", f"{rel} is {schema} {version}; this ccil reads up to {READABLE[schema]}.x")

    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {MANIFEST}
    for rel in sorted(on_disk - set(files)):
        bad("corrupt", f"{rel} is present but not listed in the manifest")

    if CANDIDATES in files and (root / CANDIDATES).is_file():
        try:
            with np.load(root / CANDIDATES, allow_pickle=False) as z:
                run_id = json.loads(str(z["meta"])).get("run_id")
            if run_id != manifest.get("run_id"):
                bad("corrupt", f"{CANDIDATES} has run_id {run_id!r}, manifest has {manifest.get('run_id')!r}")
        except (OSError, ValueError, KeyError) as e:
            bad("corrupt", f"{CANDIDATES} is unreadable: {e}")

    kinds = {p["status"] for p in problems}
    return next((k for k in ORDER if k in kinds), "complete"), problems
