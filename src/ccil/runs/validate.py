"""Classify a run directory as complete, corrupt, incompatible, or incomplete."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .layout import CANDIDATES, MANIFEST, MANIFEST_SCHEMA, MANIFEST_VERSION, READABLE, STAGING_MARKER
from .provenance import component_schema, register_artifact, sha256

ORDER = ("incompatible", "corrupt", "incomplete")


def _major(v) -> int | None:
    try:
        return int(str(v).split(".")[0])
    except ValueError:
        return None


def _component_problem(root: Path, rel) -> str | None:
    """Why rel cannot name a regular file inside root, or None.

    Run writers produce only regular files under real directories, so any
    symlink along the path is refused, wherever it points.
    """
    if not isinstance(rel, str) or not rel:
        return "is not a path string"
    parts = rel.split("/")
    if rel.startswith("/") or "\\" in rel or any(x in ("", ".", "..") for x in parts):
        return "is not a canonical relative path"
    p = root
    for part in parts:
        p = p / part
        if p.is_symlink():
            return "passes through a symlink"
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
    if mf.is_symlink():
        bad("corrupt", "manifest.json is a symlink")
        return "corrupt", problems
    if not mf.exists():
        bad("incomplete", "manifest.json is missing")
        return "incomplete", problems
    try:
        manifest = json.loads(mf.read_text())
    except (OSError, ValueError) as e:
        bad("corrupt", f"manifest.json is unreadable: {e}")
        return "corrupt", problems
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        bad("corrupt", "manifest.json is not an object with a files mapping")
        return "corrupt", problems
    if manifest.get("schema") != MANIFEST_SCHEMA or _major(manifest.get("version")) != MANIFEST_VERSION[0]:
        bad("incompatible", f"manifest is {manifest.get('schema')!r} {manifest.get('version')!r}; this ccil reads "
                            f"{MANIFEST_SCHEMA} {MANIFEST_VERSION[0]}.x")
        return "incompatible", problems
    if manifest.get("status") != "complete":
        bad("incomplete", f"manifest status is {manifest.get('status')!r}")

    verified = set()
    for rel, info in sorted(files.items()):
        why = _component_problem(root, rel)
        if why is None and rel == MANIFEST:
            why = "names the manifest itself"
        if why is None and not (isinstance(info, dict) and type(info.get("size")) is int
                                and isinstance(info.get("sha256"), str)):
            why = "has no integer size and string sha256"
        if why is not None:
            bad("corrupt", f"manifest entry {rel!r} {why}")
            continue
        p = root / rel
        if not p.is_file():
            bad("incomplete", f"{rel} is listed but missing")
            continue
        if p.stat().st_size != info.get("size") or sha256(p) != info.get("sha256"):
            bad("corrupt", f"{rel} does not match its recorded size and SHA-256")
            continue
        if not staging:  # a writer-owned artifact, matched to its run's manifest (#127)
            register_artifact(p, info["sha256"])
        schema, version = component_schema(p)
        if schema != info.get("schema") or (schema is not None and version != info.get("version")):
            bad("corrupt", f"{rel} declares {schema!r} {version!r} but the manifest records "
                           f"{info.get('schema')!r} {info.get('version')!r}")
        elif schema in READABLE and (_major(version) or 0) > READABLE[schema]:
            bad("incompatible", f"{rel} is {schema} {version}; this ccil reads up to {READABLE[schema]}.x")
        else:
            verified.add(rel)

    on_disk = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                bad("corrupt", f"{rel} is a symlink")
            elif name in filenames:
                on_disk.add(rel)
    for rel in sorted(on_disk - set(files) - {MANIFEST}):
        bad("corrupt", f"{rel} is present but not listed in the manifest")

    if CANDIDATES in verified:
        try:
            with np.load(root / CANDIDATES, allow_pickle=False) as z:
                run_id = json.loads(str(z["meta"])).get("run_id")
            if run_id != manifest.get("run_id"):
                bad("corrupt", f"{CANDIDATES} has run_id {run_id!r}, manifest has {manifest.get('run_id')!r}")
        except (OSError, ValueError, KeyError, AttributeError) as e:
            bad("corrupt", f"{CANDIDATES} is unreadable: {e}")

    kinds = {p["status"] for p in problems}
    return next((k for k in ORDER if k in kinds), "complete"), problems
