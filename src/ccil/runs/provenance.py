"""Provenance recorded in every run manifest. No timestamps, so replays are byte-identical."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .. import __version__

PACKAGES = ("numpy", "pyyaml", "torch")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code() -> dict:
    """ccil version, plus the git commit and dirty flag when the source is a git checkout."""
    src = Path(__file__).resolve().parent
    info = {"ccil_version": __version__, "git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if commit.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
            info["git_commit"] = commit.stdout.strip()
            info["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass  # no git, or not a checkout: recorded as unknown
    return info


def environment() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    devices = "cpu (torch not imported by this run)"
    torch = sys.modules.get("torch")  # never import torch just to describe it
    if torch is not None:
        devices = ([f"cuda:{i} {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
                   if torch.cuda.is_available() else "cpu (CUDA unavailable)")
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "devices": devices}


def component_schema(path: Path) -> tuple[str | None, str | None]:
    """The schema and version a component declares, if any."""
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        elif path.suffix == ".json":
            meta = json.loads(path.read_text())
        else:
            return None, None
    except (OSError, ValueError, KeyError):
        return None, None
    if not isinstance(meta, dict) or "schema" not in meta:
        return None, None
    return meta.get("schema"), str(meta.get("version"))
