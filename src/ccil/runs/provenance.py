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


PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # the ccil package that is running


class ImplementationUnknown(RuntimeError):
    """The running ccil source cannot be read, so stage outputs cannot be safely reused."""


def implementation() -> dict:
    """What decides whether a stored stage output was made by this code.

    source_sha256 hashes every file of the running ccil package (paths and
    bytes, without bytecode caches), so uncommitted edits are identified by
    their content; dependencies records the packages whose versions change
    numerics.
    """
    h = hashlib.sha256()
    try:
        files = sorted(p for p in PACKAGE_ROOT.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
        if not files:
            raise ImplementationUnknown(f"no ccil source files under {PACKAGE_ROOT}")
        for p in files:
            rel = p.relative_to(PACKAGE_ROOT).as_posix().encode()
            h.update(len(rel).to_bytes(8, "big") + rel)
            data = p.read_bytes()
            h.update(len(data).to_bytes(8, "big") + data)
    except OSError as e:
        raise ImplementationUnknown(f"cannot read the ccil source under {PACKAGE_ROOT}: {e}") from None
    return {"ccil_version": __version__, "source_sha256": h.hexdigest(), "dependencies": environment()["packages"]}


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
