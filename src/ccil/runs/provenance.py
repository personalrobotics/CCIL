"""Provenance recorded in every run manifest. No timestamps, so replays are byte-identical.

Two costs are paid once per process rather than at every stage (#127):
- Code state. code() and implementation() are captured at their first call. verify_code_state() recomputes both
  and raises CodeStateChanged on any difference; the CLI calls it when a command finishes and study runners
  before they finalize a unit's record, so a unit that saw a source edit is never finalized.
- Content hashes of writer-owned artifacts. sha256() returns a cached hash only for a file this process's
  RunWriter wrote, or one whose full rehash matched a complete run's manifest (register_artifact), and only while
  its resolved path, device, inode, size and nanosecond mtime are unchanged. Any other file, user inputs included,
  is hashed in full every time. CCIL_PROVENANCE_STRICT=1 disables the cache.
  Outside the cache's threat model: rewriting a pipeline-owned artifact in place while preserving its size,
  inode and nanosecond mtime. Those artifacts are write-once and atomically replaced; the strict mode covers
  adversarial or archival verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .. import __version__

PACKAGES = ("numpy", "pyyaml", "torch")


_ARTIFACTS: dict = {}           # resolved path -> (device, inode, size, mtime_ns, sha256): writer-owned artifacts
HASH_STATS = {"full": 0, "cached": 0}


def strict() -> bool:
    """Whether every hash is computed in full (CCIL_PROVENANCE_STRICT=1)."""
    return os.environ.get("CCIL_PROVENANCE_STRICT") == "1"


def _key(path) -> tuple[str, tuple]:
    st = os.stat(path)
    return str(Path(path).resolve()), (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _full_sha256(path) -> str:
    HASH_STATS["full"] += 1
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path) -> str:
    """SHA-256 of a file: cached for a registered writer-owned artifact whose metadata is unchanged, else in full."""
    if not strict():
        resolved, meta = _key(path)
        hit = _ARTIFACTS.get(resolved)
        if hit is not None and hit[0] == meta:
            HASH_STATS["cached"] += 1
            return hit[1]
    return _full_sha256(path)


def register_artifact(path, digest: str) -> None:
    """Record a writer-owned artifact's hash with its current metadata (RunWriter, or a manifest-verified file)."""
    resolved, meta = _key(path)
    _ARTIFACTS[resolved] = (meta, digest)


_CODE: dict | None = None
_IMPLEMENTATION: dict | None = None


class CodeStateChanged(RuntimeError):
    """The running code's state differs from the state captured when the process started: no unit is finalized."""


def code() -> dict:
    """ccil version, plus the git commit and dirty flag when the source is a git checkout (captured once)."""
    global _CODE
    if _CODE is None:
        _CODE = _code()
    return dict(_CODE)


def _code() -> dict:
    """ccil version, plus the git commit and dirty flag when the source is a git checkout."""
    src = Path(__file__).resolve().parent
    info = {"ccil_version": __version__, "git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if commit.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
            info["git_commit"] = commit.stdout.strip()
            info["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
            info["git_status_sha256"] = (hashlib.sha256(dirty.stdout.encode()).hexdigest()
                                         if dirty.returncode == 0 else None)
    except (OSError, subprocess.SubprocessError):
        pass  # no git, or not a checkout: recorded as unknown
    return info


PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # the ccil package that is running


class ImplementationUnknown(RuntimeError):
    """The running ccil source cannot be read, so stage outputs cannot be safely reused."""


def implementation() -> dict:
    """What decides whether a stored stage output was made by this code (captured once per process)."""
    global _IMPLEMENTATION
    if _IMPLEMENTATION is None:
        _IMPLEMENTATION = _implementation()
    return json.loads(json.dumps(_IMPLEMENTATION))


def verify_code_state(source_only: bool = False) -> None:
    """Raise CodeStateChanged unless the code state and ccil source equal those captured at the process's start.

    Called when a unit is finalized (a CLI command's end, a study runner's record), so a unit that saw a source
    edit is never finalized, even when the tree was already dirty (the source content hash catches it).
    source_only skips git (about 40 ms) and checks only the ccil source hash (about 4 ms): RunWriter's check
    before every commit, so no stage output computed by edited code is ever stored under the original identity."""
    if _CODE is None and _IMPLEMENTATION is None:
        return  # nothing was captured, so nothing was recorded
    changed = []
    if _CODE is not None and not source_only:
        now_code = _code()
        if now_code != _CODE:
            changed.append(f"git state {_CODE} -> {now_code}")
    now_impl = _implementation() if _IMPLEMENTATION is not None else None
    if now_impl is not None and now_impl["source_sha256"] != _IMPLEMENTATION["source_sha256"]:
        changed.append("the ccil source changed")
    if changed:
        raise CodeStateChanged("the code changed while this process ran: " + "; ".join(changed))


def _implementation() -> dict:
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
