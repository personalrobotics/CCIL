"""Atomic run directories.

Components are written into a staging directory `.<run_id>.tmp-<pid>` next to
the destination. On success the writer adds manifest.json, validates the staged
run, and moves it into place; on any exception the staging directory is removed
and an existing run is left untouched.

Replacing an existing run atomically exchanges the staging directory with the
destination (renameat2 RENAME_EXCHANGE on Linux, renamex_np RENAME_SWAP on
macOS), so a valid destination exists at every instant; the old run, now under
the staging name, is then removed. Where the filesystem cannot exchange, the
writer falls back to two renames through `.<run_id>.old-<pid>` and restores the
old run if the second fails.

A writer killed outright cannot clean up. recover() runs before any writer or
stage looks at a destination: for each directory left by a dead process, an
.old-<pid> directory is restored if the destination is missing and removed
otherwise, and a staging directory is removed.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

from .layout import CONFIG, LOG, MANIFEST, MANIFEST_SCHEMA, MANIFEST_VERSION, METRICS, METRICS_SCHEMA, STAGING_MARKER
from .provenance import code, component_schema, environment, register_artifact, sha256, verify_code_state
from .validate import validate_run


class RunError(RuntimeError):
    """A run directory could not be written or failed its own validation."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


OLD_MARKER = ".old-"


def _exchange(a: Path, b: Path) -> bool:
    """Atomically swap two existing directories; False if this platform or filesystem cannot."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux"):
            rc = libc.renameat2(-100, os.fsencode(a), -100, os.fsencode(b), 2)  # AT_FDCWD, RENAME_EXCHANGE
        elif sys.platform == "darwin":
            rc = libc.renamex_np(os.fsencode(a), os.fsencode(b), 2)  # RENAME_SWAP
        else:
            return False
    except (OSError, AttributeError):  # no libc symbol
        return False
    if rc == 0:
        return True
    err = ctypes.get_errno()
    if err in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
        return False
    raise OSError(err, os.strerror(err), str(a))


def _dead(path: Path, marker: str) -> bool:
    pid = path.name.rsplit("-", 1)[-1]
    return marker in path.name and pid.isdigit() and not _alive(int(pid))


def recover(output_dir, run_id: str) -> None:
    """Resolve what writers for run_id left behind when killed (see the module docstring)."""
    output_dir = Path(output_dir)
    dest = output_dir / run_id
    for old in sorted(output_dir.glob(f".{run_id}{OLD_MARKER}*")):
        if _dead(old, OLD_MARKER):
            if dest.exists():
                shutil.rmtree(old, ignore_errors=True)
            else:
                os.replace(old, dest)
    for stale in output_dir.glob(f".{run_id}{STAGING_MARKER}*"):
        if _dead(stale, STAGING_MARKER):
            shutil.rmtree(stale, ignore_errors=True)


def dump_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


class RunWriter:
    def __init__(self, output_dir, run_id: str, workflow: str, config, overwrite: bool = False):
        self.output_dir, self.run_id, self.workflow = Path(output_dir), run_id, workflow
        self.config, self.overwrite = config, overwrite
        self.dest = self.output_dir / run_id
        self.inputs: dict = {}
        self.staging: Path | None = None
        self.run_dir: Path | None = None

    def __enter__(self) -> "RunWriter":
        recover(self.output_dir, self.run_id)
        if self.dest.exists() and not self.overwrite:
            raise FileExistsError(f"{self.dest} exists; set run.overwrite=true to replace it")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.staging = self.output_dir / f".{self.run_id}{STAGING_MARKER}{os.getpid()}"
        shutil.rmtree(self.staging, ignore_errors=True)
        self.staging.mkdir()
        dump_json(self.config.to_dict(), self.path(CONFIG))
        return self

    def path(self, rel: str) -> Path:
        """Destination for a component inside the staged run; parents are created."""
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or str(p) == MANIFEST:
            raise ValueError(f"{rel!r} is not a writable run component path")
        out = self.staging / p
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def add_input(self, name: str, path) -> None:
        self.inputs[name] = {"path": str(path), "sha256": sha256(path)}

    def log(self, message: str) -> None:
        with open(self.path(LOG), "a") as f:
            f.write(message.rstrip("\n") + "\n")

    def metrics(self, values: dict) -> None:
        dump_json({"schema": METRICS_SCHEMA, "version": "1.0", "values": values}, self.path(METRICS))

    def _manifest(self) -> dict:
        files = {}
        for p in sorted(self.staging.rglob("*")):
            if p.is_file():
                schema, version = component_schema(p)
                entry = {"sha256": sha256(p), "size": p.stat().st_size}
                if schema is not None:
                    entry.update(schema=schema, version=version)
                files[p.relative_to(self.staging).as_posix()] = entry
        values = self.config.values
        return {
            "schema": MANIFEST_SCHEMA, "version": f"{MANIFEST_VERSION[0]}.{MANIFEST_VERSION[1]}",
            "run_id": self.run_id, "workflow": self.workflow, "status": "complete",
            "seeds": {k: v for k, v in values.items() if k.endswith(".seed")},
            "config": {"path": CONFIG, "config_hash": self.config.config_hash()},
            "inputs": self.inputs, "code": code(), "environment": environment(), "files": files,
        }

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            return False
        try:
            manifest = self._manifest()
            dump_json(manifest, self.staging / MANIFEST)
            status, problems = validate_run(self.staging, staging=True)
            if status != "complete":
                raise RunError(f"staged run failed validation ({status}): {problems}")
            verify_code_state(source_only=True)  # never commit an output the edited source may have computed (#127)
            if not self.dest.exists():
                os.replace(self.staging, self.dest)
            elif not _exchange(self.staging, self.dest):
                self._replace_by_renames()
        except BaseException:
            shutil.rmtree(self.staging, ignore_errors=True)
            raise
        # After an exchange the staging name holds the old run.
        shutil.rmtree(self.staging, ignore_errors=True)
        self.run_dir = self.dest
        for rel, info in manifest["files"].items():  # writer-owned artifacts: a rename keeps inode and mtime (#127)
            register_artifact(self.dest / rel, info["sha256"])
        return False

    def _replace_by_renames(self) -> None:
        aside = self.output_dir / f".{self.run_id}{OLD_MARKER}{os.getpid()}"
        os.replace(self.dest, aside)
        try:
            os.replace(self.staging, self.dest)
        except BaseException:
            if not self.dest.exists():
                os.replace(aside, self.dest)
            raise
        shutil.rmtree(aside)
