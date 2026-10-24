"""Atomic run directories.

Components are written into a staging directory `.<run_id>.tmp-<pid>` next to
the destination. On success the writer adds manifest.json, validates the staged
run, and renames it into place; on any exception the staging directory is
removed and an existing run is left untouched. A writer killed outright cannot
clean up, so the next writer for the same run ID removes stale staging
directories whose process is gone.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path, PurePosixPath

from .layout import CONFIG, LOG, MANIFEST, MANIFEST_SCHEMA, MANIFEST_VERSION, METRICS, METRICS_SCHEMA, STAGING_MARKER
from .provenance import code, component_schema, environment, sha256
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
        if self.dest.exists() and not self.overwrite:
            raise FileExistsError(f"{self.dest} exists; set run.overwrite=true to replace it")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.output_dir.glob(f".{self.run_id}{STAGING_MARKER}*"):
            pid = stale.name.rsplit("-", 1)[-1]
            if pid.isdigit() and not _alive(int(pid)):
                shutil.rmtree(stale, ignore_errors=True)
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
            dump_json(self._manifest(), self.staging / MANIFEST)
            status, problems = validate_run(self.staging, staging=True)
            if status != "complete":
                raise RunError(f"staged run failed validation ({status}): {problems}")
            if self.dest.exists():
                aside = self.output_dir / f".{self.run_id}.old-{os.getpid()}"
                os.replace(self.dest, aside)
                os.replace(self.staging, self.dest)
                shutil.rmtree(aside)
            else:
                os.replace(self.staging, self.dest)
        except BaseException:
            shutil.rmtree(self.staging, ignore_errors=True)
            raise
        self.run_dir = self.dest
        return False
