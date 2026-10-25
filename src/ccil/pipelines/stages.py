"""Stage runs: each pipeline stage is its own run directory (docs/runs.md).

A stage's run ID hashes its name, the configuration keys that stage consumes
(STAGE_KEYS), the components it uses (kind or name, and version), the running
implementation (the ccil source hash and dependency versions), any parameters
passed outside the configuration (such as a certificate profile's descriptor),
and the SHA-256 of every input artifact. The full identity is written to stage.json in the run.
Upstream checkpoints hand off through the manifest `inputs`, so a change
upstream produces a new downstream run, while a change that only affects later
stages (for example the policy trainer) reuses every earlier stage. Each run
still records the full configuration in config.json. Resume is owned here: a
stage whose run directory already exists and validates as complete is reused,
unless run.overwrite is set. Changed code, even uncommitted, changes the source
hash and therefore the run ID, so an output is reused only by the implementation
that made it; if the source cannot be read, the stage fails instead of reusing.
A stage that fails leaves no run directory, so a rerun resumes at the first
incomplete stage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..runs import RunError, RunWriter, layout, validate_run
from ..runs.provenance import ImplementationUnknown, implementation, sha256
from ..runs.writer import dump_json
from ..runs.writer import recover


class StageError(RuntimeError):
    """A stage failed; earlier completed stages remain valid and reusable."""

    def __init__(self, stage: str, error: BaseException):
        self.stage, self.error = stage, error
        super().__init__(f"stage {stage} failed: {error}")


@dataclass
class StageResult:
    stage: str
    run_dir: Path
    reused: bool

    def out(self, rel: str) -> Path:
        return self.run_dir / rel


# Configuration prefixes each stage consumes; run.output_dir and run.overwrite never affect identity.
STAGE_KEYS = {
    "data.prepare": ("run.seed", "data."),
    "representation.fit": ("run.seed", "representation."),
    "transition.fit": ("run.seed", "transition_model.", "representation.modality", "compute."),
    "labels.generate": ("run.seed", "generation.", "data.action_low", "data.action_high", "representation.modality",
                        "compute."),
    "labels.certify": ("representation.modality", "compute."),
    "policy.train": ("run.seed", "policy."),
    "policy.evaluate": ("run.seed", "policy.", "evaluation."),
}


def component(obj) -> dict:
    """Identity of a component: kind (representations, transition models) or name (the rest), and version."""
    key = "kind" if hasattr(obj, "kind") else "name"
    return {key: getattr(obj, key), "version": obj.version}


def stage_identity(stage: str, cfg, inputs: dict, components: dict | None = None,
                   parameters: dict | None = None) -> dict:
    """Everything the reuse decision depends on; its hash is the run ID."""
    prefixes = STAGE_KEYS[stage]
    consumed = {k: v for k, v in cfg.values.items() if any(k == p or (p.endswith(".") and k.startswith(p)) for p in prefixes)}
    try:
        impl = implementation()
    except ImplementationUnknown as e:
        raise StageError(stage, e) from e
    identity = {"stage": stage, "config": consumed, "components": components or {}, "implementation": impl,
                "inputs": {k: sha256(p) for k, p in sorted(inputs.items())}}
    if parameters:
        identity["parameters"] = parameters
    return identity


def stage_run_id(identity: dict) -> str:
    blob = json.dumps(identity, sort_keys=True)
    return f"{identity['stage'].replace('.', '-')}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


def run_stage(stage: str, cfg, inputs: dict, body, components: dict | None = None,
              parameters: dict | None = None) -> StageResult:
    """Run body(run) inside a RunWriter for this stage, or reuse a complete prior run.

    components maps roles to component objects (see component()); parameters is
    JSON data that affects the output but is not configuration.
    """
    out_dir = Path(cfg["run.output_dir"])
    identity = stage_identity(stage, cfg, inputs, {role: component(c) for role, c in (components or {}).items()},
                              parameters)
    run_id = stage_run_id(identity)
    dest = out_dir / run_id
    if out_dir.is_dir():
        recover(out_dir, run_id)
    if dest.exists() and not cfg["run.overwrite"]:
        status, problems = validate_run(dest)
        if status == "complete":
            return StageResult(stage, dest, reused=True)
        raise StageError(stage, RunError(f"{dest} exists but is {status} ({problems[:3]}); remove it or set "
                                         f"run.overwrite=true"))
    try:
        with RunWriter(out_dir, run_id, stage, cfg, overwrite=cfg["run.overwrite"]) as run:
            for name, path in sorted(inputs.items()):
                run.add_input(name, path)
            dump_json({**identity, "run_id": run_id}, run.path(layout.STAGE))
            body(run)
    except StageError:
        raise
    except Exception as e:  # the writer has already removed the staging directory
        raise StageError(stage, e) from e
    return StageResult(stage, run.run_dir, reused=False)
