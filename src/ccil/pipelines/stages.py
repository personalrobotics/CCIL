"""Content-addressed pipeline stages with validation, reuse, and resume.

Stage identity includes consumed configuration, component versions, source,
dependencies, explicit parameters, and input hashes. Failed stages leave no
completed run directory. See ``docs/runs.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..runs import RunError, RunWriter, layout, validate_run
from ..runs.provenance import CodeStateChanged, ImplementationUnknown, implementation, sha256
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
    "labels.diagnose": ("run.seed", "generation.", "diagnose.", "data.action_low", "data.action_high",
                        "representation.modality", "compute.", "certificate.profile", "certificate.reference_dynamics"),
    "labels.certify": ("certificate.", "representation.modality", "compute."),
    "labels.curate": ("curate.", "generation.evidence_mode"),
    "labels.calibrate": ("calibration.",),
    "labels.audit": ("audit.", "representation.modality"),
    "policy.train": ("run.seed", "supervision.", "integrated.", "compute."),
    "policy.evaluate": ("run.seed", "supervision.", "integrated."),
}


# Keys a prefix above would include that the stage never reads: the evidence mode is curation's.
STAGE_EXCLUDE = {"labels.generate": ("generation.evidence_mode",),
                 "labels.diagnose": ("generation.evidence_mode",)}


def component(obj) -> dict:
    """Identity of a component: kind (representations, transition models) or name (the rest), and version."""
    key = "kind" if hasattr(obj, "kind") else "name"
    return {key: getattr(obj, key), "version": obj.version}


def stage_identity(stage: str, cfg, inputs: dict, components: dict | None = None,
                   parameters: dict | None = None) -> dict:
    """Everything the reuse decision depends on; its hash is the run ID."""
    prefixes = STAGE_KEYS[stage]
    consumed = {k: v for k, v in cfg.values.items() if any(k == p or (p.endswith(".") and k.startswith(p)) for p in prefixes)
                and k not in STAGE_EXCLUDE.get(stage, ())}
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


def stage_dest(stage: str, cfg, inputs: dict, components: dict | None = None, parameters: dict | None = None) -> Path:
    """The run directory run_stage uses for these arguments (its identity's run ID under run.output_dir)."""
    identity = stage_identity(stage, cfg, inputs, {role: component(c) for role, c in (components or {}).items()},
                              parameters)
    return Path(cfg["run.output_dir"]) / stage_run_id(identity)


def completed_stage(stage: str, cfg, inputs: dict, components: dict | None = None,
                    parameters: dict | None = None) -> StageResult | None:
    """The complete run run_stage would reuse for these arguments, or None (it would run the stage).

    Study runners ask before batching work across units, so completed stages are never recomputed.
    """
    if cfg["run.overwrite"]:
        return None
    dest = stage_dest(stage, cfg, inputs, components, parameters)
    if dest.parent.is_dir():
        recover(dest.parent, dest.name)
    if dest.exists() and validate_run(dest)[0] == "complete":
        return StageResult(stage, dest, reused=True)
    return None


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
    except (StageError, CodeStateChanged):  # A code change invalidates the whole unit, not just this stage.
        raise
    except Exception as e:  # the writer has already removed the staging directory
        raise StageError(stage, e) from e
    return StageResult(stage, run.run_dir, reused=False)
