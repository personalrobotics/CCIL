"""Stage runs: each pipeline stage is its own run directory (docs/runs.md).

A stage's run ID hashes its name, the configuration keys that stage consumes
(STAGE_KEYS), any explicit identity, and the SHA-256 of every input artifact.
Upstream checkpoints hand off through the manifest `inputs`, so a change
upstream produces a new downstream run, while a change that only affects later
stages (for example the policy trainer) reuses every earlier stage. Each run
still records the full configuration in config.json. Resume is owned here: a
stage whose run directory already exists and validates as complete is reused,
unless run.overwrite is set. A stage that fails leaves no run directory, so a
rerun resumes at the first incomplete stage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..runs import RunError, RunWriter, validate_run
from ..runs.provenance import sha256
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
    "transition.fit": ("run.seed", "transition_model.", "representation.modality"),
    "labels.generate": ("run.seed", "generation.", "data.action_low", "data.action_high", "representation.modality"),
    "policy.train": ("run.seed", "policy."),
    "policy.evaluate": ("run.seed", "policy.", "evaluation."),
}


def stage_run_id(stage: str, cfg, inputs: dict, identity: dict | None = None) -> str:
    prefixes = STAGE_KEYS[stage]
    consumed = {k: v for k, v in cfg.values.items() if any(k == p or (p.endswith(".") and k.startswith(p)) for p in prefixes)}
    blob = json.dumps({"stage": stage, "config": consumed, "identity": identity or {},
                       "inputs": {k: sha256(p) for k, p in sorted(inputs.items())}}, sort_keys=True)
    return f"{stage.replace('.', '-')}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


def run_stage(stage: str, cfg, inputs: dict, body, identity: dict | None = None) -> StageResult:
    """Run body(run) inside a RunWriter for this stage, or reuse a complete prior run."""
    out_dir = Path(cfg["run.output_dir"])
    run_id = stage_run_id(stage, cfg, inputs, identity)
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
            body(run)
    except StageError:
        raise
    except Exception as e:  # the writer has already removed the staging directory
        raise StageError(stage, e) from e
    return StageResult(stage, run.run_dir, reused=False)
