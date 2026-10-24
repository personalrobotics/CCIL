"""Offline label generation: trajectories in, labels and a report out.

Stages: data.prepare -> transition.fit -> labels.generate. The accepted rows of
labels/candidates.npz together with the prepared dataset are the augmentation;
dataset writers (M4) turn them into training data. This module never imports a
policy trainer or the integrated pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ConfigError
from ..config.params import mlp_settings_problem
from ..core import (TRANSITION_MODELS, ActionBounds, BackwardEuler, ExactPreimage, Finite, GenerationFailed,
                    IdentityRepresentation, ModelResidual, NoisyAction, SupportRadius, Uniform, generate_labels,
                    load_transition_model, transition_model_class, transitions)
from ..data.prepare import prepare_stage
from ..runs import layout
from ..runs.provenance import sha256
from ..schema import read_trajectories, write_candidates
from .stages import StageResult, run_stage

MODEL_CHECKPOINT = f"{layout.CHECKPOINTS}/transition_model.npz"


def build_generator(v):
    method = v["generation.method"]
    if method in ("backward_euler", "backward_euler_fast"):
        return BackwardEuler()
    if method == "forward_euler":
        return ExactPreimage(max_iterations=v["generation.max_iterations"] or 50, tolerance=v["generation.root_tolerance"])
    if method == "noisy_action":
        return NoisyAction(v["generation.labels_per_transition"], v["generation.action_noise_std"])
    raise ConfigError([f"generation.method {method!r} is not available yet; it is ported in #22"])


def build_validators(v):
    validators = [Finite(), GenerationFailed(), ModelResidual()]
    if v["generation.support_radius"]:
        validators.append(SupportRadius(v["generation.support_radius"]))
    if v["data.action_low"] is not None:
        validators.append(ActionBounds(v["data.action_low"], v["data.action_high"]))
    return validators


def build_model(v):
    kind = v["transition_model.kind"]
    if kind not in TRANSITION_MODELS:
        raise ConfigError([f"transition_model.kind {kind!r} has no implementation; available: {sorted(TRANSITION_MODELS)}"])
    if kind == "mlp" and mlp_settings_problem(v):
        raise ConfigError([mlp_settings_problem(v)])
    try:
        cls = transition_model_class(kind)
    except ModuleNotFoundError as e:
        raise ConfigError([str(e)]) from None
    try:
        return cls.from_config(v)
    except RuntimeError as e:  # e.g. compute.device cuda without CUDA
        raise ConfigError([str(e)]) from None


def preflight(cfg) -> None:
    """Every configuration error a stage would hit, raised before any stage writes."""
    v, errors = cfg.values, []
    if v["data.path"] is None:
        errors.append("data.path is required")
    if v["generation.method"] is None:
        errors.append("generation.method is required")
    if v["generation.method"] == "noisy_action" and not (v["generation.labels_per_transition"] and v["generation.action_noise_std"]):
        errors.append("generation.method noisy_action requires generation.labels_per_transition and generation.action_noise_std")
    if v["transition_model.direction"] == "backward" and v["generation.method"] != "backward_model":
        errors.append("transition_model.direction backward fits s_t from (s_{t+1}, a_t); only generation.method "
                      "backward_model uses such a model, and it is ported in #22")
    for check in (build_model, build_generator):
        try:
            if v["generation.method"] is not None:
                check(v)
        except ConfigError as e:
            errors += e.errors
    if errors:
        raise ConfigError(errors)


def fit_transition_stage(cfg, prepared: StageResult, representation, extra_inputs=None) -> StageResult:
    """Fit the label model on every episode; score an evaluation model fitted on the train split (#13)."""
    v = cfg.values
    split_path = prepared.out(layout.SPLIT)
    inputs = {"dataset": prepared.out(layout.DATASET), **(extra_inputs or {})}
    if split_path.exists():
        inputs["split"] = split_path

    def body(run):
        dataset = read_trajectories(inputs["dataset"])
        batch = transitions(dataset, representation)
        z, a, z_next = batch.z, batch.a, batch.z_next
        if v["transition_model.direction"] == "backward":
            z, z_next = z_next, z  # a backward model maps (s_{t+1}, a_t) to s_t, as in legacy
        model = build_model(v)
        metrics = {"label_model": model.fit(z, a, z_next, seed=v["run.seed"])}
        if split_path.exists():
            import json
            split = json.loads(split_path.read_text())
            train, val = set(split["train"]), set(split["validation"])
            tr = np.isin(batch.episode_id, list(train))
            va = np.isin(batch.episode_id, list(val))
            evaluation = build_model(v)
            evaluation.fit(z[tr], a[tr], z_next[tr], seed=v["run.seed"])
            metrics["held_out"] = evaluation.evaluate(z[va], a[va], z_next[va])
            metrics["held_out"]["note"] = "evaluation model fitted on the train split; the label model saw every episode"
        model.save(run.path(MODEL_CHECKPOINT))
        run.metrics(metrics)
        run.log(f"fitted {v['transition_model.kind']} transition model on {len(batch)} transitions")
    return run_stage("transition.fit", cfg, inputs, body,
                     components={"representation": representation, "transition_model": build_model(v)})


def labels_stage(cfg, prepared: StageResult, fitted: StageResult, representation, extra_inputs=None) -> StageResult:
    v = cfg.values
    inputs = {"dataset": prepared.out(layout.DATASET), "transition_model": fitted.out(MODEL_CHECKPOINT), **(extra_inputs or {})}

    def body(run):
        dataset = read_trajectories(inputs["dataset"])
        # compute.device places torch models; the numpy linear model always runs on cpu.
        device = "cpu" if v["transition_model.kind"] == "linear" else v["compute.device"]
        model = load_transition_model(inputs["transition_model"], device=device)
        cands = generate_labels(dataset, representation, model, build_generator(v), build_validators(v),
                                run_id=run.run_id, seed=v["run.seed"], weighting=Uniform(),
                                inputs={name: sha256(path) for name, path in inputs.items()})
        write_candidates(cands, run.path(layout.CANDIDATES))
        counts = {r: int(cands.reasons[:, i].sum()) for i, r in enumerate(cands.reason_codes)}
        run.metrics({"candidates": int(len(cands.accepted)), "accepted": int(cands.accepted.sum()),
                     "acceptance_rate": float(cands.accepted.mean()) if len(cands.accepted) else 0.0,
                     "rejections_by_reason": counts, "generator": cands.generator})
        run.log(f"generated {len(cands.accepted)} candidates with {cands.generator['name']}; "
                f"accepted {int(cands.accepted.sum())}")
    return run_stage("labels.generate", cfg, inputs, body,
                     components={"representation": representation, "transition_model": build_model(v),
                                 "generator": build_generator(v), "weighting": Uniform()})


def generate_offline(cfg) -> list[StageResult]:
    """Run the offline pipeline; return the three stage results (reused or new)."""
    preflight(cfg)
    if cfg["representation.kind"] != "identity":
        raise ConfigError(["offline generation uses representation.kind=identity; learned representations are "
                           "fitted by the integrated pipeline (ccil pipeline integrated)"])
    representation = IdentityRepresentation(cfg["representation.modality"])
    prepared = prepare_stage(cfg)
    fitted = fit_transition_stage(cfg, prepared, representation)
    return [prepared, fitted, labels_stage(cfg, prepared, fitted, representation)]
