"""Offline label generation: trajectories in, labels and a report out.

Stages: data.prepare -> transition.fit -> labels.generate [-> labels.certify, with certificate.profile]. The accepted rows of
labels/candidates.npz together with the prepared dataset are the augmentation;
dataset writers (M4) turn them into training data. This module never imports a
policy trainer or the integrated pipeline.
"""
from __future__ import annotations

import contextlib

from pathlib import Path

import numpy as np

from ..config import ConfigError
from ..config.params import mlp_settings_problem
from ..config.params import BOUND_PROFILES
from ..core import (TRANSITION_MODELS, ActionBounds, BackwardEuler, ExactPreimage, Finite, GenerationFailed,
                    IdentityRepresentation, ModelMisfit, ModelResidual, NoisyAction, NoisyActionPreimage, SupportRadius,
                    Uniform, generate_labels, load_transition_model, transition_model_class, transitions)
from ..data.prepare import prepare_stage
from ..runs import layout
from ..runs.provenance import sha256
from ..core.certificates import (PROFILES, BackTrackProfile, ContinuityProfile, Declared, DisturbedActionProfile, Provided,
                                 Sources, certify, read_reference_dynamics)
from ..schema import SchemaError
from ..schema.certificates import CLASSIFICATIONS
from ..schema import read_candidates, read_trajectories, write_candidates, write_certificates
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
    if method == "noisy_action_fe":
        return NoisyActionPreimage(v["generation.labels_per_transition"], v["generation.action_noise_std"],
                                   max_iterations=v["generation.max_iterations"] or 50,
                                   tolerance=v["generation.root_tolerance"])
    raise ConfigError([f"generation.method {method!r} is not available yet; it is ported in #22"])


def build_validators(v):
    validators = [Finite(), GenerationFailed(), ModelResidual()]
    if v["generation.support_radius"]:
        validators.append(SupportRadius(v["generation.support_radius"]))
    if v["data.action_low"] is not None:
        validators.append(ActionBounds(v["data.action_low"], v["data.action_high"]))
    if v["generation.model_error_threshold"] is not None:
        validators.append(ModelMisfit(v["generation.model_error_threshold"]))
    return validators


REFERENCE = "reference"  # name of certificate.reference_dynamics among the certify stage's references


def build_certificate(v) -> tuple:
    """(profile, certify_stage keyword arguments) from certificate.*; the config rule has checked consistency."""
    name, ref = v["certificate.profile"], v["certificate.reference_dynamics"]

    def declared(key):
        return Declared(v[key], f"declared in configuration ({key})")

    def true(key):
        return Provided("reference_lipschitz", REFERENCE) if ref is not None else declared(key)

    nb = {"neighborhood_state": v["certificate.neighborhood_state"]}
    if name not in ("backtrack", "backtrack_paper_proxy"):
        nb["neighborhood_action"] = v["certificate.neighborhood_action"]
    if name in BOUND_PROFILES:
        demo = Provided("reference_residual", REFERENCE) if ref is not None else declared("certificate.demonstration_residual_bound")
        if name == "continuity":
            exact = ref is not None and v["transition_model.kind"] == "linear"
            c = [Provided("linear_discrepancy", REFERENCE) if exact else declared(k)
                 for k in ("certificate.discrepancy_state_lipschitz", "certificate.discrepancy_action_lipschitz")]
            profile = ContinuityProfile(*c, demo, **nb)
        elif name == "backtrack":
            learned = (Provided("model_lipschitz") if v["certificate.learned_lipschitz_source"] == "model"
                       else declared("certificate.learned_state_lipschitz"))
            profile = BackTrackProfile(learned, true("certificate.true_state_lipschitz"), demo, **nb)
        else:
            profile = DisturbedActionProfile(true("certificate.true_state_lipschitz"),
                                             true("certificate.true_action_lipschitz"), demo, **nb)
        admission, threshold = "upper_bound", v["certificate.max_total_error"]
    else:
        profile = PROFILES[name](**nb)
        admission, threshold = "score", v["certificate.score_threshold"]
    return profile, {"admission": admission, "threshold": threshold, "selection_quantile": v["certificate.selection_quantile"],
                     "selection_population": v["certificate.selection_population"],
                     "references": {REFERENCE: Path(ref)} if ref is not None else {}}


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
    if v["generation.method"] in ("noisy_action", "noisy_action_fe") and not (
            v["generation.labels_per_transition"] and v["generation.action_noise_std"]):
        errors.append(f"generation.method {v['generation.method']} requires generation.labels_per_transition and "
                      "generation.action_noise_std")
    ref = v["certificate.reference_dynamics"]
    if ref is not None:
        try:
            read_reference_dynamics(ref)
        except (OSError, SchemaError) as e:
            errors.append(f"certificate.reference_dynamics {ref}: {str(e).splitlines()[-1].strip()}")
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


@contextlib.contextmanager
def _jacobian_budget():
    """A Jacobian budget smaller than one row is a configuration error, not a failure of the computation."""
    try:
        yield
    except ValueError as e:
        if type(e).__name__ == "JacobianBudgetError":  # models.mlp; torch is imported only with the mlp extra
            raise ConfigError([str(e)]) from None
        raise


def labels_stage(cfg, prepared: StageResult, fitted: StageResult, representation, extra_inputs=None) -> StageResult:
    v = cfg.values
    inputs = {"dataset": prepared.out(layout.DATASET), "transition_model": fitted.out(MODEL_CHECKPOINT), **(extra_inputs or {})}

    def body(run):
        dataset = read_trajectories(inputs["dataset"])
        # compute.device places torch models; the numpy linear model always runs on cpu.
        device = "cpu" if v["transition_model.kind"] == "linear" else v["compute.device"]
        model = load_transition_model(inputs["transition_model"], device=device)
        if hasattr(model, "jacobian_memory_bytes"):
            model.jacobian_memory_bytes = v["compute.jacobian_memory_mb"] * 2**20
        with _jacobian_budget():
            cands = generate_labels(dataset, representation, model, build_generator(v), build_validators(v),
                                    run_id=run.run_id, seed=v["run.seed"], weighting=Uniform(),
                                    action_chunk_size=v["generation.action_chunk_size"],
                                    inputs={name: sha256(path) for name, path in inputs.items()})
        write_candidates(cands, run.path(layout.CANDIDATES))
        counts = {r: int(cands.reasons[:, i].sum()) for i, r in enumerate(cands.reason_codes)}
        run.metrics({"candidates": int(len(cands.accepted)), "accepted": int(cands.accepted.sum()),
                     "acceptance_rate": float(cands.accepted.mean()) if len(cands.accepted) else 0.0,
                     "rejections_by_reason": counts, "generator": cands.generator,
                     "jacobian_chunking": getattr(model, "jacobian_chunking", None)})  # its last call
        run.log(f"generated {len(cands.accepted)} candidates with {cands.generator['name']}; "
                f"accepted {int(cands.accepted.sum())}")
    return run_stage("labels.generate", cfg, inputs, body,
                     components={"representation": representation, "transition_model": build_model(v),
                                 "generator": build_generator(v), "weighting": Uniform()})


def certify_stage(cfg, prepared: StageResult, fitted: StageResult, labeled: StageResult, representation, profile,
                  admission: str = "upper_bound", threshold: float | None = None,
                  selection_quantile: float | None = None, references: dict | None = None,
                  state_scale=None, action_scale=None, extra_inputs=None,
                  selection_population: str = "admitted") -> StageResult:
    """labels.certify: write labels/certificates.npz for the labels stage's candidates (#88).

    The candidates are an input and are never modified; consumers train on
    candidate.accepted & certificate.admissible & certificate.selected.
    references are artifacts bound providers compute constants from; they are
    stage inputs. The profile's specification, the admission policy, the
    selection, and the scales enter the stage identity.
    """
    v = cfg.values
    references = dict(references or {})
    inputs = {"dataset": prepared.out(layout.DATASET), "transition_model": fitted.out(MODEL_CHECKPOINT),
              "candidates": labeled.out(layout.CANDIDATES), **(extra_inputs or {}),
              **{f"reference.{k}": p for k, p in references.items()}}

    def body(run):
        sources = Sources(inputs["candidates"], inputs["dataset"], inputs["transition_model"], representation,
                          extra=dict(extra_inputs or {}), references=references,
                          device="cpu" if v["transition_model.kind"] == "linear" else v["compute.device"],
                          jacobian_memory_bytes=v["compute.jacobian_memory_mb"] * 2**20)
        with _jacobian_budget():
            cert = certify(sources, profile, admission=admission, threshold=threshold,
                           selection_quantile=selection_quantile, state_scale=state_scale, action_scale=action_scale,
                           selection_population=selection_population)
        write_certificates(cert, run.path(layout.CERTIFICATES))
        accepted = read_candidates(inputs["candidates"]).accepted
        usable = np.asarray(accepted) & cert.admissible & cert.selected
        run.metrics({"certificates": int(len(cert.selected)), "profile": profile.name, "admission": cert.admission,
                     "by_classification": {c: int((cert.classification == c).sum()) for c in CLASSIFICATIONS},
                     "admissible": int(cert.admissible.sum()), "selected": int(cert.selected.sum()),
                     "usable": int(usable.sum())})
        run.log(f"certified {len(cert.selected)} candidates with {profile.name}; usable {int(usable.sum())}")
    scales = {k: None if x is None else [float(e) for e in x] for k, x in (("state", state_scale), ("action", action_scale))}
    return run_stage("labels.certify", cfg, inputs, body, components={"representation": representation},
                     parameters={"profile": profile.spec(), "admission": admission, "threshold": threshold,
                                 "selection_quantile": selection_quantile, "selection_population": selection_population,
                                 "scales": scales})


def generate_offline(cfg) -> list[StageResult]:
    """Run the offline pipeline; return the stage results (reused or new), with labels.certify when configured."""
    preflight(cfg)
    if cfg["representation.kind"] != "identity":
        raise ConfigError(["offline generation uses representation.kind=identity; learned representations are "
                           "fitted by the integrated pipeline (ccil pipeline integrated)"])
    representation = IdentityRepresentation(cfg["representation.modality"])
    prepared = prepare_stage(cfg)
    fitted = fit_transition_stage(cfg, prepared, representation)
    labeled = labels_stage(cfg, prepared, fitted, representation)
    if cfg["certificate.profile"] is None:
        return [prepared, fitted, labeled]
    profile, kwargs = build_certificate(cfg.values)
    return [prepared, fitted, labeled, certify_stage(cfg, prepared, fitted, labeled, representation, profile, **kwargs)]
