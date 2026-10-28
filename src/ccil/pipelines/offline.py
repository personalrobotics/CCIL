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
    if v["transition_model.fit_episodes"] == "train" and not v["data.validation_fraction"]:
        errors.append("transition_model.fit_episodes train fits the label model on the train split: it needs "
                      "data.validation_fraction > 0")
    if v["transition_model.direction"] == "backward" and v["generation.method"] != "backward_model":
        errors.append("transition_model.direction backward fits s_t from (s_{t+1}, a_t); only generation.method "
                      "backward_model uses such a model, and it is ported in #22")
    if v["transition_model.trainer_version"] == "2" and v["transition_model.kind"] == "mlp":
        from ..models.mlp_batched import SUPPORTED
        if (v["transition_model.lipschitz_type"] or "none") not in SUPPORTED:
            errors.append(f"transition_model.trainer_version 2 fits lipschitz types {list(SUPPORTED)}; "
                          f"{v['transition_model.lipschitz_type']} needs trainer_version 1")
    for check in (build_model, build_generator):
        try:
            if v["generation.method"] is not None:
                check(v)
        except ConfigError as e:
            errors += e.errors
    if errors:
        raise ConfigError(errors)


def _fit(v, z, a, z_next):
    """Fit one transition model with the configured engine: (model, fit record)."""
    model = build_model(v)
    if v["transition_model.trainer_version"] == "2" and v["transition_model.kind"] == "mlp":
        from ..models.mlp_batched import FitJob, fit_group
        return fit_group([FitJob(z, a, z_next, v["run.seed"])], dict(model.settings), device=v["compute.device"],
                         deterministic=v["compute.deterministic"])[0]
    return model, model.fit(z, a, z_next, seed=v["run.seed"])


def transition_inputs(prepared: StageResult, extra_inputs=None) -> dict:
    """transition.fit's inputs: the prepared dataset, its split when there is one, and any extra inputs."""
    split_path = prepared.out(layout.SPLIT)
    inputs = {"dataset": prepared.out(layout.DATASET), **(extra_inputs or {})}
    if split_path.exists():
        inputs["split"] = split_path
    return inputs


def transition_data(v, inputs: dict, representation):
    """(z, a, z_next, train mask, validation mask) for transition.fit; the masks are None without a split.

    The label model fits every transition; the evaluation model fits the train rows.
    """
    batch = transitions(read_trajectories(inputs["dataset"]), representation)
    z, a, z_next = batch.z, batch.a, batch.z_next
    if v["transition_model.direction"] == "backward":
        z, z_next = z_next, z  # a backward model maps (s_{t+1}, a_t) to s_t, as in legacy
    if "split" not in inputs:
        return z, a, z_next, None, None
    import json
    split = json.loads(Path(inputs["split"]).read_text())
    return (z, a, z_next, np.isin(batch.episode_id, list(split["train"])),
            np.isin(batch.episode_id, list(split["validation"])))


class PrefitMismatch(RuntimeError):
    """A model fitted elsewhere does not belong to this unit's inputs: a study-runner defect."""


def requested_execution_problem(record: dict, v) -> str | None:
    """Why a group's record was not run as this unit requests (device type, determinism, and the engine its
    group requested: never an eager fallback of a compiled one, #134), or None."""
    import torch
    group = record.get("group") or {}
    want = (torch.device(v["compute.device"]).type, bool(v["compute.deterministic"]))
    got = (torch.device(group["device"]).type if group.get("device") else None, group.get("deterministic"))
    if got != want:
        return f"was run with (device, deterministic) {got}, not the unit's {want}"
    comp = group.get("compilation") or {}
    if not comp.get("requested") or comp.get("effective") != comp["requested"]:
        return f"ran the engine {comp.get('effective')!r}, not the requested {comp.get('requested')!r}"
    return None


def _check_prefit(v, prefit: dict, z, a, z_next, tr, clean: bool) -> None:
    """Refuse prefit models whose fit identity is not this unit's (rows, split, direction, settings, seed), whose
    weights are not the ones their record describes, or that ran differently from the unit's request."""
    from ..models.mlp_batched import fit_identity, model_sha256
    settings = dict(build_model(v).settings)
    expected = {"label": (z[tr], a[tr], z_next[tr]) if clean else (z, a, z_next)}
    if tr is not None and not clean:
        expected["evaluation"] = (z[tr], a[tr], z_next[tr])
    for role, rows in expected.items():
        model, record = prefit.get(role) or (None, {})
        got = record.get("fit_identity_sha256")
        want = fit_identity(*rows, settings, v["run.seed"])
        if got != want:
            raise PrefitMismatch(f"prefit {role} model has fit identity {got}, not this unit's {want}")
        if model is None or record.get("model_sha256") != model_sha256(model):
            raise PrefitMismatch(f"prefit {role} model is not the model its record describes")
        problem = requested_execution_problem(record, v)
        if problem:
            raise PrefitMismatch(f"prefit {role} model {problem}")


def fit_transition_stage(cfg, prepared: StageResult, representation, extra_inputs=None, prefit=None) -> StageResult:
    """Fit the label model and score a model fitted on the train split on the held-out episodes (#13).

    transition_model.fit_episodes all fits the label model on every episode and a separate evaluation model on
    the train split; train fits the label model on the train split only, and scores it.

    prefit, from a study runner that fitted many units together (models/mlp_batched.fit_group), supplies
    {"label": (model, fit record), "evaluation": (model, fit record) or None} instead of fitting here; it needs
    transition_model.trainer_version 2, so its stage identity is never a per-fit one. With fit_episodes train,
    the label model is fitted on the train split and there is no separate evaluation model.
    """
    v = cfg.values
    if prefit is not None and v["transition_model.trainer_version"] != "2":
        raise ConfigError(["prefit transition models come from the batched engine: set transition_model.trainer_version 2"])
    inputs = transition_inputs(prepared, extra_inputs)

    def body(run):
        z, a, z_next, tr, va = transition_data(v, inputs, representation)
        clean = v["transition_model.fit_episodes"] == "train"
        if prefit is not None:
            _check_prefit(v, prefit, z, a, z_next, tr, clean)
            model, record = prefit["label"]
        elif clean:
            model, record = _fit(v, z[tr], a[tr], z_next[tr])
        else:
            model, record = _fit(v, z, a, z_next)
        metrics = {"label_model": record, "fit_episodes": v["transition_model.fit_episodes"]}
        if tr is not None:
            if clean:
                evaluation, note = model, "the label model, fitted on the train split only, scored on the held-out episodes"
            else:
                evaluation = (prefit["evaluation"][0] if prefit is not None and prefit.get("evaluation") is not None
                              else _fit(v, z[tr], a[tr], z_next[tr])[0])
                note = "evaluation model fitted on the train split; the label model saw every episode"
            metrics["held_out"] = evaluation.evaluate(z[va], a[va], z_next[va])
            metrics["held_out"]["note"] = note
        model.save(run.path(MODEL_CHECKPOINT))
        run.metrics(metrics)
        run.log(f"fitted {v['transition_model.kind']} transition model on {len(z)} transitions")
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
