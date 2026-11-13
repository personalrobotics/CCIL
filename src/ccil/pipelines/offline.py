"""Offline pipeline from trajectories to immutable candidates and selections.

Generation uses only an admitted recovery length and writes a valid empty
candidate set when none is admitted. This module imports no policy trainer.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil

from pathlib import Path

import numpy as np

from ..config import ConfigError
from ..config.params import mlp_settings_problem
from ..config.params import BOUND_PROFILES
from ..core import (TRANSITION_MODELS, ActionBounds, BackwardEuler, ExactPreimage, Finite, GenerationFailed,
                    IdentityRepresentation, ModelMisfit, ModelResidual, NoisyAction, NoisyActionPreimage, SupportRadius,
                    Uniform, generate_labels, load_transition_model, transition_model_class, transitions)
from ..data.prepare import prepare_stage
from ..errors import InputError
from ..runs import layout
from ..runs.provenance import sha256
from ..core.certificates import (PROFILES, BackTrackProfile, ContinuityProfile, Declared, DisturbedActionProfile, Provided,
                                 Sources, certify, read_reference_dynamics)
from ..core.batch import RequestError
from ..core.representation import JOIN, widths
from ..core.transition import CheckpointError, binding_problems, checkpoint_meta
from ..core.calibrate import calibrate, reference_audit
from ..core.curate import CurationPolicy, curate
from ..core.diagnose import diagnose, timing
from ..core.quality import SourceDisplacement, demo_metric
from ..schema import SchemaError
from ..schema.certificates import CLASSIFICATIONS
from ..schema.selection import REASON_CODES
from ..schema import read_trajectories, write_candidates, write_certificates, write_selection
from ..schema.audit import write_audit
from ..schema.calibration import write_calibration
from ..schema.diagnosis import read_diagnosis, write_diagnosis
from .stages import StageResult, run_stage

MODEL_CHECKPOINT = f"{layout.CHECKPOINTS}/transition_model.npz"
# With fit_episodes all: the model fitted on the train split only, which the diagnosis scores on held-out episodes.
EVALUATION_CHECKPOINT = f"{layout.CHECKPOINTS}/evaluation_model.npz"


def build_generator(v, action_noise_std=None):
    """The configured generator; action_noise_std, when given, is the resolved per-dimension standard deviation
    (noise_std below), otherwise generation.action_noise_std as written."""
    method = v["generation.method"]
    sigma = v["generation.action_noise_std"] if action_noise_std is None else action_noise_std
    if method in ("backward_euler", "backward_euler_fast"):
        return BackwardEuler()
    if method == "forward_euler":
        return ExactPreimage(max_iterations=v["generation.max_iterations"] or 50, tolerance=v["generation.root_tolerance"])
    if method == "noisy_action":
        return NoisyAction(v["generation.labels_per_transition"], sigma)
    if method == "noisy_action_fe":
        return NoisyActionPreimage(v["generation.labels_per_transition"], sigma,
                                   max_iterations=v["generation.max_iterations"] or 50,
                                   tolerance=v["generation.root_tolerance"])
    raise ConfigError([f"generation.method {method!r} is not implemented"])


def noise_std(v, dataset_path, split, representation):
    """generation.action_noise_std in action units: as written, or under action_noise_scale training_std its multiple of
    each action dimension's standard deviation over the training transitions (every episode without a split)."""
    if v["generation.action_noise_scale"] != "training_std" or v["generation.action_noise_std"] is None:
        return None
    batch = transitions(read_trajectories(dataset_path), representation)
    train = np.ones(len(batch.a), bool) if split is None else np.isin(batch.episode_id.astype(str), split["train"])
    std = np.asarray(batch.a, np.float64)[train].std(axis=0)
    if not (std > 0).all():
        raise ConfigError([f"generation.action_noise_scale training_std: training action dimensions "
                           f"{np.flatnonzero(std <= 0).tolist()} do not vary, so a multiple of their std is no noise"])
    return [float(v["generation.action_noise_std"] * x) for x in std]


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
    else:  # a score profile's reference dynamics serve its calibration (labels.calibrate), not certification
        return PROFILES[name](**nb), {"references": {}}
    return profile, {"references": {REFERENCE: Path(ref)} if ref is not None else {}}


def build_policy(v) -> CurationPolicy:
    """The curation policy of curate.*: admission follows the certificate profile (none: every accepted row).

    A calibrated score profile admits bounds: its calibrated rows, at calibration.level.
    """
    name = v["certificate.profile"]
    if name is None:
        return CurationPolicy(admit="accepted")
    bound = name in BOUND_PROFILES or v["calibration.level"] is not None
    return CurationPolicy(admit="bound" if bound else "score", tolerance=v["curate.tolerance"],
                          quantile=v["curate.quantile"], population=v["curate.population"])


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
    noise_missing = v["generation.method"] in ("noisy_action", "noisy_action_fe") and not (
        v["generation.labels_per_transition"] and v["generation.action_noise_std"])
    if noise_missing:
        errors.append(f"generation.method {v['generation.method']} requires generation.labels_per_transition and "
                      "generation.action_noise_std")
    for key in ("certificate.reference_dynamics", "audit.reference_dynamics"):
        if v[key] is not None:
            try:
                read_reference_dynamics(v[key])
            except (OSError, SchemaError) as e:
                errors.append(f"{key} {v[key]}: {str(e).splitlines()[-1].strip()}")
    if v["transition_model.checkpoint"] is not None and not Path(v["transition_model.checkpoint"]).is_file():
        errors.append(f"transition_model.checkpoint {v['transition_model.checkpoint']} does not exist")
    if v["transition_model.fit_episodes"] == "train" and not v["data.validation_fraction"]:
        errors.append("transition_model.fit_episodes train fits the label model on the train split: it needs "
                      "data.validation_fraction > 0")
    if v["transition_model.direction"] == "backward" and v["generation.method"] != "backward_model":
        errors.append("transition_model.direction backward fits s_t from (s_{t+1}, a_t); only generation.method "
                      "backward_model requires a fitted backward transition model")
    if v["transition_model.trainer_version"] == "2" and v["transition_model.kind"] == "mlp":
        from ..models.mlp_batched import SUPPORTED
        if (v["transition_model.lipschitz_type"] or "none") not in SUPPORTED:
            errors.append(f"transition_model.trainer_version 2 fits lipschitz types {list(SUPPORTED)}; "
                          f"{v['transition_model.lipschitz_type']} needs trainer_version 1")
    for check in (build_model, build_generator):
        if v["generation.method"] is None or (check is build_generator and noise_missing):
            continue  # already reported
        try:
            check(v)
        except ConfigError as e:
            errors += e.errors
        except InputError as e:
            errors.append(str(e))
    if errors:
        raise ConfigError(errors)


def _fit(v, z, a, z_next, parts=None):
    """Fit one transition model with the configured engine: (model, fit record). parts, the representation parts'
    widths in order, lay out a history_shift model's slots."""
    model = build_model(v)
    if getattr(model, "structured", False):
        model.set_parts(parts)
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
    group requested, never an eager fallback of a compiled one), or None."""
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
    """Fit the label model and score an independently fitted model on held-out episodes.

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
    loaded = v["transition_model.checkpoint"]
    if loaded is not None:
        inputs["checkpoint"] = Path(loaded)

    def body(run):
        z, a, z_next, tr, va = transition_data(v, inputs, representation)
        # A fitted representation (PCA) is bound by its checkpoint's hash: the same kind and shape with another
        # mean or basis is another coordinate system. The identity representation has no parameters (None).
        binding = {"representation": {"kind": representation.kind, "modality": v["representation.modality"],
                                      "checkpoint_sha256": sha256(inputs["representation"])
                                      if "representation" in inputs else None},
                   "state_dim": int(z.shape[1]), "act_dim": int(a.shape[1]),
                   "direction": v["transition_model.direction"],
                   "dt_ctrl": timing(read_trajectories(inputs["dataset"]))["dt_ctrl"]}
        if JOIN in v["representation.modality"]:  # an ordered list of modalities: bound by each part's width too
            binding["representation"]["widths"] = widths(read_trajectories(inputs["dataset"]), v["representation.modality"])
        parts = list(binding["representation"].get("widths", {}).values()) or [int(z.shape[1])]
        if v["transition_model.structure"] != "none":  # bound to its slot layout: steps and each part's width
            binding["representation"]["history"] = {"structure": v["transition_model.structure"],
                                                    "steps": v["transition_model.history_steps"], "parts": parts}
        if loaded is not None:  # an externally supplied model: bound to this run, scored, copied whole
            model = _load_bound(v, inputs["checkpoint"], binding)
            metrics = {"label_model": {"source": "loaded", "checkpoint_sha256": sha256(inputs["checkpoint"])},
                       "fit_episodes": None}
            if tr is not None:
                metrics["held_out"] = model.evaluate(z[va], a[va], z_next[va])
                metrics["held_out"]["note"] = ("the loaded model scored on the held-out episodes; whether its own fit "
                                               "saw them is not recorded here")
            shutil.copyfile(inputs["checkpoint"], run.path(MODEL_CHECKPOINT))
            metrics |= _measured_lipschitz(v, run.path(MODEL_CHECKPOINT))
            run.metrics(metrics)
            run.log(f"loaded the {v['transition_model.kind']} transition model {inputs['checkpoint']}")
            return
        clean = v["transition_model.fit_episodes"] == "train"
        if prefit is not None:
            _check_prefit(v, prefit, z, a, z_next, tr, clean)
            model, record = prefit["label"]
        elif clean:
            model, record = _fit(v, z[tr], a[tr], z_next[tr], parts)
        else:
            model, record = _fit(v, z, a, z_next, parts)
        metrics = {"label_model": record, "fit_episodes": v["transition_model.fit_episodes"]}
        if tr is not None:
            if clean:
                evaluation, note = model, "the label model, fitted on the train split only, scored on the held-out episodes"
            else:
                evaluation = (prefit["evaluation"][0] if prefit is not None and prefit.get("evaluation") is not None
                              else _fit(v, z[tr], a[tr], z_next[tr], parts)[0])
                note = "evaluation model fitted on the train split; the label model saw every episode"
            metrics["held_out"] = evaluation.evaluate(z[va], a[va], z_next[va])
            metrics["held_out"]["note"] = note
            if not clean:
                evaluation.binding = binding
                evaluation.save(run.path(EVALUATION_CHECKPOINT))
        model.binding = binding
        model.save(run.path(MODEL_CHECKPOINT))
        metrics |= _measured_lipschitz(v, run.path(MODEL_CHECKPOINT))
        run.metrics(metrics)
        run.log(f"fitted {v['transition_model.kind']} transition model on {len(z)} transitions")
    return run_stage("transition.fit", cfg, inputs, body,
                     components={"representation": representation, "transition_model": build_model(v)})


def _measured_lipschitz(v, path: Path) -> dict:
    """An MLP checkpoint's weight-based Lipschitz bound of f_hat, measured from the saved weights (a measurement,
    never a certificate); {} for other kinds."""
    if v["transition_model.kind"] != "mlp":
        return {}
    model = load_transition_model(path, device="cpu")
    out = {"scope": "the residual map f_hat in raw state and action units, recomputed from the saved checkpoint with "
                    "its normalization, output scaling and mask; a measurement, not a certificate"}
    if getattr(model, "structured", False):
        out["scope"] = ("the learned head of a history_shift model (the newest slots' residual) in raw state and action "
                        "units, recomputed from the saved checkpoint with its normalization, output scaling and mask; "
                        "a measurement, not a certificate. known_shift is the exact constant of the older slots' "
                        "residual, which is not learned")
        out["known_shift"] = {"state": model.shift_lipschitz(np.ones(model.net.state_dim)), "action": 0.0}
    for maps in ("state", "action"):
        try:
            b = model.lipschitz_bound(maps, np.ones(model.net.state_dim), np.ones(model.net.act_dim))
            out[maps] = {"value": b["value"], "factors": b["factors"], "activation": b["activation"]}
        except ValueError as e:
            out[maps] = {"value": None, "reason": str(e)}
    return {"lipschitz_measured": out}


def _load_bound(v, path: Path, binding: dict):
    """The checkpoint at path, refused (CheckpointError) unless its kind and binding match this run."""
    meta = checkpoint_meta(path, "transition_model")
    problems = binding_problems(meta.get("binding"), binding)
    if meta.get("kind") != v["transition_model.kind"]:
        problems.insert(0, f"kind: the checkpoint is {meta.get('kind')!r}, transition_model.kind is "
                           f"{v['transition_model.kind']!r}")
    if problems:
        raise CheckpointError(f"{path} cannot be loaded into this run:\n  " + "\n  ".join(problems))
    return load_transition_model(path, device=_device(v))


@contextlib.contextmanager
def _jacobian_budget():
    """A Jacobian budget smaller than one row is a configuration error, not a failure of the computation."""
    try:
        yield
    except ValueError as e:
        if type(e).__name__ == "JacobianBudgetError":  # models.mlp; torch is imported only with the mlp extra
            raise ConfigError([str(e)]) from None
        raise


def _split(prepared: StageResult):
    path = prepared.out(layout.SPLIT)
    return (path, json.loads(path.read_text())) if path.exists() else (None, None)


def _device(v):
    # compute.device places torch models; the numpy linear model always runs on cpu.
    return "cpu" if v["transition_model.kind"] == "linear" else v["compute.device"]


def diagnose_stage(cfg, prepared: StageResult, fitted: StageResult, representation, extra_inputs=None) -> StageResult:
    """Write the data-only recovery-length decision to ``diagnosis/diagnosis.json``."""
    v = cfg.values
    split_path, split = _split(prepared)
    inputs = {"dataset": prepared.out(layout.DATASET), "transition_model": fitted.out(MODEL_CHECKPOINT),
              **(extra_inputs or {})}
    if split_path is not None:
        inputs["split"] = split_path
    if fitted.out(EVALUATION_CHECKPOINT).exists():
        inputs["evaluation_model"] = fitted.out(EVALUATION_CHECKPOINT)

    sigma = noise_std(v, inputs["dataset"], split, representation)

    def body(run):
        dataset = read_trajectories(inputs["dataset"])
        metric = demo_metric(dataset, split, representation)
        model = load_transition_model(inputs["transition_model"], device=_device(v))
        if "evaluation_model" in inputs:
            evaluation = load_transition_model(inputs["evaluation_model"], device=_device(v))
        else:  # fit_episodes train: the label model never saw the held-out episodes
            evaluation = model if split is not None else None
        held_out = None
        if split is not None:
            z, a, z_next, _, va = transition_data(v, inputs, representation)
            held_out = (z[va], a[va], z_next[va])
        params = {k.split(".", 1)[1]: v[k] for k in v if k.startswith("diagnose.")}
        requested = v["generation.recovery_steps"]
        with _jacobian_budget():
            doc = diagnose(dataset, split, representation, model, evaluation, held_out, build_generator(v, sigma),
                           build_validators(v), metric, requested=requested if requested == "auto" else int(requested),
                           params=params, seed=v["run.seed"], inputs={k: sha256(p) for k, p in inputs.items()},
                           profile=v["certificate.profile"], reference=v["certificate.reference_dynamics"] is not None,
                           labels_per_transition=v["generation.labels_per_transition"])
        write_diagnosis(doc, run.path(layout.DIAGNOSIS))
        gate = doc["gates"][0]
        run.metrics({"admitted": doc["admitted"], "admitted_by": doc["admitted_by"], "reason": doc["reason"],
                     "tau_demo": doc["tau_demo"]["value"], "yield": gate["yield"]["value"],
                     "displacement": gate["displacement"]["eta"], "novelty": gate["novelty"]["eta"],
                     "fit": gate["fit"]["ratio"], "trust": gate["trust"]["evidence"]})
        run.log(f"diagnosis: admitted {doc['admitted']}" + (f" ({doc['reason']})" if doc["reason"] else ""))
    return run_stage("labels.diagnose", cfg, inputs, body,
                     components={"representation": representation, "transition_model": build_model(v),
                                 "generator": build_generator(v, sigma)})


def labels_stage(cfg, prepared: StageResult, fitted: StageResult, representation, extra_inputs=None,
                 diagnosed: StageResult | None = None) -> StageResult:
    """labels.generate: write labels/candidates.npz at the admitted recovery length, or a zero-row set.

    Without a diagnosis, generation.recovery_steps must declare the length. The
    admitted length, not the diagnosis file, is part of the identity: the diagnosis also reports evidence
    (such as the certificate's class) that does not change the labels.
    """
    v = cfg.values
    if diagnosed is None and v["generation.recovery_steps"] == "auto":
        raise ConfigError(["generation.recovery_steps auto needs labels.diagnose, which this pipeline does not run; "
                           "set generation.recovery_steps 1"])
    split_path, split = _split(prepared)
    inputs = {"dataset": prepared.out(layout.DATASET), "transition_model": fitted.out(MODEL_CHECKPOINT), **(extra_inputs or {})}
    if split_path is not None:
        inputs["split"] = split_path
    admitted = 1 if diagnosed is None else read_diagnosis(diagnosed.out(layout.DIAGNOSIS))["admitted"]
    sigma = noise_std(v, inputs["dataset"], split, representation)

    def body(run):
        dataset = read_trajectories(inputs["dataset"])
        model = load_transition_model(inputs["transition_model"], device=_device(v))
        if hasattr(model, "jacobian_memory_bytes"):
            model.jacobian_memory_bytes = v["compute.jacobian_memory_mb"] * 2**20
        metric = demo_metric(dataset, split, representation)
        # d_source is always available; the brute-force nearest-demonstration distance is opt-in.
        validators = build_validators(v) + [SourceDisplacement(metric)]
        with _jacobian_budget():
            cands = generate_labels(dataset, representation, model, build_generator(v, sigma), validators,
                                    run_id=run.run_id, seed=v["run.seed"], weighting=Uniform(),
                                    action_chunk_size=v["generation.action_chunk_size"],
                                    inputs={name: sha256(path) for name, path in inputs.items()},
                                    rows=None if admitted is not None else [])
        write_candidates(cands, run.path(layout.CANDIDATES))
        counts = {r: int(cands.reasons[:, i].sum()) for i, r in enumerate(cands.reason_codes)}
        run.metrics({"candidates": int(len(cands.accepted)), "accepted": int(cands.accepted.sum()),
                     "acceptance_rate": float(cands.accepted.mean()) if len(cands.accepted) else 0.0,
                     "rejections_by_reason": counts, "generator": cands.generator, "recovery_steps": admitted,
                     "jacobian_chunking": getattr(model, "jacobian_chunking", None)})  # its last call
        run.log(f"generated {len(cands.accepted)} candidates with {cands.generator['name']}; "
                f"accepted {int(cands.accepted.sum())}" + ("" if admitted is not None else
                                                            " (no recovery length admitted: an empty synthetic view)"))
    return run_stage("labels.generate", cfg, inputs, body,
                     components={"representation": representation, "transition_model": build_model(v),
                                 "generator": build_generator(v, sigma), "weighting": Uniform()},
                     parameters={"recovery_steps": admitted})


def certify_stage(cfg, prepared: StageResult, fitted: StageResult, labeled: StageResult, representation, profile,
                  references: dict | None = None, state_scale=None, action_scale=None, extra_inputs=None) -> StageResult:
    """Write candidate quality evidence to ``labels/certificates.npz``.

    The candidates are an input and are never modified; which rows to use is
    labels.curate's decision. references are artifacts bound providers compute
    constants from; they are stage inputs. The profile's specification and the
    scales enter the stage identity.
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
            cert = certify(sources, profile, state_scale=state_scale, action_scale=action_scale)
        write_certificates(cert, run.path(layout.CERTIFICATES))
        run.metrics({"certificates": int(len(cert.classification)), "profile": profile.name,
                     "by_classification": {c: int((cert.classification == c).sum()) for c in CLASSIFICATIONS}})
        run.log(f"certified {len(cert.classification)} candidates with {profile.name}")
    scales = {k: None if x is None else [float(e) for e in x] for k, x in (("state", state_scale), ("action", action_scale))}
    return run_stage("labels.certify", cfg, inputs, body, components={"representation": representation},
                     parameters={"profile": profile.spec(), "scales": scales})


def calibrate_stage(cfg, prepared: StageResult, labeled: StageResult, certified: StageResult) -> StageResult:
    """Write the episode-level held-out calibration to ``labels/calibration.json``.

    Reference-assisted: the observed errors come from certificate.reference_dynamics.
    """
    v = cfg.values
    split_path, split = _split(prepared)
    if split is None or not split["validation"]:
        raise ConfigError(["calibration.level calibrates on the held-out episodes; this split has none"])
    inputs = {"candidates": labeled.out(layout.CANDIDATES), "certificates": certified.out(layout.CERTIFICATES),
              "split": split_path, "reference": Path(v["certificate.reference_dynamics"])}

    def body(run):
        try:
            cal = calibrate(inputs["candidates"], inputs["certificates"], inputs["reference"], split["validation"],
                            level=float(v["calibration.level"]))
        except RequestError as e:
            raise ConfigError([str(e)]) from None
        write_calibration(cal, run.path(layout.CALIBRATION))
        b = cal["bound"]
        run.metrics({"level": cal["level"], "population_rows": cal["population"]["rows"], "episodes": b["n"], "k": b["k"],
                     "bound": b["value"]})
        run.log(f"calibrated {cal['certificate']['profile']} on {b['n']} held-out episodes ({cal['population']['rows']} rows) "
                f"at level {cal['level']}: bound {b['value']}" + ("" if b["value"] is not None else
                                                                  f" (k = {b['k']} > n; more held-out episodes give one)"))
    return run_stage("labels.calibrate", cfg, inputs, body)


def audit_stage(cfg, labeled: StageResult, certified: StageResult | None, calibrated: StageResult | None,
                curated: StageResult) -> StageResult:
    """Annotate the decided selection with reference dynamics in ``labels/audit.json``.

    Nothing reads it back: it is not an input of any other stage.
    """
    v = cfg.values
    inputs = {"selection": curated.out(layout.SELECTION), "candidates": labeled.out(layout.CANDIDATES),
              "reference": Path(v["audit.reference_dynamics"])}
    if certified is not None:
        inputs["certificates"] = certified.out(layout.CERTIFICATES)
    if calibrated is not None:
        inputs["calibration"] = calibrated.out(layout.CALIBRATION)

    def body(run):
        try:
            doc = reference_audit(inputs["selection"], inputs["candidates"], inputs["reference"],
                                  inputs.get("certificates"), inputs.get("calibration"), v["representation.modality"])
        except RequestError as e:
            raise ConfigError([str(e)]) from None
        write_audit(doc, run.path(layout.AUDIT))
        run.metrics({"selected": doc["groups"]["selected"], "bounds": doc["bounds"], "tolerance": doc["tolerance"],
                     "calibrated": doc["calibrated"], "evidence_mode": doc["selection"]["evidence_mode"]})
        run.log(f"audited the selection against {inputs['reference'].name}")
    return run_stage("labels.audit", cfg, inputs, body)


def curate_stage(cfg, labeled: StageResult, certified: StageResult | None, policy: CurationPolicy,
                 calibrated: StageResult | None = None) -> StageResult:
    """Write one policy's candidate selection to ``labels/selection.npz``.

    No input is modified, so a new tolerance writes a new selection run
    and reuses the candidate, certificate, and calibration runs unchanged.
    """
    v = cfg.values
    inputs = {"candidates": labeled.out(layout.CANDIDATES)}
    if certified is not None:
        inputs["certificates"] = certified.out(layout.CERTIFICATES)
    if calibrated is not None:
        inputs["calibration"] = calibrated.out(layout.CALIBRATION)

    def body(run):
        try:
            sel = curate(inputs["candidates"], inputs.get("certificates"), policy, v["generation.evidence_mode"],
                         inputs.get("calibration"))
        except RequestError as e:
            raise ConfigError([str(e)]) from None
        write_selection(sel, run.path(layout.SELECTION))
        counts = {code: int(sel.reasons[:, i].sum()) for i, code in enumerate(REASON_CODES)}
        run.metrics({"rows": int(len(sel.mask)), "selected": int(sel.mask.sum()), "policy": sel.policy,
                     "evidence_mode": sel.evidence_mode, "evidence_allowed": sel.evidence_allowed,
                     "threshold": sel.threshold, "reasons": counts, "calibrated": int(sel.calibrated.sum()),
                     "caveats": sel.caveats})
        run.log(f"selected {int(sel.mask.sum())} of {len(sel.mask)} candidates with {sel.policy}")
    return run_stage("labels.curate", cfg, inputs, body, parameters={"policy": policy.spec()})


def _identity_representation(cfg):
    if cfg["representation.kind"] != "identity":
        raise ConfigError(["offline generation uses representation.kind=identity; learned representations are "
                           "fitted by the integrated pipeline (ccil pipeline integrated)"])
    return IdentityRepresentation(cfg["representation.modality"])


def diagnose_offline(cfg) -> list[StageResult]:
    """`ccil diagnose`: prepare, fit, diagnose; no candidates. Generate on the same configuration reuses all three."""
    preflight(cfg)
    representation = _identity_representation(cfg)
    prepared = prepare_stage(cfg)
    fitted = fit_transition_stage(cfg, prepared, representation)
    return [prepared, fitted, diagnose_stage(cfg, prepared, fitted, representation)]


def generate_candidates(cfg) -> list[StageResult]:
    """`ccil generate`: diagnose_offline's stages, then the candidates and, with a certificate profile, their
    quality evidence. No selection: `ccil curate` writes those, any number per candidate set."""
    prepared, fitted, diagnosed = diagnose_offline(cfg)
    labeled = labels_stage(cfg, prepared, fitted, _identity_representation(cfg), diagnosed=diagnosed)
    stages = [prepared, fitted, diagnosed, labeled]
    if cfg["certificate.profile"] is not None:
        profile, kwargs = build_certificate(cfg.values)
        stages.append(certify_stage(cfg, prepared, fitted, labeled, _identity_representation(cfg), profile, **kwargs))
    return stages


def curate_candidates(cfg, generated: list[StageResult], policy: CurationPolicy | None = None) -> list[StageResult]:
    """`ccil curate`: on generate_candidates' stages, the calibration (calibration.level), one selection, and the
    reference audit (audit.reference_dynamics). The policy defaults to curate.*."""
    prepared, labeled, certified = stage(generated, "data.prepare"), stage(generated, "labels.generate"), \
        stage(generated, "labels.certify")
    stages, calibrated = [], None
    if cfg["calibration.level"] is not None:
        calibrated = calibrate_stage(cfg, prepared, labeled, certified)
        stages.append(calibrated)
    curated = curate_stage(cfg, labeled, certified, policy or build_policy(cfg.values), calibrated)
    stages.append(curated)
    if cfg["audit.reference_dynamics"] is not None:
        stages.append(audit_stage(cfg, labeled, certified, calibrated, curated))
    return stages


def generate_offline(cfg) -> list[StageResult]:
    """`ccil labels generate`: generate and curate in one command; return the stage results (reused or new).

    In order: prepare, fit, labels, [certify, [calibrate,]] curate, diagnose, [audit].
    """
    prepared, fitted, diagnosed, labeled, *certified = generate_candidates(cfg)
    curated = curate_candidates(cfg, [prepared, fitted, diagnosed, labeled, *certified])
    # labels.diagnose ran before labels.generate; it is listed after curation so the first three results keep their places.
    audit = [r for r in curated if r.stage == "labels.audit"]
    return [prepared, fitted, labeled, *certified, *[r for r in curated if r.stage != "labels.audit"], diagnosed, *audit]


def stage(results: list[StageResult], name: str) -> StageResult | None:
    """The result of the named stage among a pipeline's results, or None when it did not run."""
    return next((r for r in results if r.stage == name), None)


RESULT_SCHEMA, RESULT_VERSION = "ccil.labels-result", "1.0"
# Each artifact of a run, by name: the stage that writes it and its path in that stage's run directory.
RESULT_ARTIFACTS = {"dataset": ("data.prepare", layout.DATASET), "split": ("data.prepare", layout.SPLIT),
                    "transition_model": ("transition.fit", MODEL_CHECKPOINT),
                    "diagnosis": ("labels.diagnose", layout.DIAGNOSIS),
                    "candidates": ("labels.generate", layout.CANDIDATES),
                    "certificates": ("labels.certify", layout.CERTIFICATES),
                    "calibration": ("labels.calibrate", layout.CALIBRATION),
                    "selection": ("labels.curate", layout.SELECTION), "audit": ("labels.audit", layout.AUDIT)}


def labels_result(cfg, results: list[StageResult]) -> dict:
    """The machine-readable result of `ccil labels generate --result` (ccil.labels-result 1.0): the resolved
    configuration record, each stage's run directory and whether it was reused, and the absolute path of each
    artifact the run produced (an artifact whose stage did not run, or that its stage did not write, is absent)."""
    artifacts = {}
    for name, (stage_name, rel) in RESULT_ARTIFACTS.items():
        r = stage(results, stage_name)
        if r is not None and r.out(rel).exists():
            artifacts[name] = str(r.out(rel).resolve())
    return {"schema": RESULT_SCHEMA, "version": RESULT_VERSION, "config": cfg.to_dict(),
            "stages": [{"stage": r.stage, "run_dir": str(r.run_dir.resolve()), "reused": r.reused} for r in results],
            "artifacts": artifacts}


def write_labels_result(result: dict, path) -> None:
    """Write a labels result whole: to a temporary file beside it, then renamed into place."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
