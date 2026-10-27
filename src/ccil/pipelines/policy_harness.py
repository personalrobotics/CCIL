"""Experimental vector-policy harness: ccil policy train (#89).

    data.prepare -> transition.fit -> labels.generate [-> labels.certify] -> policy.train -> policy.evaluate

Reference infrastructure for reproducing the paper's BC, NoiseBC, and CCIL
arms (#24); CCIL itself ends at the supervision artifacts, and users train
downstream policies with any trainer (docs/supervision.md). The label stages
are the offline pipeline's, reused when complete. policy.train builds the
supervision (ccil.supervision) from those artifacts and trains the
experimental vector MLP (ccil.training.vector_bc); policy.evaluate reports
offline metrics. The trainer needs the ccil[torch] extra and is imported only
here, never by ccil labels generate.
"""
from __future__ import annotations

from ..config import ConfigError
from ..core import IdentityRepresentation
from ..runs import layout
from ..runs.writer import dump_json
from ..supervision import build_supervision
from .offline import generate_offline
from .stages import StageResult, run_stage

POLICY_CHECKPOINT = f"{layout.CHECKPOINTS}/policy.npz"
SUPERVISION = "supervision.json"
NOISY = ("noisy_action", "noisy_action_fe")


class PolicyMismatch(RuntimeError):
    """A policy trained elsewhere does not belong to this unit, or to its record: a study-runner defect."""


class VectorMLP:
    """Component identity of the experimental trainer (stage identity)."""

    name, version = "vector_mlp", "1"


def _trainer_module():
    try:
        from ..training import vector_bc
    except ImportError as e:
        raise ConfigError([f"ccil policy train needs the ccil[torch] extra (Python 3.11+): pip install 'ccil[torch]' "
                           f"({e})"]) from None
    return vector_bc


def harness_problems(v) -> list[str]:
    problems = [f"{k} is required by ccil policy train" for k in ("policy.batch_size", "policy.epochs") if v[k] is None]
    if v["representation.kind"] != "identity":
        problems.append("the vector-policy harness trains on representation.kind=identity")
    if v["policy.checkpoint_selection"] == "validation" and not (v["policy.exclude_validation_episodes"]
                                                                and v["data.validation_fraction"]):
        problems.append("policy.checkpoint_selection validation needs held-out validation episodes: "
                        "policy.exclude_validation_episodes true and data.validation_fraction > 0")
    recipe_set = [k for k in ("optimizer", "weight_decay", "lr_schedule", "warmup_steps", "step_budget",
                              "grad_clip_norm", "ema_decay", "checkpoint_selection", "validation_interval_steps")
                  if v[f"policy.{k}"] != {"optimizer": "adam", "weight_decay": 0.0, "lr_schedule": "constant",
                                          "warmup_steps": 0, "checkpoint_selection": "last"}.get(k)]
    if v["policy.warmup_steps"] and v["policy.lr_schedule"] == "constant":
        problems.append("policy.warmup_steps applies to lr_schedule cosine; a constant schedule has no warm-up")
    if recipe_set and v["policy.trainer_version"] != "2":
        problems.append(f"{['policy.' + k for k in recipe_set]} need policy.trainer_version 2 (#115)")
    if v["certificate.profile"] is not None and v["policy.method"] != "ccil":
        problems.append(f"certificate.profile certifies labels; policy.method {v['policy.method']} trains on "
                        "demonstrations only")
    return problems


def settings_of(v) -> dict:
    return {"hidden_layers": list(v["policy.hidden_layers"]), "learning_rate": v["policy.learning_rate"],
            "batch_size": v["policy.batch_size"], "epochs": v["policy.epochs"],
            "observation_scaler": v["policy.observation_scaler"], "action_scaler": v["policy.action_scaler"],
            "balance_labels": v["generation.labels_per_transition"] if v["policy.balance_noisy_labels"] else None,
            **{k: v[f"policy.{k}"] for k in ("optimizer", "weight_decay", "lr_schedule", "warmup_steps", "step_budget",
                                             "grad_clip_norm", "ema_decay", "checkpoint_selection",
                                             "validation_interval_steps")}}


def unit_supervision(cfg):
    """A unit's label stages (reused when complete) and what its policy trains on.

    Returns (stage results, inputs, supervision, NoiseBC or None, held-out supervision or None). Study runners
    call it to train many units together on exactly the supervision run_policy_harness would build.
    """
    v = cfg.values
    problems = harness_problems(v)
    if problems:
        raise ConfigError(problems)
    from ..training.noise_bc import NoiseBC
    results = generate_offline(cfg)
    prepared, labeled = results[0], results[2]
    certified = results[3] if len(results) == 4 else None
    rep = IdentityRepresentation(v["representation.modality"])
    inputs = {"dataset": prepared.out(layout.DATASET)}
    split_path = prepared.out(layout.SPLIT)
    if split_path.exists() and v["policy.exclude_validation_episodes"]:
        inputs["split"] = split_path
    if v["policy.method"] == "ccil":
        inputs["candidates"] = labeled.out(layout.CANDIDATES)
        if certified is not None:
            inputs["certificates"] = certified.out(layout.CERTIFICATES)
    sup = build_supervision(inputs["dataset"], v["policy.method"], representation=rep,
                            candidates=inputs.get("candidates"), certificates=inputs.get("certificates"),
                            split=inputs.get("split"), demonstration_weight=v["policy.demonstration_weight"])
    noise = None
    if v["policy.method"] == "noise_bc":
        noise = NoiseBC(v["policy.noise_model"], v["policy.noise_scale"], seed=v["run.seed"],
                        mask_probability=v["policy.noise_mask_probability"],
                        units="standardized observation units (after the observation scaler)")
    held = None
    if v["policy.checkpoint_selection"] == "validation":  # the held-out episodes' demonstrations
        held = build_supervision(inputs["dataset"], "bc", representation=rep, split=split_path, part="validation",
                                 demonstration_weight=v["policy.demonstration_weight"])
    return results, inputs, sup, noise, held


def run_policy_harness(cfg, trained=None, unit=None) -> list[StageResult]:
    """Run the label stages (reused when complete), then policy.train and policy.evaluate.

    trained, from a study runner that trained many units together (vector_bc_v2.train_group), supplies this
    unit's (policy, record) instead of training here; it needs policy.trainer_version 2. unit is the runner's
    unit_supervision(cfg), so the supervision is not built twice.
    """
    v = cfg.values
    vector_bc = _trainer_module()
    results, inputs, sup, noise, held = unit if unit is not None else unit_supervision(cfg)
    rep = IdentityRepresentation(v["representation.modality"])
    split_path = results[0].out(layout.SPLIT)

    def train(run):
        dump_json(sup.manifest(), run.path(SUPERVISION))
        # An interrupted policy.train restarts from scratch: training is deterministic, so a rerun reproduces it.
        if trained is not None:
            if v["policy.trainer_version"] != "2":
                raise ConfigError(["a trained policy from a group comes from trainer v2: set policy.trainer_version 2"])
            policy, record = trained
            from ..training import vector_bc_v2
            want = vector_bc_v2.training_identity_v2(sup, settings_of(v), v["run.seed"], noise, held)["sha256"]
            if record.get("training_identity_sha256") != want:
                raise PolicyMismatch(f"the trained policy's training identity {record.get('training_identity_sha256')} "
                                     f"is not this unit's {want}")
            if record.get("parameters_sha256") != vector_bc_v2.parameters_sha256(policy):
                raise PolicyMismatch("the trained policy's weights are not the ones its record describes")
            from .offline import requested_execution_problem
            problem = requested_execution_problem(record, v)
            if problem:
                raise PolicyMismatch(f"the trained policy {problem}")
        elif v["policy.trainer_version"] == "2":
            from ..training import vector_bc_v2
            [(policy, record)] = vector_bc_v2.train_group([vector_bc_v2.Job(sup, v["run.seed"], noise, held)],
                                                          settings_of(v),
                                                          device=v["compute.device"],
                                                          deterministic=v["compute.deterministic"])
        else:
            policy, record = vector_bc.train_vector_bc(sup, settings_of(v), seed=v["run.seed"], device=v["compute.device"],
                                                       deterministic=v["compute.deterministic"], noise=noise)
        policy.save(run.path(POLICY_CHECKPOINT), None, record["epochs_run"],
                    {"supervision": sup.manifest()["inputs"], "seed": v["run.seed"], "method": v["policy.method"],
                     "trainer_version": record.get("trainer_version", "1"),
                     "execution_sha256": record.get("execution", {}).get("execution_sha256"),
                     "training_identity_sha256": record["training_identity_sha256"]})
        run.metrics({"experimental": True, "supervision": sup.manifest(), "training": record})
        run.log(f"trained {record['trainer']} ({v['policy.method']}) on {record['rows']} rows for "
                f"{record['epochs_run']} epochs")
    train_result = run_stage("policy.train", cfg, inputs, train, components={"trainer": VectorMLP()})
    eval_inputs = {**inputs, "policy": train_result.out(POLICY_CHECKPOINT)}
    if split_path.exists():
        eval_inputs["split"] = split_path  # evaluated on both splits even when training used every episode

    def evaluate(run):
        import json

        import numpy as np

        from ..core import transitions
        from ..schema import read_trajectories
        policy, _, _ = vector_bc.VectorBCPolicy.load(eval_inputs["policy"])
        batch = transitions(read_trajectories(eval_inputs["dataset"]), rep)
        sq = ((policy.predict(batch.z) - np.asarray(batch.a, np.float64)) ** 2).sum(axis=1)
        metrics = {"experimental": True,
                   "expert_data_error": float(sq.mean()),
                   "expert_data_error_definition": "mean over every demonstration transition of sum_j (a - pi(s))_j^2, "
                                                   "raw action units (legacy d3rlpy continuous_action_diff_scorer)",
                   "transitions": int(len(sq)),
                   "note": "offline metrics only; simulator evaluation is #24 / M7 through VectorBCPolicy.predict"}
        if "split" in eval_inputs:
            split = json.loads(eval_inputs["split"].read_text())
            metrics["validation_episodes_in_training"] = "split" not in inputs
            for part in ("train", "validation"):
                m = np.isin(batch.episode_id, split[part])
                metrics[f"{part}_action_error"] = float(sq[m].mean()) if m.any() else None
                metrics[f"{part}_transitions"] = int(m.sum())
        run.metrics(metrics)
    evaluated = run_stage("policy.evaluate", cfg, eval_inputs, evaluate, components={"trainer": VectorMLP()})
    return [*results, train_result, evaluated]
