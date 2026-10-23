"""The parameter registry. Declared parameters are validated and documented now and
consumed by the milestone named in `milestone`; their defaults are copied from the
legacy implementation where it had one, and are otherwise left for recipes to set
(no algorithm default is invented or changed here)."""
from __future__ import annotations

from .spec import Param

D = "declared"
LIPSCHITZ = ("none", "spectral_normalization", "slack", "spectral_normalization_slack", "soft_sampling",
             "soft_sampling_slack", "soft_jac")
METHODS = ("backward_euler", "backward_euler_fast", "forward_euler", "noisy_action", "noisy_action_fe", "backward_model")

PARAMS = [
    # run / compute ---------------------------------------------------------
    Param("run.seed", "int", 42, "Seed for every random choice in the run, including the trajectory split.",
          "Legacy default seed in every config.", "run", low=0, legacy=("seed",)),
    Param("run.output_dir", "path", "runs", "Directory under which run directories are created.",
          "Replaces output.location; run directories are named by run ID.", "run", legacy=("output.location",)),
    Param("run.overwrite", "bool", False, "Replace an existing run directory with the same run ID.",
          "Legacy scripts always overwrote; the core refuses unless asked.", "run", legacy=("overwrite",)),
    Param("compute.device", "choice", "cuda", "Device for model computation.",
          "Legacy default (no_gpu false) ran on CUDA.", "compute", choices=("cpu", "cuda"), legacy=("no_gpu",),
          status=D, milestone="M3 (#21)"),
    Param("compute.deterministic", "bool", False, "Request deterministic GPU kernels.",
          "Legacy seeding set no CUDA determinism flags.", "compute", status=D, milestone="M3 (#21)"),
    # data ------------------------------------------------------------------
    Param("data.path", "path", None, "Dataset file: ccil.trajectories .npz, or an M1 ccil-trajectories/1 .npz.",
          "Required; there is no meaningful default dataset.", "data", nullable=True, legacy=("data.pkl", "data.path")),
    Param("data.validation_fraction", "float", 0.2, "Fraction of trajectories held out to score the evaluation model.",
          "Value set in every shipped config by #13; 0 skips held-out evaluation.", "data", low=0.0, high=1.0,
          high_open=True, legacy=("dynamics.validation_fraction",)),
    Param("data.action_low", "floats", None, "Declared lower action bound per dimension (or one value for all).",
          "Opt-in (#14); no shipped config declares bounds.", "data", units="action units", nullable=True,
          legacy=("data.action_low",)),
    Param("data.action_high", "floats", None, "Declared upper action bound per dimension (or one value for all).",
          "Opt-in (#14); no shipped config declares bounds.", "data", units="action units", nullable=True,
          legacy=("data.action_high",)),
    # environment -----------------------------------------------------------
    Param("environment.id", "str", None, "Simulator task identifier used for evaluation.",
          "Task-specific; set by recipes.", "environment", nullable=True, legacy=("env",), status=D, milestone="M3/M7"),
    # transition model ------------------------------------------------------
    Param("transition_model.direction", "choice", "forward", "Fit s_{t+1} from (s_t, a_t), or s_t from (s_{t+1}, a_t).",
          "Legacy dynamics.backward, absent (forward) in every config.", "transition_model",
          choices=("forward", "backward"), legacy=("dynamics.backward",), status=D, milestone="M3 (#21)"),
    Param("transition_model.lipschitz_type", "choice", None, "Lipschitz regularization of the transition model.",
          "Task-specific in legacy configs.", "transition_model", choices=LIPSCHITZ, nullable=True,
          legacy=("dynamics.lipschitz_type",), status=D, milestone="M3 (#21)"),
    Param("transition_model.lipschitz_constraint", "float", None, "Target Lipschitz bound.",
          "Task-specific (1.0-4.0) in legacy configs.", "transition_model", low=0.0, low_open=True, nullable=True,
          legacy=("dynamics.lipschitz_constraint",), status=D, milestone="M3 (#21)"),
    Param("transition_model.soft_penalty_weight", "float", None, "Weight of the soft Lipschitz penalty.",
          "Task-specific in legacy configs.", "transition_model", low=0.0, nullable=True,
          legacy=("dynamics.soft_lipschitz_penalty_weight",), status=D, milestone="M3 (#21)"),
    Param("transition_model.soft_sampling_eps", "float", 0.0003, "Perturbation radius for sampled Lipschitz estimates.",
          "Same value in every legacy config.", "transition_model", units="state units", low=0.0, low_open=True,
          legacy=("dynamics.soft_lipschitz_sampling_eps",), status=D, milestone="M3 (#21)"),
    Param("transition_model.soft_samples", "int", 5, "Samples per point for sampled Lipschitz estimates.",
          "Same value in every legacy config.", "transition_model", low=1, legacy=("dynamics.soft_lipschitz_n_samples",),
          status=D, milestone="M3 (#21)"),
    Param("transition_model.slack_weight", "float", 1.0, "Weight of the slack penalty.",
          "Legacy code default when slack_weight is absent.", "transition_model", low=0.0,
          legacy=("dynamics.slack_weight",), status=D, milestone="M3 (#21)"),
    Param("transition_model.learning_rate", "float", 0.001, "Adam learning rate.", "Same value in every legacy config.",
          "transition_model", low=0.0, low_open=True, legacy=("dynamics.lr",), status=D, milestone="M3 (#21)"),
    Param("transition_model.weight_decay", "float", 1e-05, "Adam weight decay.", "Same value in every legacy config.",
          "transition_model", low=0.0, legacy=("dynamics.weight_decay",), status=D, milestone="M3 (#21)"),
    Param("transition_model.hidden_layers", "ints", None, "Hidden layer widths.", "Task-specific in legacy configs.",
          "transition_model", low=1, nullable=True, legacy=("dynamics.layers",), status=D, milestone="M3 (#21)"),
    Param("transition_model.activation", "choice", "relu", "Hidden activation.", "Same value in every legacy config.",
          "transition_model", choices=("relu", "tanh"), legacy=("dynamics.activation",), status=D, milestone="M3 (#21)"),
    Param("transition_model.epochs", "int", None, "Training epochs.", "Task-specific in legacy configs.",
          "transition_model", low=1, nullable=True, legacy=("dynamics.train_epochs",), status=D, milestone="M3 (#21)"),
    Param("transition_model.batch_size", "int", None, "Minibatch size.", "Task-specific in legacy configs.",
          "transition_model", low=1, nullable=True, legacy=("dynamics.batch_size",), status=D, milestone="M3 (#21)"),
    # generation ------------------------------------------------------------
    Param("generation.method", "choice", None, "Corrective-label generator.", "Task-specific in legacy configs.",
          "generation", choices=METHODS, nullable=True, legacy=("aug.type",), status=D, milestone="M3 (#21, #22)"),
    Param("generation.max_iterations", "int", None, "Root-finder iterations.", "Task-specific (30 or 50) in legacy configs.",
          "generation", low=1, nullable=True, legacy=("aug.max_iter",), status=D, milestone="M3 (#21)"),
    Param("generation.root_tolerance", "float", 0.0001, "Residual norm at which the root finder stops.",
          "Legacy literal in utils.root_finder_s.", "generation", units="state units", low=0.0, low_open=True,
          status=D, milestone="M3 (#21)"),
    Param("generation.labels_per_transition", "int", None, "Noisy-action candidates per transition.",
          "Task-specific (10 or 20) in legacy configs.", "generation", low=1, nullable=True, legacy=("aug.num_labels",),
          status=D, milestone="M3 (#22)"),
    Param("generation.action_noise_std", "float", None, "Standard deviation of the noisy-action perturbation.",
          "Task-specific in legacy configs.", "generation", units="action units", low=0.0, low_open=True, nullable=True,
          legacy=("aug.delta",), status=D, milestone="M3 (#22)"),
    Param("generation.support_radius", "float", None, "Labels farther than this from the demonstration state are rejected.",
          "aug.epsilon; legacy applied it with distance < epsilon, unset meaning no check.", "generation",
          units="state units", low=0.0, low_open=True, nullable=True, legacy=("aug.epsilon",), status=D, milestone="M3 (#22)"),
    # policy ----------------------------------------------------------------
    Param("policy.method", "choice", "ccil", "Policy training data: expert plus accepted labels, expert only, or NoiseBC.",
          "Replaces the policy.naive and policy.noise_bc flag combination.", "policy", choices=("ccil", "bc", "noise_bc"),
          legacy=("policy.naive", "policy.noise_bc"), status=D, milestone="M3"),
    Param("policy.noise_scale", "float", None, "NoiseBC observation-noise scale.", "Set per suite by the legacy scripts.",
          "policy", low=0.0, nullable=True, legacy=("policy.noise_bc",), status=D, milestone="M3"),
    Param("policy.noise_model", "choice", "legacy_uniform_half_batch", "NoiseBC noise distribution.",
          "Legacy CustomBC adds noise_cov * U[0,1) to half of each batch (non-negative, not zero-mean).", "policy",
          choices=("legacy_uniform_half_batch",), status=D, milestone="M3"),
    Param("policy.learning_rate", "float", 0.001, "Policy learning rate.", "Same value in every legacy config.", "policy",
          low=0.0, low_open=True, legacy=("policy.lr",), status=D, milestone="M3"),
    Param("policy.batch_size", "int", None, "Policy minibatch size.", "Task-specific in legacy configs.", "policy",
          low=1, nullable=True, legacy=("policy.batch_size",), status=D, milestone="M3"),
    Param("policy.epochs", "int", None, "Policy training epochs.", "Task-specific in legacy configs.", "policy", low=1,
          nullable=True, legacy=("policy.train_epochs",), status=D, milestone="M3"),
    Param("policy.balance_noisy_labels", "bool", False, "Repeat expert data num_labels times for noisy-action labels.",
          "Legacy aug.balance_data.", "policy", legacy=("aug.balance_data",), status=D, milestone="M3 (#22)"),
    Param("policy.action_scaler", "choice", "min_max", "Action normalization during policy training.",
          "Legacy literal in train_bc_policy.py.", "policy", choices=("min_max", "none"), status=D, milestone="M3"),
    # evaluation --------------------------------------------------------------
    Param("evaluation.episodes", "int", 100, "Evaluation episodes per policy.", "Legacy literal in eval_bc_policy.py.",
          "evaluation", low=1, status=D, milestone="M7"),
    Param("evaluation.noise_levels", "floats", [0.0], "Observation and action noise levels to evaluate.",
          "Legacy eval.noise (empty meant [0]).", "evaluation", low=0.0, legacy=("eval.noise",), status=D, milestone="M7"),
    Param("evaluation.noise_model", "choice", "legacy_uniform_mix", "Evaluation noise distribution.",
          "Legacy mixes in a uniform sample of the space; M7 (#51) predeclares zero-mean Gaussian.", "evaluation",
          choices=("legacy_uniform_mix", "gaussian"), status=D, milestone="M7 (#51)"),
    Param("evaluation.max_episode_steps", "int", None, "Episode step cap; unset uses the environment's own limit.",
          "Legacy hard-coded 500 for Meta-World.", "evaluation", units="steps", low=1, nullable=True, status=D, milestone="M7"),
]
REGISTRY = {p.key: p for p in PARAMS}

# Cross-field rules: (name, keys, check(values) -> error or None). Checked after types and domains.


def _bounds(v):
    lo, hi = v["data.action_low"], v["data.action_high"]
    if (lo is None) != (hi is None):
        return "data.action_low and data.action_high must be declared together"
    if lo is not None:
        if len(lo) != len(hi):
            return f"data.action_low has {len(lo)} values but data.action_high has {len(hi)}"
        if any(a > b for a, b in zip(lo, hi)):
            return "data.action_low must not exceed data.action_high"
    return None


def _noise(v):
    if v["policy.method"] == "noise_bc" and not v["policy.noise_scale"]:
        return "policy.method noise_bc requires a positive policy.noise_scale"
    return None


RULES = [
    ("action_bounds", ("data.action_low", "data.action_high"), _bounds),
    ("noise_bc_scale", ("policy.method", "policy.noise_scale"), _noise),
]
