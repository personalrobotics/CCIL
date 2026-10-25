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
          "Legacy default (no_gpu false) ran on CUDA; cuda without CUDA fails, never falls back.", "compute",
          choices=("cpu", "cuda"), legacy=("no_gpu",)),
    Param("compute.deterministic", "bool", False, "Request deterministic torch kernels while fitting and predicting.",
          "Legacy seeding set no determinism flags. On CUDA this needs CUBLAS_WORKSPACE_CONFIG=:4096:8.", "compute"),
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
    # representation ----------------------------------------------------------
    Param("representation.kind", "choice", "identity", "Representation the labels live in.",
          "identity reproduces vector CCIL; pca is the integrated pipeline's reference learned representation.",
          "representation", choices=("identity", "pca")),
    Param("representation.modality", "str", "state", "Stored modality the representation reads.",
          "Vector CCIL reads the state modality.", "representation"),
    Param("representation.dim", "int", None, "Latent dimension of a learned representation.",
          "Required by pca; unused by identity.", "representation", low=1, nullable=True),
    # environment -----------------------------------------------------------
    Param("environment.id", "str", None, "Simulator task identifier used for evaluation.",
          "Task-specific; set by recipes.", "environment", nullable=True, legacy=("env",), status=D, milestone="M3/M7"),
    # transition model ------------------------------------------------------
    Param("transition_model.kind", "choice", "mlp", "Transition-model family: linear (least squares) or mlp.",
          "Legacy code used an MLP (WorldModel), ported in #21; mlp needs the ccil[torch] extra (Python 3.11+).",
          "transition_model", choices=("linear", "mlp")),
    Param("transition_model.direction", "choice", "forward", "Fit s_{t+1} from (s_t, a_t), or s_t from (s_{t+1}, a_t).",
          "Legacy dynamics.backward, absent (forward) in every config.", "transition_model",
          choices=("forward", "backward"), legacy=("dynamics.backward",)),
    Param("transition_model.lipschitz_type", "choice", None, "Lipschitz regularization of the transition model.",
          "Task-specific in legacy configs.", "transition_model", choices=LIPSCHITZ, nullable=True,
          legacy=("dynamics.lipschitz_type",)),
    Param("transition_model.lipschitz_constraint", "float", None, "Target Lipschitz bound.",
          "Task-specific (1.0-4.0) in legacy configs.", "transition_model", low=0.0, low_open=True, nullable=True,
          legacy=("dynamics.lipschitz_constraint",)),
    Param("transition_model.soft_penalty_weight", "float", None, "Weight of the soft Lipschitz penalty.",
          "Task-specific in legacy configs.", "transition_model", low=0.0, nullable=True,
          legacy=("dynamics.soft_lipschitz_penalty_weight",)),
    Param("transition_model.soft_sampling_eps", "float", 0.0003, "Perturbation radius for sampled Lipschitz estimates.",
          "Same value in every legacy config.", "transition_model", units="state units", low=0.0, low_open=True,
          legacy=("dynamics.soft_lipschitz_sampling_eps",)),
    Param("transition_model.soft_samples", "int", 5, "Samples per point for sampled Lipschitz estimates.",
          "Same value in every legacy config.", "transition_model", low=1, legacy=("dynamics.soft_lipschitz_n_samples",),
          status=D, milestone="M3 (#21)"),
    Param("transition_model.slack_weight", "float", 1.0, "Weight of the slack penalty.",
          "Legacy code default when slack_weight is absent.", "transition_model", low=0.0,
          legacy=("dynamics.slack_weight",)),
    Param("transition_model.learning_rate", "float", 0.001, "Adam learning rate.", "Same value in every legacy config.",
          "transition_model", low=0.0, low_open=True, legacy=("dynamics.lr",)),
    Param("transition_model.weight_decay", "float", 1e-05, "Adam weight decay.", "Same value in every legacy config.",
          "transition_model", low=0.0, legacy=("dynamics.weight_decay",)),
    Param("transition_model.hidden_layers", "ints", None, "Hidden layer widths.", "Task-specific in legacy configs.",
          "transition_model", low=1, nullable=True, legacy=("dynamics.layers",)),
    Param("transition_model.activation", "choice", "relu", "Hidden activation.", "Same value in every legacy config.",
          "transition_model", choices=("relu", "tanh"), legacy=("dynamics.activation",)),
    Param("transition_model.epochs", "int", None, "Training epochs.", "Task-specific in legacy configs.",
          "transition_model", low=1, nullable=True, legacy=("dynamics.train_epochs",)),
    Param("transition_model.batch_size", "int", None, "Minibatch size.", "Task-specific in legacy configs.",
          "transition_model", low=1, nullable=True, legacy=("dynamics.batch_size",)),
    # generation ------------------------------------------------------------
    Param("generation.method", "choice", None,
          "Corrective-label generator. backward_model is ported in #22.",
          "Task-specific in legacy configs.", "generation", choices=METHODS, nullable=True, legacy=("aug.type",)),
    Param("generation.max_iterations", "int", None, "Root-finder iterations.", "Task-specific (30 or 50) in legacy configs.",
          "generation", low=1, nullable=True, legacy=("aug.max_iter",)),
    Param("generation.root_tolerance", "float", 0.0001, "Residual norm at which the root finder stops.",
          "Legacy literal in utils.root_finder_s.", "generation", units="state units", low=0.0, low_open=True),
    Param("generation.labels_per_transition", "int", None, "Noisy-action candidates per transition.",
          "Task-specific (10 or 20) in legacy configs.", "generation", low=1, nullable=True, legacy=("aug.num_labels",)),
    Param("generation.action_noise_std", "float", None, "Standard deviation of the noisy-action perturbation.",
          "Task-specific in legacy configs.", "generation", units="action units", low=0.0, low_open=True, nullable=True,
          legacy=("aug.delta",)),
    Param("generation.max_label_error", "float", None, "Reject labels whose estimated error exceeds this.",
          "Legacy aug.max_label_err (label-filtering). The estimate uses the residual Jacobian (#22).", "generation",
          units="representation units", low=0.0, low_open=True, nullable=True, legacy=("aug.max_label_err",)),
    Param("generation.label_error_quantile", "float", None,
          "Keep labels whose estimated error is at most this quantile of the produced candidates' estimates.",
          "Legacy aug.label_err_quantile (label-filtering); 0.3 in its pendulum configs.", "generation", low=0.0,
          low_open=True, high=1.0, nullable=True, legacy=("aug.label_err_quantile",)),
    Param("generation.model_error_threshold", "float", None,
          "Reject the candidates of transitions whose one-step model error is not below this.",
          "Legacy aug.model_err_thresh (label-filtering).", "generation", units="representation units", low=0.0,
          low_open=True, nullable=True, legacy=("aug.model_err_thresh",)),
    Param("generation.support_radius", "float", None, "Labels farther than this from the demonstration state are rejected.",
          "aug.epsilon; legacy applied it with distance < epsilon, unset meaning no check.", "generation",
          units="state units", low=0.0, low_open=True, nullable=True, legacy=("aug.epsilon",)),
    # policy ----------------------------------------------------------------
    Param("policy.method", "choice", "ccil", "Policy training data: expert plus accepted labels, expert only, or NoiseBC (ported in M3).",
          "Replaces the policy.naive and policy.noise_bc flag combination.", "policy", choices=("ccil", "bc", "noise_bc"),
          legacy=("policy.naive", "policy.noise_bc")),
    Param("policy.trainer", "choice", "linear_least_squares", "Downstream policy trainer of the integrated pipeline.",
          "Reference trainers for the integrated pipeline (#19); real policy classes plug in through PolicyTrainer.",
          "policy", choices=("linear_least_squares", "mean_action")),
    Param("policy.noise_scale", "float", None, "NoiseBC observation-noise scale: the maximum noise magnitude.",
          "Set per suite by the legacy scripts.", "policy", units="observation units", low=0.0, nullable=True,
          legacy=("policy.noise_bc",), status=D, milestone="#89"),
    Param("policy.noise_model", "choice", "symmetric_uniform_half_batch", "NoiseBC noise distribution.",
          "symmetric_uniform_half_batch adds scale*(2U-1) to half of each batch, zero-centered (#22). "
          "legacy_uniform_half_batch reproduces legacy CustomBC's scale*U[0,1), which is always positive, for parity.",
          "policy", choices=("symmetric_uniform_half_batch", "legacy_uniform_half_batch"), status=D, milestone="#89"),
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


def _pca_dim(v):
    if v["representation.kind"] == "pca" and not v["representation.dim"]:
        return "representation.kind pca requires representation.dim"
    return None


# Settings each Lipschitz regularizer reads; legacy WorldModel failed on a missing one only mid-training.
LIPSCHITZ_NEEDS = {
    "spectral_normalization": ("transition_model.lipschitz_constraint",),
    "spectral_normalization_slack": ("transition_model.lipschitz_constraint",),
    "soft_sampling": ("transition_model.lipschitz_constraint", "transition_model.soft_penalty_weight"),
    "soft_sampling_slack": ("transition_model.lipschitz_constraint", "transition_model.soft_penalty_weight"),
    "soft_jac": ("transition_model.lipschitz_constraint", "transition_model.soft_penalty_weight"),
}


def mlp_settings_problem(v):
    """Why an mlp transition model cannot be built from v, or None; checked when a stage builds one."""
    missing = [k for k in ("transition_model.hidden_layers", "transition_model.epochs", "transition_model.batch_size")
               if v[k] is None]
    missing += [k for k in LIPSCHITZ_NEEDS.get(v["transition_model.lipschitz_type"], ()) if v[k] is None]
    if missing:
        return f"transition_model.kind mlp with lipschitz_type {v['transition_model.lipschitz_type']!r} requires {missing}"
    return None


RULES = [
    ("pca_dim", ("representation.kind", "representation.dim"), _pca_dim),
    ("action_bounds", ("data.action_low", "data.action_high"), _bounds),
    ("noise_bc_scale", ("policy.method", "policy.noise_scale"), _noise),
]
