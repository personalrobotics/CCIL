"""The parameter registry. Declared parameters are validated and documented now and
consumed by the milestone named in `milestone`; their defaults are copied from the
legacy implementation where it had one, and are otherwise left for recipes to set
(no algorithm default is invented or changed here)."""
from __future__ import annotations

from .spec import Param

D = "declared"
LIPSCHITZ = ("none", "spectral_normalization", "slack", "spectral_normalization_slack", "soft_sampling",
             "soft_sampling_slack", "soft_jac")
CERTIFICATE_PROFILES = ("continuity", "backtrack", "disturbed_action", "backtrack_paper_proxy",
                        "disturbed_action_paper_proxy", "legacy_label_filtering")
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
    Param("transition_model.trainer_version", "choice", "1", "How mlp transition models are fitted (#110).",
          "1 fits each model with its own loop (MLPTransitionModel.fit); 2 with the batched engine "
          "(models/mlp_batched.py), which study runners use to fit many seeds together. Same protocol, not "
          "bit-compatible, so the version is part of the stage identity.", "transition_model", choices=("1", "2")),
    Param("transition_model.fit_episodes", "choice", "all", "Episodes the label model is fitted on.",
          "all (legacy) fits it on every episode, and a separate evaluation model on the train split is scored on "
          "the held-out episodes; train fits the label model on the train split only, so held-out episodes "
          "influence no label (a clean holdout), and that model is the one scored.", "transition_model",
          choices=("all", "train")),
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
    Param("generation.action_chunk_size", "int", None,
          "Add action chunks of this size to every label: the label action, then the demonstration's continuation.",
          "Legacy aug.chunk_size (label-filtering), whose inverted condition ignored a configured value (#22).",
          "generation", units="actions", low=1, nullable=True, legacy=("aug.chunk_size",)),
    Param("generation.model_error_threshold", "float", None,
          "Reject the candidates of transitions whose one-step model error is not below this.",
          "Legacy aug.model_err_thresh (label-filtering).", "generation", units="representation units", low=0.0,
          low_open=True, nullable=True, legacy=("aug.model_err_thresh",)),
    Param("generation.support_radius", "float", None, "Labels farther than this from the demonstration state are rejected.",
          "aug.epsilon; legacy applied it with distance < epsilon, unset meaning no check.", "generation",
          units="state units", low=0.0, low_open=True, nullable=True, legacy=("aug.epsilon",)),
    # certificate (#88) ------------------------------------------------------
    Param("certificate.profile", "choice", None, "Certify the labels with this profile (labels.certify); null skips it.",
          "Replaces the label-filtering estimate (#88). Bound profiles: continuity, backtrack, disturbed_action. "
          "Score profiles: backtrack_paper_proxy, disturbed_action_paper_proxy, legacy_label_filtering.",
          "certificate", choices=CERTIFICATE_PROFILES, nullable=True),
    Param("certificate.reference_dynamics", "path", None,
          "ccil.reference-dynamics file of the true dynamics: the source of analytic constants and demonstration residuals.",
          "Known only for analytic systems; without it the true constants and the demonstration residual bound are "
          "declared, and the certificate is conditional.", "certificate", nullable=True),
    Param("certificate.true_state_lipschitz", "float", None, "Declared state Lipschitz constant K_z of the true dynamics.",
          "An assumption, recorded as such (conditional); used without certificate.reference_dynamics.", "certificate",
          units="error per scaled state", low=0.0, nullable=True),
    Param("certificate.true_action_lipschitz", "float", None, "Declared action Lipschitz constant K_a of the true dynamics.",
          "An assumption, recorded as such (conditional); used without certificate.reference_dynamics.", "certificate",
          units="error per scaled action", low=0.0, nullable=True),
    Param("certificate.learned_lipschitz_source", "choice", "model", "Source of the learned state constant K_hat_z.",
          "model computes it from the frozen checkpoint (verified for linear models, conditional for MLPs); declared "
          "uses certificate.learned_state_lipschitz.", "certificate", choices=("model", "declared")),
    Param("certificate.learned_state_lipschitz", "float", None, "Declared state Lipschitz constant K_hat_z of the model.",
          "Used when certificate.learned_lipschitz_source is declared.", "certificate", units="error per scaled state",
          low=0.0, nullable=True),
    Param("certificate.discrepancy_state_lipschitz", "float", None,
          "Declared state constant C_z of the continuity profile (variation of f - f_hat).",
          "Used by continuity unless the reference dynamics and a linear model give it exactly.", "certificate",
          units="error per scaled state", low=0.0, nullable=True),
    Param("certificate.discrepancy_action_lipschitz", "float", None,
          "Declared action constant C_a of the continuity profile (variation of f - f_hat).",
          "Used by continuity unless the reference dynamics and a linear model give it exactly.", "certificate",
          units="error per scaled action", low=0.0, nullable=True),
    Param("certificate.demonstration_residual_bound", "float", None,
          "Declared bound on every demonstration residual ||T(z_t, a_t) - z_{t+1}||.",
          "Bound profiles add the residual once; without certificate.reference_dynamics it must be declared.",
          "certificate", units="scaled state", low=0.0, nullable=True),
    Param("certificate.neighborhood_state", "float", None, "Labels farther than this from their anchor state are unsupported.",
          "Null: the constants are global.", "certificate", units="scaled state", low=0.0, low_open=True, nullable=True),
    Param("certificate.neighborhood_action", "float", None, "Labels farther than this from their anchor action are unsupported.",
          "Null: the constants are global; backtrack keeps the anchor's action and takes none.", "certificate",
          units="scaled action", low=0.0, low_open=True, nullable=True),
    Param("certificate.max_total_error", "float", None, "Bound profiles: admit labels whose bound is at most this.",
          "Admission is upper_bound: only bound rows (certified or conditional) are admitted.", "certificate",
          units="scaled state", low=0.0, nullable=True),
    Param("certificate.score_threshold", "float", None, "Score profiles: admit labels whose score is at most this.",
          "Admission is score: rows stay empirical and are never presented as bounds. Legacy aug.max_label_err.",
          "certificate", units="scaled state", low=0.0, nullable=True, legacy=("aug.max_label_err",)),
    Param("certificate.selection_quantile", "float", None,
          "Select admitted, accepted labels whose total is at most this quantile of theirs.",
          "A data budget; it never changes acceptance or classification. Legacy aug.label_err_quantile (0.3 in the "
          "label-filtering pendulum configs).", "certificate", low=0.0, low_open=True, high=1.0, nullable=True,
          legacy=("aug.label_err_quantile",)),
    Param("certificate.selection_population", "choice", "admitted", "Rows the selection quantile is taken over.",
          "admitted: accepted, admissible rows. produced: every supported row a legacy run would have generated, "
          "independently of the admission threshold and support radius, as the label-filtering branch did; migrated "
          "legacy configurations use it.", "certificate", choices=("admitted", "produced")),
    # policy ----------------------------------------------------------------
    Param("policy.method", "choice", "ccil", "Policy training data: expert plus accepted labels, expert only, or NoiseBC (ported in M3).",
          "Replaces the policy.naive and policy.noise_bc flag combination.", "policy", choices=("ccil", "bc", "noise_bc"),
          legacy=("policy.naive", "policy.noise_bc")),
    Param("policy.trainer", "choice", "linear_least_squares", "Downstream policy trainer of the integrated pipeline.",
          "Reference trainers for the integrated pipeline (#19); real policy classes plug in through PolicyTrainer.",
          "policy", choices=("linear_least_squares", "mean_action")),
    Param("policy.noise_scale", "float", None, "NoiseBC observation-noise scale: the maximum noise magnitude.",
          "Set per suite by the legacy scripts; applied after the observation scaler, as legacy CustomBC did.", "policy",
          units="standardized observation units", low=0.0, nullable=True, legacy=("policy.noise_bc",)),
    Param("policy.noise_model", "choice", "symmetric_uniform_half_batch", "NoiseBC noise distribution.",
          "symmetric_uniform_half_batch adds scale*(2U-1) to half of each batch, zero-centered (#22). "
          "legacy_uniform_half_batch reproduces legacy CustomBC's scale*U[0,1), which is always positive, for parity.",
          "policy", choices=("symmetric_uniform_half_batch", "legacy_uniform_half_batch")),
    Param("policy.noise_mask_probability", "float", 0.5, "Probability that NoiseBC perturbs a sample in a minibatch.",
          "Legacy CustomBC rand > 0.5.", "policy", low=0.0, high=1.0),
    Param("policy.learning_rate", "float", 0.001, "Policy learning rate.", "Same value in every legacy config.", "policy",
          low=0.0, low_open=True, legacy=("policy.lr",)),
    Param("policy.batch_size", "int", None, "Policy minibatch size.", "Task-specific in legacy configs.", "policy",
          low=1, nullable=True, legacy=("policy.batch_size",)),
    Param("policy.epochs", "int", None, "Policy training epochs.", "Task-specific in legacy configs.", "policy", low=1,
          nullable=True, legacy=("policy.train_epochs",)),
    Param("policy.balance_noisy_labels", "bool", False, "Repeat expert data num_labels times for noisy-action labels.",
          "Legacy aug.balance_data: demonstrations appear labels_per_transition + 1 times and epochs are divided by "
          "labels_per_transition, for ccil with noisy-action labels.", "policy", legacy=("aug.balance_data",)),
    Param("policy.action_scaler", "choice", "min_max", "Action normalization during policy training.",
          "Legacy literal in train_bc_policy.py (d3rlpy MinMaxActionScaler to [-1, 1]).", "policy",
          choices=("min_max", "none")),
    Param("policy.observation_scaler", "choice", "standard", "Observation normalization during policy training.",
          "Legacy literal scaler='standard' in train_bc_policy.py: (x - mean) / (std + 1e-3), population std.",
          "policy", choices=("standard", "none")),
    Param("policy.trainer_version", "choice", "2", "Experimental vector-policy trainer version (#108).",
          "2 trains groups of policies as one batched, compiled program on the device (vector_bc_v2.py); 1 is the "
          "original per-step loop (vector_bc.py), kept for comparison. Same protocol, not bit-compatible.",
          "policy", choices=("1", "2")),
    Param("policy.optimizer", "choice", "adam", "Optimizer of the trainer v2 recipe (#115).",
          "adam is v1's (no weight decay); adamw applies decoupled weight decay.", "policy", choices=("adam", "adamw")),
    Param("policy.weight_decay", "float", 0.0, "AdamW decoupled weight decay (#115).",
          "0 is v1's protocol; nonzero needs policy.optimizer adamw.", "policy", low=0.0),
    Param("policy.lr_schedule", "choice", "constant", "Learning-rate schedule (#115).",
          "constant is v1's; cosine warms up linearly over policy.warmup_steps, then decays to 0 at the last step.",
          "policy", choices=("constant", "cosine")),
    Param("policy.warmup_steps", "int", 0, "Linear warm-up steps of the cosine schedule (#115).",
          "Used with policy.lr_schedule cosine.", "policy", low=0),
    Param("policy.step_budget", "int", None, "Optimizer steps per model, replacing policy.epochs (#115).",
          "A budget independent of the dataset size; the last epoch is cut at the budget. Null keeps epochs.",
          "policy", low=1, nullable=True),
    Param("policy.grad_clip_norm", "float", None, "Global gradient-norm clipping per model (#115).",
          "Null disables clipping (v1's).", "policy", low=0.0, low_open=True, nullable=True),
    Param("policy.ema_decay", "float", None, "Exponential moving average of the weights (#115).",
          "When set, the averaged weights are the policy that is selected and returned. Null disables it (v1's).",
          "policy", low=0.0, low_open=True, high=1.0, nullable=True),
    Param("policy.checkpoint_selection", "choice", "last", "Which epoch's weights are returned (#115).",
          "last is v1's; validation keeps the epoch with the lowest loss on the split's validation-episode "
          "demonstrations, never on evaluation data, and needs policy.exclude_validation_episodes.",
          "policy", choices=("last", "validation")),
    Param("policy.validation_interval_steps", "int", None,
          "With checkpoint_selection validation: validate every this many optimizer steps and at the last step, "
          "rather than once per epoch (#115).",
          "A step cadence gives arms of very different dataset sizes the same number of selection points.",
          "policy", low=1, nullable=True),
    Param("policy.hidden_layers", "ints", [256, 256], "Hidden-layer widths of the experimental vector policy.",
          "d3rlpy's default VectorEncoderFactory, which legacy BC used.", "policy", low=1),
    Param("policy.exclude_validation_episodes", "bool", True,
          "Train the policy on the training episodes only, holding out the split's validation episodes.",
          "Legacy BC trained on every expert transition; its validation split (#13) only evaluated the dynamics model. "
          "Converted legacy configurations with fidelity historical set false.", "policy"),
    Param("policy.demonstration_weight", "float", 1.0, "Loss weight of every demonstration row.",
          "Legacy weighted every row equally; synthetic rows keep their recorded weights (docs/supervision.md).",
          "policy", low=0.0),
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
    if v["policy.method"] != "noise_bc" and v["policy.noise_scale"] is not None:
        return f"policy.noise_scale applies to policy.method noise_bc, not {v['policy.method']}"
    if v["policy.balance_noisy_labels"] and (v["policy.method"] != "ccil"
                                             or v["generation.method"] not in ("noisy_action", "noisy_action_fe")):
        return "policy.balance_noisy_labels applies to policy.method ccil with noisy-action labels"
    return None


def _pca_dim(v):
    if v["representation.kind"] == "pca" and not v["representation.dim"]:
        return "representation.kind pca requires representation.dim"
    return None


BOUND_PROFILES = ("continuity", "backtrack", "disturbed_action")
# Generator family each profile certifies, and the family of each generation.method.
PROFILE_FAMILY = {"backtrack": "backtrack", "backtrack_paper_proxy": "backtrack", "disturbed_action": "disturbed_action",
                  "disturbed_action_paper_proxy": "disturbed_action"}
METHOD_FAMILY = {"backward_euler": "backtrack", "backward_euler_fast": "backtrack", "forward_euler": "backtrack",
                 "noisy_action": "disturbed_action", "noisy_action_fe": "disturbed_action"}
# Declared-value keys each bound profile reads when no reference dynamics are given.
CERTIFICATE_DECLARED = {"continuity": ("certificate.discrepancy_state_lipschitz", "certificate.discrepancy_action_lipschitz"),
                        "backtrack": ("certificate.true_state_lipschitz",),
                        "disturbed_action": ("certificate.true_state_lipschitz", "certificate.true_action_lipschitz")}
_VALUE_KEYS = ("certificate.true_state_lipschitz", "certificate.true_action_lipschitz", "certificate.learned_state_lipschitz",
               "certificate.discrepancy_state_lipschitz", "certificate.discrepancy_action_lipschitz",
               "certificate.demonstration_residual_bound")


def _certificate(v):
    prof, ref = v["certificate.profile"], v["certificate.reference_dynamics"]
    defaults = {"certificate.learned_lipschitz_source": "model", "certificate.selection_population": "admitted"}
    settings = [k for k in v if k.startswith("certificate.") and k != "certificate.profile"
                and v[k] is not None and v[k] != defaults.get(k)]
    if prof is None:
        return f"{settings} require certificate.profile" if settings else None
    problems = []
    if v["certificate.selection_population"] != "admitted" and v["certificate.selection_quantile"] is None:
        problems.append("certificate.selection_population applies to a selection; set certificate.selection_quantile")
    if v["certificate.learned_lipschitz_source"] != "model" and prof != "backtrack":
        problems.append(f"certificate.learned_lipschitz_source declared applies to the backtrack bound profile only, "
                        f"not {prof}")
    if prof in ("backtrack", "backtrack_paper_proxy") and v["certificate.neighborhood_action"] is not None:
        problems.append(f"{prof} keeps the anchor's action; certificate.neighborhood_action does not apply")
    family, method = PROFILE_FAMILY.get(prof), v["generation.method"]
    if family and method in METHOD_FAMILY and METHOD_FAMILY[method] != family:
        problems.append(f"certificate.profile {prof} certifies {family} labels; generation.method {method} makes "
                        f"{METHOD_FAMILY[method]} labels")
    if prof not in BOUND_PROFILES:
        extra = [k for k in settings if k in _VALUE_KEYS + ("certificate.reference_dynamics", "certificate.max_total_error")]
        if extra:
            problems.append(f"score profile {prof} takes no constants, reference, or bound threshold: unset {extra}")
        return "; ".join(problems) or None
    if v["certificate.score_threshold"] is not None:
        problems.append(f"bound profile {prof} admits by certificate.max_total_error, not certificate.score_threshold")
    if ref is not None and v["certificate.demonstration_residual_bound"] is not None:
        problems.append("certificate.demonstration_residual_bound is declared, but certificate.reference_dynamics computes "
                        "the residuals; set one")
    if ref is None and v["certificate.demonstration_residual_bound"] is None:
        problems.append(f"{prof} needs certificate.reference_dynamics or certificate.demonstration_residual_bound")
    exact_discrepancy = prof == "continuity" and ref is not None and v["transition_model.kind"] == "linear"
    needed = () if (ref is not None and prof != "continuity") or exact_discrepancy else CERTIFICATE_DECLARED[prof]
    if prof == "backtrack" and v["certificate.learned_lipschitz_source"] == "declared":
        needed += ("certificate.learned_state_lipschitz",)
    missing = [k for k in needed if v[k] is None]
    unused = [k for k in _VALUE_KEYS if k != "certificate.demonstration_residual_bound" and k not in needed and v[k] is not None]
    if missing:
        problems.append(f"{prof} without {'exact discrepancy' if prof == 'continuity' else 'reference dynamics'} needs {missing}")
    if unused:
        problems.append(f"{prof} does not use {unused} here; unset them")
    return "; ".join(problems) or None


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
    ("noise_bc_scale", ("policy.method", "policy.noise_scale", "policy.balance_noisy_labels", "generation.method"), _noise),
    ("certificate", ("certificate.profile", "certificate.reference_dynamics", "generation.method", "transition_model.kind"),
     _certificate),
]
