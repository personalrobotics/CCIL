# Vector dynamics and backward-Euler labels: migration map (#21)

This page maps the original vector CCIL dynamics model and backward-Euler
labels to the `ccil` core.

- **Source:** `archive/upstream-main-2026-10-22`, commit
  `477ddb81c7adc3527c77b094c08181a98077eff3`, the files
  `correct_il/models/nn_dynamics.py`, `correct_il/models/spectral_norm.py` and
  `correct_il/gen_aug_label.py`.
- **Method:** Ke et al., *CCIL: Continuity-Based Data Augmentation for
  Corrective Imitation Learning*, ICLR 2024,
  [arXiv:2310.12972](https://arxiv.org/abs/2310.12972). Its method section
  learns a dynamics model with local Lipschitz continuity, then generates
  corrective labels where the model is continuous.
- **Recorded provenance:** each saved `mlp` checkpoint records this provenance
  in its meta (`derived_from`).

## Function map

| Legacy | New | Notes |
|---|---|---|
| `WorldModel` | `ccil.models.mlp.MLPTransitionModel` (`kind="mlp"`, `version="1"`) | Implements `TransitionModel`. Settings come from `transition_model.*` and `compute.*`. |
| `DynamicsNet` | `ccil.models.mlp.DynamicsNet` | Same layers, normalization, residual output and `out_scale >= 1e-8` mask. The normalization tensors are registered buffers, so they move with `.to(device)` and are saved. |
| `WorldModel.init_normalization_to_data` | `MLPTransitionModel.fit` | Mean and mean-absolute-deviation of s, a and s' − s. |
| `WorldModel.fit_dynamics`, `fit_model` | `MLPTransitionModel.fit` | Adam, a new permutation each epoch, and the last partial minibatch dropped. |
| `construct_loss_fn` and its losses | `mse_loss`, `soft_sampling_loss`, `soft_sampling_slack_loss`, `slack_loss`, `soft_jac_loss` | Same formulas. `functorch` becomes `torch.func`. |
| `spectral_norm.SpectralNorm` | `ccil.models.spectral_norm` | The legacy bound `W / max(σ/c, 1)` with one power iteration per training-mode pass. The PyTorch BSD-3 notice ships as `ccil/models/LICENSE-pytorch.txt`. |
| `WorldModel.f`, `predict` | `MLPTransitionModel.delta` | Batched, with numpy in and out. |
| `eval_prediction_error` | `evaluate` (`one_step_metrics`) | Mean and max one-step L2 error. |
| `save_world_model`, `load_world_model` (#12) | `save`, `load`, `load_transition_model` | `.npz` with `allow_pickle=False` instead of a `torch.save` payload. Strict keys and shapes. `cuda` without CUDA raises. |
| `train_dynamics_model.py` | the `transition.fit` stage | Labels use a model fitted on every episode. An evaluation model fitted on the train split is scored on held-out episodes (#13). `transition_model.direction=backward` swaps s and s' as legacy did. |
| `gen_backward_euler`, `gen_backward_euler_fast` | `ccil.core.BackwardEuler` | See below. |
| `_candidates` support check, `collect_labels` accounting | `SupportRadius`, `Finite`, `GenerationFailed`, `ActionBounds`, `generate_labels` | `distance < ε` as in legacy. A row is accepted exactly when no validator rejects it. |
| Label weighting (implicit: every accepted label counted once) | `Uniform` | Weight 1 for accepted rows and 0 for rejected rows. |

**Backward Euler.**
- Legacy `backward_euler` root-finds `s' + f(s*, a*) = s*`. The model is
  evaluated at the demonstration state `s*`, so the unknown `s'` appears only
  linearly, and the solution is the closed form `s' = s* − f(s*, a*)`. That is
  exactly `backward_euler_fast`.
- Both map to `BackwardEuler`, which computes the closed form directly. The
  legacy iterative solve only added a convergence tolerance (1e-4).

## Configuration

All values that affect behavior are named and validated. They are listed in
[`../config/parameters.md`](../config/parameters.md).

| Setting | Values |
|---|---|
| Architecture | `transition_model.hidden_layers`, `activation` |
| Optimization | `learning_rate`, `weight_decay`, `epochs`, `batch_size` |
| Regularizer | `lipschitz_type` (`none`, `spectral_normalization`, `slack`, `spectral_normalization_slack`, `soft_sampling`, `soft_sampling_slack`, `soft_jac`) |
| Regularizer settings | `lipschitz_constraint`, `soft_penalty_weight`, `soft_sampling_eps`, `soft_samples`, `slack_weight` |
| Compute | `compute.device` and `compute.deterministic` |

Building an `mlp` model rejects a configuration that is missing its
architecture, epochs, batch size, or the settings its regularizer reads. The
check runs in pipeline preflight, before any stage writes, and only when a
stage fits a model, so `data prepare` keeps working with the defaults. Legacy
code failed on these only partway through training.

Two legacy literals stay literals: the normalization epsilon (1e-8) and the
mask threshold (1e-8). They are part of the method, not tunable settings.

## Intentional differences

| Difference | Why | Evidence |
|---|---|---|
| Random streams | `fit(…, seed)` draws initialization and soft-sampling noise from a forked torch RNG seeded with `run.seed`, and minibatch order from `numpy.random.default_rng(seed)`. Legacy used the global RNGs. Fits are deterministic and leave no global state, but don't reproduce legacy fits bit for bit; #24 judges them statistically (contract §5). | `same_seed_byte_identical_checkpoints` |
| Fewer transitions than one minibatch | Raises. Legacy trained zero steps and divided by zero when averaging losses. | — |
| Device | `compute.device=cuda` without CUDA fails before any stage writes. Normalization buffers move with the model. | `checkpoint_persistence` |
| Checkpoint format | `.npz`, never unpickled (`docs/security.md`). | `checkpoint_persistence` |

There is no difference in prediction at fixed weights. Legacy `WorldModel`
weights loaded into `MLPTransitionModel` give bit-identical eval-mode
predictions for all seven Lipschitz types. Training-mode predictions and
power-iteration buffers agree within 2e-7 relative, across torch 2.1.0 and the
locked torch (`legacy_worldmodel_equivalence_at_fixed_weights`).

## Kept as-is, pending a decision

These legacy behaviors are ported unchanged. Contract §6 says a correctness
change needs its own pull request with before-and-after evidence.

1. **Training-mode inference.** Legacy never calls `eval()`, so a
   spectral-normalized model runs a power iteration on every prediction. That
   updates `u` and `v` in place and makes labels depend on call order. The
   equivalence evidence shows legacy eval-mode and training-mode predictions
   differ for exactly the two spectral-normalization types. This is fixed by
   the follow-up correctness PR for #21.
2. **The units of the soft-sampling penalty.** `soft_sampling` and
   `soft_sampling_slack` compare the network's *normalized residual* `Y_pred`
   with the *next-state prediction* `s + f(s + δ, a + δ)` of the perturbed
   input. The two are in different units, so the penalty doesn't measure the
   model's local Lipschitz ratio as its name says. The Pendulum recipes use
   `soft_sampling`. Changing this changes a published configuration and needs
   a decision.
3. **Broadcasting in the soft-sampling slack penalty.** In
   `soft_sampling_slack`, the per-sample slack `(B, 1)` multiplies the
   per-perturbation penalty `(B·n,)`, which broadcasts to `(B, B·n)` instead of
   pairing each sample with its own perturbations.

## Evidence

```bash
uv run --group evidence python tools/m3_dynamics/check.py --out <new-dir> [--device cuda] [--only CASE ...]
```

The reports are in [`../fixtures/evidence/m3-dynamics/`](../fixtures/evidence/m3-dynamics/):

| Report | What it covers |
|---|---|
| `report-cpu.json` | All seven cases on CPU. |
| `report-cuda.json` | The Pendulum case on CUDA (siddh, RTX PRO 6000 Blackwell, torch 2.14.1+cu130). |

The legacy-equivalence case runs `tools/m3_dynamics/legacy_export.py` in the
pinned legacy environment (`tools/dynamics_checkpoint/requirements.txt`). It
exchanges only numpy arrays with the new code.
