# Vector dynamics and backward-Euler labels: migration map (#21)

The port is faithful where the legacy method was correct. It is corrected where
the implementation contradicted the paper or the per-sample semantics the
method intends. Every correction has before-and-after evidence, listed below.

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
| `construct_loss_fn` and its losses | `mse_loss`, `soft_sampling_loss`, `soft_sampling_slack_loss` (with `slack_sampling_terms`), `slack_loss`, `soft_jac_loss` (with `residual_jacobian_norm`) | MSE and `slack_loss` are unchanged. The three Lipschitz penalties are corrected (see below). `functorch` becomes `torch.func`. |
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

At fixed weights the network computes the same function as legacy.
- Legacy `WorldModel` weights loaded into `MLPTransitionModel` give
  bit-identical eval-mode predictions for all seven Lipschitz types.
- The network's training-mode pass and power-iteration buffers agree within
  2e-7 relative, across torch 2.1.0 and the locked torch
  (`legacy_worldmodel_equivalence_at_fixed_weights`).

What changed is when each mode is used, and the soft-sampling penalty. See the
next section.

## Correctness changes (`mlp` version 2)

Four legacy defects are corrected under contract §6. Their before-and-after
evidence compares the port as first landed (`6074e33`, `mlp` version 1) with
the corrected code, using the same seeds:

```bash
uv run --group evidence python tools/m3_dynamics/correctness.py --out <new-dir>
```

The report is at
[`../fixtures/evidence/m3-dynamics/correctness.json`](../fixtures/evidence/m3-dynamics/correctness.json).
Version-1 checkpoints are refused (`CheckpointError`).

1. **Eval-mode inference.** Legacy never calls `eval()`, so a
   spectral-normalized model runs a power iteration on every prediction. That
   updates `u` and `v` in place and makes labels depend on call order. `delta`
   and `evaluate` now predict in eval mode and restore each network's prior
   mode. Training still runs the power iteration on every training-mode pass.

   | Spectral-norm model, CPU | Before | After |
   |---|---|---|
   | Repeated `delta` on the same inputs | differs (1.6e-7) | identical |
   | Power-iteration buffers changed by `delta` | yes (8.4e-3) | no |
   | Labels, batch size 1 vs one batch | 4.4e-6 | 1.5e-8 |

   - The remaining 1.5e-8 is float32 rounding that depends on batch shape. Every
     torch model has it, including one without spectral normalization (see
     [`../interfaces.md`](../interfaces.md)).
   - Spectral-norm labels move by at most 1.7e-6.
   - A model without spectral normalization gives bit-identical labels before
     and after.

2. **The soft-sampling penalty.** The paper constrains the residual model f̂:
   Sec. 4.2 and Eqs. 3–4, with the hinge form in Appendix C.1, Eq. 10. The
   penalty now matches it:

       ratio = ‖f̂(s + δs, a + δa) − f̂(s, a)‖ / ‖(δs, δa)‖,   δ ~ ε·N(0, I)
       penalty = λ · mean(max(ratio − L, 0))

   Both terms are in raw units. As in legacy, state and action are perturbed
   together. Legacy computed `‖Y_pred − (s + f̂(s + δs, a + δa))‖ − L‖n‖`, which
   has two errors:
   - it compares the network's normalized residual `Y_pred` with a next-state
     prediction;
   - it divides by the unscaled noise `n` instead of the step `ε·n`.

   With ε = 3e-4, the bound it enforced was therefore about 3,000× looser than
   L, applied to a quantity that isn't the Lipschitz ratio.

   | Pendulum, `pendulum_cont` recipe, `backward_euler`, CPU | Before | After |
   |---|---|---|
   | Final Lipschitz penalty / MSE | 7.9e-4 / 9.9e-5 | 0 / 2.9e-4 |
   | Measured ratio of f̂ (p50 / p95 / max), L = 2 | 0.074 / 0.163 / 0.229 | 0.074 / 0.163 / 0.220 |
   | Held-out one-step L2 (mean / max) | 6.9e-4 / 4.5e-3 | 6.4e-4 / 3.4e-3 |
   | Label model train L2 (mean) | 5.3e-4 | 7.1e-4 |

   - Labels move by 1.1e-3 on average and 4.5e-3 at most.
   - **The learned Pendulum dynamics are about 10× below the constraint, so the
     paper's penalty never activates there.** The legacy penalty didn't make the
     model smoother either: the measured ratio is the same before and after. It
     acted as an unrelated regularizer.
   - The Pendulum parity run (#24) has to account for this when it compares
     against legacy results.

3. **`soft_jac` constrains the residual.** Legacy penalized the spectral norm
   of the Jacobian of the next-state map `s + f̂`, which is `I + ∂f̂/∂s`. That
   norm is about 1 for any small residual, so the penalty measures the identity,
   not the model's continuity. The penalty now uses `∂f̂/∂(s, a)`, as in the
   paper's Eq. 3.

   | `f(s, a) = 0.05 (sin s + a)`: ‖∂f/∂(s, a)‖ ≤ 0.087, ‖I + ∂f/∂s‖ ≈ 1; L = 0.5, λ = 1 | Before | After |
   |---|---|---|
   | Final penalty | 0.548 (the floor ‖I + J‖ − L) | 0 |
   | Fitted ‖∂f̂/∂(s, a)‖, max | 0.102 | 0.104 |
   | Residual fit error (mean; residual scale 0.051) | 0.0023 | 0.0019 |

   Against the normalized MSE, the legacy floor contributed little gradient
   here, so the fits are similar. Its value tracked the identity term rather
   than the residual.

4. **`soft_sampling_slack` pairs each sample with its own perturbations.**
   Legacy multiplied the `(B, 1)` slack by the `(B·n,)` per-perturbation hinge,
   which broadcasts to `(B, B·n)`, so every sample's slack weighted every
   sample's perturbations. Each sample's term is now its slack times the mean
   hinge over its own `n` perturbations (`slack_sampling_terms`). The legacy
   batch-level regularizer on slack magnitude, `1 − exp(−0.1·mean|slack|)`, is
   unchanged.
   - With explicit noise, permuting the batch permutes the terms exactly, and a
     sample's term is identical in a different batch. The maximum difference
     is 0.0 for both.
   - The pre-fix expression has shape `(8, 40)` for B = 8 and n = 5.

**Legacy behavior** stays available only at the pinned pre-fix commit
`6074e33`. `correctness.py` runs that commit as the "before" side.

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
