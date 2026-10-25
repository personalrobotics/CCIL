# Label methods: migration map (#22)

This page maps the legacy corrective-label variants and the NoiseBC baseline to
the `ccil` core. #22 runs as three PRs, and this page grows with each:

| PR | Contents |
|---|---|
| 1 | Noisy-action generation and NoiseBC |
| 2 | Filtering, weighting, and the label-filtering branch inventory |
| 3 | Action chunks (to come) |

**Ownership.** #22 owns the *measurements* each label carries. #88 owns their
mathematical interpretation: proof profiles, bounds, and the certified,
conditional or empirical classification. #89 owns the vector BC trainer, which
applies NoiseBC.

**Source:** `archive/upstream-main-2026-10-22`, files
`correct_il/gen_aug_label.py` and `correct_il/CustomBC.py`.

## Method map

| Legacy | New | Status |
|---|---|---|
| `gen_noisy_action_be` (`aug.type: noisy_action`) | `ccil.core.NoisyAction` (`generation.method=noisy_action`, version 2) | Same mathematics: for k draws ã = a + N(0, σ²), z_G = z_{t+1} − f(z_{t+1}, ã). Version 2 adds the measurements below. |
| `gen_noisy_action_fe` (`aug.type: noisy_action_fe`) | `ccil.core.NoisyActionPreimage` (`generation.method=noisy_action_fe`) | Same mathematics: solve z_G + f(z_G, ã) = z_{t+1} by Newton from z_t. Uses the row-wise solver of `ExactPreimage` (#17). A failed proposal is rejected as `generation_failed` and never refilled. |
| `CustomBC.compute_loss_weight` with `noise_cov` (NoiseBC) | `ccil.training.noise_bc.NoiseBC` | Corrected (below). Training integration is #89. |

## Measurements every noisy-action proposal records

Proposals record these whether accepted or rejected. The definitions, norms
and units travel in the candidate file as `measurement_specs`
(`ccil.candidates` 2.2), and the generator's full `params` (k, σ, and the
noise distribution) travel in `generator.params`.

| Name | Definition | Norm | Units |
|---|---|---|---|
| `generator.action_noise_norm` | ‖a_G − a_t‖, the sampled action perturbation | l2 | action units |
| `generator.state_displacement_norm` | ‖z_G − z_t‖, the distance from the demonstration state of the source transition | l2 | representation units |
| `generator.solver_residual` (solved variant only) | ‖z_G + f(z_G, a_G) − target‖ of the returned iterate | l2 | representation units |
| `distance` (`SupportRadius`) | ‖z_G − z_t‖, accepted when below the radius | l2 | representation units |
| `residual` (`ModelResidual`) | ‖z_G + f(z_G, a_G) − target‖ under the model | l2 | representation units |

The source episode and frame are part of every row (`ccil.candidates`).

**Measurements describe the stored candidate.**
- Each measurement is computed from the final, dtype-cast arrays written to the
  file.
- At float32 and a large action magnitude, rounding can shrink the persisted
  perturbation, even to zero. `action_noise_norm` then reports the stored
  value, not the sample. The evidence has a case where it is 0 for every row:
  actions near 1e4, where float32 spacing is 9.8e-4, and σ = 1e-4.
- A solved candidate's `solver_residual` and its `generation_failed` decision
  are judged on the stored state, not on the float64 iterate.

**The contract is enforced.** Every generator and validator declares
`measurement_specs`. A producer whose specs are missing, or don't name exactly
the measurements it returned, raises `InterfaceError`. Specs are never silently
dropped.

**Sample identity.** Noisy-action draws and NoiseBC draws key each transition
by a BLAKE2b-128 digest of `(episode_id, frame)` (`ccil.core.identity`). The
earlier CRC32 of the episode ID let distinct episodes share a stream: the
evidence uses the colliding pair `4300bByhudVrUGx5` and `0BZ5Gzkl9ccfsEbx`.

## NoiseBC correction

Legacy NoiseBC behaved as follows:
- on every loss computation it drew a half-batch mask (`rand(B) > 0.5`) and
  noise `noise_cov · U[0, 1)`;
- that noise is always positive, so it biases states by `noise_cov / 2`;
- `x = obs_t` aliases the batch, so the noise was written into the caller's
  tensor in place.

The paper (Appendix D.4) describes only a small input-state disturbance and
gives no distribution.

`NoiseBC` keeps the parts that were intended:
- online resampling at every training draw;
- the masking probability;
- `policy.noise_scale` as the maximum magnitude.

It removes the positive bias and the in-place write:

| `policy.noise_model` | Perturbation of masked samples | Use |
|---|---|---|
| `symmetric_uniform_half_batch` (default) | s̃ = s + σ(2U − 1), in [−σ, σ) | canonical |
| `legacy_uniform_half_batch` | s̃ = s + σU, in [0, σ) | historical parity (#24) |

**Determinism.**
- Each uniform is a function of (seed, optimizer step, sample identity, draw
  index, component), computed with a splitmix64 counter hash over the
  128-bit sample-identity digest.
- A draw is therefore reproducible from its key, fresh at the next step or
  draw, and independent of batch order and composition.
- `perturb` returns a new array. It rejects inputs that it could not perturb as configured: non-floating observations, which would truncate the noise; identity arrays that don't match the batch; and negative frames, steps or draws.

**Recording.** `NoiseBC.descriptor()` gives what a run records: distribution,
scale, units, truncation, masking probability, seed, RNG, and draw schedule.
`NoiseTrace` gives a running SHA-256 over every draw, for reproducibility
without storing the draws.

## Filtering and weighting (PR 2)

The label-filtering branch (`archive/upstream-label-filtering-2026-10-22`) adds
two filters to legacy `main`'s support radius. Validity, acceptance and
training weight stay separate:
- a **validator** rejects a row with a reason, and records measurements;
- a row is **accepted** exactly when no validator rejects it;
- a **weighting** sets the training weight of accepted rows, and can't change
  acceptance or reasons.

| Legacy | New | Reason | Notes |
|---|---|---|---|
| `aug.epsilon` (`distance < ε`) | `SupportRadius` (`generation.support_radius`) | `out_of_support` | Unchanged (#11). |
| `aug.model_err_thresh` (`model_fits_data`) | `ModelMisfit` (`generation.model_error_threshold`) | `model_misfit` | Legacy generated nothing for a misfit transition, recording only a failure count. Its candidates are now generated and rejected, so the decision is in provenance. It measures `anchor_model_error`. |
| `label_error` with `aug.max_label_err` / `aug.label_err_quantile` | `LabelError` (`generation.max_label_error`, `generation.label_error_quantile`) | `label_error` | Corrected estimate (below). The quantile is taken over the finite estimates of candidates generation produced (not failed), as in legacy. The applied threshold is recorded per row as `label_error_threshold`. |

`model_misfit` and `label_error` are built-in reasons added in
`ccil.candidates` 2.3. A file has their columns only when a validator uses
them.

**The label-error estimate (correctness change).**
- **The branch** computed `label_err = L · ‖(s_G, a_G) − (s_t, a_t)‖`, with `L`
  the spectral norm of the Jacobian of `WorldModel.predict`, the next-state map
  `s + f`. That Jacobian is `[I + ∂f/∂s, ∂f/∂a]`, whose norm is about 1 for any
  small residual. The estimate therefore came out close to the bare distance,
  whatever the model's continuity.
- **`LabelError`** uses the residual Jacobian `∂f/∂(z, a)` at the anchor, which
  is the continuity the method assumes of f. It is the same correction as
  `soft_jac` in #21.
- **Its scope:** `L` is a pointwise value at the anchor, so the estimate is
  empirical, not a bound over the neighborhood. #88 classifies it.
- The evidence shows both formulas on the analytic system.

**Weighting.**
- **No per-label soft weighting exists in any branch.** The only weights are
  the per-action-dimension policy-loss weights in `CustomBC` and the expert
  repetition `aug.balance_data`. Both are policy-side, owned by #89.
- `Uniform` remains the label weighting.
- The schema keeps weights from affecting acceptance: rejected rows must have
  weight 0, and a weight of 0 on an accepted row leaves it accepted.

## Label-filtering branch inventory

Each capability of `docs/inventory/capabilities.json` that the branch changes,
with its terminal classification. The evidence checks that none is missing.

| Capability | Branch change | Classification |
|---|---|---|
| `dynamics-model` | `local_lipschitz_coeff()`; unreduced prediction errors | Ported: `jacobian_za` and `residual_jacobian_za` (residual, corrected); per-row errors as `anchor_model_error` |
| `corrective-labels` | `forward_euler` as fixed-point iteration; `model_err_thresh`; original actions recorded | Fixed-point iteration superseded by `ExactPreimage` (same equation; Newton doesn't need ‖∂f/∂z‖ < 1, and no hardcoded 1e-4); `model_err_thresh` ported as `ModelMisfit`; source actions are carried by the source identity |
| `label-filtering` | label-error estimate; `label_err_quantile`; `max_label_err`; combined with `epsilon` | Ported, with a correctness change: `LabelError`, alongside `SupportRadius` |
| `action-chunking-labels` | `gen_aug_label_chunking.py` | #22 PR 3 |
| `bc-policy` | removes the tycho case; configurable encoder layers; callable loss weights | Moved to #89 (vector BC training) |
| `data-fraction-ablation` | `data.data_frac` | Superseded by the M7 low-data subsets (#51–#55) |
| `dynamics-training-tool` | paper-style plots; error-distribution plot | Out of scope; plotting is left to analysis tools |
| `f1-evaluation` | success-rate metric; hardcoded `/home/prl/dev/CCIL` | Out of scope: the F1 suite is retired (#3) |
| `results-analysis` | `tabulate_sweep.py`, `plot_ablation.py` | Out of scope (analysis scripts) |
| `task-configs` | all 14 configs retuned, e.g. pendulum to `backward_euler_fast` with `label_err_quantile: 0.3` | Recorded as variants; each migrates to the keys above. Which variant produced each figure is a ledger question (#24) |
| `pipeline-scripts` | eval flag; training steps commented out | Out of scope: superseded by `ccil` pipelines |
| `cluster-sweeps` | sweep scripts | Excluded (inventory disposition) |
| `repo-hygiene` | `.gitignore` | Out of scope |
| `documentation` | README usage | Out of scope; superseded by the `ccil` docs |

## Evidence

```bash
uv run --group evidence python tools/m3_labels/check.py --out <new-dir>
```

The report is at
[`../fixtures/evidence/m3-labels/report.json`](../fixtures/evidence/m3-labels/report.json).
