# Label methods: migration map (#22)

This page maps the legacy corrective-label variants and the NoiseBC baseline to
the `ccil` core. #22 runs as three PRs, and this page grows with each:

| PR | Contents |
|---|---|
| 1 | Noisy-action generation and NoiseBC (this page so far) |
| 2 | Filtering, weighting, and the label-filtering branch inventory |
| 3 | Action chunks |

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
  index, component), computed with a splitmix64 counter hash
  (`splitmix64-counter/1`).
- A draw is therefore reproducible from its key, fresh at the next step or
  draw, and independent of batch order and composition.
- `perturb` returns a new array.

**Recording.** `NoiseBC.descriptor()` gives what a run records: distribution,
scale, units, truncation, masking probability, seed, RNG, and draw schedule.
`NoiseTrace` gives a running SHA-256 over every draw, for reproducibility
without storing the draws.

## Evidence

```bash
uv run --group evidence python tools/m3_labels/check.py --out <new-dir>
```

The report is at
[`../fixtures/evidence/m3-labels/report.json`](../fixtures/evidence/m3-labels/report.json).
