# The experimental vector-policy harness

> **Experimental reference infrastructure.** CCIL ends at its supervision
> artifacts, and users train downstream policies with any trainer
> ([supervision.md](supervision.md)). This harness exists to reproduce the
> paper's vector BC, NoiseBC and CCIL arms (#24) and to show that the
> artifacts are consumable. CCIL does not require it. It is not a general
> policy-training framework, and its interfaces may change.

```bash
uv run ccil policy train --recipe analytic_double_integrator --set data.path=rollouts.npz \
    --set generation.method=noisy_action --set generation.labels_per_transition=4 \
    --set generation.action_noise_std=0.05 --set policy.method=ccil \
    --set policy.batch_size=32 --set policy.epochs=4
```

    data.prepare → transition.fit → labels.generate [→ labels.certify] → policy.train → policy.evaluate

**Stage reuse.** The label stages are the offline pipeline's, and are reused
when complete. The arms (`bc`, `noise_bc`, `ccil`) share them.

**Outputs.**
- `policy.train` writes:
  - `checkpoints/policy.npz`;
  - `supervision.json`, the supervision manifest;
  - the training record, in `metrics.json`.
- `policy.evaluate` writes offline metrics.

**Dependencies.** It needs the `ccil[torch]` extra.
[`src/ccil/training/vector_bc.py`](../src/ccil/training/vector_bc.py) is
imported only by this command, never by `ccil labels generate`.

## The legacy protocol

Read from `archive/upstream-main-2026-10-22` (`train_bc_policy.py`,
`CustomBC.py`, `utils.py`) and the pinned d3rlpy fork
(`personalrobotics/d3rlpy@e89a7da`):

| Item | Legacy | Here |
|---|---|---|
| Network | d3rlpy default `VectorEncoder` [256, 256], ReLU after every hidden layer, `Linear`, tanh (`DeterministicRegressor`) | Same; `policy.hidden_layers` |
| Observation scaler | `scaler='standard'`: (x − mean) / (std + 1e-3), population std over all training rows | Same; `policy.observation_scaler` |
| Action scaler | `action_scaler='min_max'`: ((a − min) / (max − min + 1e-8)) · 2 − 1, inverted at prediction | Same; `policy.action_scaler` |
| Optimizer | Adam(`policy.lr`), betas (0.9, 0.999), eps 1e-8, weight decay 0 | Same; `policy.learning_rate` |
| Iteration | `RoundIterator`: a fresh shuffle each epoch, ⌊N / batch_size⌋ minibatches, remainder dropped | Same |
| Training rows | Expert transitions (t ≤ T − 2) plus the selected labels | The supervision contract's rows |
| `aug.balance_data` | Expert set × (num_labels + 1); epochs ÷ num_labels | Same; `policy.balance_noisy_labels` (`ccil` with noisy-action labels) |
| Loss | `F.mse_loss` in scaled action space | The weighted form below, equal to `F.mse_loss` with unit weights |
| NoiseBC | In `compute_loss`, after the scaler: mask `rand > 0.5` and `noise_bc · U[0, 1)`, written into the batch in place | The same placement and mask, through [`noise_bc.py`](../src/ccil/training/noise_bc.py): `symmetric_uniform_half_batch` (default) or `legacy_uniform_half_batch`, counter-keyed, never mutating |
| Offline score | `continuous_action_diff_scorer` on the expert data | `expert_data_error` (below) |

## Loss and weights

Per minibatch, in scaled action space:

    L = Σᵢ wᵢ · mean_j (π(sᵢ) − aᵢ)²_j / Σᵢ wᵢ

- The weights come from the supervision contract. Demonstration rows use
  `policy.demonstration_weight`, and synthetic rows keep their recorded
  weights.
- With every weight 1, L is legacy `F.mse_loss`.
- A minibatch whose weights sum to 0 takes no optimizer step. It is still
  counted as a step, and recorded as `zero_weight_steps`.

## Randomness, determinism and resume

**RNG streams.**

| Stream | Source |
|---|---|
| Initialization | `torch.manual_seed(run.seed)`, in a forked torch RNG |
| Epoch e's permutation | `SeedSequence([run.seed, e])` |
| NoiseBC | The #22 counter hash of (seed, optimizer step, sample identity, draw index = 0). A row occurs at most once per step, so every step draws afresh. |

**Determinism.** On CPU, the same seed gives byte-identical checkpoints.

**Resume.** A checkpoint written at an epoch boundary
(`policy.checkpoint_every`) resumes exactly. It holds the parameters,
scalers, Adam state, epoch and NoiseBC trace chain, and no RNG state is
needed. On CPU the resumed run is byte-identical to an uninterrupted one. On
CUDA this is not claimed: the evidence reports the measured difference.

## Checkpoint and records

`checkpoints/policy.npz` contains no executable serialization:
- **Arrays:** `net/*` (parameters), `scaler/*`, and `adam/<i>/{step, exp_avg,
  exp_avg_sq}`.
- **Meta:** the trainer (`vector_mlp`), version, dimensions, scalers, epoch,
  run seed, method, and the supervision's input hashes.

**The training record** (`metrics.json` → `training`):
- the settings and the optimizer;
- the rows and row sources, steps per epoch, and rows dropped per epoch;
- per epoch: the mean loss, the rows seen per source, and the zero-weight
  steps;
- for `noise_bc`: the full NoiseBC descriptor (mode, distribution, scale and
  units, truncation, mask probability, seed and RNG, draw schedule), the
  per-epoch trace hashes, and the combined `trace_sha256`.

**`policy.evaluate`** reports offline metrics only:
- `expert_data_error`: the mean over every demonstration transition of
  Σ_j (a − π(s))², in raw action units, as legacy d3rlpy's
  `continuous_action_diff_scorer`;
- the same on the train and validation episodes.

Simulator evaluation belongs to #24 and M7, through
`VectorBCPolicy.predict(observations)`, which returns raw actions.

## Parameters

| Key | Default | Units / domain | Source of value | Ownership |
|---|---|---|---|---|
| `policy.method` | `ccil` | `bc`, `noise_bc`, `ccil` | Legacy `policy.naive` + `policy.noise_bc` | #89 |
| `policy.hidden_layers` | [256, 256] | widths ≥ 1 | d3rlpy `VectorEncoderFactory` default | #89 (experimental) |
| `policy.learning_rate` | 0.001 | > 0 | Legacy `policy.lr`, the same in every config | #89 |
| `policy.batch_size` | (required) | ≥ 1 | Legacy `policy.batch_size`, per suite | #89 / #24 |
| `policy.epochs` | (required) | ≥ 1 | Legacy `policy.train_epochs`, per suite | #89 / #24 |
| `policy.observation_scaler` | `standard` | `standard`, `none` | Legacy `scaler='standard'` | #89 |
| `policy.action_scaler` | `min_max` | `min_max`, `none` | Legacy `action_scaler='min_max'` | #89 |
| `policy.balance_noisy_labels` | false | bool | Legacy `aug.balance_data` | #89 / #24 |
| `policy.demonstration_weight` | 1.0 | ≥ 0 | Legacy weighted every row equally | #89 |
| `policy.noise_scale` | (`noise_bc` only) | standardized observation units, ≥ 0 | Legacy `policy.noise_bc` | #89 / #24 |
| `policy.noise_model` | `symmetric_uniform_half_batch` | or `legacy_uniform_half_batch` | #22 | #89 |
| `policy.noise_mask_probability` | 0.5 | [0, 1] | Legacy `rand > 0.5` | #89 |
| `policy.checkpoint_every` | null | epochs ≥ 1 | — | #89 |
| `compute.device`, `compute.deterministic` | `cuda`, false | | M2 | M2 |

These keys and their domains are generated into
[`config/parameters.md`](config/parameters.md).

## Evidence

```bash
uv run --group evidence python tools/m3_policy/check.py --out <new-dir> [--device cuda]
```

The reports are in
[`fixtures/evidence/m3-policy/`](fixtures/evidence/m3-policy/), and each
arm's manifest, training record and offline metrics are under `artifacts/`.
The report shows:
- **Arms.** The three arms run from the same prepared data.
- **Legacy protocol.** It holds, with the scalers checked against
  transcriptions of d3rlpy.
- **Weights.** They follow the declared loss.
- **Reconciliation.** The rows seen reconcile with the manifest.
- **Determinism.** CPU training is deterministic.
- **NoiseBC.**
  - It replays from its key, draws afresh each step, and never mutates its
    inputs.
  - Legacy and corrected noise are compared at the same scale.
- **Resume.** It is byte-identical on CPU; the CUDA difference is measured.
- **Packaging.**
  - Checkpoints contain no pickle.
  - `ccil labels generate` runs without torch.
