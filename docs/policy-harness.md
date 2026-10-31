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

## Randomness, determinism and resume (trainer v1)

**RNG streams.**

| Stream | Source |
|---|---|
| Initialization | `torch.manual_seed(run.seed)`, in a forked torch RNG |
| Epoch e's permutation | `SeedSequence([run.seed, e])` |
| NoiseBC | The #22 counter hash of (seed, optimizer step, sample identity, draw index = 0). A row occurs at most once per step, so every step draws afresh. |

**Determinism.** On CPU, the same seed gives byte-identical checkpoints.

**Resume, at the command level.** `ccil policy train` reuses complete stages.
An interrupted `policy.train` is not resumed: it restarts from scratch.
Training is deterministic, so the rerun reproduces the same policy and costs
only time.

**Resume, at the library level.** `train_vector_bc(..., on_epoch=...)` can
save a **resumable** checkpoint at an epoch boundary, with
`policy.save(path, optimizer, epoch, state)`.
- **What it holds:** the parameters, scalers, complete Adam state, epoch,
  per-epoch records, NoiseBC trace chain and **training identity**. No RNG
  state is needed.
- **What the identity covers:** the supervision manifest's hash and inputs,
  method, seed, model, scaler, optimizer and batching settings, balancing,
  NoiseBC descriptor, and trainer version.
- **What `resume=` refuses:**
  - a checkpoint whose identity differs from the requested run's;
  - an incomplete Adam state;
  - an inference checkpoint.
- **Exactness.** On CPU the resumed run is byte-identical to an
  uninterrupted one. On CUDA this is not claimed: the evidence reports the
  measured difference.

## Trainer v2 (the default, #108)

`policy.trainer_version` selects the trainer. `2`, the default, is
[`vector_bc_v2.py`](../src/ccil/training/vector_bc_v2.py). `1` is the original
per-step loop described above, kept for comparison. Both train the same
`VectorBCPolicy` under the same protocol: architecture, initialization,
scalers, loss, Adam settings, batch size, epochs and balancing.

v2 is **not bit-compatible** with v1. Its shuffle and its arithmetic order
differ, so it is validated by engine-correctness checks and a comparison on
Pendulum that found no detected difference (below), not by parameter
agreement.

**One batched program for a group of models.** `vector_bc_v2.train_group`
trains M jobs (supervision, seed and, for NoiseBC, a perturbation) together.
`ccil policy train` runs a group of one; batched groups come through #112.

**Groups have fixed buckets and an execution identity.**
- **Buckets.** A group is padded to the next of 1, 2, 4, 8, …, 128 models
  with masked copies of its first job, which never update and are not
  returned. Compilation therefore specializes per bucket.
- **Why identity matters.** float32 rounding can depend on the bucket and
  the members.
- **What is recorded.** Every record and checkpoint carries the group's
  execution identity: the ordered members, the model's position, the bucket,
  the engine, the data mode and the precision. Two runs of the same job in
  different groups share a training identity, but not an execution
  identity.

**Data:**
- **Shared rows.** Rows are stored once in a shared table, and blocks are
  deduplicated by content, so seeds that share demonstrations hold them once.
- **No per-model arrays.** A model's training rows are a few segments of
  the table, and balancing repeats a segment.
- **Shuffle.** Minibatches come from a counter-based permutation: a keyed
  6-round Feistel network with cycle-walking, keyed by (seed, epoch).
- **Memory.** Working memory is O(M · chunk · B) indices plus O(M · P)
  parameters, besides the table.

**Modes** (`data_mode`, recorded):
- **Resident:** the table is on the device, and each step gathers its rows
  inside the step.
- **Streaming:** the table stays in pinned host memory. Indices are computed
  on the device, rows are gathered on the host, and chunks bounded by a byte
  budget are copied on a side stream, double-buffered. The source flag is
  streamed with the rows. Device memory does not grow with the table or the
  row width.
  - A failure in the producer reaches the trainer as an error, never a
    hang.
  - A trainer failure stops and joins the producer.

Both modes give bit-identical parameters.

**Step.**
- **Batching:** `torch.func` vmap over the stacked parameters.
- **Optimizer:** a functional Adam in `torch.optim.Adam`'s update order, with
  per-model masks. Models past their last step, and zero-weight minibatches,
  take no step.
- **Compilation:** `torch.compile` (default mode) once per process on CUDA,
  so groups of the same shapes reuse the compiled step. CUDA graphs
  (`reduce-overhead`) were measured slower: each replay copies the resident
  table into the graph's buffers.
  - The step specializes only on the bucket, the NoiseBC flag and the recipe
    (#134). The learning rate is a tensor, the resident table's row count is
    dynamic, and a chunk's indices are step-major, so every step has the same
    strides. Groups that differ in table size or learning rate reuse one
    compiled step, and results do not depend on the chunk size.
  - Past `torch.compile`'s recompile limit (8 specializations per function and
    process), a compiled function would run uncompiled with only a warning.
    A group adds at most one specialization, so a group whose step is
    already at the limit starts a fresh compiler context
    (`torch._dynamo.reset`). The compiled steps are unchanged; only how many
    are kept changes. A step that would still exceed the limit is refused with
    `CompilationRefused` before it runs. The batched fits share both.
  - Compilation is lazy, so the first compiled step is guarded. A compiler
    failure there falls back to eager for the whole run, with nothing
    updated, and the outcome is recorded as the engine. A study process
    (`ccil.experiments.runner`) refuses it instead.
- **Precision:** float32 with TF32 off is enforced during training, and the
  caller's settings are restored afterwards.
- **NoiseBC:** the #22 counter draws are computed on the device, bit-identical
  to the host draws.
- **Synchronization:** the host synchronizes once per chunk and once per
  epoch, never per step.

**Not in v2:**
- **Resume.** A group trains in seconds to minutes.
- **The per-step SHA-256 NoiseBC trace.** The draws themselves are
  reproducible from their keys.

**Records.** The training record and the checkpoint carry `trainer_version`,
and the training identity includes it. The record's `group` field holds:
- the group size and table blocks;
- the data mode and chunk size;
- the engine (compiled or eager), the device and the TF32 setting;
- `compilation`: the requested and effective engine, the step's
  specializations in the compiler context (in all, and new in this group),
  and whether the group started a fresh context;
- the shuffle rule;
- the setup, training and per-epoch times.

**Measured on siddh** (RTX PRO 6000 Blackwell, compiled, commit `d7257e0`;
the scaling report below):
- **Steady time per step** is flat at 0.34–0.38 ms from 10⁵ to 10⁸ rows.
  Groups of 1 to 16 models take the same time per step, so per-model cost
  falls linearly; 64 models take 0.89 ms per step.
- **For comparison,** v1 takes about 1.7 ms per step for one model (phase 0,
  #114).
- **Real Pendulum workloads**, 20 distinct label sets trained as one group,
  against one model alone:

  | Task, arm | 20 models | 1 model |
  |---|---|---|
  | Continuous CCIL, 100 epochs | 0.45 ms per step, 95 s | 0.33 ms per step, 69 s |
  | Discontinuous CCIL, balanced | 0.49 ms per step, 53 s | 0.37 ms per step, 35 s |
  | BC and NoiseBC, either task | 0.44–0.50 ms per step, 7–25 s | |

- **Peak device memory:**
  - streaming stays at 0.12 GB from 10⁷ to 10⁸ rows;
  - 16 models at a 512-dimensional observation use 2.1 GB resident, of
    which the table is 1.9 GB, and 0.84 GB streaming.
- **One-time costs:** compiling takes a few seconds per new shape, and a
  repeated group of the same shapes reuses it. Host setup at 10⁸ rows is
  23 s.
- **Limitation.** Streaming wide observations is bound by host memory
  bandwidth: 2.65 ms per step at D = 512, against 0.51 ms resident. Resident
  mode is chosen whenever the table fits the memory budget.
- **CUDA graphs** (`reduce-overhead`) were measured and rejected: 2.6 against
  1.0 ms per step at 64 models, and 3.4 against 1.2 ms per step with 1.8 GB
  more memory at D = 512, because each replay copies the resident table.

### The training recipe (#115)

Trainer v2 also offers a modern recipe for larger datasets. Every key
defaults to v1's protocol, so the paper's settings are unchanged unless a key
is set.

| Key | Default | Meaning |
|---|---|---|
| `policy.optimizer` | `adam` | `adamw` for decoupled weight decay |
| `policy.weight_decay` | 0 | AdamW weight decay; nonzero needs `adamw` |
| `policy.lr_schedule` | `constant` | `cosine`: linear warm-up, then cosine decay to 0 at the last step |
| `policy.warmup_steps` | 0 | Warm-up steps of the cosine schedule; refused with `constant` |
| `policy.step_budget` | null | Optimizer steps per model, replacing epochs; the last epoch is cut at the budget |
| `policy.grad_clip_norm` | null | Global gradient-norm clipping, per model |
| `policy.ema_decay` | null | An exponential moving average of the weights, which then becomes the policy |
| `policy.checkpoint_selection` | `last` | `validation`: keep the weights with the lowest weighted loss on the split's validation-episode demonstrations |
| `policy.validation_interval_steps` | null | With `validation`: validate every this many optimizer steps and at the last step; null validates once per epoch |

**Validation selection.**
- **Data:** it uses only held-out demonstrations, never evaluation data or
  generated labels.
- **Requirements:** `policy.exclude_validation_episodes true` and a
  validation fraction.
- **Loss:** the training loss, Σ w·mean((π(s) − a)²) / Σ w in the scaled
  action space, on the weights that would be returned: the EMA weights when
  EMA is on, which are then the policy.
- **Cadence:** once per epoch, or every `policy.validation_interval_steps`
  optimizer steps and at each model's last step. Steps are counted as v1
  counts them (epoch × steps per epoch + step), so arms of different dataset
  sizes get the same selection points under a shared step budget.
- **Records:** every validation (step, epoch, loss), the selected one, and
  its step and loss go in the training record.
- **Memory:** identical held-out sets (the arms of one seed) are stored once
  in a host table. The loss reads each model's set in chunks of at most
  `stream_chunk_bytes`, so device memory does not grow with the number or
  size of the validation sets.
- **Identity:** the training identity includes the held-out rows' content
  (`validation_sha256`), since the returned policy depends on them.
- **Refused at ingress:**
  - a held-out set with zero total weight;
  - a cosine warm-up that is not shorter than the model's optimizer steps.
  - a warm-up with the constant schedule, which has none.

**Step budget.** Each epoch runs only as many steps as its longest remaining
budget, so a budget bounds the work whatever the dataset size. Each epoch's
record gives the steps run, the optimizer updates and the rows seen,
including a last epoch cut short by the budget.

**Schedule details.** The cosine schedule warms up linearly from lr/W at the
first step to lr at step W (W = `policy.warmup_steps`), then decays to 0 at
the model's last step. AdamW uses β = (0.9, 0.999) and ε = 1e-8, as
`torch.optim.AdamW`.

**Trainer v1** refuses every recipe key.

### Stopping and exposure for the shared harness (il-supervision-benchmarks#11)

More `train_group` settings serve the shared study harness. They are library settings, not configuration keys. Unset, they leave training and every training identity unchanged.

| Setting | Default | Meaning |
|---|---|---|
| `patience` | null | With `checkpoint_selection: validation`: stop a model after this many validations without improvement. The step budget or epochs become only a ceiling. |
| `min_delta` | 0 | The improvement a validation must make (loss below the best minus `min_delta`) to count, both for patience and for the selected checkpoint. Needs `patience`. |
| `synthetic_fraction` | null | Every minibatch carries `round(fraction × B)` synthetic rows and the rest demonstrations. Excludes `balance_labels`. |
| `lr_schedule: plateau` | | Lower the learning rate by `lr_factor` after `lr_patience` validations without improvement, never below `min_lr`; the model stops only at `min_lr`, after `patience` more validations without improvement. Needs `patience`, `lr_factor`, `lr_patience` and `min_lr` (below the learning rate). |
| `validation_interval_epochs`, `validation_min_steps` | null | An exposure-aware cadence instead of `validation_interval_steps`: each model validates every max(S_min, E × ⌈N_demo / b_d⌉) steps, b_d being the demonstrations per minibatch ((1 − α)B with `synthetic_fraction` α, else B). This demonstration-equivalent epoch is independent of the synthetic rows, so patience counts roughly epochs on large datasets and a minimum of steps on small ones. Set together. |
| `synthetic_coverage_before_stop` | null | The stopping rule may fire only once every synthetic row has been drawn, at step ⌈N_syn / b_s⌉ (the synthetic stream is one counter over keyed permutations). `exposure.synthetic_coverage` records the draws, the distinct rows and the per-row minimum and maximum, for the trained steps and at the stop. Needs `synthetic_fraction` and `patience`. |
| `extend_after_stop` | null | A convergence check: a model whose rule fires at step s trains on to the first validation at or after `factor × s` (within the ceiling), then returns the checkpoint selected at s. Needs `patience`. |

**Patience.**
- **Stopping:** a model that stops takes no further step, and the rest of its group continues.
- **Record:** `stopping` gives the rule, the step at which it fired, and the steps run.
- **Ceiling reached first:** `stopping.outcome` is `incomplete_fit`, never convergence.
- **Recomputable:** the stop and the selected checkpoint follow from the saved validation history alone. A run cut at the selected step returns the same weights.
- **Plateau:** each validation records the learning rate it trained under, and `stopping` lists the reductions, so the reductions, the stop and the selection recompute from the history. Each reduction restarts the stopping window.
- **Extension:** `stopping.extension` gives the loss selected at the stop, the best loss up to the extended step, and the relative gain. The returned weights are those of the same run without the extension.

**Synthetic fraction.**
- **Epoch:** one pass over the demonstrations, shuffled as before.
- **Synthetic stream:** synthetic rows run as their own stream of keyed permutations, so each is drawn once before any repeats.
- **Record:** `exposure` gives the requested fraction, the rows and weight mass actually consumed per source, and the realized row and mass fractions.
- **Tolerance:** the realized row fraction is within the rounding tolerance 1/(2B).
- **Epoch records:** they also carry `weight_mass`.

Evidence: the `contract` part of `tools/m3_trainer_v2/check.py`.

## Checkpoint and records

Checkpoints contain no executable serialization. `meta.checkpoint_kind` says
which kind a checkpoint is:
- **Inference** (`checkpoints/policy.npz`, written by `policy.train`):
  - `net/*` (parameters) and `scaler/*`;
  - meta: the trainer (`vector_mlp`), version, dimensions, scalers, epoch,
    run seed, method, the supervision's input hashes, and the training
    identity's hash.
- **Resumable** (library level only): the inference arrays, plus
  `adam/<i>/{step, exp_avg, exp_avg_sq}` for every parameter, plus the full
  training identity and the per-epoch state in the meta.

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
| `policy.trainer_version` | `2` | `1`, `2` | v2 (#108); v1 is the original loop | #108 |
| `policy.hidden_layers` | [256, 256] | widths ≥ 1 | d3rlpy `VectorEncoderFactory` default | #89 (experimental) |
| `policy.learning_rate` | 0.001 | > 0 | Legacy `policy.lr`, the same in every config | #89 |
| `policy.batch_size` | (required) | ≥ 1 | Legacy `policy.batch_size`, per suite | #89 / #24 |
| `policy.epochs` | (required) | ≥ 1 | Legacy `policy.train_epochs`, per suite | #89 / #24 |
| `policy.observation_scaler` | `standard` | `standard`, `none` | Legacy `scaler='standard'` | #89 |
| `policy.action_scaler` | `min_max` | `min_max`, `none` | Legacy `action_scaler='min_max'` | #89 |
| `policy.balance_noisy_labels` | false | bool | Legacy `aug.balance_data` | #89 / #24 |
| `policy.demonstration_weight` | 1.0 | ≥ 0 | Legacy weighted every row equally | #89 |
| `policy.exclude_validation_episodes` | true | bool | Legacy BC trained on every episode (false); converted with `--fidelity historical` ([legacy-conversion.md](migration/legacy-conversion.md)) | #23 / #24 |
| `policy.noise_scale` | (`noise_bc` only) | standardized observation units, ≥ 0 | Legacy `policy.noise_bc` | #89 / #24 |
| `policy.noise_model` | `symmetric_uniform_half_batch` | or `legacy_uniform_half_batch` | #22 | #89 |
| `policy.noise_mask_probability` | 0.5 | [0, 1] | Legacy `rand > 0.5` | #89 |
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
  Checkpoints from another training run (method, supervision, seed, learning
  rate, batch size, balancing, NoiseBC), incomplete Adam state, and inference
  checkpoints are all refused.
- **Packaging.**
  - Checkpoints contain no pickle.
  - `ccil labels generate` runs without torch.

### Trainer v2

```bash
uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \
    python tools/m3_trainer_v2/check.py --out <new-dir> --device cuda --parts correctness,robustness,modes,equivalence
# timing, on a quiet host:
uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \
    python tools/m3_trainer_v2/check.py --out <new-dir> --device cuda --parts realworkload,scaling
```

The reports are in [`fixtures/evidence/m3-trainer-v2/`](fixtures/evidence/m3-trainer-v2/), all from the
clean commit `d7257e0`:
- `report.json`: macOS, CPU, 10/10 (correctness, robustness, modes);
- `report-siddh.json`: CUDA, 11/11 (adds the comparison with v1), with
  per-seed returns in `equivalence-siddh.json`;
- `scaling-siddh.json`: CUDA, 2/2 (real workloads and scaling), on a quiet
  host.

- **Engine correctness:**
  - one full-batch step equals a `torch.optim.Adam` step of v1's model on
    v1's loss;
  - a zero-weight model is left exactly unchanged inside a training group;
  - NoiseBC draws are bit-identical to the host's;
  - each epoch visits every training row once (less the dropped
    remainder), and balancing matches v1.
- **Modes and determinism:** resident equals streaming, and reruns are
  identical, on CPU and CUDA.
- **Robustness:**
  - a failing producer raises promptly;
  - a synthetic block identical to the demonstration block keeps its source;
  - execution identities differ between groups;
  - a compiler failure falls back to eager with unchanged results;
  - TF32 is enforced off and restored.
- **No detected difference from v1 on Pendulum:** 10 training seeds, on BC
  and CCIL.
  - The rule was declared before running: each 95% union interval of
    mean(R_v2 − R_v1) must contain 0.
  - The CCIL intervals are wide, so this is **not** evidence of equivalence.
  - A tighter 40-seed margin test (`--parts equivalence_margin`) is
    declared in the tool and was not run; #108 records why. v2 rests on
    engine correctness, and #113 compares arms trained by the same trainer.
- **Real workloads:** 20 distinct Pendulum label sets per task, trained as
  20-seed groups of BC, NoiseBC and CCIL (balanced on discontinuous).
- **Scaling:** steady-state time per step, one-time compile cost, setup and
  peak memory, as models, rows, observation dimension and width grow.
