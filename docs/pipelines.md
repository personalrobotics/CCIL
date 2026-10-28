# Pipelines

The `ccil` core has two public workflows (#19). They are implemented in
[`src/ccil/pipelines/`](../src/ccil/pipelines/) and share the configuration
contract ([`config/`](config/README.md)), the core interfaces
([`interfaces.md`](interfaces.md)) and the run directories ([`runs.md`](runs.md)).

## Stages are runs

Each stage writes its own run directory.

- **Identity.** A stage's run ID hashes:
  - the stage name and the configuration keys that stage consumes;
  - the components it uses, by kind or name and version (representation,
    transition model, generator, weighting, trainer);
  - the running implementation: the ccil version, a SHA-256 over every file of
    the ccil package, and the numpy, pyyaml and torch versions;
  - the SHA-256 of every input artifact.

  The full identity is written to the stage's `stage.json`, listed in its
  manifest.
- **Checkpoint handoff.** Upstream outputs, such as a dataset, a representation,
  a transition model or labels, become the downstream manifest's `inputs`, with
  their hashes.
- **Resume.** A stage whose run directory already validates as `complete` is
  reused unless `run.overwrite=true`. A stage that fails leaves no run directory
  and no staging, so a rerun resumes at the first incomplete stage. A change
  upstream changes every downstream run ID, so stale outputs are never reused.
  Reuse also requires byte-identical ccil source: any edit, committed or not,
  writes new runs and leaves the old ones intact. If the source can't be read,
  the stage fails rather than reusing.
- **Custom trainers.** A trainer passed from Python is outside the ccil package,
  so its `name` and `version` are its identity. Bump `version` when its
  behavior changes.
- **Failures.** Configuration problems are reported before any stage writes.
  A stage failure is reported with the stage name, and earlier stages stay valid.

## Offline generation: `ccil labels generate`

    data.prepare → transition.fit → labels.generate

The output is a label set, `labels-generate-*/labels/candidates.npz`, with
accepted and rejected rows, plus a report in its `metrics.json`: counts by
reason, the acceptance rate, and the generator identity. The label set records
its own provenance: the run seed, the uniform weighting, a weight per row (0 on
rejected rows), and the SHA-256 of the stage's inputs (dataset and transition
model, plus the representation in the integrated pipeline). These match the
stage manifest.

**`transition.fit`** fits the label model and reports a held-out one-step error
(#13):
- **`transition_model.fit_episodes all`** (the default, as legacy): the label
  model fits every episode, and a separate evaluation model, fitted on the train
  split, is scored on the held-out episodes.
- **`train`:** the label model fits the train split only, and it is the model
  scored. Held-out episodes then influence no training label: a clean
  holdout, for studies that select or tune on those episodes. Labels are
  still generated for held-out episodes' transitions; supervision excludes
  them when `policy.exclude_validation_episodes` is true.
- **`transition_model.trainer_version 2`** fits mlp models with the batched
  engine (`models/mlp_batched.py`): the same protocol (initialization,
  regularizers, Adam with L2 decay, epochs and batch), vmapped and compiled so a
  study runner can fit many seeds as one group. It is not bit-compatible with
  version 1, so the version is part of the stage identity. It supports
  lipschitz types `none`, `slack`, `soft_sampling` and `soft_sampling_slack`
  (preflight refuses the others), and honors `compute.deterministic`.
- **Grouped results are authenticated.** Each batched fit records a content
  identity: its exact rows, settings, seed and engine. `transition.fit` with a
  prefit model recomputes it from the unit's own inputs and refuses a mismatch.
  Likewise, `policy.train` with a group-trained policy recomputes the training
  identity and checks the policy's parameter hash against its record. Both
  refuse a group whose effective engine is not the one it requested (#134).
- **`transition_model.lipschitz_coordinates`** (#125) says which ratio the
  sampled Lipschitz penalties (`soft_sampling`, `soft_sampling_slack`) hinge
  at `lipschitz_constraint`:
  - `model_normalized` (the default). It perturbs the model's normalized input
    x = (q − shift) / (scale + 1e-8) by δx ~ ε N(0, I), and takes
    ‖g(x + δx) − g(x)‖ / ‖δx‖ with g the normalized residual. This ratio is
    dimensionless, in the space the network is optimized in.
    - **Invariance.** It is invariant, up to the normalization's 1e-8 and
      ReLU kinks, under a consistent per-coordinate rescaling of states,
      actions and residuals.
    - **Timestep.** A different timestep changes it only as far as the
      refitted normalization absorbs the change.
    - **Units of ε.** ε is a radius in normalized input units.
  - `raw` (#21). Raw residuals over raw perturbations, so the constant
    depends on the units and the timestep. On continuous Pendulum the paper's
    L = 2 never binds in raw coordinates.
  - `soft_jac` supports `raw` only, and needs it set explicitly.
  - **Compatibility.** The default is `config_version 2`'s. A configuration
    of `config_version` 0 or 1 that does not set the key resolves to `raw`,
    its meaning when it was written (the resolved configuration notes this).
    Converted legacy configurations are version 1. A checkpoint written
    before #125 loads as `raw`, explicitly.
  - **Checkpoint provenance.** Checkpoints store the coordinates' descriptor
    (coordinates, units, constraint and normalization SHA-256), and loading
    verifies it against the weights.
  - Fit records state the coordinates, the units and a SHA-256 of the
    normalization.
- **The penalty is a training regularizer, not a certificate.** The value of
  `lipschitz_constraint` is never a proven bound in either coordinate system.
  Certificates recompute model bounds from the weights and normalization
  transforms (no certificate code reads it). A normalized bound used directly
  in a theorem would need every distance in that theorem measured in the same
  weighted norm, or a conservative conversion to raw coordinates.
- **Evidence:** `tools/m3_fit_v2/check.py`.
  - One step equals version 1's step from the same initialization.
  - The held-out error of batched fits is compared with per-fit ones on
    Pendulum.
  - The clean holdout is checked by perturbing the held-out episodes, which
    changes no training label.

The augmentation is the accepted rows together with the prepared dataset. The
M4 dataset writer turns them into training data. Offline generation never
imports the integrated pipeline or a policy trainer.

```bash
uv run ccil labels generate --recipe analytic_double_integrator --set data.path=fixture.npz
```

A `labels.certify` stage can follow `labels.generate`. It writes
`labels-certify-*/labels/certificates.npz` and leaves the candidates unchanged.
The profile's specification, admission policy, selection and references are part of its identity. See
[certificates.md](certificates.md).

## Experimental vector-policy harness: `ccil policy train`

    data.prepare → transition.fit → labels.generate [→ labels.certify] → policy.train → policy.evaluate

Reference infrastructure for reproducing the vector BC, NoiseBC and CCIL
arms. It needs the `ccil[torch]` extra, and CCIL does not require it. See
[policy-harness.md](policy-harness.md).

## Integrated pipeline: `ccil pipeline integrated`

    data.prepare → representation.fit → transition.fit → labels.generate → policy.train → policy.evaluate

CCIL owns stages 1–4. The downstream trainer plugs in through `PolicyTrainer`,
either `policy.trainer` or `run_integrated(cfg, trainer=...)`. Changing only the
trainer reuses stages 1–4 and reruns the two policy stages.

A trainer implements `fit`, `predict`, `save` and `load`. `policy.train` takes
its rows from the supervision contract ([supervision.md](supervision.md)),
and records the supervision manifest (arm, input hashes, row-count
reconciliation) as its metrics. `policy.evaluate`
always loads the policy from the `policy.train` checkpoint, so it evaluates the
same policy whether training ran now or was reused. Changing only
`evaluation.*` settings reuses stages 1–5.

The reference components stand in for what M5 and later milestones will
provide, and claim no image-CCIL parity:

| Reference component | Stands in for |
|---|---|
| `PCARepresentation`, over state or flattened image pixels, written as `latent.pca` labels | M5's learned image encoders |
| `LinearPolicyTrainer`, `MeanActionTrainer` | real policies |
| Offline action-prediction error on held-out episodes | simulator evaluation (M7) |

## Configuration

| Parameter | Choices | Status |
|---|---|---|
| `generation.method` | `backward_euler`, `backward_euler_fast`, `forward_euler` (exact preimage), `noisy_action` | Implemented |
| `generation.method` | `noisy_action_fe` | Implemented (#22). See [`migration/label-methods.md`](migration/label-methods.md). |
| `generation.method` | `backward_model` | Fails with guidance to #22 |
| `generation.model_error_threshold` | number | Implemented (#22). It adds `ModelMisfit`, the model-error gate. |
| `certificate.*` | profile, constants, thresholds | Implemented (#88). With `certificate.profile` set, `ccil labels generate` adds the `labels.certify` stage. `generation.max_label_error` and `generation.label_error_quantile` were withdrawn in favour of these keys; see [certificates.md](certificates.md). |
| `generation.action_chunk_size` | int | Implemented (#22). Adds action chunks: the label action, then the demonstration's continuation, masked at the episode end. Only position 0 carries the one-step claim. See [`migration/label-methods.md`](migration/label-methods.md). |
| `transition_model.kind` | `linear` | Implemented |
| `transition_model.kind` | `mlp` (the legacy default; needs `ccil[torch]`, Python 3.11+) | Implemented (#21). See [`migration/vector-dynamics.md`](migration/vector-dynamics.md). |
| `representation.kind` | `identity` (offline), `pca` (integrated) | Implemented |
| `policy.method` | `ccil`, `bc` | Implemented |
| `policy.method` | `noise_bc` | The input perturbation is implemented (`ccil.training.noise_bc`, #22). Training integration arrives with #89. |

## Evidence

```bash
uv run --group evidence python tools/m2_pipelines/check.py --out <new-dir>
```

[`fixtures/evidence/m2-pipelines/report.json`](fixtures/evidence/m2-pipelines/report.json)
has 15 cases:

- **Offline:** analytic generation passes the #10 oracle, replays byte for
  byte, and is fully reused on rerun. It imports no trainer. Its labels carry
  provenance equal to the stage manifest.
- **Implementation identity:**
  - each `stage.json` re-derives its run ID;
  - a one-comment source change or a component version bump writes distinct
    runs, leaving the old runs byte-identical;
  - an unreadable implementation exits 2 without reusing anything.
- **Integrated:** all six stages complete, with checkpoint handoff verified by
  hash.
- **Trainer swap:** reuses the four CCIL stages.
- **Evaluation-only change:** reuses stages 1–5. The new evaluation loads the
  reused policy and reproduces the original metrics.
- **Failure and resume:** a `policy.train` failure leaves the earlier stages
  intact, and the rerun resumes there. A `policy.evaluate` failure leaves
  `policy.train` complete, and the rerun reuses it and completes.
- **Preflight:** unsupported configurations fail before any stage writes.
- **Help:** each command states its inputs, outputs, optional dependencies and
  exit codes.
