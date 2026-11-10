# Pipelines

> The user-facing workflow (Diagnose, Generate, Curate, Inspect) and its artifact vocabulary are specified in [contract.md](contract.md).

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

## Offline generation: `ccil diagnose`, `ccil generate`, `ccil curate`

    ccil diagnose:  data.prepare → transition.fit → labels.diagnose
    ccil generate:  ccil diagnose's stages (reused) → labels.generate [→ labels.certify]
    ccil curate:    [labels.calibrate →] labels.curate [→ labels.audit], any number per generate result

`ccil generate` writes an immutable `run.output_dir/generate-<id>.json` (`ccil.generate-result` 1.0), copied to
`generate-result.json`, which `ccil curate` reads; curate first verifies every artifact it records and that each
path stays in that directory ([contract.md](contract.md#commands)). `ccil inspect` prints a result's configuration,
provenance, quality, rejections and usable rows. The Python entry points are `ccil.diagnose`,
`ccil.generate`, `ccil.curate` and `ccil.inspect` ([contract.md](contract.md#commands)). `ccil labels
generate`, deprecated, runs generate and curate in one command and keeps `--result` (`ccil.labels-result` 1.0).

`labels.calibrate` and `labels.audit` read reference dynamics as declared
evidence ([certificates.md](certificates.md#calibration-and-the-reference-audit)).

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
  them when `supervision.exclude_validation_episodes` is true.
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

**Machine-readable result.** `--result PATH` also writes the run's result as
JSON (`ccil.labels-result` 1.0), whole, so a harness never parses the printed
lines:
- `config`: the resolved configuration record, as each stage's `config.json`
  holds it;
- `stages`: each stage in order, with its absolute `run_dir` and whether it was
  `reused`;
- `artifacts`: the absolute path of each artifact the run produced, by name:
  `dataset`, `split`, `transition_model`, `diagnosis`, `candidates`,
  `certificates`, `calibration`, `selection` and `audit`. A name whose stage did
  not run, or whose file the stage did not write (a split with
  `data.validation_fraction` 0), is absent.

A new field is a minor version; a changed or removed one is a major version.

A `labels.certify` stage follows `labels.generate` when `certificate.profile`
is set. It writes `labels-certify-*/labels/certificates.npz`, the labels'
quality evidence, and leaves the candidates unchanged. The profile's
specification and references are part of its identity.

`labels.curate` always runs last. It writes `labels-curate-*/labels/selection.npz`,
the rows one policy selects (`curate.*`), and leaves the candidates and
certificates unchanged, so a new tolerance reruns only this stage. See
[certificates.md](certificates.md#curation).

## Diagnosis: `labels.diagnose`

`labels.diagnose` runs after `transition.fit` and before `labels.generate`. It
writes `labels-diagnose-*/diagnosis/diagnosis.json` (`ccil.diagnosis` 1.0):
the data-only decision whether one-step labels are worth generating. Its
inputs are the prepared dataset and split, the label model, and, with
`transition_model.fit_episodes all`, the evaluation model that
`transition.fit` now saves (`checkpoints/evaluation_model.npz`), which never
saw the held-out episodes.

- **The metric.** Every distance is ‖(x − y) / s‖, where s is the
  per-coordinate standard deviation of the training demonstration states.
  The action is not a coordinate.
- **τ_demo** is the `diagnose.tau_quantile` (0.95) of the distance from
  held-out demonstration states to their nearest training state: the
  demonstrations' own spacing. It uses at most `diagnose.tau_sample` (2048)
  evenly spaced held-out states, so its cost is linear in the training set,
  and records the sample. It needs held-out episodes.
- **The probe** is stratified by training episode under
  `diagnose.probe_transitions` (256). With at most that many episodes, each
  gets an equal share of evenly spaced transitions. With more, that many
  evenly spaced episodes get one transition each. The diagnosis records how
  many episodes the probe covers. The configured generator and validators
  label the probe, and only their summary is kept.
- **The gates, for M = 1.** Each asks a distinct question, and M is admitted
  only when all pass:

  | Gate | Question | Passes when |
  |---|---|---|
  | yield | Did generation work? | at least `diagnose.min_yield` (1%) of the probe labels are accepted |
  | displacement | How far did labels move from their sources? | the `diagnose.label_quantile` (0.9) of d_source is at least `diagnose.min_displacement` (1) τ_demo |
  | novelty | Do they reach states the dataset does not cover? | the same quantile of d_demo, the exact distance to the nearest training state, is at least `diagnose.min_novelty` (1) τ_demo |
  | fit | Does the model fit where it has data? | the `diagnose.fit_quantile` (0.95) of the held-out one-step error, over the median d_source, is at most `diagnose.fit_ratio` (0.5) |
  | trust | Is the model credible at the proposed states? | a bound certificate profile supplies off-data evidence: constants that are declared assumptions or computed from the model, never from reference dynamics |

  - **Displacement is not novelty.** A label can move far from its source and
    land beside another demonstration.
  - **Fit is necessary, never sufficient.** It is measured at demonstration
    states, where the data are.
  - **Trust cannot come from data alone.** A model can fit the
    demonstrations' manifold and be wrong off it: when the demonstrator's
    actions are an exact linear function of the state, the action response
    is unidentified, fit passes, and labels land thousands of units away.
    So `auto` admits only under a bound profile's recorded assumptions
    (`conditional`), and refuses without one.
  - **Query extrapolation is reported beside trust:** the distance of each
    proposed (z_G, a_G) from the nearest training (z, a) pair.
  - **The certificate follows admission.** Under declared assumptions, its
    bound on each label then decides which labels curation keeps.

- **`generation.recovery_steps`.**
  - `auto` (the default) admits M = 1 only when every gate passes.
  - `1` generates at M = 1 and still reports the gates. Configurations
    written before `config_version` 3, including the shipped recipes,
    resolve to `1`, with a note.
- **When nothing is admitted,** the diagnosis names the first failing gate
  and a next step. `labels.generate` then writes a schema-valid zero-row
  candidate set, and certification and curation follow with zero rows. Every
  run directory is complete, and supervision has no synthetic row.
- **Also recorded:**
  - `dt_ctrl`, from uniform timestamps, and the horizon H = M · dt_ctrl (null without timing);
  - the accepted probe labels counted by d_demo / τ_demo;
  - the recommended nested views for noisy generators;
  - the configured certificate's data-only evidence class.
- **Novelty on every label.** `labels.generate` adds `quality.d_source`, the
  distance from the source state in the same metric, to every candidate. The
  distance to the nearest training state, `quality.d_demo`, is the opt-in
  validator `ccil.core.quality.NearestDemonstration`: a brute-force search
  whose cost is candidates × training states, too slow for study-scale
  datasets until it is indexed.
- **Admission uses only data and declared assumptions.** Reference dynamics
  and policy outcomes never enter it. The diagnosis records
  `evidence_mode: data_only`.

## Integrated pipeline: `ccil pipeline integrated`

    data.prepare → representation.fit → transition.fit → labels.diagnose → labels.generate → policy.train → policy.evaluate

CCIL owns stages 1–4. The downstream trainer plugs in through `PolicyTrainer`,
either `integrated.trainer` or `run_integrated(cfg, trainer=...)`. Changing only the
trainer, or only `supervision.*`, reuses stages 1–4 and reruns the two policy
stages.

A trainer implements `fit`, `predict`, `save` and `load`. `policy.train` takes
its rows from the supervision contract ([supervision.md](supervision.md)),
built from `supervision.source`, `supervision.demonstration_weight` and
`supervision.exclude_validation_episodes`, and records the supervision manifest (arm, input hashes, row-count
reconciliation) as its metrics. `policy.evaluate`
always loads the policy from the `policy.train` checkpoint, so it evaluates the
same policy whether training ran now or was reused.

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
| `supervision.source` | `demonstrations_plus_selection`, `demonstrations` | Implemented (integrated pipeline). Replaces `policy.method` (#168); NoiseBC moved to il-supervision-benchmarks. |

## Evidence

```bash
uv run --group evidence python tools/m2_pipelines/check.py --out <new-dir>
```

Reports: [`m2-pipelines/report.json`](fixtures/evidence/m2-pipelines/report.json).
Each case, and the contract it protects, is listed in the docstring of [`tools/m2_pipelines`](../tools/m2_pipelines/).
