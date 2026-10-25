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
stage manifest. `transition.fit` also
reports the held-out error of an evaluation model fitted on the train split
(#13).

The augmentation is the accepted rows together with the prepared dataset. The
M4 dataset writer turns them into training data. Offline generation never
imports the integrated pipeline or a policy trainer.

```bash
uv run ccil labels generate --recipe analytic_double_integrator --set data.path=fixture.npz
```

A `labels.certify` stage can follow `labels.generate`. It writes
`labels-certify-*/labels/certificates.npz` and leaves the candidates unchanged.
The profile's constants and thresholds are part of its identity. See
[certificates.md](certificates.md).

## Integrated pipeline: `ccil pipeline integrated`

    data.prepare → representation.fit → transition.fit → labels.generate → policy.train → policy.evaluate

CCIL owns stages 1–4. The downstream trainer plugs in through `PolicyTrainer`,
either `policy.trainer` or `run_integrated(cfg, trainer=...)`. Changing only the
trainer reuses stages 1–4 and reruns the two policy stages.

A trainer implements `fit`, `predict`, `save` and `load`. `policy.evaluate`
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
| `generation.model_error_threshold`, `max_label_error`, `label_error_quantile` | numbers | Implemented (#22). They add `ModelMisfit` and `LabelError`, which uses the generator's BackTrack or DisturbedAction definition. Label-error filtering requires `model_error_threshold`. See [`migration/label-methods.md`](migration/label-methods.md). |
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
