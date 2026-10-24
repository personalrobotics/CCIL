# Pipelines

The `ccil` core has two public workflows (#19). They are implemented in
[`src/ccil/pipelines/`](../src/ccil/pipelines/) and share the configuration
contract ([`config/`](config/README.md)), the core interfaces
([`interfaces.md`](interfaces.md)) and the run directories ([`runs.md`](runs.md)).

## Stages are runs

Each stage writes its own run directory.

- **Identity.** A stage's run ID hashes the stage name, the configuration keys
  that stage consumes, any explicit identity (such as a Python-supplied
  trainer), and the SHA-256 of every input artifact.
- **Checkpoint handoff.** Upstream outputs, such as a dataset, a representation,
  a transition model or labels, become the downstream manifest's `inputs`, with
  their hashes.
- **Resume.** A stage whose run directory already validates as `complete` is
  reused unless `run.overwrite=true`. A stage that fails leaves no run directory
  and no staging, so a rerun resumes at the first incomplete stage. A change
  upstream changes every downstream run ID, so stale outputs are never reused.
- **Failures.** Configuration problems are reported before any stage writes.
  A stage failure is reported with the stage name, and earlier stages stay valid.

## Offline generation: `ccil labels generate`

    data.prepare → transition.fit → labels.generate

The output is a label set, `labels-generate-*/labels/candidates.npz`, with
accepted and rejected rows, plus a report in its `metrics.json`: counts by
reason, the acceptance rate, and the generator identity. `transition.fit` also
reports the held-out error of an evaluation model fitted on the train split
(#13).

The augmentation is the accepted rows together with the prepared dataset. The
M4 dataset writer turns them into training data. Offline generation never
imports the integrated pipeline or a policy trainer.

```bash
uv run ccil labels generate --recipe analytic_double_integrator --set data.path=fixture.npz
```

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
| `generation.method` | `noisy_action_fe`, `backward_model` | Fail with guidance to #22 |
| `transition_model.kind` | `linear` | Implemented |
| `transition_model.kind` | `mlp` (the legacy default) | Fails with guidance to #21 |
| `representation.kind` | `identity` (offline), `pca` (integrated) | Implemented |
| `policy.method` | `ccil`, `bc` | Implemented |
| `policy.method` | `noise_bc` | Arrives in M3 |

## Evidence

```bash
uv run --group evidence python tools/m2_pipelines/check.py --out <new-dir>
```

[`fixtures/evidence/m2-pipelines/report.json`](fixtures/evidence/m2-pipelines/report.json)
has 10 cases:

- **Offline:** analytic generation passes the #10 oracle, replays byte for
  byte, and is fully reused on rerun. It imports no trainer.
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
