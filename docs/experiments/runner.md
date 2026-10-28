# Predeclared studies: `ccil.experiments`

`ccil.experiments` runs predeclared studies: many seeds × arms × tuning values
of the CCIL pipeline. The predeclaration (for example
[pendulum-ccil-study-predeclaration.md](pendulum-ccil-study-predeclaration.md))
is the scientific contract. This package turns the study's plan into stage
artifacts and records and verifies them, without changing what any unit
computes.

| Module | Role |
|---|---|
| `plan` | A validated YAML study spec, expanded into units (task, arm, seed, value) |
| `runner` | One phase of a study, scheduled for throughput |
| `records` | Write-once unit records, phase verification, the tuning choice, deferred returns |
| `gate` | The code-state gate against a freeze manifest |
| `analysis` | Paired union intervals and the predeclared decision rules |

A **suite** supplies what is task-specific: configurations and data subsets,
simulator evaluation, the label-accuracy report and the rollout audit. The
Pendulum suite is `tools/m3_experiments/pendulum_suite.py`.

## Plans

A plan names the tasks, the study-wide settings, the phases with their seeds
and arms, every arm's settings, noise levels and data fraction, and the
tuning grids. `tools/m3_experiments/pendulum_v6.yaml` is the version 6
Pendulum design. Validation refuses:
- unknown keys;
- seeds shared between phases;
- a tuning phase with untuned arms;
- a tuned arm without a grid.

A unit's overrides are the study's settings, then its arm's, then its tuned
value, so later settings win.

Tuned arms select by one of two rules:
- `highest_mean_noise_free_R`: the audited noise-free return over the tuning
  seeds.
- `lowest_mean_validation_loss`: the selected checkpoint's held-out loss.
  - A tuning phase computes **no return** for these arms (`Study.deferred`).
  - `records.record_choice` writes the choice once (`tuning-choice.json`).
  - `runner.evaluate_deferred` then computes their tuning-seed returns into
    records that name the choice and their tuning records by SHA-256.
  - `records.verify_deferred` refuses any that do not. It also makes
    `verify_phase`'s checks: each record's path and phase, its gate and
    code state (shared with the tuning records), exactly the planned noise
    levels, each rollout's audited task, seed and noise level, and a status
    consistent with finite returns.

## The runner

`runner.run_phase` runs every unit of a phase that has no record. Its stage
artifacts and identities are exactly those `ccil policy train` would write:
the CLI with a unit's overrides reuses every stage. What changes is the
scheduling:

| Stage | How it runs |
|---|---|
| `data.prepare` | per unit |
| `transition.fit` | deduplicated by stage identity, then fitted in groups by `models/mlp_batched.fit_group` (trainer version 2). Each fit's content identity is recomputed by the stage, so a grouped result is never written under another unit |
| `labels.generate`, `labels.certify` | in worker processes, in waves that never let two processes write one stage |
| `policy.train` | in groups by `training/vector_bc_v2.train_group`. The stage checks each policy's training identity and parameter hash |
| simulator evaluation | the suite's batched evaluation. The Pendulum suite steps every unit and noise level together, with results bit-identical to evaluating each policy alone |

More scheduling details:
- **Group sizes.** Groups are sized by a measured cost model, a group's
  size plus a per-step overhead, so groups avoid both padding to the next
  bucket and many small groups run in sequence.
- **Out of memory.** A group that runs out of device memory is split in
  half, down to single units. A single unit out of memory is retried once,
  then recorded as an infrastructure fault.
- **Failures.** A unit's stage failure, a non-finite training loss or a
  non-finite return is an algorithmic failure. A configuration error stops
  the phase.
- **Sharding.** `shard=(i, n)` runs every n-th seed, so n processes, one
  per GPU, run a phase together.
- **Resume.** Records are written per chunk, and completed stages are
  reused, so a stopped phase resumes without recomputing.
- **Code state.** Before a chunk's records are written, the runner checks two
  things, and writes no record if either check fails:
  - the process's code state still equals what it captured at start
    (`provenance.verify_code_state`, docs/runs.md);
  - the caller's `recheck` passes. The Pendulum driver passes its gate again,
    at the same HEAD.

  A phase stopped this way resumes from a clean tree.

Driver scripts must guard their entry point (`if __name__ == "__main__":`):
label workers are spawned processes and re-import it.

## Records and verification

A record is one JSON file per unit, at
`results/<phase>/<task>/<arm>[@value]/s<seed>.json`. It is written atomically
and never overwritten. It carries:
- the code-state gate;
- the input hashes;
- the resolved configuration;
- the training, labels and certification reports;
- every evaluation's rollout path and SHA-256.

`records.verify_phase` refuses the phase unless:
- the records are exactly the plan's units;
- they share one code state;
- every rollout file has its recorded hash and passes the suite's audit;
- every record agrees with its audit.

The analysis reads only verified records.

## The Pendulum study driver

```bash
uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
    python tools/m3_experiments/run_study.py STEP --root DIR [--shard i/n] [--smoke]
```

The steps run in order: `tuning` (one process per shard), `choose`, `returns`,
`study` (one process per shard), `verify`.
- **Real runs** need `tools/m3_experiments/freeze.json`, and a clean tree
  equal to the frozen commits.
- **`--smoke`** runs the same steps before the freeze, on seeds 9101–9103 at
  one epoch, with budgeted recipes scaled down.

## Evidence

```bash
uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
    python tools/m3_experiments/check.py --out <new-dir> --runner
```

The check covers:
- **The #24 tooling.** Analysis, gate, plan and records match it.
- **The runner:**
  - a smoke phase verifies;
  - the CLI reuses every stage the runner wrote;
  - resume, after a code change and a failed gate each refused to write a
    record;
  - batched evaluation is bit-identical to solo evaluation;
  - a validation-tuned arm chooses with no return until the choice is
    recorded;
  - concurrent shards;
  - injected memory exhaustion.
