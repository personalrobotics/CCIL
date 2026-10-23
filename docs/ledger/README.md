# CCIL reproducibility ledger

Captured **2026-10-22**. Every previously runnable example and every reported
result has an entry here. Each entry records how to reproduce it, its current
status, and what blocks it. The ledger is [`ledger.json`](ledger.json), and its
versioned schema is [`schema.json`](schema.json). Refs and capability IDs come
from the [capability inventory](../inventory/README.md). This M0 ledger covers
CCIL and its preserved forks. The actively changing `image_ccil` repository is
deferred to M5 issue #31 and will receive its own baseline capture before image
integration begins.

To validate:

```bash
git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'
python3 scripts/check_ledger.py     # needs PyYAML, already a CCIL dependency
```

The validator checks:

- **Fields and enums.** Every entry has its required fields, and every
  enumerated value is valid.
- **Status rules.** `reproduced*` needs evidence links, `blocked` needs
  blockers with a next action, and `out-of-scope` needs a rationale.
- **Coverage.** Every README command, every script and config on `main`,
  `label-filtering` and the fork's `image` branch, and every inventory
  capability maps to an entry or an explicit exclusion. Every config key is
  classified in the parameter registry.

On capture it reported 30 entries, 63 parameters, and 0 errors. After the migration contract (schema v2) it reports 2 pending, 3 blocked, 25 out of scope, and 0 errors.

## Statuses

| Status | Meaning |
|---|---|
| `pending` | Not attempted yet. |
| `reproduced` | Matches within the entry's parity class. Evidence required. |
| `reproduced-with-correctness-change` | Matches after an explained correctness fix. Evidence and explanation required. |
| `blocked` | Cannot proceed. Names the missing artifact, dependency, license, hardware, knowledge or decision, and a next action. |
| `out-of-scope` | Intentionally not reproduced, with a rationale. |

Nothing has been rerun yet. An entry may not be marked reproduced from source
inspection alone, and historical output is not the oracle when it violates the
intended algorithm.

## Entries

Statuses and targets follow the
[migration contract](../migration-contract.md) §3 (schema v2).

| Entry | Claim | Status | Target |
|---|---|---|---|
| `sim-pendulum-cont` | Fig. 4(c) | pending | M3, the refactor-parity anchor |
| `sim-pendulum-disc` | Fig. 4(c), tracked in #3 | pending | M3 |
| `sim-pendulum-label-bound` | Sec. 5.1 error bound | blocked: procedure unrecorded | M3 |
| `internal-label-filtering-configs` | label-filtering configs | blocked: no linked result | M3, resolved through #3 and #25 |
| `internal-action-chunking-labels` | Sec. 6, App. C.3 | blocked: real-robot data | M3, capability ported in #22 |
| `sim-hopper`, `-walker2d`, `-ant`, `-halfcheetah` | Table 1 | out-of-scope: re-baselined on Minari in M7 | none |
| `sim-coffee-pull`, `-button`, `-coffee-push`, `-drawer-close` | Table 1 | out-of-scope: re-baselined on Meta-World v3 in M7 | none |
| `sim-f1tenth`, `sim-drone-*` | Figs. 5(c), 6(b) | out-of-scope: benchmark retired | none |
| `sim-model-based-baselines` | MOReL and MILO rows | out-of-scope: external baselines retired | none |
| `real-*` (6 entries) | Figs. 8–13, real robot | out-of-scope: cited as published | none |
| `setup-install`, `readme-*` | README install and commands | out-of-scope: legacy CLI retired | none |
| `internal-data-ablation` | Simulated data ablation | out-of-scope: subsumed by M7 low-data subsets | none |
| `internal-cluster-sweep` | sweep.sh / launch.slurm | out-of-scope | none |
| `deferred-image-pendulum` | Fork image branch | out-of-scope (deferred) | none |

Target commands are **provisional** until #17 fixes the CLI names.

## Findings that affect the claims

These are recorded in the entries' `discrepancies`. None of them has been
fixed.

- **The drone evaluation discards observations.** In
  `utils.evaluate_on_environment`, `mix if not drone_task else U[-1,1]`
  replaces every drone observation with uniform noise whenever
  `eval.noise > 0`. All drone configs use 0.0003, so the Fig. 6(b) numbers
  were produced this way.
- **The noise model differs from the paper.** The paper describes Gaussian
  sensor and actuator noise. The code mixes in a uniform sample of the
  observation or action space.
- **The NoiseBC noise is not zero-mean.** It is `noise_cov × U[0,1)`, added
  to half of each batch. `mujoco.sh` uses noise 1.0 for Ant and 0.0001
  elsewhere.
- **Datasets differ from paper Table 2.** Pendulum uses 100 demonstrations,
  not 50, and drone circle uses 50, not 30. Two Meta-World files don't state
  a count, and the paper's claim that Meta-World data comes from D4RL can't
  be right.
- **Seeds are inconsistent.** Pendulum uses seeds 41–50, and every other
  suite uses 40–49. Seeding sets no CUDA determinism flags.
- **Two F1 metrics share one label.** `main` prints return and
  `label-filtering` prints success rate, both as "Avg return".
- **The real-robot results are absent.** Figs. 8–13 depend on hardware and
  data that no preserved CCIL ref contains.

## Updating an entry

Change `status` only together with `evidence`. Evidence is an https link, or a
repository path to a run artifact that records commit, config, seeds,
environment and per-seed results. Then rerun `scripts/check_ledger.py`.
