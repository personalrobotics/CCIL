# CCIL reproducibility ledger

Captured **2026-10-22**. Every previously runnable example and every reported
result has an entry here. Each entry records how to reproduce it, its current
status, and what blocks it. The ledger is [`ledger.json`](ledger.json), and its
versioned schema is [`schema.json`](schema.json). Refs and capability IDs come
from the [capability inventory](../inventory/README.md). `image_ccil` is out of
scope; its integration is deferred.

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

On capture it reported 30 entries, 63 parameters, and 0 errors.

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

| Entry | Claim | Status |
|---|---|---|
| `sim-pendulum-cont` | Fig. 4(c) | pending |
| `sim-pendulum-disc` | Fig. 4(c), tracked in #3 | pending |
| `sim-pendulum-label-bound` | Sec. 5.1 error bound | blocked: procedure unrecorded |
| `sim-f1tenth` | Fig. 5(c) | pending |
| `sim-drone-hover`, `-circle`, `-flythrugate` | Fig. 6(b) | pending |
| `sim-hopper`, `-walker2d`, `-ant`, `-halfcheetah` | Table 1 | pending |
| `sim-coffee-pull`, `-button`, `-coffee-push`, `-drawer-close` | Table 1 | pending |
| `sim-model-based-baselines` | MOReL and MILO rows | blocked: scope decision (#9) |
| `real-*` (6 entries) | Figs. 8–13, real robot | blocked: hardware and data |
| `setup-install` | README install | pending |
| `readme-tabulate`, `readme-baselines` | README commands | pending |
| `internal-label-filtering-configs` | label-filtering configs | blocked: no linked result |
| `internal-data-ablation` | Simulated data ablation | blocked: no linked result |
| `internal-action-chunking-labels` | Sec. 6, App. C.3 | blocked: real-robot data |
| `internal-cluster-sweep` | sweep.sh / launch.slurm | out-of-scope |
| `deferred-image-pendulum` | Fork image branch | out-of-scope (deferred) |

The target commands are **provisional** (`ccil run <task> --method ...`). The
migration contract (#9) ratifies the unified CLI.

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
