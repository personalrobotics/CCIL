# Pendulum parity: predeclaration (#24)

**Version 1.** Committed before any #24 policy outcome is run or inspected.
Results are reported against the version of this document that governed them.
A change after any outcome is seen is a new version, with its rationale, and
is never applied retroactively (migration contract §5).

This document fixes the arms, data, seeds, evaluation, metrics, deterministic
tolerances, statistics and decision rules for the ledger entries
`sim-pendulum-cont` and `sim-pendulum-disc`. #25 publishes the report.

## 1. Question

Does the unified implementation reproduce the historical implementation's
Pendulum results? Where it does not, is every difference a named, intentional
correctness change rather than a porting regression?

Two questions are answered alongside, informationally, and do not change the
parity decision:
- **Paper comparison.** How do both implementations compare with paper
  Fig. 4(c)? The reported values are:

  | Task | BC | BC+CCIL |
  |---|---|---|
  | Continuous | −3244 | −2269 |
  | Discontinuous | −3204 | −2556 |

- **CCIL effect.** How much does CCIL change the return over BC, in each
  implementation?

## 2. Arms

| Arm | Code | Used for |
|---|---|---|
| **Historical** | `archive/upstream-main-2026-10-22`, with only the import shims needed to run without D4RL, mujoco-py or the gym registry. The shim diff is published, and contains no behavioral edit. | Every stage |
| **Faithful port** | Immutable commit `50cfc535`, the vector-dynamics port before the #21 repairs | Deterministic fixed-weight comparisons only (§6) |
| **Corrected** | `main` at the run commit, which includes #21, #22, #88, #89 and #23 | Every stage |

**Corrected-arm configuration.** It uses the #23 conversions of
`config/pendulum_{cont,disc}.yml` at **fidelity `historical`**:
- the policy trains on every expert transition, as legacy BC did
  (`policy.exclude_validation_episodes: false`);
- NoiseBC uses `legacy_uniform_half_batch`.

Every remaining difference from the historical arm is therefore code: a port
or a correctness change. It is not a protocol choice.

**The historical arm** runs the legacy scripts exactly as
`scripts/train_ccil.sh` invokes them. The legacy configs are unchanged, and
only `seed` and `output.location` are overridden, as the script does.

## 3. Data, methods and seeds

**Datasets.** The arms read the same data, each in its own form:
- the historical arm reads `data/pendulum_cont_100.pkl` and
  `data/pendulum_disc_500.pkl`, after SHA-256 verification;
- the corrected arm reads their #23 conversions.

| Task | Pickle SHA-256 | Canonical SHA-256 |
|---|---|---|
| Continuous | `ca9ee2f2…` | `a417269b…` |
| Discontinuous | `05c46cc5…` | `b30cb074…` |

The full hashes are in
[`../fixtures/evidence/legacy-conversion/artifacts/conversion.json`](../fixtures/evidence/legacy-conversion/artifacts/conversion.json).

**Methods**, per task and arm. These are the arms `scripts/train_ccil.sh` runs:

| Method | Historical | Corrected |
|---|---|---|
| BC | `policy.naive true` | `policy.method bc` |
| NoiseBC | `policy.naive true`, `policy.noise_bc 0.0001` | `policy.method noise_bc`, `policy.noise_scale 0.0001` |
| CCIL | the config's `noisy_action` labels; `lipschitz_type` `soft_sampling` (continuous) or `slack` (discontinuous) | the same, converted |

**Seeds.** Exactly 41–50, the seeds `scripts/pendulum.sh` uses. Each seed is
one full pipeline per arm: dynamics fit, labels, the three policies, and
evaluation.

**No parameter is tuned or searched.** Every value comes from the configs.

## 4. Evaluation

**Environment.** The legacy environment needs gym 0.19, so all arms use one
**ported Pendulum environment** that matches legacy `correct_il/envs/pendulum_env.py`:

| Element | Value |
|---|---|
| State | (sin θ, cos θ, θ̇) |
| Dynamics | l = 1, g = 9.81, dt = 0.02, `scipy.integrate.solve_ivp` RK45 |
| Discontinuous variant | θ̇ flips sign within 1e-2 of π/6 or π/3 |
| Reward | −½‖(θ, θ̇) − (π, 0)‖² − ½a², with θ wrapped to [0, 2π) |
| Actions | in [−3, 3] |
| Termination | none: 500 steps (`max_episode_steps=500`) |

The port is used only if it passes the equivalence check in §6.

**Protocol** (`eval_bc_policy.py`):
- 100 episodes per policy per seed, with `eval.noise` [0], so there is no
  sensor or actuator noise.
- **Initial states** are drawn exactly as legacy did: `np.random.seed(seed)`,
  then θ ~ U(−1, 1) and θ̇ ~ U(−0.5, 0.5) per episode, in order. Within a
  seed, every method in every arm faces the same 100 initial states.
- **Policies** act deterministically on raw observations:
  - historical: d3rlpy's exported `policy.pt` through `D3Agent`;
  - corrected: `VectorBCPolicy.predict`.

## 5. Metrics

**Primary.** For each task, arm, method and seed: R = the mean return over the
100 episodes.

**Secondary.** Reported, but not part of the parity decision:
- the standard deviation of returns within each seed;
- the per-seed CCIL − BC and NoiseBC − BC differences, within each arm;
- the dynamics model's training and held-out errors;
- the label counts, and their acceptance by reason;
- the #88 certificates of the corrected arm's labels. The true Pendulum
  Lipschitz constants are unknown, so constants are **declared**, and any
  bound is reported as conditional, never as a measured fact.

**Exploratory.** Reported separately, with no decision:
- the `lipschitz_type none` CCIL ablation in `pendulum.sh`;
- corrected-arm runs at fidelity `corrected` (symmetric NoiseBC; validation
  episodes held out).

## 6. Deterministic stages and tolerances

Each check passes or fails on its own. A failure blocks the statistical
comparison until it is explained.

| Stage | Compared | Tolerance | Rationale |
|---|---|---|---|
| Dataset conversion | pickle against canonical file | exact, bitwise | Done in #23. |
| Environment port | port against the legacy environment class, run under the gym stub. 20 action sequences (seeded uniform in [−3, 3]) × 500 steps from legacy initial states. | max \|Δstate\| ≤ 1e-9, max \|Δreward\| ≤ 1e-9 | The same scipy integrator on the same float64 inputs. Any difference indicates a library or port error. |
| Model prediction, fixed weights | f̂(s, a) on the Pendulum transitions, from legacy-exported weights (#21 tooling, Pendulum settings), loaded into the faithful port and into the corrected implementation | faithful port against legacy: max \|Δ\| ≤ 1e-6. Corrected against legacy: ≤ 1e-6 in eval mode for non-spectral-norm models. Otherwise the difference is reported and attributed. | The networks are the same float32 networks. 1e-6 is about 8 ulps of the output magnitude. Legacy predicted in training mode for spectral-norm models (#21). |
| Label generation, fixed weights | `noisy_action` labels from the same weights, with the same injected action perturbations (legacy draws exported and replayed) | max \|Δz_G\| ≤ 1e-5, the same acceptance decisions | The same closed form from the same f̂. The bound allows float32 accumulation over the 3-D state. |

## 7. Statistics

For each task and method, the paired per-seed differences are
dₛ = R_corrected(s) − R_historical(s), over the valid seeds.
- **d̄** is their mean, and **L̄** is the historical arm's mean of R.
- **CI:** the 95% percentile bootstrap interval of d̄, from 10,000 resamples
  of the seeds with replacement, using `numpy.random.default_rng(20261026)`.
- **Band:** [−0.1·|L̄|, +0.1·|L̄|], the contract §5 rule. #25 did not tighten
  it before this version.
- **Aggregates:** mean ± std over seeds, as the paper reports, plus every
  per-seed value.

## 8. Seeds that fail

**Infrastructure faults** (out of memory, a lost host, a killed process) are
rerun once, with identical inputs. The rerun is recorded.

**Algorithmic failures** are results: a non-finite loss, a crash inside the
algorithm, or a non-finite return. Such a seed is reported, and is excluded
from the paired differences for every method of that task.

**Fewer than 8 valid seeds** for a task makes that task **inconclusive**.

## 9. Decision rule

Each ledger entry (task) gets exactly one outcome:

| Outcome | When |
|---|---|
| **reproduced** | All §6 checks pass, and the parity CI lies inside the band for BC, NoiseBC and CCIL. |
| **reproduced with explained correctness change** | All §6 checks pass, BC and NoiseBC are within the band, and every CCIL failure passes the **label-swap attribution** below. |
| **inconclusive** | A CI overlaps a band edge (neither inside nor disjoint) and is not attributed, or a task has fewer than 8 valid seeds. |
| **not reproduced (porting regression)** | A CI is disjoint from the band and not attributed, or a §6 check fails without explanation. This blocks M3. The fix lands in its own pull request, and the rerun is a new, predeclared run. Both runs are reported. |
| **blocked** | An arm cannot run. The missing component, its owner and the next action are named. Deterministic checks that can run are still reported. |

**Label-swap attribution.** The corrected trainer is trained on the
historical arm's own CCIL labels, `aug_data.pkl`, converted through the
trusted boundary. If the resulting CCIL returns satisfy the parity band
against the historical CCIL returns, then:
- the trainer, data path and evaluation are reproduced;
- the CCIL difference lies in label generation;
- §6 then attributes it to the named #21 corrections.

Otherwise the difference is unexplained.

The paper comparison (§1) and the CCIL effect (§5) are reported whatever the
outcome, and neither affects it. "Inconclusive" is an acceptable answer to
#3.

## 10. Execution and reporting

- **Hosts.** Runs execute on siddh (CUDA), one process per arm, task and
  seed, in `runs/` with full logs.
- **Timing.** It is recorded, but never compared across arms or hosts.
- **Records.** For every run:
  - the commit and the clean or dirty flag;
  - the resolved configuration;
  - the package versions and the GPU;
  - the input SHA-256;
  - the per-seed results;
  - the run manifest.
- **The legacy harness** environment is a pinned requirements file, published
  with its shim diff.
- **Reporting.** Every number in the #25 report links to its raw per-seed
  file. Negative and inconclusive outcomes are kept.
