# Pendulum parity: predeclaration (#24)

> **Superseded on 2026-10-26, before any run.** The legacy refactor-parity
> oracle was withdrawn by the migration contract amendment of that date (§5).
> Nothing in this document was ever run. #24 is now governed by
> [`pendulum-ccil-study-predeclaration.md`](pendulum-ccil-study-predeclaration.md).
> This file is kept as the record of what was planned.

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
| **Corrected** | **Frozen at `847b06057e8d36a16d13aa2f3b2e375ceb9147e5`** (the #99 merge, which includes #21, #22, #88, #89 and #23) | Every stage |

**The corrected arm is frozen.**
- Every corrected-arm run executes the `ccil` package at commit `847b060`,
  from a clean tree.
- Experiment tooling (the environment port, run drivers and analysis, under
  `tools/m3_pendulum/`) may be added in later commits. It may never modify
  `src/ccil`, `pyproject.toml` or `uv.lock`.
- Each run records the tooling commit, and verifies that
  `git diff 847b060 -- src pyproject.toml uv.lock` is empty.
- A behavior-changing fix to `ccil` after this point needs a new version of
  this document and a separately reported rerun.

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
| Label generation, fixed weights | `noisy_action` labels from the same weights, with the same injected action perturbations (legacy draws exported and replayed), over the complete training handoff (the table below) | per field, below | The same closed form, z_G = z_{t+1} − f̂(z_{t+1}, a_t + Δ), from the same f̂ |

**The label handoff.** Every field that reaches policy training is compared:

| Field | Rule |
|---|---|
| Source episode and frame of every candidate, in row order | exact |
| Row order and the total number of candidates | exact |
| Candidate action a_G = a_t + Δ | max \|Δ\| ≤ 1e-6 (one float32 addition of replayed draws) |
| Candidate observation z_G | max \|Δ\| ≤ 1e-5 (float32 accumulation over the 3-D state) |
| Target y = z_{t+1} | exact (read from the dataset) |
| Acceptance, and the rejection reason: legacy `‖z_G − z_t‖ < aug.epsilon` against `out_of_support` | exact. Candidates within 1e-5 of the support radius are listed, and counted as agreeing whichever way they fall. |
| Training weight of accepted rows (legacy weight 1) | exact |
| Accepted and rejected counts | exact, up to the boundary candidates listed above |

## 7. Statistics

For each task and method, the paired per-seed differences are
dₛ = R_corrected(s) − R_historical(s), over the n valid seeds (§8).
- **d̄** is their mean, and **L̄** is the historical arm's mean of R over the
  same seeds.
- **CI.** Each interval uses a fresh `numpy.random.default_rng(20261026)`,
  so every interval resamples the same seeds:
  - `idx = rng.integers(0, n, size=(10000, n))`;
  - the resampled means are `d[idx].mean(axis=1)`;
  - the endpoints are `np.quantile(means, [0.025, 0.975], method="linear")`.
- **Band:** B = [−0.1·|L̄|, +0.1·|L̄|], the contract §5 rule. #25 did not
  tighten it before this version.
- **Classifying an interval [lo, hi]:**

  | Class | Condition |
  |---|---|
  | **inside** | −0.1·\|L̄\| ≤ lo and hi ≤ +0.1·\|L̄\|. Equality at an edge counts as inside. |
  | **disjoint** | hi < −0.1·\|L̄\| or lo > +0.1·\|L̄\| |
  | **overlapping** | Otherwise. |

- **Within-arm effects** (CCIL − BC and NoiseBC − BC, informational): the
  same procedure applied to R_CCIL(s) − R_BC(s), and to R_NoiseBC(s) − R_BC(s),
  within each arm, over that arm's valid seeds. No band applies.
- **Aggregates:** mean ± std over seeds, as the paper reports, plus every
  per-seed value.

## 8. Seeds that fail

**Infrastructure faults** (out of memory, a lost host, a killed process) are
rerun once, with identical inputs. The rerun is recorded. If the rerun fails
again with an infrastructure fault, the task is **blocked**: the fault is
named, and nothing more is rerun under this version.

**Algorithmic failures** are results: a non-finite loss, a crash inside the
algorithm, or a non-finite return. A failing (task, method, seed) is reported.
It is omitted from the numerical intervals, but it counts in the decision:

| Failure | Effect on the outcome |
|---|---|
| **Shared** (both arms fail on the same task, method and seed) | It reproduces legacy behavior. It does not by itself prevent "reproduced". |
| **Corrected-arm only** | **not reproduced (porting regression)**, unless the §9 attribution counterfactual for a named correction also removes it. Then it is an explained correctness change. |
| **Historical-arm only** | **inconclusive**, unless a named correction's counterfactual reproduces the legacy failure. Then it is an explained correctness change. |

Any unexplained arm-specific failure prevents **reproduced**.

**Valid seeds** are those with no failure in either arm, for any method of
the task. Fewer than 8 valid seeds makes the task **inconclusive**, unless a
failure rule above gives a stronger outcome.

## 9. Decision rule

**Per method** (BC, NoiseBC, CCIL), within a task, the result is one of:
- **inside**;
- **explained**: not inside, and every difference passes the attribution
  below;
- **overlapping** or **disjoint**, when unexplained;
- **failed**, for an unexplained arm-specific failure;
- **blocked**.

**Per task.** The first matching row, in this order of precedence, decides
the ledger outcome:

| Order | Outcome | When |
|---|---|---|
| 1 | **blocked** | Any arm or method cannot run (§8), or a §6 check cannot run. The missing component, its owner and the next action are named. |
| 2 | **not reproduced (porting regression)** | A §6 check fails without explanation, any method is disjoint, or there is an unexplained corrected-arm-only failure. This blocks M3. The fix lands in its own pull request, and the rerun is a new, predeclared run. Both runs are reported. |
| 3 | **inconclusive** | Any method is overlapping, there is an unexplained historical-arm-only failure, or there are fewer than 8 valid seeds. |
| 4 | **reproduced with explained correctness change** | Every method is inside or explained, and at least one is explained. |
| 5 | **reproduced** | Every method is inside, and every §6 check passes. |

### Attribution

A difference is **explained** only by the counterfactual predeclared for a
named correction. Each counterfactual arm is judged by the same §7 paired
bootstrap and band, against the historical arm's returns for the same method
and seeds: "explained" means its interval is **inside**.

**Step 1: label swap** (localization). The corrected trainer is trained on
the historical arm's own CCIL labels (`aug_data.pkl` per seed, converted
through the trusted boundary).
- **Inside:** the trainer, data path and evaluation are reproduced, and the
  difference lies upstream of policy training.
- **Not inside:** the difference lies in policy training or evaluation. No
  #21 correction lives there, so it is unexplained.

**Step 2: dynamics swap** (run only if step 1 is inside). The corrected label
generator and trainer use the historical arm's trained dynamics weights for
each seed. They are exported to numpy in the pinned legacy environment
(#21 tooling), and loaded with `MLPTransitionModel`.
- **Inside:** the difference lies in dynamics fitting. It is explained only
  if the table below names a correction that changes dynamics fitting for
  that task.
- **Not inside:** it lies in label generation, filtering, conversion, row
  order or weights. §6 asserts those equal for fixed weights, so it is
  unexplained.

**Correction attribution table**, fixed now:

| Correction | Stage | `pendulum_cont` (`soft_sampling`) | `pendulum_disc` (`slack`) | Counterfactual |
|---|---|---|---|---|
| #21 eval-mode prediction | dynamics prediction | not applicable: no spectral norm | not applicable | none needed (§6 shows prediction equality) |
| #21 `soft_sampling` units and step | dynamics training loss | **applies** | not applicable | dynamics swap (step 2) |
| #21 `soft_jac` residual Jacobian | dynamics training loss | not applicable | not applicable | — |
| #21 `soft_sampling_slack` pairing | dynamics training loss | not applicable | not applicable (`slack_loss` is unchanged) | — |
| #22 NoiseBC distribution | NoiseBC training | not applicable: the primary run uses the legacy mode | not applicable | exploratory corrected-fidelity runs only |
| #22 counter-keyed draws, #89 RNG streams | label perturbations, initialization, batch order | statistical only | statistical only | not a correction: a difference attributed only to RNG streams is unexplained |

So on `pendulum_disc` no named correction changes dynamics fitting. A CCIL
difference there that steps 1 and 2 localize to dynamics fitting is
**unexplained**.

The paper comparison (§1) and the within-arm effects (§7) are reported
whatever the outcome, and neither affects it. "Inconclusive" is an
acceptable answer to #3.

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
