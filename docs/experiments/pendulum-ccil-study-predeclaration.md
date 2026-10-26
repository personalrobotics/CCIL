# Pendulum CCIL efficacy study: predeclaration (#24)

**Version 3.** Committed before any run of this study. Results are reported
against the version that governed them. A change after any outcome is seen
is a new version, with its rationale, and is never applied retroactively.

**Changes from version 1**, made before any tooling, tuning or study run, so
no outcome informed them:
- **Intervals (§8).** Version 1 used percentile bootstrap intervals, which
  tend to undercover at 20 seeds, worst in the tails the Bonferroni interval
  uses. The primary interval is now the union of a BCa bootstrap interval and
  a paired t-interval.
- **CCIL-tuned arm (§§5, 6).** The paper's CCIL configuration perturbs
  actions very little (σ = 1e-5 on continuous Pendulum), so a null primary
  result could describe that configuration rather than the method, while
  NoiseBC gets a scale search. CCIL-tuned gets the same search budget, on
  the same tuning seeds, as a secondary arm. The paper-configured comparison
  remains the only primary result.
- **Positive control (§8).** BC on all demonstrations against BC on 10%
  must show a detectable difference before a "no detectable difference"
  primary outcome is interpreted as one.

**Clarifications** made while building the tooling, before any tuning or
study run, which state what the data generator already did: the reward is
evaluated on the pre-step state (§3), and the noise index k (§7).

**Changes from version 2**, made before any tuning or study run:
- **Zero-action reference (§§7, 8).** This addition was motivated by an
  observed outcome: in the §3 acceptance checks (seed 9001, which is neither
  a tuning nor a study seed), the zero-action policy scored R ≈ −2,350 to
  −2,395. That is above the paper's reported BC returns and close to its
  BC+CCIL returns, while an upright pendulum scores near 0. Absolute returns
  therefore cannot be read without a passive-policy reference. The reference
  is descriptive only, and it changes no arm, metric or decision rule.

This study replaces the legacy refactor-parity comparison, which was
withdrawn by the migration contract's 2026-10-26 amendment (§5) before it
ran.

## 1. Questions

- **Primary.** On continuous and discontinuous Pendulum, does BC+CCIL change
  closed-loop return relative to BC, in which direction, and by how much?
- **Secondary.** These are descriptive and inform no decision:
  - CCIL compared with a tuned NoiseBC;
  - CCIL with its action noise tuned like NoiseBC's scale, compared with BC
    and with tuned NoiseBC;
  - the paper's label-error filtering compared with plain CCIL;
  - the effect of the dynamics model's Lipschitz regularization;
  - robustness to evaluation noise;
  - the effect of demonstration count;
  - how accurate CCIL's labels are against the true dynamics;
  - how the results compare with paper Fig. 4(c).

## 2. Implementation, frozen

**The `ccil` package.**
- Every run uses the `ccil` package at commit
  `847b06057e8d36a16d13aa2f3b2e375ceb9147e5`, from a clean tree. `ed737b8`
  is identical in `src`, `pyproject.toml` and `uv.lock`.
- Experiment tooling lives only under `tools/m3_pendulum/`: the environment,
  run drivers, evaluation and analysis. It may never modify `src/ccil`,
  `pyproject.toml` or `uv.lock`.
- Tooling-only dependencies come from a pinned
  `tools/m3_pendulum/requirements.txt`, for example scipy for the
  environment's integrator and the analysis's intervals, layered with
  `uv run --group evidence --with-requirements`.
- A behavior-changing `ccil` fix needs a new version of this document and
  separately reported runs.

**The tooling is frozen too, before any outcome.** The tooling is developed,
and its acceptance checks (§3) pass, without producing any tuning or study
outcome. Then **one tooling commit is frozen**:
- It is recorded in this document by a pull request that changes nothing
  else (the "tooling freeze" addendum, §11), and merged before the first
  tuning run.
- Every tuning and study run uses exactly that commit. Each run verifies that
  `HEAD` is the frozen commit and that the tree is clean, which also
  re-verifies the `ccil` freeze above.
- A behavior-changing tooling fix after the freeze needs a new version of
  this document and a **complete rerun** of tuning and the study. The earlier
  runs are reported, not mixed with the new ones.

**Host.** All study runs execute on one host (siddh, CUDA). Timings are not
compared.

## 3. Environment

There is one canonical Pendulum environment in `tools/m3_pendulum/`. It has
the physics that generated the demonstrations:

| Element | Value |
|---|---|
| State | (sin θ, cos θ, θ̇) |
| Dynamics | θ̈ = −(g/l) sin θ + a, with l = 1 and g = 9.81 |
| Step | dt = 0.02, integrated by `scipy.integrate.solve_ivp` RK45 at its default tolerances, evaluated at dt |
| Discontinuous variant | After each step, θ̇ changes sign when θ is within 1e-2 of π/6 or π/3 |
| Reward | r = −½‖(θ, θ̇) − (π, 0)‖² − ½a², with θ = atan2(sin θ, cos θ) wrapped to [0, 2π), evaluated on the pre-step state and the applied action (as the data generator) |
| Actions | Not clipped (as the data generator) |
| Episode | 500 steps; no termination |

**Acceptance checks, run before the tooling freeze with no tuning or study
outcome:**
1. **Transitions.** For every transition of both datasets, stepping the
   environment from the stored (s_t, a_t) must reproduce the stored s_{t+1}
   within max |Δ| ≤ 1e-6.
2. **Evaluator audit.** The evaluator runs fixed, non-learned policies:
   - zero action;
   - constant actions ±1;
   - a fixed linear controller.

   For each, it saves every step's state, action and reward. An independent
   implementation must then reproduce, from the saved states and actions
   alone:
   - every per-step reward (max |Δ| ≤ 1e-9);
   - the 500-step horizon, with no early termination;
   - each episode return as the sum of its rewards;
   - each seed's R as the mean of its 100 returns (to 1e-9).

If either check fails, nothing proceeds until the cause is found and
reported.

**Every evaluation is auditable.** Each tuning and study evaluation saves
every episode's per-step states, actions and rewards, plus its initial state
and noise draws. The analysis recomputes every reported return from those
files with the independent implementation, and refuses any mismatch.

**Initial states.** For each seed s, 100 episodes start from
θ ~ U(−1, 1) and θ̇ ~ U(−0.5, 0.5), drawn from
`np.random.default_rng(SeedSequence([s, 7]))`. Within a seed, every arm faces
the same 100 initial states.

## 4. Data

The paper's Pendulum demonstrations, converted by #23:

| Dataset | Canonical SHA-256 | Size |
|---|---|---|
| `pendulum_cont_100` | `a417269b62c3c5b444f9ac71aca5407fef18cf2c04f59a60df73f3234122f962` | 100 episodes |
| `pendulum_disc_500` | `b30cb074…` (in [`conversion.json`](../fixtures/evidence/legacy-conversion/artifacts/conversion.json)) | 500 episodes |

**Use of the data.**
- Policies train on every episode (`policy.exclude_validation_episodes:
  false`), as in the paper.
- The dynamics model's 20% validation split is used only for its held-out
  error metric.

## 5. Arms

Every arm uses the #23 conversion of the task's paper configuration, with the
values below. Nothing else is tuned.

| Arm | Role | Settings |
|---|---|---|
| **BC** | primary baseline | `policy.method bc` |
| **CCIL** | primary treatment | `policy.method ccil`: `noisy_action` labels, 20 per transition, with the config's action noise (1e-5 continuous, 1e-4 discontinuous), support radius (1.0 continuous, 0.01 discontinuous), and dynamics model (`soft_sampling` continuous, `slack` discontinuous). `balance_noisy_labels` is as configured (false continuous, true discontinuous). Labels are uncertified. |
| **NoiseBC** | secondary baseline | `policy.method noise_bc`, `symmetric_uniform_half_batch`, with the scale chosen as in §6 |
| **CCIL-filtered** | secondary | CCIL labels filtered by the paper's DisturbedAction label-error score (Def. 4.6, the `disturbed_action_paper_proxy` profile), keeping those at or below the 0.3 selection quantile, which is the label-filtering branch's value (`certificate.selection_quantile 0.3`, score admission). It shares CCIL's dynamics model and labels. |
| **CCIL-noLip** | secondary ablation | CCIL with `transition_model.lipschitz_type none` (the ablation in `scripts/pendulum.sh`) |
| **CCIL-tuned** | secondary | CCIL with `generation.action_noise_std` chosen as in §6. Every other setting, including the support radius, is CCIL's. |

**Shared policy settings:** the experimental vector harness (#89), with
batch 512, 100 epochs (÷20 when balanced), learning rate 1e-3, [256, 256],
and standard and min_max scalers.

**Asymmetry, stated.** In the primary comparison, BC and CCIL use the paper's
settings, while NoiseBC receives a scale search (§6), so any bias this
creates favors the baseline. CCIL-tuned receives the same search budget as
NoiseBC, so that CCIL − NoiseBC can also be read with both sides tuned.

## 6. Seeds and tuning

**Tuning seeds 1–3.** These are used only to choose the NoiseBC scale and the
CCIL-tuned action noise, **separately per task**. Each search has five values:

| Arm | Parameter | Grid |
|---|---|---|
| NoiseBC | `policy.noise_scale` (standardized observation units) | {1e-4 (the paper's value), 1e-3, 1e-2, 1e-1, 3e-1} |
| CCIL-tuned | `generation.action_noise_std` (action units) | {the paper's value (1e-5 continuous, 1e-4 discontinuous), 1e-3, 1e-2, 1e-1, 3e-1} |

The support radius is not searched: it is the paper's per-task constraint (on
discontinuous Pendulum, 0.01 keeps labels from crossing the discontinuity),
and searching it would give CCIL-tuned a larger budget than NoiseBC.

For each arm and task:
- **Selection:** the value with the highest mean noise-free return over
  seeds 1–3. Ties go to the smaller value.
- **Failed tuning runs.** A value with any failed tuning run is ineligible.
  If every value is ineligible, the arm uses the paper's value, and this is
  recorded.
- **Reporting.** Every value-by-seed tuning result is reported, with
  CCIL-tuned's label acceptance rate.
- Both choices are committed to the run record before any study seed runs.

**Study seeds 101–120** (20 seeds). Each seed runs one complete pipeline per
arm: the dynamics fit, labels, the policy, and evaluation. Tuning seeds are
never reported as study results.

## 7. Metrics

- **Primary:** R(s, arm), the mean return of the 100 noise-free episodes for
  seed s.
- **Secondary:**
  - **Robustness.** Evaluation noise uses the legacy mix model: observation
    o ← (1 − n)·o + n·U(observation space), and action a ← (1 − n)·a + n·U(action space).
    - The spaces are the legacy bounds: observation [−1, −1, −4π] to
      [1, 1, 4π], and action [−3, 3].
    - n ∈ {0.01, 0.05, 0.1}, with draws from
      `default_rng(SeedSequence([s, 8, k]))` shared across arms, with k = 0, 1, 2 for n = 0.01, 0.05, 0.1.
    - It applies to BC, NoiseBC, CCIL and CCIL-tuned.
  - **Data size.** Nested episode subsets of 10%, 25% and 50%, chosen by
    `default_rng(SeedSequence([s, 9])).permutation`, for BC and CCIL, with
    noise-free evaluation.
    - Each data-size run uses only its subset.
    - For CCIL, it refits the dynamics model, regenerates the labels, and
      retrains the policy on that subset.
    - BC trains on the subset.
  - **Label accuracy.** For CCIL's labels:
    - the proposal acceptance rate, by reason;
    - quantiles of the displacements ‖z_G − z_t‖ and ‖a_G − a_t‖;
    - the true label error ‖T(z_G, a_G) − y‖ from one environment step
      (mean and quantiles per seed), compared descriptively with the paper's
      reported empirical mean of 0.02367 and bound of 0.065.

    The ledger entry `sim-pendulum-label-bound` stays **blocked**: the
    procedure behind 0.065 is unrecorded, so this measurement is reported as
    related evidence, not as a reproduction.
  - **Training diagnostics:** the dynamics model's held-out one-step error,
    the label counts by reason, and the supervision manifests.
  - **Zero-action reference.** R(s, zero), the zero-action policy (a = 0 at
    every step), on each study seed's initial states, noise-free and at
    each noise level, for both tasks. It needs no training and has no tuning.

## 8. Statistics and decisions

**Paired differences.** For each task and comparison,
dₛ = R(s, A) − R(s, B) over the seeds valid for both arms.

**Intervals.** At confidence level c, for the n paired differences d:
- **BCa bootstrap:** `scipy.stats.bootstrap((d,), np.mean,
  confidence_level=c, n_resamples=10000, method="BCa",
  rng=np.random.default_rng(20261026))`, with the scipy pinned in
  `tools/m3_pendulum/requirements.txt`;
- **Paired t:** d̄ ± t_{(1+c)/2, n−1} · s_d / √n, with s_d the sample standard
  deviation (ddof 1).

If the BCa interval is undefined (non-finite endpoints, for example when
every dₛ is equal), the t-interval is used alone, and this is reported.

**Primary decision**, per task, for CCIL − BC. Two tasks are tested, so
c = 0.975 (Bonferroni). The decision interval is the **union** of the two
intervals: from the lower of their lower endpoints to the higher of their
upper endpoints. A direction is claimed only when both methods agree.

| Outcome | Condition |
|---|---|
| **CCIL improves on BC** | the lower endpoint > 0 |
| **CCIL is worse than BC** | the upper endpoint < 0 |
| **No detectable difference** | the interval contains 0, and the positive control passes |
| **Inconclusive** | the interval contains 0, and the positive control fails; or §9 applies |

**Positive control**, per task. The same procedure (c = 0.975, the union
interval) applied to dₛ = R(s, BC on all episodes) − R(s, BC on the 10%
subset of §7) must give a lower endpoint > 0. If it does not, the setup has
not shown it can detect a known difference on that task, so a null primary
result there is not read as evidence of no effect. A directional primary
result stands either way. The control is reported in full whether it passes
or fails.

**Reported alongside the decision:**
- d̄, and both intervals separately;
- the relative effect d̄ / |mean R_BC|, with its BCa interval:
  `scipy.stats.bootstrap((d, R_BC), lambda d, b: d.mean() / abs(b.mean()),
  paired=True, vectorized=False, confidence_level=0.975, n_resamples=10000,
  method="BCa", rng=np.random.default_rng(20261026))`;
- every per-seed value.

A "no detectable difference" outcome is reported with its interval, which
bounds the size of any effect the study could have missed.

**Secondary comparisons** are CCIL − NoiseBC, CCIL-tuned − BC,
CCIL-tuned − NoiseBC, CCIL-filtered − CCIL and CCIL-noLip − CCIL, and per
noise level and data size. Each gets the union interval at c = 0.95. They are
descriptive: no decisions, and no multiplicity claims.

**Paper comparison** (informational): the mean ± std over seeds of BC and
CCIL, beside Fig. 4(c):

| Task | BC | BC+CCIL |
|---|---|---|
| Continuous | −3244 | −2269 |
| Discontinuous | −3204 | −2556 |

The paper used different seeds and the legacy implementation. No agreement
or disagreement is claimed beyond what the intervals show.

**Zero-action reference** (descriptive). For each task, R(s, zero) is
reported per seed beside every arm. For each primary and secondary arm
(noise-free), and for BC, NoiseBC, CCIL and CCIL-tuned at each noise level,
arm − zero is reported with the union interval at c = 0.95. These intervals
inform no decision, are not part of the positive control, and make no
multiplicity claim.

## 9. Failures

**Infrastructure faults** (out of memory, a lost host, a killed process) are
rerun once, with identical inputs. A second fault excludes that seed for that
arm, and the exclusion is reported.

**Algorithmic failures** are results: a non-finite loss, a crash inside the
algorithm, or a non-finite return. They are reported per arm and seed.
- **Primary arms.** Any algorithmic failure in BC or CCIL on a task makes
  that task's primary outcome **inconclusive**. A comparison is never
  conditioned on the seeds where an arm succeeded.
- **Secondary arms.** The failed arm-seed is missing from that descriptive
  comparison, and the failure is reported.
- **Positive control.** Any algorithmic failure in BC on the full data or on
  the 10% subset, or fewer than 16 seeds valid for both, means the control
  fails on that task.

**Inconclusive.** A task's primary outcome is also **inconclusive** if
infrastructure exclusions leave either primary arm with more than 2 excluded
seeds, or fewer than 16 seeds valid for both.

## 10. Reporting

#25 publishes the report. Every number in it links to:
- a raw per-seed result file;
- the resolved configuration;
- the run manifest;
- the tooling and `ccil` commits;
- the package versions and GPU.

Negative and null outcomes are reported the same way as positive ones.

## 11. Tooling freeze (addendum, to be completed before any tuning run)

- Frozen tooling commit: *(recorded by its own pull request, after the §3
  acceptance checks pass)*
- Acceptance check evidence: *(linked here)*
