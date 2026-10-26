# Pendulum CCIL efficacy study: predeclaration (#24)

**Version 1.** Committed before any run of this study. Results are reported
against the version that governed them. A change after any outcome is seen
is a new version, with its rationale, and is never applied retroactively.

This study replaces the legacy refactor-parity comparison, which was
withdrawn by the migration contract's 2026-10-26 amendment (§5) before it
ran.

## 1. Questions

- **Primary.** On continuous and discontinuous Pendulum, does BC+CCIL change
  closed-loop return relative to BC, in which direction, and by how much?
- **Secondary.** These are descriptive and inform no decision:
  - CCIL compared with a tuned NoiseBC;
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
  environment's integrator, layered with
  `uv run --group evidence --with-requirements`.
- Each run records the tooling commit, and verifies that
  `git diff 847b060 -- src pyproject.toml uv.lock` is empty.
- A behavior-changing `ccil` fix needs a new version of this document and
  separately reported runs.

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
| Reward | r = −½‖(θ, θ̇) − (π, 0)‖² − ½a², with θ = atan2(sin θ, cos θ) wrapped to [0, 2π) |
| Actions | Not clipped (as the data generator) |
| Episode | 500 steps; no termination |

**Acceptance check (before any study run).** For every transition of both
datasets, stepping the environment from the stored (s_t, a_t) must reproduce
the stored s_{t+1} within max |Δ| ≤ 1e-6. If it does not, the study stops,
and the mismatch is investigated and reported.

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
| **CCIL-filtered** | secondary | CCIL plus the paper's label-error filter: `certificate.profile disturbed_action_paper_proxy`, score admission, `certificate.selection_quantile 0.3` (the label-filtering branch's value). It shares CCIL's dynamics model and labels. |
| **CCIL-noLip** | secondary ablation | CCIL with `transition_model.lipschitz_type none` (the ablation in `scripts/pendulum.sh`) |

**Shared policy settings:** the experimental vector harness (#89), with
batch 512, 100 epochs (÷20 when balanced), learning rate 1e-3, [256, 256],
and standard and min_max scalers.

**Asymmetry, stated.** NoiseBC receives a scale search (§6); CCIL and BC use
the paper's settings. Any bias this creates therefore favors the baseline.

## 6. Seeds and the NoiseBC scale

**Tuning seeds 1–3.** These are used only to choose the NoiseBC scale.
- **Grid** (standardized observation units): {1e-4 (the paper's value),
  1e-3, 1e-2, 1e-1, 3e-1}.
- **Selection:** the scale with the highest mean noise-free return over seeds
  1–3. Ties go to the smaller scale.
- The choice is committed to the run record before any study seed runs.

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
      `default_rng(SeedSequence([s, 8, k]))` shared across arms.
    - It applies to BC, NoiseBC and CCIL.
  - **Data size.** Nested episode subsets of 10%, 25% and 50%, chosen by
    `default_rng(SeedSequence([s, 9])).permutation`, for BC and CCIL, with
    noise-free evaluation.
  - **Label accuracy** (the `sim-pendulum-label-bound` question). For CCIL's
    accepted labels, ‖T(z_G, a_G) − y‖ from one environment step, as the mean
    and quantiles per seed. It also reports the share of labels within the
    support radius.
  - **Training diagnostics:** the dynamics model's held-out one-step error,
    the label counts by reason, and the supervision manifests.

## 8. Statistics and decisions

**Paired differences.** For each task and comparison,
dₛ = R(s, A) − R(s, B) over the seeds valid for both arms.

**Intervals.** Each interval uses a fresh
`numpy.random.default_rng(20261026)`:
- `idx = rng.integers(0, n, size=(10000, n))`;
- the endpoints are `np.quantile(d[idx].mean(axis=1), q, method="linear")`.

**Primary decision**, per task, for CCIL − BC. Two tasks are tested, so the
interval is Bonferroni-adjusted: 97.5%, q = (0.0125, 0.9875).

| Outcome | Condition |
|---|---|
| **CCIL improves on BC** | the lower endpoint > 0 |
| **CCIL is worse than BC** | the upper endpoint < 0 |
| **No detectable difference** | the interval contains 0 |

**Reported alongside the decision:**
- d̄;
- the relative effect d̄ / |mean R_BC|, with its interval;
- every per-seed value.

**Secondary comparisons** are CCIL − NoiseBC, CCIL-filtered − CCIL and
CCIL-noLip − CCIL, and per noise level and data size. Each gets a 95%
interval (q = 0.025, 0.975). They are descriptive: no decisions, and no
multiplicity claims.

**Paper comparison** (informational): the mean ± std over seeds of BC and
CCIL, beside Fig. 4(c):

| Task | BC | BC+CCIL |
|---|---|---|
| Continuous | −3244 | −2269 |
| Discontinuous | −3204 | −2556 |

The paper used different seeds and the legacy implementation. No agreement
or disagreement is claimed beyond what the intervals show.

## 9. Failures

**Infrastructure faults** (out of memory, a lost host, a killed process) are
rerun once, with identical inputs. A second fault excludes that seed for that
arm, and the exclusion is reported.

**Algorithmic failures** are results: a non-finite loss, a crash inside the
algorithm, or a non-finite return. They are reported per arm and seed, and
the arm-seed is missing from that comparison's paired differences.

**Inconclusive.** A task's primary outcome is **inconclusive** if:
- either primary arm has more than 2 missing seeds; or
- fewer than 16 seeds are valid for both arms.

## 10. Reporting

#25 publishes the report. Every number in it links to:
- a raw per-seed result file;
- the resolved configuration;
- the run manifest;
- the tooling and `ccil` commits;
- the package versions and GPU.

Negative and null outcomes are reported the same way as positive ones.
