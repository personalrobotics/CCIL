# Pendulum CCIL efficacy study: predeclaration (#24)

**Version 6.** Results are reported against the version that governed them.
A change after any outcome is seen is a new version, with its rationale, and
is never applied retroactively.

**History.**
- **Version 4** ran under frozen tooling `cba34fe` and `ccil` `847b060`. Its
  tuning phase completed. Its study was stopped by the operator after 21 of
  520 units, so that the study could run on a faster execution stack.
- **Version 5** was a draft (#121), never frozen and never run. It kept every
  version 4 rule and changed only the execution stack.
- **Version 6** supersedes it. It also changes the continuous CCIL algorithm,
  so it reruns everything on seeds no run has used: fresh tuning, a fresh
  choice, the study and the analysis (§2). Version 4's records are reported in
  #25 as a separate, superseded run, and are never mixed with version 6's.

**What had been seen when version 6 was written:**
- **Version 4's tuning choice,** on seeds 1–3, the same on both tasks:
  NoiseBC scale 1e-4, and CCIL-tuned action noise 3e-1, which is the top of
  its grid.
- **Continuous CCIL − BC,** noise-free, for version 4 study seeds 101–110.
  The stop was decided before any study outcome was examined, but as the
  session ended a progress monitor printed these differences:

  | Seed | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | CCIL − BC | +1071.2 | −153.5 | +239.1 | +1082.8 | +1046.4 | +275.5 | +585.4 | +21.4 | −77.8 | +633.3 |

- **Training diagnostics of the dynamics models** from dev timing runs of
  the new stack, on seeds 1–3 and 101–120. These were the recorded Lipschitz
  penalty and the finite-difference ratios of fitted transition models. No
  return of those runs was read; the runs used the paper's values in place of
  a tuning choice.
- **Nothing else.** No other study outcome was seen.

**Changes from version 4.** Every other version 4 arm, grid, metric and rule
is kept exactly, including CCIL-tuned's grid although its version 4 choice
was at the edge.
- **The execution stack (§2).** Policies train with trainer v2 (#108).
  Dynamics models fit in batched groups (#110). Label generation uses stacked
  Newton solves and chunked Jacobians (#110). Evaluation steps every episode
  together (#111). Units run through the batched study runner (#112). None of
  these is bit-identical to the version 4 stack. Each has its own equivalence
  evidence, listed in §2, which must pass before the freeze.
- **The coordinates of the continuous dynamics model's Lipschitz penalty
  (§5, #125).** This changes the primary continuous CCIL algorithm.
  - **Before:** in raw coordinates, as ported in #21, the paper's
    `soft_sampling` penalty (constraint 2, ε = 3e-4) never binds on continuous
    Pendulum. The trained residual's ratio peaks near 0.24; a one-step residual
    over dt = 0.02 has a small raw Lipschitz constant. The penalty was exactly
    0 in every epoch of every continuous fit.
  - **Now:** version 6 measures the ratio in the model's normalized
    coordinates (`transition_model.lipschitz_coordinates model_normalized`).
    That is a dimensionless metric, invariant to units and timestep, in the
    space the network is optimized in.
  - **What it rests on:** only the pre-outcome dynamics diagnostics above. No
    return informed it.
  - **What stays fixed:** the constraint 2, ε and the penalty weight keep
    the paper's values and are not tuned after this change.
  - **Discontinuous Pendulum** uses the `slack` regularizer, which has no
    coordinate choice, and is unchanged.
- **Fresh seeds (§6).** Tuning seeds 11–13 and study seeds 1201–1220 replace
  1–3 and 101–120. No run of this study, its tooling evidence or its dev
  timing used them. The ten outcomes above therefore come from seeds version
  6 never uses.
- **An addition: two modern-recipe arms (§§5, 6),** BC-modern and
  CCIL-modern. They are secondary and descriptive and never enter the primary
  decision. Their tuning uses validation loss only, and their tuning-seed
  returns are computed only after the choice is recorded. Adding them changes
  no version 4 arm.

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

**Changes from version 3**, made before any tuning or study run. Only the
smoke runs of the tooling's evidence (1-epoch policies on seeds 9001–9003)
had run, and their returns were not reported or used:
- **Label-accuracy sample (§7).** The true label error is measured on
  10,000 accepted labels per CCIL unit, drawn without replacement by
  `default_rng(SeedSequence([s, 10]))`, rather than on every label: a
  discontinuous unit produces about 5 million candidates.
- **Balancing (§5).** BC and NoiseBC train without label balancing.
  Balancing (legacy `aug.balance_data`) repeats demonstrations against
  noisy labels, so it applies to CCIL only, and legacy BC trained on the
  demonstrations as they are.
- **The freeze gate (§§2, 11).** The freeze pull request adds a manifest
  naming the tooling commit. Each run verifies the paths it executes or
  reads against that commit, because the freeze pull request itself
  creates a new `HEAD`.

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
- Every run uses the `ccil` package at the commit named by the version 6
  freeze (§11), from a clean tree. (Version 4 used
  `847b06057e8d36a16d13aa2f3b2e375ceb9147e5`.)
- Experiment tooling lives only under `tools/`: `tools/m3_pendulum/` (the
  canonical environment, the inputs and the audit), `tools/vector_env/` (the
  batched environment) and `tools/m3_experiments/` (the study plan and the
  Pendulum suite of the runner). It may never modify `src/ccil`,
  `pyproject.toml` or `uv.lock`.
- Tooling-only dependencies come from the pinned
  `tools/m3_pendulum/requirements.txt` and `tools/vector_env/requirements.txt`,
  layered with `uv run --group evidence --with-requirements`.
- A behavior-changing `ccil` fix needs a new version of this document and
  separately reported runs.

**The execution stack (version 6)** and the evidence each part must pass
before the freeze:

| Part | Issue | Equivalence evidence |
|---|---|---|
| Trainer v2: every unit's policy trained in vmapped, compiled groups | #108 | One full-batch step equals v1's Adam step; a model's results do not depend on unrelated group members except through float32 rounding (recorded execution identity); no detected difference from v1 in closed-loop return on 10 seeds per task and arm |
| Normalized Lipschitz coordinates | #125 | The penalty equals its definition; it is invariant to a change of units; the two engines give the same penalty and gradients; fit records state the coordinates and the normalization hash |
| Batched dynamics fits | #110 | One step equals `MLPTransitionModel.fit` for each regularizer used; held-out one-step error on the Pendulum data matches v1's |
| Stacked Newton solves | #110 | Candidates identical to per-row solves on the real Pendulum labels |
| Chunked Jacobians | #110 | Certificates identical on the 4.3 M discontinuous labels; one Jacobian row in 4.3 M differs (a ReLU kink) |
| Batched evaluation | #111 | Reproduces the canonical environment (§3) within a few ulps on every stored transition; every rollout replays bit for bit through the integrator that produced it |
| Batched study runner | #112 | Writes the stage artifacts `ccil policy train` would (the CLI reuses every one); resumes without recomputation; a unit's evaluation does not depend on what it was batched with |

**The tooling is frozen too, before any outcome.** The tooling is developed,
and its acceptance checks (§3) pass, without producing any tuning or study
outcome. Then **one tooling commit is frozen**:
- It is recorded in this document by a pull request that changes nothing
  else (the "tooling freeze" addendum, §11), and merged before the first
  tuning run.
- Every tuning and study run, the tuning choice and the analysis use exactly
  that commit's tooling. The freeze pull request adds a manifest
  (`tools/m3_experiments/freeze.json` for version 6;
  `tools/m3_pendulum/freeze.json` for version 4) naming the commit. Each refuses to
  run unless:
  - the tree is clean;
  - every path the study executes or reads (`tools/m3_pendulum`,
    `tools/vector_env`, `tools/m3_experiments`, `tools/legacy_conversion`,
    `correct_il`, `data`, `config`, and the conversion and artifact
    manifests) equals that commit;
  - `src`, `pyproject.toml` and `uv.lock` equal the frozen `ccil` commit.
- A behavior-changing tooling fix after the freeze needs a new version of
  this document and a **complete rerun** of tuning and the study. The earlier
  runs are reported, not mixed with the new ones.

**Host.** All study runs execute on one host (siddh, CUDA). Timings are not
compared.

**Batching.** The runner trains many units' models in one group and records
each group's members, bucket and engine. A unit's results can depend on its
group only through float32 rounding. The grouping is a function of the plan
and the order of units, so a rerun from the frozen commit reproduces it,
except where a group runs out of device memory and is split (§9), which is
recorded.

## 3. Environment

There is one canonical Pendulum environment in `tools/m3_pendulum/`. It has
the physics that generated the demonstrations. Version 6 evaluates with the
batched implementation in `tools/vector_env/` (§2), which runs the same RK45
algorithm on every episode at once:

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
outcome, for the canonical and the batched environment:**
1. **Transitions.** For every transition of both datasets, stepping the
   environment from the stored (s_t, a_t) must reproduce the stored s_{t+1}
   within max |Δ| ≤ 1e-6. The batched environment must also agree with the
   canonical one on every transition, and the maximum difference is
   reported.
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
files with the independent implementation, and refuses any mismatch. It also
replays every saved transition through the batched integrator that produced
it and requires the saved next state bit for bit.

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
| **BC-modern** | secondary (version 6) | BC with the modern recipe below, its learning rate chosen as in §6 |
| **CCIL-modern** | secondary (version 6) | CCIL (labels and balancing as the CCIL arm) with the modern recipe below, its learning rate chosen as in §6 |

**Shared policy settings:** the experimental vector harness (#89), trained
by trainer v2 (`policy.trainer_version 2`), with batch 512, 100 epochs (÷20
when balanced), learning rate 1e-3, [256, 256], and standard and min_max
scalers. Dynamics models fit with `transition_model.trainer_version 2` and
`transition_model.lipschitz_coordinates model_normalized`. Balancing is the configuration's for the
CCIL arms; BC and NoiseBC never balance
(`policy.balance_noisy_labels false`).

**The Lipschitz ablation (CCIL-noLip).** CCIL's dynamics model uses the
configuration's regularizer: `soft_sampling` on continuous Pendulum and
`slack` on discontinuous. CCIL-noLip removes it (`lipschitz_type none`).
- **Continuous Pendulum.** The `soft_sampling` ratio is measured in the
  model's normalized coordinates (`transition_model.lipschitz_coordinates
  model_normalized`, #125). Its constraint (2), ε (3e-4) and penalty weight
  are the paper's. CCIL − CCIL-noLip is then the actual Lipschitz ablation.
- **No raw-coordinate arm.** In raw coordinates the penalty is never active
  on this task, so such an arm would train the same model as CCIL-noLip.
- **Legacy differed again.** Legacy's term, corrected in #21, mixed the two
  spaces: it compared the normalized residual with the next-state
  prediction, against 2 times the norm of the unscaled noise. So the paper's
  continuous ablation removed a different term than this one; the comparison
  with paper Fig. 4(c) reports this.
- **A regularizer, not a certified bound.** The penalty is a training
  regularizer. Its constraint is never used as a certified bound, and the
  CCIL-filtered certificates recompute model bounds from the weights (#125).

**The modern recipe** (#115), for BC-modern and CCIL-modern only. Each value
is a fixed design choice, with the reason for it; only the learning rate is
tuned (§6).

| Setting | Value | Reason |
|---|---|---|
| Optimizer | AdamW (`policy.optimizer adamw`), β = (0.9, 0.999), ε = 1e-8, decoupled weight decay 1e-4 | The standard optimizer for this model class; light decay regularizes long training without becoming a second tuned axis |
| Learning rate | peak chosen as in §6; linear warm-up from peak/1,000 at step 1 to the peak at step 1,000 (`policy.warmup_steps 1000`), then cosine decay to 0 at step 50,000 (`policy.lr_schedule cosine`) | Warm-up stabilizes the first steps of AdamW; the cosine endpoint makes the last steps small, which the checkpoint rule then benefits from |
| Length | 50,000 optimizer steps per model (`policy.step_budget 50000`), the same for both arms; epochs are cut at the budget | Equal compute: CCIL has about 20 times BC's rows, so equal epochs would give it far more steps |
| Gradient clipping | global norm 1.0 per model (`policy.grad_clip_norm 1.0`) | Guards against rare large gradients from noisy labels; inactive in ordinary steps |
| Weight averaging | EMA of the weights, decay 0.999 (`policy.ema_decay 0.999`); the EMA weights are the policy, and they are what validation judges and returns | Averaging over about the last 1,000 steps smooths the late iterates |
| Checkpoint | the EMA weights with the lowest validation loss, judged every 2,000 optimizer steps and at step 50,000 (`policy.checkpoint_selection validation`, `policy.validation_interval_steps 2000`): 25 selection points for both arms | A shared step cadence gives both arms the same selection opportunities; an epoch cadence would give BC about 20 times more |
| Validation loss | Σ w·mean_j(π(s) − a)_j² / Σ w over the held-out episodes' demonstrations, in the scaled action space: the training loss on held-out data | The quantity training minimizes, on data training never sees |
| Held-out data | 20% of episodes (the data split, `data.validation_fraction 0.2`, at the unit's seed) are held out of policy training (`policy.exclude_validation_episodes true`) | A clean holdout: see below |
| Dynamics model | CCIL-modern's label model is fitted on the training episodes only (`transition_model.fit_episodes train`); its labels and acceptance are otherwise CCIL's | Held-out episodes then influence no training label, so CCIL-modern and BC-modern see the same data |

The held-out episodes are used only to select checkpoints and, through their
losses, to tune the learning rate. Everything else (architecture, batch,
scalers, balancing) is the corresponding paper-settings arm's.

**Asymmetry, stated.** In the primary comparison, BC and CCIL use the paper's
settings, while NoiseBC receives a scale search (§6), so any bias this
creates favors the baseline. CCIL-tuned receives the same search budget as
NoiseBC, so that CCIL − NoiseBC can also be read with both sides tuned.

## 6. Seeds and tuning

**Tuning seeds 11–13** (version 4: 1–3). These are used only to choose the NoiseBC scale and the
CCIL-tuned action noise, **separately per task**. Each search has five values:

| Arm | Parameter | Grid |
|---|---|---|
| NoiseBC | `policy.noise_scale` (standardized observation units) | {1e-4 (the paper's value), 1e-3, 1e-2, 1e-1, 3e-1} |
| CCIL-tuned | `generation.action_noise_std` (action units) | {the paper's value (1e-5 continuous, 1e-4 discontinuous), 1e-3, 1e-2, 1e-1, 3e-1} |
| BC-modern | `policy.learning_rate` (peak) | {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} |
| CCIL-modern | `policy.learning_rate` (peak) | {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} |

The support radius is not searched: it is the paper's per-task constraint (on
discontinuous Pendulum, 0.01 keeps labels from crossing the discontinuity),
and searching it would give CCIL-tuned a larger budget than NoiseBC.

For each arm and task:
- **Selection:** for NoiseBC and CCIL-tuned, the value with the highest mean
  noise-free return over the tuning seeds. For BC-modern and CCIL-modern, the
  value with the lowest mean selected-checkpoint validation loss over the
  tuning seeds.
  Ties go to the smaller value.
- **Order, for the modern arms.** Their tuning runs compute no return. The
  choice is computed from validation losses and written once to the run
  record (`tuning-choice.json`). Only then are their tuning-seed returns
  computed, into separate records that name the choice and the tuning record
  by SHA-256. The analysis refuses returns that do not, so no return can have
  informed the choice. The returns are reported afterwards.
- **Failed tuning runs.** A value with any failed tuning run is ineligible.
  If every value is ineligible, the arm uses the paper's value, and this is
  recorded.
- **Reporting.** Every value-by-seed tuning result is reported, with
  CCIL-tuned's label acceptance rate.
- Both choices are committed to the run record before any study seed runs.

**Study seeds 1201–1220** (20 seeds; version 4: 101–120). Each seed runs one complete pipeline per
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
    - It applies to BC, NoiseBC, CCIL, CCIL-tuned, BC-modern and CCIL-modern.
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
      reported empirical mean of 0.02367 and bound of 0.065. It is measured
      on 10,000 accepted labels per CCIL unit (all of them if fewer), drawn
      without replacement by `default_rng(SeedSequence([s, 10]))`. Version 6
      takes the environment step with the batched integrator (§3), one call for
      all sampled labels.

    The ledger entry `sim-pendulum-label-bound` stays **blocked**: the
    procedure behind 0.065 is unrecorded, so this measurement is reported as
    related evidence, not as a reproduction.
  - **Training diagnostics:** the dynamics model's held-out one-step error,
    the label counts by reason, and the supervision manifests; for the
    modern arms, the per-epoch validation losses and the selected epoch.
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
noise level and data size; and, for the modern arms, CCIL-modern − BC-modern,
BC-modern − BC and CCIL-modern − CCIL, noise-free and per noise level. Each gets the union interval at c = 0.95. They are
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
(noise-free), and for BC, NoiseBC, CCIL, CCIL-tuned, BC-modern and
CCIL-modern at each noise level,
arm − zero is reported with the union interval at c = 0.95. These intervals
inform no decision, are not part of the positive control, and make no
multiplicity claim.

## 9. Failures

**Infrastructure faults** (out of memory, a lost host, a killed process) are
rerun once, with identical inputs. (Version 6: a training group that runs out
of device memory is first split in half and retried, down to single units;
only a single unit's memory exhaustion is a fault.) A second fault excludes that seed for that
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

The version 4 run is reported separately, labeled incomplete and superseded:
its tuning tables and choice, and its 21 study units, with the disclosure
above.

## 11. Tooling freeze

**Version 6:** to be recorded by a pull request that changes nothing else,
merged before the first version 6 tuning run. It names the frozen tooling
and `ccil` commits in `tools/m3_experiments/freeze.json` and lists the §2 and
§3 evidence, generated from one clean commit on each host.

**Version 4 (superseded; its runs are reported, not reused).**

- **Frozen tooling commit:** `cba34fea7241cdbf4a48bb00083da76bb2383d40`, the
  merge of #106. It is named by
  [`tools/m3_pendulum/freeze.json`](../../tools/m3_pendulum/freeze.json),
  which every real run, the tuning choice and the analysis check (§2).
- **Evidence.** It was generated from the clean commit `b7267f0`. Every frozen
  tooling path, and `src`, `pyproject.toml` and `uv.lock`, are identical at
  `cba34fe`; only the evidence reports differ.
  - §3 acceptance checks, 8/8 on each host:
    [`report.json`](../fixtures/evidence/m3-pendulum/report.json) (macOS,
    CPU) and
    [`report-siddh.json`](../fixtures/evidence/m3-pendulum/report-siddh.json)
    (Linux, CUDA).
  - Run drivers and analysis, 7/7 on each host:
    [`study-report.json`](../fixtures/evidence/m3-pendulum/study-report.json)
    and
    [`study-report-siddh.json`](../fixtures/evidence/m3-pendulum/study-report-siddh.json).
    These cover known answers, interval coverage, smoke runs on seeds
    9001–9003 only, refusals, and the frozen-tooling gate.
- **No tuning or study outcome** existed when this was recorded.
