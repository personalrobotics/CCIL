# Pendulum v6: behavioral diagnostics (declared before inspection)

Written on 2026-10-28, after the v6 analysis (`docs/fixtures/evidence/pendulum-study-v6/analysis.json`) and before any per-episode state, action or reward of the v6 rollouts was inspected.

The aggregate per-seed returns in the analysis had been seen. Three of them could not be interpreted:
- the continuous positive control fails because of one seed: BC on 10% of the demonstrations, seed 1209, returned −19068;
- the zero-action policy returns more than every learned arm, except CCIL-tuned on continuous;
- CCIL − BC is negative at every evaluation noise level.

These diagnostics explain v6. They do not change its hypothesis test, its intervals or its acceptance criteria (predeclaration §8). No broader analysis is added until these three are done.

## Decision rule

- **If replay or recomputation disagrees with the stored evaluations** (tolerances below), stop: v6 is treated as invalid.
- **If they agree,** the predeclared statistical result stands, and its interpretation is withheld until these diagnostics explain it.

## Facts from the evaluator's code (`tools/m3_pendulum/env.py`), before any data

- **State.** (sin θ, cos θ, θ̇). θ = 0 hangs down, and the target is upright, θ = π. The equation of motion is θ̈ = −9.81 sin θ + a.
- **Start.** θ ~ U(−1, 1), θ̇ ~ U(−0.5, 0.5), near the bottom.
- **Reward,** from the pre-step state and the applied action: r = −½[(θ − π)² + θ̇²] − ½a², with θ wrapped to [0, 2π).
- **Episodes.** 500 steps of dt = 0.02 (10 s), with no termination.
- **No clipping.** Actions are applied unclipped. The declared action range [−3, 3] bounds only the noise draws.
- **Evaluation noise** at level n acts on both the observation and the action:
  - observation: o = (1 − n)·s + n·U(low, high), where θ̇'s bounds are ±4π;
  - action: a = (1 − n)·a_policy + n·U(−3, 3).

  The noise therefore also shrinks the commanded action by a factor (1 − n).

## 1. Seed 1209 (first)

**Unit:** continuous, `bc_frac10`, seed 1209, noise-free. **Controls:** continuous `bc` and `ccil`, seed 1209, noise-free.

**Integrity checks:**
- The policy checkpoint's sha256 equals the record's `checkpoint_sha256`, and the rollout's sha256 equals `rollout_sha256`.
- **Recomputation.** Rewards recomputed from the stored states and applied actions, and returns summed from them, equal the stored rewards, returns and R within 1e-9 absolute.
- **Open-loop policy replay.** The checkpoint, loaded as the runner loads it (`VectorBCPolicy.load`, CUDA, `predict`), applied to the stored observations, reproduces the stored `policy_actions`. Tolerance: 1e-6 absolute, float32 policy. Any difference is reported with its maximum.
- **Closed-loop replay through the study's evaluator** (`tools/vector_env/pendulum.rollouts`, same device):
  - states, actions and rewards equal the stored arrays bit for bit;
  - each episode's return is within 1e-6 absolute.
- **Closed-loop replay through the canonical scipy evaluator** (`tools/m3_pendulum/env.rollout`). This is an independent check. A per-episode return difference above 1e-6 is a disagreement, unless it is explained by rounding amplified by the dynamics: a first divergence at the 1e-12 level that grows over time, reported with the step where it first appears. Anything else counts as a disagreement.

**Description of the unit:**
- every episode's return, sorted;
- the number of episodes that account for half of the total shortfall against the controls' median episode;
- for the worst episodes: θ, θ̇, the commanded action and the reward over time;
- the maximum |a| and |θ̇|;
- the count of non-finite values, and of steps with |a| > 3.

**Classification,** stated in the report:
- (a) a genuinely catastrophic policy, poor on most episodes;
- (b) a few pathological episodes;
- (c) an evaluator problem, if any integrity check fails.

## 2. Zero action against the learned policies

Arms: `zero`, `bc`, `ccil`, `noise_bc`, `ccil_tuned`, on both tasks, noise-free, all 20 study seeds and 100 episodes. The data are the stored rollouts; nothing is re-run.

**Per episode:**
- **Angle error** e_t = |wrap(θ_t) − π| ∈ [0, π].
- **Upright:** e_t < 0.3 rad. The sensitivity is reported at 0.1 and 0.5 rad.
- **Reaches upright:** upright at some step.
- **First upright step;** **steps upright in total;** **the longest consecutive upright run.**
- **Holds:** upright on each of the last 100 steps (2 s).
- **Swings up and falls:** reaches upright, and later has e_t > π/2.
- **Never swings up:** e_t > π/2 at every step.
- **Return decomposition:** the episode's return split into its angle term, velocity term and action term.

**Summaries:**
- per arm and task, the fraction of episodes in each category, and the mean of each return term, over all 2000 episodes;
- per seed, the same means, as a table.

**Representative seeds,** fixed by rule, per task:
- the seed whose `bc` R is the median of the 20. With 20 seeds, this is the lower of the two middle seeds;
- the seed with the largest CCIL − BC difference;
- the seed with the smallest (most negative) CCIL − BC difference. This one is added so the selection is not one-sided.

For each representative seed: θ, θ̇, the applied action and the reward against time, for each arm. The episodes shown are the one at the median of BC's return on that seed, plus episodes 0, 1 and 2.

## 3. The evaluation-noise mechanism

Arms: `bc` and `ccil`, on both tasks, at noise 0.01, 0.05 and 0.1, all seeds. Both arms share the same initial states and noise draws.

**Per step, from the stored rollouts:**
- the commanded action a_policy;
- the action perturbation a − a_policy;
- the observation perturbation o − s;
- the applied action a.

There is no clipping, so "saturation" is reported as the fraction of steps with |a_policy| > 3 and with |a| > 3, and as the distribution of |a_policy| relative to 3.

**Per arm, task and level:**
- those fractions;
- the mean |a_policy| and the mean |a − a_policy|;
- the return decomposition of §2;
- the §2 behavior categories.

**Paired comparison, BC against CCIL on identical starts and draws:**
- the per-seed difference of each quantity, and the paired change in each category from noise-free to each noise level;
- noise-free per-seed CCIL − BC differences against the corresponding noisy differences.

**Separating the two noise sources:**
- **action noise only:** the stored noise draws applied to the action, with the observation left clean;
- **observation noise only:** the reverse.

Both are re-run through the study's evaluator from the stored checkpoints, for `bc` and `ccil` at each level. These are evaluations of existing checkpoints; nothing is trained. They are diagnostic only, and never enter the v6 analysis.

## Artifacts

- Scripts in `tools/pendulum_diagnostics/`, run from a clean commit on siddh against the v6 run root.
- Each diagnostic writes a JSON report and figures. Logs go in `runs/` on siddh.
- The reports state which result of this document each figure or number answers.
