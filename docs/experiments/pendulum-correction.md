# Pendulum correction treatment, Gate 3A: predeclaration, version 1 (#145, #167)

> **Historical record (2026-11-03).** The current account of the Pendulum investigation is [the Pendulum account](https://github.com/personalrobotics/il-supervision-benchmarks/blob/main/docs/studies/pendulum.md). The text below is unchanged.

**Status: version 1, pushed before any candidate in this document is generated.** If the gate fails, the failure is reported, no learner is trained from this treatment, and any change is made as a new version.

## Result (version 1: run 20261101T222500Z from `d36fc58`)

**No treatment at any r.** No σ passes, so no learner is trained from this treatment. The evidence:
- the summary, `docs/fixtures/evidence/pendulum-correction/correction.json`;
- the per-condition funnel, `funnel.log`;
- the portrait, `docs/experiments/figures/pendulum-correction-geometry.svg`.

- **Construction validity is not the problem.** Every accepted row is on the manifold at every σ, so solver b was never needed. For moved rows, the exact map reaches the target within 1e-3.
- **Near-identity at small σ:** every accepted row at σ ≤ 1e-4 is returned at initialization, as are 90% at 1e-3. That reproduces PR #16. The fraction falls to 42% at 3e-3 and to 4% or less from 3e-2.
- **Demonstrator consistency falls as σ grows:** the median |ã − π_D(z̃)| is 0.18 at σ = 0.1, 0.53 at 0.3 and 1.25 at 1.
- **The binding condition is coverage.** At every σ and every r, no accepted training row lies τ_demo from the training demonstrations.
- **This is structural, not a σ choice.** A one-step preimage x₁ = (θ₁, ω₁) and its source x₂ = (θ₂, ω₂) both reach the same successor in one interval dt = 0.02, under actions at most δu = 6 rad/s² apart.
  - **Comparison system:** going backward from that common successor, the differences δθ and δω obey δθ̇ = δω and δω̇ = −g(sin θ₁ − sin θ₂) + Δu, with |sin θ₁ − sin θ₂| ≤ |δθ| and |Δu| ≤ δu. Both start at zero at the successor, so they are bounded by the solution of the majorant system δθ̈ = g·δθ + δu.
  - **Bound** (√g·dt = 0.0626):
    - |δθ₀| ≤ (δu/g)(cosh √g·dt − 1) = 0.0012004 rad;
    - |δω₀| ≤ (δu/√g) sinh √g·dt = 0.120078 rad/s;
    - so d_source = √(δθ₀² + (δω₀/√g)²) ≤ 0.038357.
  - **Correction.** Version 1 of this section stated 0.0383 = dt·δu/√g. That counted only the velocity difference from the actions and omitted the angle difference and the gravity term. The corrected bound is 0.1% larger, and no conclusion changes.
  - **Scope of the bound.** It is proved for the continuous dynamics θ̈ = −g sin θ + u. The exact map integrates that flow with RK45 under scipy's default tolerances, and its deviation from the flow is not bounded here. On the map, the observed maximum d_source over every accepted row of the scan is 0.0383, at σ = 1, consistent with the bound.
  - **Consequence.** The source is itself a training-demonstration state, so d_demo ≤ d_source ≤ 0.038357. That is below τ_demo for each declared subset: 0.116, 0.088 and 0.053 at r = 0.1, 0.25 and 1. The smallest margin is 0.0146.
  - **Conclusion.** For the continuous dynamics, no σ or K of this one-step generator can pass this gate at this timestep and torque bound, for these subsets. For the RK45 map, the measured result covers the declared σ grid, and the bound together with the observed maximum explains it.

## Why

il-supervision-benchmarks#12 can test whether CCIL's synthetic supervision helps a learner only if CCIL first produces a real treatment: labeled states that are physically valid, meaningfully outside the demonstrations, in places where a policy needs recovery, with actions that a known-good controller would endorse. PR #16 did not have one. Its σ was chosen by validation loss, and at σ ≤ 1e-4 every exact-dynamics row was returned unmoved.

Gate 3A asks only that question. It trains nothing and computes no return.

## Evidence classes

Each candidate label (z̃, ã) carries three trust levels, kept separate:

| Level | Question | Status here |
|---|---|---|
| Construction validity | Does applying ã at z̃ under the exact dynamics reach the intended target? | checked directly |
| Demonstrator consistency | Is ã close to what the corrected demonstrator does at z̃? | checked directly |
| Task-optimality | Is ã near-optimal for the task? | **unsupported**: the DP oracle did not converge (`docs/experiments/pendulum-oracle.md`, "Result") |

No task regret, value, or optimal action is reported, and no figure here is an optimal-control portrait.

## Inputs, fixed

- **Data:** the canonical continuous dataset, sha256 `59c332bc…cf43c`: 100 episodes of 500 steps.
- **Subset and split:** non-study seed 9401, as in `il-supervision-benchmarks/tools/sigma_geometry.py`. The subsets follow the harness: the first round(r·100) episodes of `default_rng(SeedSequence([9401, 9])).permutation(100)`, for nested r ∈ {0.1, 0.25, 1.0}. Each subset is split with `ccil.data.split.make_split` (seed 9401, validation fraction 0.2). Rows are reported for training-episode sources only.
- **Dynamics:** the exact map `tools/vector_env/pendulum.py` with the action clipped to ±3, wrapped as a CCIL model (`delta = step(z, a) − z`), as the harness's `ExactModel` does.
- **Generator (solver a), the harness's configuration:**
  - `NoisyActionPreimage` with K = 20, max_iterations 30, tolerance 1e-4, action bounds ±3;
  - validators `Finite`, `GenerationFailed`, `SupportRadius(1.0)` and `ActionBounds(±3)`, through `ccil.core.generate_labels`;
  - representation (sin θ, cos θ, θ̇), and Newton starts at the source state.
- **σ grid (normalized; action-noise std = 3σ):** 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3 and 1. It includes every σ PR #16 selected. Draws depend only on (seed, episode, frame), so the r < 1 rows are exactly the r = 1 rows of their episodes, and one generation per σ serves all three r.
- **Solver b (manifold-constrained Newton in (θ, ω)):** run only for a σ where, at some r, more than 1% of solver (a)'s accepted training rows are off the manifold. Otherwise it is not needed and is reported as not run.
  - It reuses solver (a)'s draws: the same source, target and ã per row.
  - Newton runs in (θ, ω) from the source state, with central-difference Jacobians and the residual in the physical metric below. The tolerance is 1e-4 and the cap is 30 iterations.
  - Every row is on the manifold by construction. It is checked with the same four validators' rules.

## Coordinates and metric

- **Coordinates:** x = (e, ω), with e = wrap(θ − π) and θ = atan2(sin, cos).
- **Distance:** d(x, y) = √(wrap(e_x − e_y)² + ((ω_x − ω_y)/√g)²), the oracle's physical metric (radians, with ω scaled by √(g/l)).
- **Manifold:** a state is on the manifold when |sin² + cos² − 1| ≤ 6.7e-7 (the oracle's bound, 10 × the dataset's own drift).

## Per-row fields

Every generated row, accepted or not, gets:
- **Identity:** the source episode and frame, the draw, σ and the solver.
- **Harness acceptance** and the rejection reasons.
- **on_manifold**, and the off-manifold error.
- **Construction residual:** r_c = d(F(z̃, ã), z_target) through the exact map, from z̃ as stored.
- **Bounded:** |ã| ≤ 3.
- **Solver class:**
  - *returned at initialization:* the residual at the source state is already ≤ 1e-4, so Newton never iterates (the PR #16 case);
  - *zero displacement:* z̃ = z_source exactly, for any other reason;
  - *converged:* an iterated solve that met the tolerance;
  - *failed.*
- **Displacement:** d_source = d(x̃, x_source).
- **Coverage:** d_demo, the distance to the nearest training-demonstration state of the same r.
- **Demonstrator consistency:** Δ_D = |ã − π_D(z̃)| in rad/s², where π_D is the corrected demonstrator (`tools/pendulum_eval/expert.py`, primary variant). Rows within 0.01 rad of π_D's own LQR switch (`switch_margin` < 0.01) are flagged *switch-ambiguous* and reported separately.
- **Region**, from the demonstrator and the energy only:
  - *LQR window:* |e| < 0.1, which is π_D's own LQR branch;
  - *pumping:* H < 0 outside the window;
  - *overshoot:* H ≥ 0 outside the window;
  - H = ½ω² − g(1 − cos e), so H = 0 at upright rest.
- **Demonstrator recovery from x̃:** π_D, rolled out for 500 steps through the exact map from z̃ projected onto the manifold, ends captured: |e| ≤ 0.1 and |ω| ≤ 0.5 at every one of its last 100 steps. This is computed only for rows that are candidates for the gate's supported set.

## Thresholds, fixed before any candidate exists

- **Construction-valid:** r_c ≤ 1e-3.
- **Demonstrator-consistent:** Δ_D ≤ 0.3 rad/s², which is 5% of the action range. The oracle's convergence rule used the same 0.3 for action agreement.
- **Outside demonstration support:** d_demo ≥ τ_demo(r). τ_demo(r) is the 95th percentile of d_demo over the validation-episode demonstration states of the same subset. It is how far a fresh demonstration typically lies from the training demonstrations, computed from data alone before any candidate.
- **Supported row:** one that meets all of these:
  - accepted by the harness validators;
  - on the manifold;
  - construction-valid;
  - bounded;
  - demonstrator-consistent;
  - not switch-ambiguous;
  - outside demonstration support;
  - outside the LQR window;
  - recovered by the demonstrator.

## The gate, per (solver, r, σ) cell

A cell passes when both hold:
1. **Not near-identity:** at most half of its accepted rows are *returned at initialization* or *zero displacement*.
2. **A real recovery population:** at least max(100, 1% of accepted rows) supported rows, from at least 5 distinct training episodes, in pumping or overshoot.

**Gate 3A passes for r** when solver (a), or solver (b) where it was run, passes at some σ. The **treatment cell** for r is the smallest passing σ. If no σ passes at r, the outcome for r is **"no treatment"**: it is reported with its full table, and no learner is trained at that r.

Every cell is reported, passing or not, with:
- the accepted fraction and rejections by reason;
- the solver-class fractions and the off-manifold fraction;
- quantiles of r_c, d_source, d_demo and Δ_D, overall and by region;
- the supported count.

## What the learner test receives (il-supervision-benchmarks#12; not run here)

For each r with a treatment cell, the **matched states** are that cell's supported rows' z̃. The three smallest arms then train on:
1. **BC:** the training demonstrations;
2. **demonstrator relabels:** the demonstrations plus (z̃, π_D(z̃)) at the matched states;
3. **exact-dynamics CCIL:** the demonstrations plus (z̃, ã) at the same matched states.

Seeds, learners, exposure and statistics belong to #12.

## Figure: a descriptive phase portrait

`docs/experiments/figures/pendulum-correction-geometry.svg`, on the oracle's axis contract: e ∈ [−π, π) horizontal and periodic, and ω ∈ [−8, 8].

- **One row per r**, at its treatment cell, or at the largest σ with the label "no treatment" if none passes.
- **(a) Background and arrows:** the demonstrator action field, energy contours with H = 0 bold, and the LQR window as a band. On top, the training demonstrations, and sparse source-to-synthetic arrows for accepted on-manifold rows, colored by Δ_D. Supported rows are marked.
- **(b) Coverage:** the d_demo map of the supported rows against τ_demo.
- **(c) What is excluded:** rejected rows, near-identity rows, and off-manifold rows, kept in their own panel and never drawn as phase-space corrections.

It is titled as a correction-geometry portrait, never as an optimal-control portrait.

## Artifacts and running

- **Per-row file:** one `.npz` per σ, a release asset with a recorded sha256.
- **Summary:** `docs/fixtures/evidence/pendulum-correction/correction.json`, committed. It holds the cell table, τ_demo, the treatment cells and the code state.
- **Regeneration:** `tools/pendulum_correction/scan.py --check` regenerates the summary and compares it with the committed copy, and `figure.py --check` does the same for the figure.
- **Runs:**
  - on siddh in tmux, with logs in `runs/`;
  - progress at least every 5 minutes;
  - each σ checkpointed atomically, with `--resume`;
  - a 30-minute stall threshold;
  - a watcher for completion, failure and stall.

## Not covered

Learned dynamics; task-optimal regret; the discontinuous variant; certificates (the exact model has no checkpoint for `certify`, so the certificate class is reported as *not computed*); and any learner.
