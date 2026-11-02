# Pendulum recovery-segment diagnostic: predeclaration, version 1 (#171, #167)

**Status: version 1, pushed before any transition model is fitted, any segment is generated, or any audit input is opened.** The draft's review decisions are recorded under "Version history". This step trains no policy. If the selected recovery segments fail coverage or trust, the outcome is reported, and no learner runs.

## Why

il-supervision-benchmarks#12 established a positive control, exploratory and against exposure-matched duplication: for states BC visits inside the demonstrator LQR-branch capture region (not the oracle's terminal set Ω_c), corrected-demonstrator labels improve both held-out action accuracy and final capture (`docs/studies/pendulum-capture-control.md` in il-supervision-benchmarks). Gate 3A showed that one-step CCIL cannot leave demonstration support on this task (`docs/experiments/pendulum-correction.md`, "Result").

This step asks whether CCIL's recovery segments (#171) can reproduce that supervision **from demonstrations alone**: states in the capture region, distinct from the demonstrations, with labels the demonstrator would endorse, confirmed by exact dynamics.

## Two phases, kept separate

1. **Generation (data-only).** The inputs are the demonstrations and the declared representation and metric. Nothing else: no exact map, no demonstrator, no environment, no capture set, no policy.
   - It fits the transition model, probes the recovery-length ladder, selects M automatically, and generates and freezes the segments.
   - The tool writes the selection and segment artifacts with their sha256 and refuses to start the audit until both exist.
   - The generation code imports nothing from the Pendulum task, demonstrator or harness.
2. **Audit (external information).** After phase 1 is frozen, the audit opens:
   - the exact map;
   - the corrected demonstrator;
   - the positive control's capture-state sets.

   It answers the six questions below. It cannot change M, the segments, or any phase 1 threshold.

## Inputs, fixed

- **Data:** the canonical continuous dataset (sha256 `59c332bc…cf43c`) at r = 0.1. These are the positive control's subsets and splits for seeds 3311–3320, built by the same rule (il-supervision-benchmarks `tools/capture_control.py`, `stage_data`), so each seed's audit target is its own capture set.
- **Seeds:** s ∈ {3311, …, 3320} (n = 10). Each seed is a separate dataset: a separate model, selection and segments.
- **Representation:** (sin θ, cos θ, θ̇), the dataset's declared encoding.
  - **Chart:** segments are parameterized in the chart (θ, ω).
  - **Re-encoding:** every state passed to or returned by the model is re-encoded as (sin θ, cos θ, ω), with θ = atan2(sin, cos), so every row is on the circle by construction. The model's un-projected circle drift, |sin² + cos² − 1| before re-encoding, is recorded per transition as a model diagnostic.
- **Metric (declared input, as in Gate 3A):** d(x, y) = √(wrap(e_x − e_y)² + ((ω_x − ω_y)/√g)²), with e = wrap(θ − π).
  - It is fixed before any data is read. It is the only physical constant phase 1 uses.
  - The periodic nearest-neighbor tree is the one Gate 3A uses.
- **Support scale:** τ_demo(s), the 95th percentile of d_demo over the validation-episode demonstration states, as in Gate 3A and the positive controls.

## Transition model and the held-out dynamics split

- **Model:** the il-supervision-benchmarks study's `dynamics` block, unchanged:
  - `mlp`, 512-512, ReLU;
  - soft-sampling Lipschitz penalty 2.0 in model-normalized coordinates;
  - 100 epochs, batch 512, learning rate 1e-3.
- **Fit:** `transition_model.fit_episodes: train`, so it is fitted on the seed's 8 training episodes only.
- **Check model:** a second model with the same settings and the same training episodes, independently initialized (fit seed s + 100000). The solver never uses it. Its only role is the disagreement criterion below.
- **Held-out dynamics split:** the seed's 2 validation episodes, never seen by either model. For each M on the ladder, ε_M is the 95th percentile of d(F̂_M(x_t, U_t), x_{t+M}).
  - Every contiguous window t…t+M inside a validation episode is used, never crossing an episode end.
  - F̂_M is the model composed M times with the recorded actions, re-encoding after each step.
  - The one-step error ε_1 is reported too.

## Recovery-length ladder and generator

- **Ladder:** M ∈ {1, 2, 4, 8, 16, 32}. The cap of 32 is 0.64 s at the dataset's 20 ms interval, about a third of the small-swing period 2π/√g ≈ 2.0 s.
  - The cap is set from that a priori scale, before generation.
  - If no M is admissible, the outcome stands: M = 64 is not added afterward.
  - M = 1 is the existing one-step case.
- **Source windows:** a training-episode window t…t+M with t + M ≤ 499. Episodes hold 500 states and 500 actions, and action t takes state t to state t + 1. The target is the demonstration state z_target = x_{t+M}.
- **Proposals, the persistent action-bias family:** K = 20 per window.
  - Each proposal draws one requested offset δ ~ N(0, (b/3)²), where b = 3 is the action half-range, so the standard deviation is one third of the half-range. One offset holds for the whole segment, drawn from `default_rng(SeedSequence([seed, episode index, t, M, k]))`.
  - The actions are Ũ = clip(U_t + δ, ±b), so the realized offset at step i is ũ_i − u_{t+i}.
  - **Recorded per segment:**
    - the requested offset;
    - the realized offset at every step;
    - the fraction of steps clipped;
    - the impulse, Σ_i (ũ_i − u_{t+i})·dt;
    - the action-deviation energy, Σ_i (ũ_i − u_{t+i})²·dt.
  - **Scope:** this diagnostic tests the persistent-bias proposal family only. It is not a test of every possible recovery-segment proposal.
- **Solve:** find the start state x̃_0 in the chart (θ, ω) such that F̂_M(x̃_0, Ũ) = z_target, residual measured in the metric d.
  - Newton from the source state x_t, with central-difference Jacobians through the composed model.
  - Tolerance 1e-4; at most 30 iterations.
  - The intermediate states x̃_i = F̂_i(x̃_0, Ũ) are recorded for i = 1…M.
- **Validators (per segment):**
  - finite;
  - solver converged;
  - every ũ_i within ±3;
  - every x̃_i strictly within representation-space L2 distance 1.0 of its demonstration counterpart x_{t+i}: the rule of `SupportRadius(1.0)`, the harness's validator, applied at every offset;
  - target residual ≤ 1e-3.
- **Solver classes:** as in Gate 3A, *returned at initialization*, *zero displacement*, *converged* and *failed*.

## Automatic M selection (per seed; data-only)

The probe uses a fixed subset: windows with t ≡ 0 (mod 10) in each training episode, K = 20. For each M it records:
- **acceptance:** the fraction of segments accepted;
- **degeneracy:** the fraction returned at initialization or with zero displacement;
- **reach:** r_reach(M), the median over accepted segments of d_demo(x̃_0), the start state's distance to the nearest training-demonstration state;
- **η_reach(M)** = r_reach(M) / τ_demo(s);
- **ε_M;**
- **check-model disagreement:** κ_M, the 95th percentile over accepted probe segments of each segment's largest disagreement between the check model and the primary model. A segment's disagreement is the larger of two values:
  - the largest one-step value, d(F̂_check(x̃_i, ũ_i), x̃_{i+1});
  - the composed value, d(F̂_check,M(x̃_0, Ũ), z_target);
- the target residual quantiles.

**M is admissible** when all of these hold:
1. **nondegenerate:** degeneracy ≤ 0.5;
2. **reaches beyond support:** η_reach(M) ≥ 1;
3. **trustworthy on demonstrations:** ε_M ≤ 0.25·τ_demo(s). The primary model's held-out M-step error must be at most a quarter of the support scale the segment claims to leave;
4. **trustworthy off demonstrations:** κ_M ≤ 0.25·τ_demo(s). Held-out error along demonstrations does not test the off-demonstration states the generator creates, so the independent check model must agree with the primary model on the generated segments;
5. **workable:** acceptance ≥ 0.5.

Criteria 3 and 4 are empirical evidence, not a certificate.

**Selected M(s)** is the smallest admissible M. If none is admissible, the seed's output is an **empty synthetic view** with the per-M diagnostic. An empty seed has no segments to audit and fails coverage.

## Generation at the selected M

- Every window of every training episode, with K = 20, through the same validators.
- **Segment identity:** (seed, source episode, t, M, k), plus the sha256 of the model checkpoint, the configuration and the code commit.
- **Frozen artifacts:** the per-seed segment files (x̃_0…x̃_M, Ũ, target identity, residual, solver class, validator outcomes), the selection record, and a manifest of their sha256.

### Pointwise output view (the default consumer)

- **Rows:** (x̃_i, ũ_i) for i = 0…M − 1 of every accepted segment.
- **Per-row fields:** segment identity, offset i, M, target identity, the evidence class *learned-model recovery segment (empirical, not certified)*, and weight 1.
- **Per-segment field:** the check-model disagreement. It is recorded for every generated segment and does not filter any.
- **Scope of the claim:** the segment-level claim, that the target is reached after M actions, belongs to the segment, not to any single row.
- **Reconciliation:** every row reconciles to exactly one segment. The tool refuses to write a view that fails this check.

### Exact-dynamics reference segments (for the later learner comparison; not used for selection)

After phase 1 is frozen, the same generator runs with the exact map (`tools/vector_env/pendulum.py`, action clipped to ±3) at each seed's already selected M, with the same proposals. These segments are audited by the same questions and reported separately. They cannot change M.

## Audit (phase 2)

**Audit inputs, with identities recorded:**
- the exact map;
- the corrected demonstrator `tools/pendulum_eval/expert.py`, primary variant;
- the positive control's capture-state sets: il-supervision-benchmarks run 20261102T003101Z, `select/s{seed}.npz`, with the sha256 of each file recorded at freeze. The collection set is the primary target and the diagnostic set is secondary.

**Regions:**
- **capture region:** |e| < 0.1, with π_D capturing within 500 steps (|e| ≤ 0.1 and |ω| ≤ 0.5 over the last 100);
- **boundary annulus:** 0.1 ≤ |e| ≤ 0.5.

**Empirical transfer radius:** ρ_cap(s), the 95th percentile, over the seed's diagnostic capture states, of the distance to the nearest collection capture state.
- In the positive control, labels at the collection states improved the policy at the diagnostic states. ρ_cap is the radius over which that supervision was observed to transfer in this experiment. It is not a general support radius or a guarantee.
- It is used by the audit only. Phase 1 never computes or reads it, so it cannot influence M or generation.
- Its value and its ratio ρ_cap/τ_demo are reported for every seed.

### The six questions, per seed

| # | Question | Measure | Criterion |
|---|---|---|---|
| 1 | Do accepted synthetic states enter the capture region? | distinct accepted segments with at least one state x̃_i, i < M, in the capture region (a row count would reward longer M) | ≥ 100 segments |
| 2 | Do they cover the fixed capture-state set? | for the collection capture states not already within ρ_cap of a training-demonstration state, the fraction within ρ_cap of a pointwise row | ≥ 0.5 |
| 3 | Are they distinct from the demonstrations? | the fraction of in-region rows with d_demo ≥ ρ_cap; η_reach at the selected M | ≥ 0.5 (η_reach ≥ 1 already holds by selection) |
| 4 | Do in-region labels agree with the demonstrator? | \|ũ_i − π_D(x̃_i)\| over in-region rows | ≥ 90% of in-region labels within 0.3 rad/s² (Gate 3A's consistency threshold, 5% of the action range) |
| 5 | What happens in the boundary annulus? | row count, the \|ũ_i − π_D(x̃_i)\| quantiles, and a phase plot of the label field against π_D | descriptive only: it neither selects nor rejects |
| 6 | Do exact dynamics confirm every segment and intermediate transition? | for each accepted segment, replay x̃_0 with Ũ through the exact map: every one-step error d(F(x̃_i, ũ_i), x̃_{i+1}) and the final d(F_M(x̃_0, Ũ), z_target) | ≥ 95% of segments have every one-step error and the final error ≤ 0.25·τ_demo(s) |

**Reporting the coverage denominator:** for question 2, the count of collection capture states already covered by the demonstrations is reported too.
- **Coverage is not the missing mechanism:** if fewer than 8 seeds have at least 100 capture states that the demonstrations do not cover, the outcome is "coverage is not the missing mechanism at ρ_cap". It is reported, and no learner runs.

### Decision (fixed)

A seed **passes** when it has a selected M and meets questions 1, 2, 3, 4 and 6.

The diagnostic **passes** when at least 8 of the 10 seeds pass. Otherwise it fails, and the outcome is reported with the full per-M frontier. No learner runs, and the ladder, proposal scale, thresholds and radius are not changed after the fact.

**If it passes,** the next learner comparison, separately predeclared in il-supervision-benchmarks#12, is:
- BC;
- duplication;
- demonstrator capture relabeling;
- exact-dynamics recovery segments;
- learned-dynamics data-only recovery segments.

## Figure (descriptive)

On the Gate 3A axis contract, e ∈ [−π, π) periodic and ω ∈ [−8, 8], with a near-upright inset (e ∈ [−0.5, 0.5], ω ∈ [−3, 3]), for the first seed:
- the demonstrations;
- accepted segments drawn as polylines from x̃_0 to the target;
- the capture states, colored covered or uncovered;
- the label-minus-demonstrator field |ũ − π_D| in the region and the annulus.

## Running and artifacts

- **Tool:** `tools/pendulum_recovery/`. Phase 1 and phase 2 are separate commands, and phase 2 refuses to run without phase 1's frozen manifest.
- **Runs:** on siddh in tmux, with logs in `runs/`, headed by the command, commit, dirty state, host and versions.
- **Progress:** at least every 5 minutes; each (seed, stage) is checkpointed atomically and resumes with `--resume`; the stall threshold is 30 minutes; a watcher runs.
- **Smoke first:** a smoke run on off-protocol seeds exercises crash and resume, the phase-2 refusal, and row reconciliation.
- **Committed evidence:** `docs/fixtures/evidence/pendulum-recovery/`, holding the selection records, the audit summary, the logs and the figure. The segment files are a release asset with their sha256.

## Not covered

Policy training; any M, σ, K or threshold chosen from audit outcomes; the chunk view; certificates for composed segments (#171 work order step 2, so the certificate class is *not computed*); and other tasks.

## Version history

**Version 1** froze the draft's review points as amended at review:
- **Proposal:** the constant offset is named the persistent action-bias family. Its scale is stated as one third of the action half-range, and its realized offsets, clipping, impulse and deviation energy are recorded. The draft's claim that i.i.d. per-step noise largely cancels was removed: that depends on the dynamics.
- **Ladder:** M ∈ {1, …, 32} is approved, and no M = 64 is added afterward.
- **Trust:** ε_M ≤ 0.25·τ_demo is kept. An independently initialized check model is added, and its disagreement on the generated segments must meet the same threshold.
- **Radius:** ρ_cap is named the empirical transfer radius and is used by the audit only.
- **Criteria:**
  - "≥ 100 rows" became "≥ 100 distinct segments";
  - "median agreement ≤ 0.3" became "≥ 90% of labels within 0.3";
  - exact confirmation was raised from 90% to 95%;
  - the boundary annulus stays descriptive.
