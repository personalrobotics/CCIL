# Pendulum discrete DP oracle: predeclaration, version 6.1 (#145)

**Status: version 6.1, pushed before any oracle run that a check below reports on.** Version 1 (`d0d13ed`) was pushed before any oracle value was computed; "Version history" records why it changed. The checks below are fixed here. If any fails, the oracle is not used: the failure is reported, and any change is made as a new version in a new pull request.

## Result (version 6.1: run 20261101T191213Z from `e7e9f95`)

The oracle stopped at L3, as declared, because the L2-L3 convergence rule fails. It is not used. `docs/fixtures/evidence/pendulum-oracle/oracle.json` is the failure artifact.

- **Established:** the deterministic formulation is behaviorally proper. At L1, L2 and L3, for every β, the greedy policy arrives at Ω_c from every node of R within the horizon and keeps the |ω| ≤ 10 margin, and the bracketing gap and Bellman residual are within tolerance.
- **Established:** the domain and boundary treatment are stable. Values, action sets and greedy trajectories agree between ω_max = 12 and 14, and no boundary transition is unsupported.
- **Not established:** continuum accuracy. The 95th-percentile relative change in V is 0.12–0.14 from L1 to L2 and 0.047–0.078 from L2 to L3, against 0.05. It passes only at β = 1. The value from hanging rest (S1) still changes by about 3% from L2 to L3.
- **Consequence:** task-optimal regret is unavailable. Downstream work uses construction validity and demonstrator consistency, and reports task-optimality as unsupported (`docs/experiments/pendulum-correction.md`). No objective is selected, and β = 1 is not adopted after the fact.
- **Closed:** oracle work ends here (#145). No version 7 is drafted, and the β grid is not rerun. Higher-order interpolation can break the monotonicity that makes the bracketed scheme trustworthy, and refining selected starts repairs neither the phase field nor per-label regret. Exact dynamics may audit a later frozen result, but they cannot select the objective or the recovery length.

## Why

il-supervision-benchmarks#12 (Stage A) needs a task oracle in order to describe CCIL's synthetic labels by their correction geometry and their action quality, before any policy is trained. This document defines that oracle for the regime of the canonical continuous dataset.

It is a **discrete Bellman (dynamic-programming) oracle**: it optimizes the exact sampled transition map. "HJB" is reserved for a later continuous-time PDE comparison. Its evidence class is **numerically converged DP oracle**, not "certified": no rigorous discretization bound is produced.

## Task semantics

- **Transition map:** `tools/vector_env/pendulum.py`, `step(states, actions, discontinuous=False)`, which is the evaluator's map.
  - It integrates θ̈ = −9.81 sin θ + u over 0.02 s with adaptive RK45, starting from (sin θ, cos θ, θ̇).
  - There is no damping. θ = π is upright, and θ = 0 hangs down.
  - The commanded action is clipped to |u| ≤ 3.
  - Only the continuous variant is covered here.
- **Units:**
  - u is an angular acceleration;
  - the normalized action is a = u/9.81, so ρ = 3/9.81 ≈ 0.3058;
  - the error angle is e = wrap(θ − π) ∈ [−π, π), and ω = θ̇.
- **Cost per step:** dt · [(1 − cos e)/2 + λa²], with dt = 0.02.
- **Effort price:** β = λρ², the cost of saturated effort relative to the largest height deficit. The β grid is {1e-3, 1e-2, 1e-1, 1}, that is λ ≈ {0.0107, 0.107, 1.07, 10.7}.
  - β = 0 is descriptive only. It is a first exit to the β = 1e-3 terminal set with zero terminal value, it is never selectable, and it never reuses a positive-β terminal value.
- **Physical metric:** S = diag(1 rad, √(g/l) = 3.132 rad/s) on (e, ω), with distance ‖S⁻¹Δ‖₂. It is fixed and independent of the data, and every later distance uses it.
- **Energy relative to upright:** H = ½ω² − g(1 − cos e). It is 0 at upright rest and −2g hanging at rest.

## Local LQR terminal set Ω_c (per β > 0)

- **Linearization:** A and B are the exact one-step map's Jacobians at upright in x = (e, ω), by central differences with step 1e-6.
- **LQR solution:** P is the DARE solution (`scipy.linalg.solve_discrete_are`) with Q = dt·diag(1/4, 0) and R = dt·λ/9.81², and K = (R + BᵀPB)⁻¹BᵀPA.
- **Ω_c** = {x : xᵀPx ≤ c}, with c = min(c_input, c_decrease, c_accuracy) (version 4):
  - **c_input** = 9 / (K P⁻¹ Kᵀ), so that |Kx| ≤ 3 on all of Ω_c.
  - **c_decrease** is the largest c, by bisection to a relative 1e-3 over (0, c_input], such that every point of a fixed sample satisfies V(F_cl(x)) < V(x) for x ≠ 0, and so F_cl(x) ∈ Ω_c. The sample is 20,000 points uniform in Ω_c plus 2,000 on its boundary, seed 1451.
  - **c_accuracy** is the largest c, by the same bisection, for which xᵀPx is a valid terminal value. At deterministic points on four radial shells inside Ω_c (xᵀPx = s²c for s ∈ {0.25, 0.5, 0.75, 1}, 64 equally spaced angles each), the exact closed-loop cost must be within 5% of x₀ᵀPx₀.
    - **Closed-loop cost:** the running cost under u = −Kx until first entry into the inner ellipsoid xᵀPx ≤ 1e-4·c, plus xᵀPx there for the negligible tail.
    - **Failure to enter:** a point that does not enter the inner ellipsoid within 2,000 steps fails the test.
  - **No positive c:** if none passes, that β fails, and so does the run. It is never dropped silently.
  - **Closed-loop map:** F_cl is the exact map under u = −Kx.
  - **The 8-point LQR reference behavior below is unchanged.** It stays an independent check, with its own points and its looser 10% tolerance.

## The DP

**Grids.** The grid is periodic in e, with N_e nodes spaced 2π/N_e and one node at e = 0. In ω it spans [−ω_max, ω_max] with N_ω nodes, ω_max = 12. Actions are N_u uniform points on [−3, 3], including 0 and ±3.

| Level | N_e × N_ω | N_u |
|---|---|---|
| L1 | 200 × 241 | 31 |
| L2 | 400 × 481 | 61 |
| L3 | 800 × 961 | 121 |

**Running order.** L1 and L2 always run. L3 runs only if the convergence rule fails between L1 and L2.

**Problem.** A stationary first-exit shortest path:
- V(x) = xᵀPx on Ω_c;
- elsewhere, V(x) = min_u [c(x, u) + V(F(x, u))];
- the transition is the deterministic exact map, x⁺ = F(x, u). Interpolation only approximates V(x⁺); its weights are numerical coefficients, never transition probabilities (version 6);
- V(x⁺) is bilinear on the grid, periodic in e, when |ω⁺| ≤ ω_max;
- a transition with |ω⁺| > ω_max is not clamped. Its V(x⁺) is the **boundary continuation**: the corrected demonstrator is rolled out through the exact map from the exact continuous successor x⁺ for up to 2,000 steps, and its first-exit cost (running cost plus terminal value at entry) is used. That is a finite admissible upper bound;
- a continuation that does not reach Ω_c marks that transition **unsupported**, with V = +∞. Unsupported transitions are counted per β and level, with the smallest |ω| of a node that has one. The domain is not expanded within a run.

**Transitions** are computed once per (level, ω_max) through the exact map and shared by every β.

**Value iteration** runs from two initializations:
- **Lower:** V₀ = 0 off Ω_c. Its iterates must be monotone non-decreasing.
- **Upper:** V₀ = the corrected demonstrator's own cost-to-go. It is evaluated by simulating `tools/pendulum_eval/expert.py:act` (primary variant) through the exact map for up to 2,000 steps. Where the demonstrator alone does not reach Ω_c, V₀ is the best one-step lookahead: the minimum over the action grid of c(x, u) plus the demonstrator's continuation from the exact successor, or the terminal value there (version 6). That is still a feasible policy's cost. V₀ = +∞ only where no first action recovers.

Each run stops when the largest change over finite nodes is ≤ 1e-10, with a cap of 20,000 iterations.

**Behavioral properness** (version 6). Properness is established by rollouts through the exact deterministic map, not by reading the interpolation as a Markov chain. From every grid node in R, for every β > 0, each level requires all of:
- **Arrival:** the greedy policy enters Ω_c within 500 steps.
- **No domain exit and a margin:** along every reported trajectory, |ω| ≤ 10, that is ω_max − 2 rad/s. The reported trajectories are the greedy rollouts under both the ω_max = 12 and the ω_max = 14 fields, and the corrected demonstrator's rollouts (up to 2,000 steps), the policy behind the upper initialization and the continuation. The physical bound |ω| ≲ 9.03 from R is not assumed; it is checked on these trajectories.
- **No nonterminal cycle:** with a deterministic map and a stationary policy, a trajectory that revisits a state repeats forever and never enters Ω_c, so arrival within the horizon excludes it. For a trajectory that does not arrive, the least physical-metric distance from its final state to its earlier states is reported, to separate a cycle from slow progress.
- **Agreement between ω_max = 12 and 14,** at the same spacing:
  - **values:** |ΔV| ≤ 1e-3 · max(1, V), with the same finite/unreachable status, on R;
  - **action sets:** on R outside Ω_c, each field's greedy action is within 1e-3 · max(1, V) of the other field's minimum Q;
  - **trajectories:** the ω_max = 14 greedy policy also arrives and keeps the margin from every node of R, and the two rollout costs agree within 1e-2 · max(1, J).

**Strict hierarchy, and what a failure produces.**
1. **L1, then L2.** Each level runs its boundary continuations, its value iterations, the behavioral properness gates and the checks below. Any failed check for a β > 0 ends the run at that level; L2 runs only if L1 passes.
2. **Comparisons:** only on converged, finite fields, which is what the convergence rule below compares.
3. **L3:** runs only when the L1–L2 rule fails. It never rescues a level that stopped or failed its checks.

Every end of a run writes `oracle.json`. A stopped run records `"status": "stopped"`, the reason, every diagnostic computed so far and the reference behaviors. It writes no fields, makes no β selection and publishes no figure. The figure script refuses any run that did not pass every check.

β = 0 is descriptive: its diagnostics are reported, and it never stops the run.

**Reported region R:** every e with |ω| ≤ 6.5. It contains every state of the dataset, whose largest |ω| is 6.26.

**Convergence argument.** The operator is not a contraction, so a level is valid only if all of the following hold:
1. **Both runs stop** before the cap. The lower run is monotone, and the Bellman residual of the result is ≤ 1e-9 on R.
2. **Initialization gap:** the two runs' values differ by at most 1e-6 on R.
3. **Behavioral properness,** as above: arrival, the margin, and no cycle. The greedy policy is the argmin over the action grid of c(x, u) + V(x⁺), with V(x⁺) as in "Problem".
4. **No boundary dependence:** the level rerun with ω_max = 14, at the same spacing, agrees in values, action sets and trajectories, as above.

**Convergence rule between levels** (L_k against L_{k+1}). Compare at L_k's nodes in R outside Ω_c, interpolating L_{k+1}. All three must hold:
- **Named starts:** at the 8 named starts, |ΔV| ≤ 0.02·V + 0.01.
- **All nodes:** the 95th percentile of |ΔV| / (V + 0.01) is ≤ 0.05.
- **Greedy actions:** |Δu| ≤ 0.3 at ≥ 90% of nodes, excluding nodes that are tied at either level and nodes within 2 cells of a switching boundary.

If L1–L2 passes, L2 is the oracle. If not, L3 runs and L2–L3 is tested by the same rule; on a pass L3 is the oracle. If L2–L3 also fails, the oracle is "not converged", it is not used, and that result is reported.

**Ties and switching boundaries:**
- **Optimal-action set:** {u : Q(x, u) ≤ min_u Q + ε_tie}.
- **ε_tie:** the 95th percentile of |ΔV| on R between the oracle level and the next coarser level, floored at 1e-6. This is the measured discretization error.
- **Tie:** a node is tied when its set contains some u ≤ −0.3 and some u ≥ +0.3 and is not contiguous on the action grid (a clockwise/counterclockwise tie). Ties are recorded as sets, never averaged.
- **Switching boundary:** an edge between neighboring nodes whose greedy actions differ by ≥ 3.

**Q at continuous actions.** Q*(x, a) = c(x, a) + V(F(x, a)), computed with the exact map at the given continuous a, never snapped to the action grid. Task regret is Q*(x, a) − V*(x).

## Consistency checks (not proofs)

- **LQR:**
  - **Values:** at oracle nodes outside Ω_c with xᵀPx ≤ 2c, |V − xᵀPx| ≤ 0.05·xᵀPx + ε_tie.
  - **Actions:** |u_greedy + Kx| ≤ 0.3 at ≥ 90% of the non-tied nodes where |Kx| ≤ 3.
- **Symmetry:** max over R of |V(e, ω) − V(−e, −ω)| ≤ ε_tie.
- **Direct shooting, an upper-bound check.**
  - **Starts:** the 8 named starts.
  - **Parameterization:** 25 piecewise-constant segments of 20 steps, bounded to [−3, 3].
  - **Optimizer:** L-BFGS-B with a batched finite-difference gradient, from three initializations: the greedy DP actions averaged per segment, zeros, and seeded uniform (seed 1452).
  - **Cost:** the running cost until the first entry into Ω_c plus xᵀPx at entry, with a penalty of 1e3 if Ω_c is not entered within 500 steps.
  - **Lower test:** the best shooting cost J must be ≥ V − (ε_tie + 0.02·V). A lower J would mean the DP overestimates.
  - **Consistency gap:** J must be ≤ 1.10·V + 0.01.
  - The greedy DP policy's own continuous rollout cost is also reported.

**Named starts** (e, ω): S1 (π, 0), hanging at rest; S2 (π − 0.3, 0); S3 (π/2, 0); S4 (−π/2, 0); S5 (π, 2); S6 (2.0, −3.0); S7 (0.6, 0); S8 (0, 4).

## Reference behaviors and the scorer

**Scoring horizon:** 500 steps (10 s) for every reference total. Hanging down for the horizon costs 500·dt = 10, not the infinite first-exit cost.

**One canonical scorer** reports, per trajectory:
- running cost, split into its height part and its effort part;
- physical effort Σu²·dt and normalized effort mean (u/3)²;
- action variation Σ|u_{t+1} − u_t|;
- upright occupancy (within 8°), and final hold (the last 100 steps);
- time to Ω_c (the first entry, or ∞);
- swing count: the sign changes of ω before the first entry;
- direction: the sign of ω at the first entry.

**Expectations derived by hand,** each of which must hold:

| Behavior | Expectation |
|---|---|
| Hanging at rest, u = 0 | Height total = 10.0 to within 1e-9 (the state is an exact fixed point of the map). |
| Upright at rest, u = 0 | Height total ≤ 1e-4. The only motion is the float error in sin π, which grows by about e^{3.13 t}. |
| Constant u = +3 from hanging rest | I = ½ω² − 9.81 cos θ − uθ stays within 1e-2 of its initial value. The largest θ is within 1e-2 rad of θ* > 0 solving 9.81(1 − cos θ) = 3θ. |
| Constant u = −3 | The exact mirror (θ → −θ, ω → −ω) of the u = +3 trajectory, to within 1e-12. |
| LQR inside Ω_c | From 8 points on 0.9 × the boundary of Ω_c, xᵀPx decreases at every step and ends ≤ 1e-6·c. The accumulated cost plus the final xᵀPx is within 10% of x₀ᵀPx₀. |
| Corrected demonstrator, greedy DP policy | Scored from the dataset's 100 initial states (no expectation besides those below). |

## Selecting β

Selection uses the dataset's 100 initial states, which are not final-test starts. The selected objective is called the **demonstrator-compatible DP objective**. It is not the unique true task objective, and no learner result plays any role in choosing it.

1. **Admissibility.** The β's greedy policy captures (reaches Ω_c) and holds (final hold) from every start the demonstrator captures.
2. **Compatibility.** Each feature's median discrepancy from the demonstrator, over the captured starts, is within its scale:

   | Feature | Scale |
   |---|---|
   | Swing count | 1 swing |
   | Direction agreement | at least 90% of starts (a tied start counts as agreeing) |
   | Saturation fraction before capture | 0.1 |
   | Time to Ω_c | 1 s |
   | Physical effort | max(1, 10% of the demonstrator's median) |
   | Action variation | max(0.1, 10% of the demonstrator's median) |

3. **Selection.** Among compatible β, the least sum of scaled discrepancies; a tie goes to the smaller β. Direction counts as (1 − agreement)/0.1.
4. **No match.** "No compatible β" is a permitted outcome. Every β's fields and regrets are still reported, and the Stage B DP-relabel arm then needs a separate decision.

The time to Ω_c and the saturation fraction are measured against each β's own Ω_c, for the demonstrator as well.

## Named phase-space regions

These are defined from the oracle alone, at the selected β (or, under "no compatible β", at β = 1e-2, labeled as such). They are frozen here, before any candidate is scanned.

| Region | Definition |
|---|---|
| Ω_c | The LQR terminal set. |
| Capture neighborhood | Outside Ω_c; the greedy DP policy enters Ω_c within 25 steps (0.5 s). |
| Pumping | Otherwise, with H < 0. |
| Overshoot | Otherwise, with H ≥ 0. |

The recovery region needs the demonstrations, so it is declared with the correction scan (#167's Pendulum instance). It is the set of states outside the demonstration tube from which the greedy DP policy captures.

## State validity

A state is on the cylinder when |sin² + cos² − 1| ≤ 6.7e-7. That is 10 × the largest drift of the dataset's own states (6.74e-8), which comes from the integrator carrying sin and cos separately. A state outside it is not a physical Pendulum state. This bound is used by the correction scan, not by the oracle.

## Artifacts

- **Fields:** the oracle level's fields per β, as one release asset with a recorded sha256 (`.npz`). They are V, the optimal-action sets, the tie mask, the switching edges, the saturation map, Ω_c (P, K, c) and the region masks.
- **Evidence:** `docs/fixtures/evidence/pendulum-oracle/oracle.json` holds every check above with pass or fail, the reference scores, the β-selection table, the field hashes, and the code commit and dirty state.
- **Panel 1** ("task anatomy"), `docs/experiments/figures/pendulum-oracle-anatomy.svg`, on the axis contract below:
  - the cost-to-go background, sparse DP action arrows, and tie and switching boundaries;
  - energy contours, the saturation region, the target and Ω_c;
  - the demonstrations, and the demonstrator's action field as a separate panel.
- **Axis contract** for every later phase portrait: e ∈ [−π, π) horizontal and periodic, ω ∈ [−8, 8] vertical, one orientation, shared color scales, and the named starts marked.
- **Regeneration:** `check.py` has no in-place `--check` mode. The evidence is regenerated by a full rerun, about 2.3 hours on a 64-core host (the official run took 8,161 s), then compared deterministically:
  1. From commit `e7e9f95`, with numpy 2.4.6, scipy 1.16.3 and Python 3.11.17, run `uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_oracle/check.py --data DATASET.npz --out NEW_DIR`, where DATASET is the canonical continuous dataset (sha256 `59c332bc…cf43c`). It exits 1, stopping at L3, as the official run did.
  2. Run `python tools/pendulum_oracle/compare.py docs/fixtures/evidence/pendulum-oracle/oracle.json NEW_DIR/oracle.json`. It exits 0 only when every field is exactly equal except wall time and the run's code commit and dirty flag. The numerical environment is compared like any other field.
  - The oracle is deterministic: version 6.1 reproduced version 6's L1 and L2 results exactly.
  - An independent full rerun at `e7e9f95` (2026-11-01, same versions) passed `compare.py`: every substantive field matched exactly, with only wall time differing (8,027 s against 8,161 s).

## Running

- **Progress:** every parallel stage logs completed and total jobs, and every value iteration its iteration, last change and elapsed time, at least every 5 minutes (`PROGRESS_SECONDS`).
- **Stall threshold:** 30 minutes without a new log line, which is six progress intervals. The watcher also reports process exit and the `EXIT=` marker. A stall is diagnosed with the process kept; it is not restarted silently.
- **Checkpoints:** each value iteration's V and its iteration statistics are written atomically to `checkpoints/` in the output directory. `--resume` continues an incomplete directory (one without `oracle.json`) at the same commit; anything else is refused. Everything else is recomputed, which is deterministic.
- **Incomplete:** a run is complete only when `oracle.json` exists. A dead worker fails the run with an error instead of hanging it.

## Version history

**Version 2** changes only the ω domain: ω_max goes from 8 to 12, and the boundary check from ω_max = 10 to 14. The grid spacing, R and every tolerance are unchanged.

**Why.** Version 1's domain could not contain the motions that start in its own reported region. A state in R at |ω| = 6.5 near upright has H ≈ 21. Swinging down through the bottom, it reaches |ω| of about 9.0 even under full braking, since braking removes at most 3π of energy over half a turn. That exceeds ω_max = 8, so the greedy-reachability and boundary checks would have failed by construction, not by any property of the oracle. From R the bound is |ω| ≤ √(2(21.1 + 2g)) ≈ 9.03 without braking, which ω_max = 12 contains with margin.

**How it was found.** A development probe of the implementation: L1, β = 1e-2, lower value iteration only, run before any check reported here. It found 42 unreachable nodes in R, all at |ω| = 6.5 with H ≈ 19–21. The probe also showed that the implementation let a finite sentinel leak through interpolation. That was fixed so that +∞ propagates as declared. The probe's values are not reported or used.

**Version 6.1** (operational only; no definition, grid, tolerance, rule or check changes):
- **What changes:** value iteration updates nodes in blocks of 20,000 from the previous iterate and shares the bilinear weights across β, so its memory no longer grows with the number of concurrent jobs times the full N × N_u temporaries. Parallel stages fail on a dead worker instead of hanging. Progress logging, checkpoints and `--resume` are added (under "Running").
- **Equivalence evidence:** on the declared L1 grids (ω_max 12 and 14), for β 0, 1e-2 and 1, lower and upper value iteration, and block sizes 20,000 and 7,777, V, Q, the residual and every statistic are bit-identical to version 6's (`docs/fixtures/evidence/pendulum-oracle/vi-equivalence.log`). The rerun's L1 and L2 check lines must reproduce the aborted run's exactly.
- **Why:** the version 6 official run, `runs/20261101T143027Z-dp-oracle` on siddh from 644f5fb, hung at L3. Its 15 concurrent L3 value iterations needed about 16 GB each on a 125 GB host, the kernel killed 7 workers for memory (`oom_kill 7` in the session's cgroup), and the process pool waited for them indefinitely. The log stopped at 06:48 and the hang was found at 10:47 by the stall watcher.
- **What had been observed:** L1 and L2: every level check passed for every β, and the L1-L2 convergence rule failed (its pass flag only; the comparison's numbers were not logged), which triggered L3 as declared. L3: the boundary continuation report and the demonstrator's reach from R. No L3 value function, check, β selection, reference, figure or field was produced. The log and a process and memory snapshot are preserved beside an `ABORTED.md` note. That run is excluded, and the rerun under version 6.1 is the official one.

**Version 6** (boundary semantics and properness; before any value iteration of an official run):
- **What version 5 showed:** under version 5's own definition, almost every L1 node was improper (official run `runs/20261101T062610Z-dp-oracle`, `docs/fixtures/evidence/pendulum-oracle/oracle.json`; all verifications passed).
  - **The mechanism:** a safety cascade from the 238 nodes at |ω| ≥ 11.9 with no in-domain action, inward over 1,193 removal batches (`improper-cascade-L1.svg`).
  - **Why it is not a pendulum fact:** reading every positive interpolation weight as a possible transition turns grid interpolation into adversarial process noise. That is a property of the declared numerical abstraction, not of the pendulum. It does not show that the states in R are improper, and the interpolation-as-stochastic-process reading is inappropriate for this deterministic oracle.
- **What changes:**
  - transitions stay the deterministic exact map, and interpolation only approximates V(x⁺);
  - exits are valued by the demonstrator's boundary continuation from the exact successor. They are not clamped, which would hide the modeling error and add artificial dissipation at the boundary;
  - the Markov-chain improper analysis, its stop rule and its certificate are removed. Properness is behavioral (arrival, margin, no cycle) and is checked for agreement between ω_max = 12 and 14.
- **Upper initialization where the demonstrator stalls:** the best one-step lookahead followed by the demonstrator, as above.
  - **Why:** the corrected demonstrator never leaves exact hanging rest, where energy shaping applies u = 0. So version 5's +∞ initialization there was absorbing: every action's interpolation includes that node, and +∞ is a fixed point of the operator. That made the initialization gap infinite by construction.
  - **How it was found:** in the version 6 smoke test on the coarse smoke grids, where 1 node of R stayed unreached. No declared level had been run under version 6.
- **Unchanged:** grids, levels, tolerances, the terminal sets, the convergence rule, the consistency checks, the references and β selection.

**Version 5** (a performance-only amendment to the improper-state analysis; nothing in the definition changes):
- **What changes:** the implementation of the improper-state fixed point. The repeated full sweeps are replaced by an event-driven predecessor queue. Interpolation supports, boundary semantics, tolerances and the definition of improper are unchanged.
- **What is added:** an independent verification of each improper set, which stops the run if it fails, and a per-node removal certificate (both under "The DP").
- **Equivalence evidence:** on every smoke grid (40×49 to 160×193 nodes, 7–25 actions; ω_max 12 and 14; all five β), the new and version-3 implementations return identical improper, self-loop and no-live-action sets, and every verification passes.
- **Why:** the first official run, `runs/20261101T020624Z-dp-oracle` on siddh from 698262f, spent 3 h 19 min in the version-3 sweeps of its first L1 improper analysis, on one core. A local timing of the same L1 computation showed each outer round taking about 1.5 s and removing about 30 nodes. The run was stopped on the user's instruction. It had logged only the terminal sets, which are reported unchanged, and no improper count, value function, check, β selection or figure. Its log is preserved beside an `ABORTED.md` note. That run is excluded, and the run under version 5 is the official one.

**Version 4** (terminal-set rule; before the official run):
- **The rule it replaces:** in versions 1–3, c was the largest value in (0, c_sat] for which a seeded sample satisfied |Kx| ≤ 3 and strict decrease of xᵀPx under the exact closed loop.
- **Why it failed:** stability and invariance alone do not make xᵀPx a valid terminal value. At β = 1 that rule gave c = 0.061, below c_sat = 0.169. There, the declared 8-point LQR reference, which is unchanged, failed: the closed-loop cost plus the final xᵀPx ranged from 0.74 to 1.53 of x₀ᵀPx₀, against a 10% tolerance.
- **How it was found:** in the version 3 smoke test. The reference behaviors do not depend on the grid.
- **The new rule:** adds c_accuracy as above. Under it, β ≤ 0.1 is unchanged: c_input, c_decrease and c_accuracy coincide. At β = 1, c_accuracy is binding.

**Version 3** (safeguards before the official run; no tolerance changed):
- adds the exact improper-state analysis, its report, and its stop rule;
- makes the level hierarchy strict;
- requires a failure artifact for every stop;
- gates β selection and the figure on a fully passing run.

It was prompted by a smoke test of the implementation on deliberately coarse grids (40×49 to 160×193 nodes, 7–25 actions; not a declared level, and nothing from it is reported). There, the lower value iteration ran to its iteration cap because some nodes have only positive-cost self-loops.

## Not covered

The (ρ, λ) regime map beyond the dataset's ρ; the discontinuous variant; any learned dynamics or policy; and a continuous-time HJB comparison.
