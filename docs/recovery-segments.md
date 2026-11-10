# Recovery segments

This page is the normative mathematics of recovery segments (#171): what a
segment is, how CCIL constructs one, what it may claim, and how the recovery
length M is admitted. One-step CCIL is the case M = 1, and every statement
below reduces to the current one-step definitions there. The implementation
follows in later steps; [`tools/recovery_segments`](../tools/recovery_segments/check.py)
verifies this page numerically on an analytic system.

## Transitions and their composition

States z live in the representation's coordinates (state units for the
identity representation), actions a in action units. A dataset's recorded
transition (z_t, a_t, z_(t+1)) is one represented action interval: the action
a_t governs the interval from frame t to frame t + 1. The true one-interval
map and a learned one are

    T(z, a)  = z + f(z, a)        the system, unknown
    T̂(z, a) = z + f̂(z, a)        the transition model

f is the residual map. The composition over M consecutive intervals with
actions U = (u_0, ..., u_(M-1)) is

    F_M(z, U) = T(... T(T(z, u_0), u_1) ..., u_(M-1)),   and F̂_M likewise with T̂.

A learned discrete map needs no integrator. A continuous-time model may
integrate internally with step h_int; h_int is numerical provenance and is
never part of a segment's meaning (see [Invariants](#invariants)).

Norms are the scaled norms the certificates use: ‖v‖ means ‖S_z⁻¹ v‖₂ on
states and ‖S_a⁻¹ v‖₂ on actions, S_z = diag(certificate.state_scale), S_a =
diag(certificate.action_scale) ([certificates.md](certificates.md)). Quality
distances (d_source, d_demo, τ_demo) use the demonstration metric instead:
‖(x − y) / s‖₂, s the training states' per-coordinate standard deviation
([quality](contract.md#recovery-length)).

## The recovery segment

A demonstration window of length M is

    x_t --u_t--> x_(t+1) --> ... --u_(t+M-1)--> x_(t+M),

within one episode (t + M ≤ T − 1), each interval with its recorded action.
A recovery segment for that window is

    X̃ = (x̃_0, ..., x̃_M),   Ũ = (ũ_0, ..., ũ_(M-1)),   target = x_(t+M),

with x̃_M = x_(t+M). It is synthetic: a model-constructed path, never an
observed episode. Its canonical claim is about the whole segment: applying Ũ
from x̃_0 rejoins the demonstration at x_(t+M) after M intervals, to within
the error terms below.

## Construction

CCIL builds segments backward from the target, with a persistent action bias.

1. **Proposal.** For each window and each draw k, one offset δ ~ N(0, σ²I)
   (σ = generation.action_noise_std, in action units) is drawn from the
   window's stream, `transition_rng(seed, episode, t)`, in the same order and
   layout as one-step `noisy_action` draws. Every action of the segment
   carries it:

       ũ_i = u_(t+i) + δ,   i = 0, ..., M − 1,

   stored in the action dtype. The **realized** deviation Δa_i = ũ_i − u_(t+i)
   is computed on the stored values. With action bounds, a proposal is
   skipped unless every ũ_i lies inside them, so nothing is clipped and Δa_i
   equals δ up to storage rounding.

2. **Backward chain.** With x̃_M = x_(t+M), for i = M − 1 down to 0,

       x̃_i = P(x̃_(i+1), ũ_i),

   where P is the configured generator's one-step **construction operator**:
   - `noisy_action`: P(z⁺, a) = z⁺ − f̂(z⁺, a), a backward-Euler
     (fixed-point) proposal. It is not an inverse of T̂: T̂(P(z⁺, a), a) ≠ z⁺
     in general.
   - `noisy_action_preimage`: P(z⁺, a) is the Newton solution of
     z + f̂(z, a) = z⁺, started from the demonstration state x_(t+i). It
     attempts the inverse; a step that does not converge fails the segment.

3. **Step residuals.** Every step records

       s_i = ‖T̂(x̃_i, ũ_i) − x̃_(i+1)‖,

   the amount by which the model's own step from x̃_i misses the next segment
   state. For `noisy_action_preimage`, s_i is the solver residual (at most its
   tolerance when it converged). For `noisy_action`, s_i = ‖f̂(x̃_i, ũ_i) −
   f̂(x̃_(i+1), ũ_i)‖, first order in the step and not zero.

**What the model says about the segment.** The model's rollout from x̃_0
misses the target by at most the step residuals, propagated:

    ‖F̂_M(x̃_0, Ũ) − x_(t+M)‖ ≤ Σ_(i=0)^(M-1) L̂^(M-1-i) s_i,     L̂ ≥ Lip(T̂) over the segment,

by telescoping F̂ along the segment. CCIL claims nothing about the model
rollout beyond what the recorded s_i support; with `noisy_action` it is not
claimed to reach the target exactly.

**M = 1 is today's generator.** With M = 1 the chain is one step,
x̃_0 = P(x_(t+1), ũ_0) with ũ_0 = u_t + δ, which is exactly the one-step
label of `noisy_action` (z_G = z_(t+1) − f̂(z_(t+1), ã)) or
`noisy_action_preimage` (Newton from z_t), from the same draws in the same
order. The check verifies this byte for byte.

**Ties.** Many segments can reach one target: every δ gives one, and P need
not be unique. CCIL keeps each draw as its own segment and claims no
uniqueness or optimality.

## The composed `disturbed_action` bound

**Assumptions** (declared constants, so the bound is *conditional* on them):
- the true residual map is Lipschitz,

      ‖f(x, u) − f(y, v)‖ ≤ K_z ‖x − y‖ + K_a ‖u − v‖,

  for all pairs in a region R that contains the demonstration window and the
  true rollout from x̃_0 under Ũ. A global constant satisfies this; a local
  one must be argued to cover that rollout, not only the neighborhood of x_t.
  K_z and K_a are today's `certificate.true_state_lipschitz` and
  `true_action_lipschitz`. They bound f, the residual map, so the state factor
  is (1 + K_z). A constant L_T bounding the full map T would enter as L_T in
  place of (1 + K_z);
- each demonstration step's residual under the true map is bounded:
  ‖T(x_j, u_j) − x_(j+1)‖ ≤ ρ_j (`certificate.demonstration_residual_bound`,
  one value for every j, or a provided per-step residual).

**Bound.** Let y_0 = x̃_0 and y_(i+1) = T(y_i, ũ_i) be the true rollout, and
d_i = ‖y_i − x_(t+i)‖. Then

    y_(i+1) − x_(t+i+1) = (y_i − x_(t+i)) + [f(y_i, ũ_i) − f(x_(t+i), u_(t+i))]
                          + [T(x_(t+i), u_(t+i)) − x_(t+i+1)],

so d_(i+1) ≤ (1 + K_z) d_i + K_a ‖Δa_i‖ + ρ_(t+i), and unrolling,

    ‖F_M(x̃_0, Ũ) − x_(t+M)‖ = d_M
        ≤ (1 + K_z)^M d_0 + Σ_(i=0)^(M-1) (1 + K_z)^(M-1-i) (K_a ‖Δa_i‖ + ρ_(t+i)),

with d_0 = ‖x̃_0 − x_t‖ and Δa_i the realized deviations.

**M = 1.** The bound is (1 + K_z) ‖Δz‖ + K_a ‖Δa‖ + ρ_t, today's
`disturbed_action` total with target_offset 0.

**What it is.** A statement of where the true end state may land, conditional
on the constants. It does not use the model, so the step residuals s_i are
reported beside it and never added. It is at least (1 + K_z)^M d_0, so it
grows with both reach and M: a segment that starts further from the
demonstration, or runs longer, carries a larger bound by construction. Curate
compares it with `curate.tolerance` per segment.

## Admission of M

Diagnose evaluates each M on the ladder {1, 2, 4, 8, 16}, capped, on a fixed
probe of training windows. Every quantity is today's at M = 1:

- **yield**: accepted / produced segments.
- **displacement**: q(d_source) / τ_demo, d_source = distance of x̃_0 from x_t
  in the demonstration metric.
- **novelty**: q(d_demo) / τ_demo, d_demo = distance of x̃_0 from the nearest
  training state.
- **fit**: ε_M / median d_source, with ε_M the held-out quantile of
  ‖F̂_M(x_t, U_t) − x_(t+M)‖ over held-out windows of length M, from the model
  that never saw them. It is measured on demonstration states: necessary,
  never sufficient.
- **trust**: unchanged; it passes only with off-data evidence, a bound profile
  from declared or model-computed constants (for M > 1, the composed bound).

The admitted M is the smallest that passes every gate. If none passes, the
result is a valid empty synthetic view with the reason; weak labels are never
emitted to raise a row count. Longer is not assumed better: a longer segment
reaches further and compounds model error, which ε_M sees on the data.

**Pointwise novelty** is reported beside the gates and never gates: the
fraction of flattened states x̃_i (i < M) with d_demo ≥ τ_demo, by offset i.
Selection uses x̃_0; a segment with a novel start followed by M − 1 states that
rejoin the data is reported as such, never as M new corrective examples.

## Views

**Segment (canonical).** The immutable artifact holds X̃, Ũ, the target, the
s_i, the window's identity, and the evidence.

**Pointwise (default).** Each usable segment flattens to M rows
(x̃_i, ũ_i), i = 0, ..., M − 1. Every row carries its segment identity, its
offset i, the target, the timing, the evidence class, and a weight:

- a weight is confidence per row, and flattening copies the segment's weight
  to each of its M rows, so an M-step segment carries M rows of exposure;
- the supervision manifest reports the segment count, the row count, and the
  summed row weight;
- a comparison of supervision sources is exposure-matched: duplication of
  demonstration rows to the same total row weight is the control.

A row inherits only row-level action supervision: ũ_i at x̃_i is the action
of a model-consistent (up to s_i) path that rejoins the demonstration. The
segment-level claim, that the target is reached after M actions, is never
attached to one isolated row; using it requires the whole segment or an
additional completion argument.

## Clocks

M counts represented action intervals. The physical horizon is derived only
from timing in the data:
- constant rate (every interval dt_ctrl, within the timestamps' precision):
  H = M · dt_ctrl;
- variable rate: H for a segment is the sum of its window's recorded
  durations, t_(t+M) − t_t;
- no timestamps: M only, and no physical-time, resampling or deployment-rate
  claim.

## Invariants

1. Refining the internal integration step h_int of a continuous-time model
   changes neither M, nor the actions, nor any segment's meaning; segment
   states change only within the integration tolerance.
2. Saving and loading a model preserves its transitions, and so the segments.
3. Every pointwise row reconciles to exactly one segment and offset.
4. Reserializing a dataset without changing which action governs which
   interval does not change the supervision.
5. Changing the action period changes the system contract (T itself); it is
   not resampling.

## Artifacts and compatibility

At M = 1 no artifact changes: candidates stay `ccil.candidates` 2.x and the
`disturbed_action` certificate descriptor stays version "2". A file holding
M > 1 segments is `ccil.candidates` 3.0, so a 2.x reader refuses it by its
major version instead of reading x̃_0 and ũ_0 as a one-step label; its
certificate uses `disturbed_action` profile version "3".

## The analytic fixture

[`tools/recovery_segments/check.py`](../tools/recovery_segments/check.py)
records an inverted pendulum, θ'' = g sin θ − c θ' + u (g = 30, c = 0.1),
every Δt = 0.05 s, integrated with RK4 over 4 substeps as the true map T. A
stabilizing demonstrator, u = −2g θ − 2√g θ' plus N(0, 1) exploration, records
24 episodes of 60 frames from θ₀ ~ U(−1.2, 1.2); the last 6 are held out. The
learned map is the linear model fitted on the 18 training episodes. Segments
use `noisy_action` with 4 draws and σ = 1.0, on the diagnosis probe (256
training windows) and the default thresholds. These settings were chosen while
exploring this fixture so that it exhibits the regime #171 requires; they are
fixed in the tool and never come from a policy outcome. The report is
[`recovery-segments/report.json`](fixtures/evidence/recovery-segments/report.json).

**What it establishes.**
- **M = 1 is today's generators**, byte for byte, for `noisy_action` and
  `noisy_action_preimage`: states, actions, targets and failures.
- **The composed bound covers the exact miss** of every probed segment at
  every M, with the system's own global constants (K_z = 3.48, K_a = 0.116
  from the vector field's Lipschitz constant 30, by Grönwall) and per-step
  residuals ρ from the exact map. The model rollout misses the target by at
  most the propagated step residuals, as stated above.
- **The regime holds on each of three data seeds**: M = 1 is not admitted
  (it fails displacement or novelty), an intermediate M is, and M = 16 fails
  the fit gate (0.61–0.73 against 0.5). The exact miss confirms that M = 16 is
  untrustworthy: its median true end state lies 9–20 τ_demo from x_(t+M).
- **Integration refinement** (4 → 16 substeps) leaves the admitted M at 8 and
  moves segment states by at most 4e-6 and actions by at most 8e-6, within the
  declared 1e-4.

**What it does not establish, and the reader should weigh.**
- **The composed `disturbed_action` bound is vacuous beyond small M.** With
  honest constants, 1 + K_z = 4.48 per interval, and the bound amplifies the
  segment's own reach d_0 by (1 + K_z)^M. Its median is 0.27 at M = 1, 76 at
  M = 4, 7.5e4 at M = 8 and 6e10 at M = 16, against median exact misses of
  0.008, 0.037, 0.12 and 1.3. It is valid, but at the M this fixture admits it
  selects nothing under any useful tolerance. Admission is unaffected (trust
  requires the profile, not a small bound); curation by tolerance is.
- **Which intermediate M is admitted depends on the data**: 8, 4 and 2 on the
  three seeds, and the admitted M's fit ratios (0.40–0.48) sit close to the
  0.5 threshold.
- **The fit gate is necessary, never sufficient.** Even at the admitted M, the
  exact miss is large for a minority of segments: its q90 is 5.5, 2.4 and 2.4
  τ_demo on the three seeds. No data-only gate sees this.
- **Pointwise novelty is low.** At the admitted M, 6%, 8% and 10% of the
  flattened states lie beyond τ_demo: segments start novel and rejoin the data
  within a few intervals. An admitted M-step segment is one novel start plus
  mostly near-duplicate rows, and is reported as such.
