# Recovery segments

This page is the normative mathematics of recovery segments (#171, Stage 1):
- what a segment is;
- the recovery problem it solves, independent of any solver;
- which errors it records, and how;
- the artifact and views that carry it.

One-step CCIL is exactly the case M = 1, and every definition below reduces to
the current one-step quantities there.
[`tools/recovery_segments`](../tools/recovery_segments/check.py) verifies that
reduction on the artifacts `ccil generate` writes.

**Status: specified, not implemented.** CCIL ships M = 1, and its ladder is
{1} ([contract](contract.md#recovery-length)). This page adds no product code,
config key or schema module. `ccil.segments` below is a specification, which
the evidence tool encodes for reference only. Implementation waits on #171's
exact-dynamics feasibility gate (Stage 2) and learned-dynamics trust gate
(Stage 3). If either gate is negative, the work ends there.

**Notation.** M is the number of recorded action intervals in a segment. H is
physical time, H = M · dt_ctrl ([Clocks](contract.md#recovery-length)).
#171's issue text writes H for M.

## Transitions and their composition

States x lie in the representation's coordinates, and actions u lie in action
units. A recorded transition (x_t, u_t, x_(t+1)) is one action interval. The
learned transition and its M-fold composition are

    F̂(x, u) = x + f̂(x, u)
    F̂_M(x, U) = F̂(… F̂(F̂(x, u_0), u_1) …, u_(M-1)),   U = (u_0, …, u_(M-1)).

F denotes the true system, which is unknown. A continuous-time model may
integrate internally with step h_int. That step is numerical provenance and
never part of a segment's meaning.

## The recovery segment

A segment is a **single-shooting model rollout** together with a demonstrated
target:

    ξ = (x_0, u_0, x_1, …, u_(M-1), x_M),   x_(k+1) = F̂(x_k, u_k),
    target y = the demonstration state at (episode e, frame τ).

- The stored states *are* the rollout.
- The target y is stored separately; it is not x_M.
- The segment's claim concerns the whole segment: applying U from x_0 reaches
  y, to within the terminal target residual and the trust evidence below.
- A segment is synthetic. It is never an observed episode.

## The recovery problem

The problem is stated independently of any solver. It declares:

| Element | Meaning |
|---|---|
| Start | Synthetic (solved for), or fixed and observed (a rollout's identity and step) |
| Targets | The admissible demonstrated states y: one given state, or a declared set |
| Ladder | A finite set of M values |
| Constraints | State and action bounds, and the region where the model is declared trustworthy |
| Model | F̂, with its identity and its smoothness and access contract: closed form, differentiable, or query-only |
| Terminal tolerance | ε_y on ‖x_M − y‖ |
| Trust condition | The rule that decides whether the segment's states and actions are where F̂ is credible ([below](#trust)) |
| Objective | How to choose among feasible segments, for example ‖x_M − y‖ + λ_u · effort(U) + λ_M · M |

**Solvers.** Any solver may be used: inverse construction, single or multiple
shooting, iLQR, CEM, MPPI, dynamic programming, or another method. It must be
chosen from the model's smoothness and access contract **before** any
downstream outcome is read. The solver is part of the segment's identity, and
its own convergence measure is recorded separately from the residuals below.

**Two modes, one artifact.**
- **Backward (data-only, default).** Fix the target, propose U, and solve for
  x_0. Then roll forward from x_0 and record the rollout. No environment,
  reward, expert or learner is involved.
- **Forward (interactive).** Fix an observed start x_0 from a rollout, and
  search jointly over U, the target y and, optionally, M.

The mode and the start's provenance are fields of the same artifact, and
nothing else differs.

**Horizon selection fails closed.** Choose the **smallest** M on the declared
ladder for which reach, validity (constraints and terminal tolerance) and
trust all pass. If none passes, the result is empty and names the failed
conditions. Trust is never relaxed to admit a longer segment, and no policy
return or downstream outcome enters the choice
([contract](contract.md#recovery-length)).

## Errors, kept apart

Each segment records four distinct quantities. They are never summed into one
figure without saying so.

| Quantity | Definition | Where it comes from |
|---|---|---|
| Per-step dynamics defect | e_k = ‖F̂(x_k, u_k) − x_(k+1)‖ | Zero by construction. Only evaluation and the stored dtype's rounding make it nonzero |
| Terminal target residual | r = ‖F̂_M(x_0, U) − y‖ | The rollout's miss of the demonstrated target, under the model |
| Solver convergence residual | The solver's own stopping measure, against its own tolerance | Absent for an explicit construction |
| Model-versus-reference error | How far F̂ is from F on the segment | On data: anchor_model_error and the held-out M-step error ε_M. Off data: certificate constants. Where reference dynamics exist: replay of (x_0, U) through F |

Numerical provenance, such as floating-point bounds (`model_evaluation`,
`delta_error_bound`) and h_int, is recorded as provenance, separate from all
four.

### Trust

The true miss of the target splits into the terminal target residual and the
divergence between the true rollout and the model rollout:

    ‖F_M(x_0, U) − y‖ ≤ ‖F_M(x_0, U) − F̂_M(x_0, U)‖ + r.

The divergence term needs off-data evidence; the data alone cannot give it.
The composed `continuity` form bounds it without amplifying the segment's own
displacement. It applies on a declared convex region R × A, in certificate
norms ([certificates.md](certificates.md#profiles)), and assumes:
- ‖F(x, u) − F(x′, u)‖ ≤ L_F ‖x − x′‖;
- Δ = F − F̂ satisfies ‖Δ(x, u) − Δ(x′, u′)‖ ≤ K_Δz ‖x − x′‖ + K_Δa ‖u − u′‖;
- the demonstration residual is bounded by ρ_j.

Let z_k be the true rollout from x_0 under U, and d_k = ‖z_k − x_k‖. Then
d_0 = 0 and d_(k+1) ≤ L_F d_k + b_k + e_k, where

    b_k = ‖x_(j+1) − F̂(x_j, u_j)‖ + ρ_j + K_Δz ‖x_k − x_j‖ + K_Δa ‖u_k − u_j‖

is anchored at a demonstration transition j near step k. Unrolling,

    d_M ≤ D_M = Σ_(k=0)^(M-1) L_F^(M-1-k) (b_k + e_k).

A segment is **supported** only if every state is in R, every action is in A,
and the ball of radius D_k around x_k stays in R. An unsupported segment gets
no bound. At M = 1, with e_0 = 0, the bound D_1 + r is today's `continuity`
total: the residual, anchor model error, ρ_t, and the two discrepancy terms.

The older composed `disturbed_action` form amplifies the segment's own reach
d_0 by (1 + K_z)^M. PR #194 found it valid but vacuous beyond small M. It is
not a trust rule for M > 1.

Evidence classes stay those of [certificates.md](certificates.md): proven
constants certify, declared constants give a conditional bound, and held-out
errors are empirical scores.

## M = 1 is one-step CCIL, exactly

Every current generator writes a `ccil.candidates` row (z_G, a_G, target).
That row *is* an M = 1 backward segment:

    x_0 = z_G,   u_0 = a_G,   x_1 = F̂(z_G, a_G),   y = target,   τ = t + offset.

| Generator | Profile | y | Target offset | Solver residual |
|---|---|---|---|---|
| `backward_euler` | backtrack | x_t | 0 | absent: explicit z_G = x_t − f̂(x_t, u_t) |
| `forward_euler` (`exact_preimage`) | backtrack | x_t | 0 | `generator.solver_residual` (Newton) |
| `noisy_action` | disturbed_action | x_(t+1) | 1 | absent: explicit z_G = x_(t+1) − f̂(x_(t+1), ã) |
| `noisy_action_fe` (`noisy_action_preimage`) | disturbed_action | x_(t+1) | 1 | `generator.solver_residual` (Newton) |

At M = 1:
- **Terminal target residual.** r is the existing `residual` measurement,
  ‖z_G + f̂(z_G, a_G) − target‖.
- **Explicit generators.** For backward Euler and `noisy_action`, r is a
  construction error, not a solver tolerance.
- **The certificates' term.** Their `solver_error` term is the same r, in
  scaled state ([certificates.md](certificates.md)), whatever the generator.
  Its name predates this separation. The additive total
  `solver_error + model_error` is the M = 1 case of D_1 + r.
- **The Newton generators' own measure.** Only `generator.solver_residual`
  is a solver convergence residual in the sense above.

**Compatibility is literal.** Requesting M = 1 dispatches to the existing
generator. Its `ccil.candidates` 2.x file is byte-for-byte what it is today,
and it never gains horizon, segment or weight fields. Adding any of them would
change its bytes and its identity. A segment record is therefore a separate,
derived artifact that names the candidates file by SHA-256.

## The `ccil.segments` 1.0 artifact (specified)

The record is immutable. It references its inputs by SHA-256 and is never
rewritten.

**Per segment (n_seg rows):**

| Field | Meaning |
|---|---|
| `segment_id` | Position in the record |
| `M` | Recorded action intervals |
| `row_offset` | First action row; the M_i + 1 states of segment i start at `row_offset[i] + i` |
| `start_source` | Synthetic, or the observed rollout's identity and step (forward mode) |
| `candidate_row` | Row in the source candidates file, when derived from one |
| `source_episode_id`, `source_frame` | The demonstration transition the segment was built from |
| `target_episode_id`, `target_frame`, `target` | y and where it lies; the frame the segment reaches |
| `terminal_residual` | r |
| `solver_residual` | The solver's own measure, only when the solver reports one |
| `accepted`, `reasons` | Admission, with every rejection reason |
| `W` | Total training weight of the segment |
| `trust` | Profile, evidence class, D_M or the empirical score, support flag, and declared assumptions |

**Per action row (Σ M_i rows):** u_k and e_k. **Per state row (Σ (M_i + 1)
rows):** x_k.

**Meta:**
- `mode`;
- model, solver and configuration identities;
- `candidates_sha256`, when the record is derived from a candidates file;
- the run identity;
- the target-frame offset rule;
- dt_ctrl and H, when timing exists.

**W.** At M = 1, W is the candidate row's `weight`: 1 if accepted, 0 if
rejected, or whatever a future weighting assigns. It is inherited, never
reassigned. A new accepted segment defaults to W = 1. A rejected segment has
W = 0.

**Reconciliation.** Every row of every view maps to exactly one
(`segment_id`, offset), and the offsets of a segment are 0 … M − 1.

## Views

**Pointwise.** For each segment and each k < M, the row (x_k, u_k) is given
weight

    w_k = W / M,

so a segment contributes total weight W whatever its length. Longer segments
gain no influence from having more rows. Each row carries `segment_id` and
offset k, which [contract.md](contract.md) promises. A pointwise row never
carries the segment's terminal residual or reach claim, because those hold
only for the whole segment. At M = 1, w_0 = W, the candidate's weight.

**Chunk.** The chunk view is (x_0, U), padded to a requested size K:
- positions 0 … M − 1 are U;
- each position k ≥ M is the source episode's action at frame τ + k − M,
  continuing the demonstration from the target;
- positions past the episode's last frame are masked and zero.

Only positions 0 … M − 1 carry the segment's claim. At M = 1 this is exactly
the current `action_chunks` and `action_chunk_mask`
([`core.chunks`](../src/ccil/core/chunks.py)) for both profiles. A consumer
that does not declare chunk consumption is refused, as today.

## Invariants

1. Refining h_int changes neither M, nor U, nor a segment's meaning. States
   move only within the integration tolerance.
2. Saving and loading the model preserves its transitions, and so the segments.
3. Every view row reconciles to one segment and offset. Pointwise weights sum
   to W per segment.
4. The candidates file a record derives from is unchanged, and the record
   names it by SHA-256.
5. Changing dt_ctrl changes the system, and is not resampling.

## Evidence

Run:

    uv run --group evidence python tools/recovery_segments/check.py --out DIR

The tool runs `ccil generate` on the [quickstart](quickstart.md)
demonstrations for:
- all four one-step generators;
- a linear and an MLP transition model;
- action chunks of size 3.

The diagnosis's displacement and novelty gates are set to 0 so that every
generator yields rows; admission is not under test here. A reference encoder
derives the M = 1 segments from each candidates file and its model, and
checks:
- **Pointwise view.** It equals the candidates' observations, actions,
  accepted, reasons and weight bit for bit, with W inherited. The target
  equals the candidates' target.
- **Residuals.** r equals `residual` bit for bit. e_0 is zero for float64
  states, and within float32 rounding (3e-8) for the Newton generators,
  which store float32. The solver residual equals `generator.solver_residual`
  exactly where the generator reports one, and is absent otherwise.
- **Chunk view.** It equals `action_chunks` and `action_chunk_mask` bit for
  bit.
- **Candidates file.** It is unchanged by the derivation.
- **M = 2 and M = 4.** These are segments rolled forward from the same
  starts: a schema exercise, not a recovery claim. Weights sum to W per
  segment, rows reconcile one to one, and the chunk view returns U.
- **The reference reader** refuses:
  - an offset outside its segment;
  - weights that do not sum to W;
  - a terminal claim on a pointwise row;
  - a changed candidates file.

The report holds no paths or times, and two runs on one host are
byte-identical. The committed report is
[`report.json`](fixtures/evidence/recovery-segments/report.json).

## What comes next

Stages 2–5 are owned by #171 and are not restated here:
- Stage 2 is exact-dynamics feasibility on three predeclared systems. At
  least one of them must be independent of PushT and Pendulum.
- Stage 3 is learned-dynamics trust.
- Only after both pass does product implementation begin. That would add a
  `ccil.segments` schema module and one justified solver behind this
  contract.
- The downstream benchmark belongs to il-supervision-benchmarks.
