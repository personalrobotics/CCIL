# Pendulum expert reconstruction: frozen validation (#143, decision 1)

**Status: frozen before either comparison was run.** This document, the controller (`tools/pendulum_eval/expert.py`) and every tolerance below were committed before any reconstructed action or trace was compared with a stored demonstration. Decision 1 of `docs/experiments/pendulum-eval-v2-predeclaration.md` requires this ordering. A failure is reported as observed. No gain, tolerance or variant is changed after the comparisons, and nothing is fitted to the data. Any later revision is a new declaration with its own frozen commit, and this one stays on record.

## What was looked at before freezing

- **The paper's controller,** from CCIL (ICLR 2024), Appendix D:
  - LQR when |θ − π| < 0.1: π(s) = −20.11(θ − π) − 7.08 θ̇;
  - energy shaping otherwise: π(s) = −θ̇(½θ̇² − 9.81 cos θ − 9.81);
  - torque clamped to [−3, 3].
- **The printed gains are rounded.** No standard LQR formulation of the stated linearization and cost reproduces them (continuous: 19.67 and 6.35; discrete at 0.02 s: 19.06 to 19.08, and 6.15 to 6.16). They are therefore used as printed, with their rounding (±0.005) treated as specification uncertainty.
- **The stored action range only,** to establish the action-bound behavior that decision 1 requires:
  - actions are float32, in [−3, 3], with |a| = 3 exactly on 24.1% (continuous) and 24.0% (discontinuous) of steps, and never beyond;
  - the stored action is therefore the clamped action;
  - stored states are float64.
- **The generator's dynamics,** from the legacy environment (`correct_il/envs/pendulum_env.py`, in v6's `tools/m3_pendulum/env.py`): scipy `solve_ivp` RK45 at default tolerances over 0.02 s, sin θ and cos θ integrated as independent coordinates, and velocity negated after a step that lands within 1e-2 of π/6 or π/3.

No stored action was compared with a reconstructed one, and no reconstructed trace was computed.

## Declared variants

The paper leaves two details unstated. Each is enumerated here, and nothing else is varied.

| Detail | Primary | Alternative |
|---|---|---|
| Angle wrap | θ = atan2(sin, cos) in [0, 2π), the paper's normalized angle | (−π, π], where the LQR window covers only θ > π − 0.1 |
| cos θ in the energy term | cosine of the wrapped angle | the state's cosine coordinate |

**Selection.**
- All four combinations run the per-state check.
- If the primary passes, it is the expert. Otherwise the first passing variant in the order (wrap, cosine) of the table is the expert, and the report says that the per-state check identified it.
- If none passes, the reconstruction fails and the expert stays reported on its own starts only.

The closed-loop replay is then run once, with the selected variant only. It is the independent check.

## Check 1: per-state actions

At every stored state s_t of every demonstration (100 continuous and 500 discontinuous episodes, 500 steps each), the reconstructed clamped action a(s_t) is computed in float64 and compared with the stored float32 action. The tolerances are:

- **Energy branch:** |a_stored − a| ≤ 1e-6. This bounds float32 storage rounding (≤ 1.2e-7 at |a| ≤ 3) with room for operation order.
- **LQR branch:** |a_stored − a| ≤ 1e-6 + 0.005(|θ − π| + |θ̇|), which covers the gains' rounding. The clamp is 1-Lipschitz, so the same bound holds on clamped steps.
- **Switch-ambiguous states:** where | |θ − π| − 0.1 | < 1e-9, the state passes if either branch meets its own tolerance.

**Pass:** every state passes. There is no allowed fraction of failures.

**Report:**
- per branch: the maximum and 99.9th-percentile error, and the count of states;
- the count of switch-ambiguous states;
- every failing state, if any.

## Check 2: closed-loop replay

For every demonstration, the replay starts from its stored first state s_0 (float64) and runs 499 steps of the selected controller through v6's scalar generator step (`tools/m3_pendulum/env.step`, with the task's discontinuity), applying the float64 clamped action. This gives 500 state frames and 500 actions, the same as the stored episode.

**Conditioning envelope.** Each episode is also replayed under the specification's and the numerics' own uncertainty:
- four replays with the gains at the corners (20.11 ± 0.005, 7.08 ± 0.005);
- one replay with the applied action rounded to float32;
- one replay with the batched integrator (`tools/vector_env/pendulum.step`).

D(t) is the running maximum, over steps ≤ t and over those six replays, of the largest absolute difference from the main replay in any state coordinate or in the action.

**Comparison.** E(t) is the largest absolute difference between the stored episode and the main replay at step t, over the three state coordinates and the action. The action is compared as float32.

**Conditioned prefix.** These are the steps with D(t) ≤ 0.01. An episode whose prefix is shorter than 500 steps is reported as conditioning-limited, with its prefix length. Beyond that point, the stored trace and the replay can legitimately differ in a discrete event: a switch or a flip one step apart.

**Episode pass:** E(t) ≤ 1e-6 + 2·D(t) at every step of the conditioned prefix.

**Check pass:** both conditions hold:
- every episode passes;
- at least 90% of episodes are conditioned over all 500 steps. If fewer are, the check is inconclusive, not passed.

**Velocity flips** (discontinuous task):
- For each episode, the report gives the main replay's flip steps.
- For each flip inside the conditioned prefix, it checks that the stored θ̇ changes sign at the same step.
- It reports the count of episodes with flips, the total count of flips, and any disagreement. Inside the prefix the trace tolerance already enforces the flips, and this check makes them explicit.

## After both checks

- If the selected variant passes both checks, it is scored, with no change, on the Part B fresh starts (evaluation seeds 2201–2220) as a reference.
- If either check fails or is inconclusive, the expert is reported only on its own demonstration starts, labeled "different starts". The failure is reported as observed.

## Artifacts

- `tools/pendulum_eval/expert_validation.py` runs both checks from the SHA-256-verified conversion boundary. It writes `docs/fixtures/evidence/pendulum-eval-v2/expert-reconstruction.json`, and its `--check` option compares a regeneration against the committed report.
- The report's verdicts are recorded as outcomes, never as exit statuses.

## Outcome (appended after the comparison; nothing above was changed)

Check 1 failed for every declared variant on both tasks (`docs/fixtures/evidence/pendulum-eval-v2/expert-reconstruction.json`, commit 786d158), so check 2 was not run. The post hoc diagnosis is `tools/pendulum_eval/expert_diagnosis.py`. Revision 2 of `docs/experiments/pendulum-eval-v2-predeclaration.md` terminates the historical expert reconstruction. A corrected demonstrator is a separate study: `docs/experiments/pendulum-corrected-demonstrator-predeclaration.md`.
