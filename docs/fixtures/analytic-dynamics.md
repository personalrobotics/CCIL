# Analytic dynamics fixture

This is a behavior-level oracle for corrective-label generation (#10). It
uses a linear system whose correct labels are known in closed form, so a
generator's output can be judged without trusting any implementation.
[`tools/analytic_fixture/system.py`](../../tools/analytic_fixture/system.py)
holds the system, [`oracle.py`](../../tools/analytic_fixture/oracle.py) the
invariants, and [`config.json`](../../tools/analytic_fixture/config.json) the
fixture's data and methods; the tools under `tools/` check ccil against them.
The runs that showed the legacy implementation failing it, and passing after
#11, are at `surface-archive-v1`.

## System

The state is s = (x, v), the action a is a scalar, and h = 0.1:

    s' = s + f(s, a),   f(s, a) = A s + B a,   A = [[0, h], [0, 0]],   B = [0, h]^T

Generators receive the **exact** f through the legacy `model.f(s, a)`
interface. Model error is therefore zero, and every discrepancy is caused by
generation or acceptance logic.

## Closed-form labels

Each label claims that applying action a_G at state s_G reaches a target state.

**Backward Euler** (`backward_euler`, `backward_euler_fast`). The target is s*.

    s_G = s* − f(s*, a*) = (x* − h v*,  v* − h a*)
    distance ‖s_G − s*‖ = h·√(v*² + a*²)

Its true one-step error is s_G + f(s_G, a*) − s* = −A f(s*, a*) = (−h² a*, 0),
so the residual is h²·|a*|. This is the method's inherent approximation error,
not a defect.

**Exact preimage** (legacy `forward_euler`, which solves s_G + f(s_G, a*) = s*).
The target is s*.

    s_G = (I + A)⁻¹ (s* − B a*) = (x* − h v* + h² a*,  v* − h a*)
    residual = 0

**Noisy action** (`noisy_action`), with ã = a* + δ and δ ~ N(0, delta²). The
target is s*ₜ₊₁ = s* + f(s*, a*).

    s_G = s*ₜ₊₁ − f(s*ₜ₊₁, ã) = (x* − h² a*,  v* − h δ)
    distance = h·√(h² a*² + δ²)
    residual = h²·|ã|

The oracle evaluates the δ values each implementation actually drew. It
recovers them from the returned a_G, so implementations may use different
random number generators.

## Validity rule and rejection classes

The validity rule is declared per method in the config. It is not inferred
from any implementation.

| Class | Rule |
|---|---|
| `non_finite` | Any component of s* or a* is not finite, or of a_G for noisy action. The candidate must be rejected. |
| `out_of_support` | Not ‖s_G − s*‖ < ε, where ε is the `aug.epsilon` passed to the method. This is the rule legacy `load_demo_for_policy` applied when loading labels for training, and it is skipped when `aug.epsilon` is unset. The fixture sets ε = 0.01 for backward-Euler-fast and 0.005 for noisy action, and leaves it unset for backward Euler and exact preimage, to cover both cases. The candidate must be rejected. |
| `boundary` | Within 5% of ε. Either decision is allowed, which avoids judging float32 rounding at the threshold. |
| `accepted` | Everything else must be accepted. |

The oracle judges the **generation stage**: what `aug_data.pkl` contains. Legacy generation accepted every candidate and applied the support rule only later, at policy-load time. The 0.01 in legacy `gen_backward_euler_fast` was a failure *counter*, not its validity rule; the code reads `count_success = 1 # - count_fail`.

## Data

The config defines three episodes that are rolled out exactly:
- `slow`: small v and a. Most backward-Euler-fast labels fall within ε, and one (a = 0.12) does not.
- `fast`: v ≈ 0.5. Every backward-Euler-fast label falls outside ε.
- `corrupt`: action 1 is replaced by NaN *after* rollout, so the states stay consistent.

The reference run yields these expected classes:

| Method | accepted | out_of_support | non_finite | boundary |
|---|---|---|---|---|
| backward_euler_fast | 6 | 5 | 1 | 0 |
| backward_euler | 11 | 0 | 1 | 0 |
| forward_euler | 11 | 0 | 1 | 0 |
| noisy_action (4 per transition) | 26 | 16 | 4 | 2 |

The noisy-action counts depend on the δ values drawn. These are the reference
implementation's draws; each run records its own.

## Invariants

| ID | Invariant |
|---|---|
| I1 | Every accepted label is finite. |
| I2 | Every accepted label of a method that declares ε lies within ε(1 + 5%) of s*. |
| I3 | Accounting reconciles, per transition: attempted = expected count = number of candidate records = accepted + rejected + generation_failed, and the reported accepted count equals the accepted records. Every attempted candidate, rejected ones included, needs a record. |
| I4 | Every finite candidate matches its closed-form s_G and target within 1e-5, and its true residual matches the predicted residual. A `generation_failed` candidate is exempt, because it holds the solver's last iterate rather than a label. |
| I5 | Each accept or reject decision matches the expected class, and each rejected candidate's reason equals that class. Boundary candidates are exempt. |
| I6 | Transitions equal the exact rollout, in order. No transition crosses an episode boundary, and there are Σ(Tᵢ − 1) of them. |
| I7 | Running the implementation twice gives byte-identical transitions and candidates. |
