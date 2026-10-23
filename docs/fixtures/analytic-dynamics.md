# Analytic dynamics fixture

This is a behavior-level oracle for corrective-label generation (#10). It
uses a linear system whose correct labels are known in closed form, so a
generator's output can be judged without trusting any implementation, the
legacy one included. #11, #13, #14 and every later refactor use it as their
acceptance check.

## Reproduce

```bash
git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'   # the legacy run reads the archive tag
uv run --no-project --python 3.11 --with-requirements tools/analytic_fixture/requirements.txt \
  python tools/analytic_fixture/run.py --impl {legacy,reference} --out <new-dir>
uv run ... python tools/analytic_fixture/run.py --check <artifact-dir>   # re-evaluate saved candidates
```

The run uses the CPU only, one versioned config
([`tools/analytic_fixture/config.json`](../../tools/analytic_fixture/config.json)),
and a fixed seed. It exits 0 only if every invariant passes.

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
| `out_of_support` | ‖s_G − s*‖ > ε, for methods that declare ε. These are backward-Euler-fast with 0.01 and noisy action with 0.005, each the method's own legacy threshold. The candidate must be rejected. |
| `boundary` | Within 5% of ε. Either decision is allowed, which avoids judging float32 rounding at the threshold. |
| `accepted` | Everything else must be accepted. Backward Euler and exact preimage declare no ε at generation time. Legacy applies `aug.epsilon` later, in `load_demo_for_policy`, and that stage is out of scope here. |

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
| I4 | Every finite candidate matches its closed-form s_G and target within 1e-5, and its true residual matches the predicted residual. |
| I5 | Each accept or reject decision matches the expected class. Boundary candidates are exempt. |
| I6 | Transitions equal the exact rollout, in order. No transition crosses an episode boundary, and there are Σ(Tᵢ − 1) of them. |
| I7 | Running the implementation twice gives byte-identical transitions and candidates. |

**Pending #22:** weighting invariants. Weights must be finite, non-negative,
recorded per accepted label, and zero or absent for rejected labels. Legacy
`main` has no weighting, so this is not yet checked.

## Implementations

- **`legacy`** ([`legacy.py`](../../tools/analytic_fixture/legacy.py)) loads
  `gen_*`, `root_finder_s` and `build_dataset` **verbatim** from
  `archive/upstream-main-2026-10-22` with `ast`. It mirrors the legacy
  `main()` acceptance loop (`gen_aug_label.py:250-260`): every returned row
  is kept whenever the generator reports `count_success > 0`. Legacy counts
  map to `accepted = count_success` and `rejected = count_fail`.
- **`reference`** ([`reference.py`](../../tools/analytic_fixture/reference.py))
  is a positive control that applies the declared rule to the closed forms.
  It shows that the oracle passes on a correct implementation.

## Artifact

Each run directory contains:

| File | Contents |
|---|---|
| `config.resolved.json` | The config used. |
| `transitions.json` | The transitions the implementation built. |
| `candidates.json` | Every candidate, with the implementation's decision, the oracle's expected class, distance and residual. |
| `metrics.json` | The per-invariant, per-method verdicts, with examples. |
| `plot.png` | Per method, each candidate's distance with ε, coloured by decision and ringed when the decision contradicts the oracle. |
| `manifest.json` | SHA-256 of every file, the code and legacy commits, and package versions. |

Non-finite floats are written as JSON strings (`"nan"`). The artifact is
byte-identical across processes on the same platform and package versions.

## Pre-fix evidence (legacy)

[`evidence/legacy-pre-fix/`](evidence/legacy-pre-fix/) is the legacy run that
#10 requires to fail. It does:

| Method | Failing invariants | Cause |
|---|---|---|
| backward_euler_fast | I1, I2, I3, I5 | Hard-coded `count_success = 1`. The label is returned even when `distance > 0.01` counts it as a failure, and a NaN label is accepted. |
| noisy_action | I1, I2, I3, I5 | Hard-coded `count_success = num_labels`. All candidates are returned, including those beyond ε and the NaN ones. |
| forward_euler | I3 | On success, `count_fail` is the root finder's iteration count. The code itself carries `# TODO fix the count_fail`. |
| backward_euler | I3 | The NaN transition is counted as rejected but has no candidate record or reason. This is a provenance gap, not a wrongly accepted label. |

These pass for every method: I4 (the label mathematics is correct), I6 and
I7. The suspected stale-residual defect in `root_finder_s`, whose success
test reads the residual from before its final update, does **not** occur
with `max_iter = 50`. The labels converge, and I4 passes. The post-fix
evidence belongs to #11.

## Verification procedure

1. The `reference` run exits 0, and every invariant passes.
2. The `legacy` run exits 1 with exactly the pattern above.
3. Two runs of each implementation give identical `manifest.json` `sha256`
   maps.
4. Flipping any `accepted` flag in a reference `candidates.json` and running
   `--check` makes I3 and I5 fail.
