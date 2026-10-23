# Data invariants

This is the normative specification for #14. The implementation is
[`correct_il/invariants.py`](../../correct_il/invariants.py) (ingress) together
with `collect_labels` in [`correct_il/gen_aug_label.py`](../../correct_il/gen_aug_label.py)
(candidates).

## Dataset contract

A dataset is a non-empty list of episodes. Each episode is a dict with
`observations` of shape (T, state_dim) and `actions` of shape (T, act_dim) or
(T,), both floating-point and of equal length T ≥ 2. Transition t pairs
`observations[t]`, `actions[t]` and `observations[t+1]`, so the last action is
never used. Transitions are built per episode and never cross an episode
boundary.

## Ingress invariants

These are checked before any model is constructed, any split is made, or any
file is written. Both `train_dynamics_model.py` and `gen_aug_label.py` check
them, and label generation checks them **before** it loads the model.

| Invariant | Rule |
|---|---|
| `dataset_structure` | The dataset is a non-empty list. |
| `episode_structure` | Each episode is a dict with `observations` and `actions`. |
| `dtype` | Both arrays have a floating dtype. Integers are rejected, never cast. |
| `shape` | Observations are 2-D; actions are 1-D or 2-D. |
| `length_mismatch` | `len(observations) == len(actions)`. Legacy `build_dataset` would otherwise misalign the flattened arrays across episodes. |
| `too_short` | T ≥ 2. |
| `dimension_mismatch` | Every episode has episode 0's `(state_dim, act_dim)`. |
| `non_finite` | Every observation and every *used* action is finite. |
| `float32_overflow` | The values fit in float32, which training and generation compute in. |
| `action_out_of_bounds` | Used actions lie within `data.action_low` / `data.action_high`, **only when they are declared**. |

A violation raises `DataValidationError`. Each violation records the
invariant, the episode index, the frame index (or none, for whole-episode
violations) and a detail. Counts are exact; examples are capped at 20 per
invariant. Validation never truncates, broadcasts, clamps, coerces or mutates
its input.

## Candidate invariants

| Check | Outcome |
|---|---|
| Non-finite candidate (from the model or the inputs) | Rejected with `non_finite` (#11). |
| Accepted candidate whose action is outside the declared bounds | Rejected with `action_out_of_bounds`. Never clamped. |
| Generator output whose shape isn't `(n, state_dim)`, `(n, act_dim)` or `(n,)` | `ValueError`. A programming defect, not a rejected label. |

`label_decisions.npz` records each candidate's `episode` and `frame`, along
with `datapoint`, `accepted` and `reason`.

## Action bounds are opt-in

The shipped configs declare **no** bounds, so the behavior of every shipped
task is unchanged. Declaring bounds is a scientific decision, because the
shipped data shows two things:

- **Meta-World actions exceed the environment's [-1, 1] action space.** The
  observed ranges go up to 12.5 (button-press), because the scripted expert's
  output was saved unclipped. Declaring the true bounds would reject these
  datasets. #52 owns that decision for the Meta-World v3 port.
- **Pendulum actions saturate at ±3.** With bounds declared, the noisy-action
  labels drawn near saturation would be rejected instead of clamped.

## Pending

**Action chunks** (#22): chunks must not cross an episode boundary and must
keep their source frame. There is no chunking code on `main` yet; the
invariant will be checked when #22 ports it.

## Evidence

```bash
uv run --no-project --python 3.11 --with-requirements tools/invariants/requirements.txt \
  python tools/invariants/check.py --out <new-dir>
```

[`evidence/data-invariants/report.json`](evidence/data-invariants/report.json)
has 18 cases, each with its accepted and rejected examples and reasons:

- **A:** one malformed dataset per ingress invariant, each with the expected
  episode and frame and an unmodified input, plus a clean pass and a NaN
  confined to the unused last action.
- **B:** through `collect_labels`, action-bound rejections (9 of 36 noisy
  candidates, none clamped), NaN model output rejected with its episode and
  frame, and a malformed generator output raising an error.
- **C:** the real `train_dynamics_model.py` and `gen_aug_label.py` on Pendulum
  (CPU, legacy-harness stubs). A clean run writes labels that carry their
  episode and frame. A copy with one NaN action stops both scripts at ingress
  with `non_finite: episode 7, frame 123`, before writing anything.
