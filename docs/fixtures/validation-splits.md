# Validation splits and transition-model metrics

This covers how dynamics training reports model quality (#13). The
implementation is [`correct_il/validation.py`](../../correct_il/validation.py),
which `correct_il/train_dynamics_model.py` uses.

## Two models

| Model | Trained on | Used for |
|---|---|---|
| **Label model** (`dynamics.pt`) | every trajectory, exactly as before #13 | corrective-label generation |
| **Evaluation model** (not saved) | the train split only, with the label model's settings and seed | held-out metrics |

The held-out numbers validate the *fitting procedure*, not the exact label
model. The label model saw the held-out trajectories too. This keeps labels,
and the Pendulum parity check in #24, unchanged, at the cost of a second
dynamics fit.

## Split

- **Size.** `dynamics.validation_fraction` (default 0.2 in the shipped configs)
  of the trajectories, rounded, are held out.
- **Whole trajectories.** Trajectories are assigned whole, so no episode
  contributes frames to both splits.
- **Seed.** The split seed is the run's `seed`.
- **Persistence.** The split is written atomically to `split.json` in the
  dynamics output folder (format `ccil-split/1`: seed, fraction, trajectory
  count, sorted train and validation indices).
- **Reuse.** A rerun reuses `split.json` when the trajectory count, fraction
  and seed all match. Changing any of them writes a new split.
- **Fail fast.** If either split would be empty, training stops *before* any
  model is fitted, with a message that names the trajectory count, the
  fraction, and the fix (more data, another fraction, or 0 to skip).
  `f1.yml` (1 trajectory) sets 0.
- **No model selection.** Dynamics training does no model selection, so there
  is no separate selection split.

## Metrics (`validation.json`)

The metric is the one-step L2 error, ‖f(s, a) − s_target‖₂, in state units. A
backward model uses (s_{t+1}, a_t) → s_t. For each trajectory, the report
records the trajectory's dataset index, its transition count, the mean and
maximum error, and `relative_l2`: the mean error divided by the mean true
displacement ‖s_target − s‖₂, which is unitless.

| Aggregate | Definition |
|---|---|
| `pooled_mean_l2` | Mean over all transitions, weighted by transition count. |
| `macro_mean_l2` | Mean of the per-trajectory means. |
| `max_l2` | The largest single-transition error. |

`held_out` is computed only from validation trajectories. `train_split` is the
same evaluation model scored on its own training trajectories, for comparison.

`statistics.txt` now labels the label model's error as a training-set error
("not a held-out metric") and reports the held-out aggregate beside it.
`eval_prediction_error` now includes the final partial batch. Before, it
skipped trailing data and returned NaN whenever there were fewer transitions
than one batch.

## Reproduce

```bash
uv run --no-project --python 3.11 --with-requirements tools/validation_split/requirements.txt \
  python tools/validation_split/check.py --out <new-dir>
```

The check runs the real `train_dynamics_model.py` on `pendulum_cont` (CPU, 2
epochs) through the [legacy harness stubs](../../tools/legacy_harness/README.md).
It checks:
- that the split is a partition of whole trajectories;
- that held-out metrics come only from validation trajectories;
- aggregate consistency;
- the honest `statistics.txt` wording;
- split reuse, and a new split when the config changes;
- that the label model has **byte-identical weights** with and without validation;
- that a 1-trajectory dataset fails before training;
- partial-batch error accounting.

The evidence is [`evidence/validation-splits/report.json`](evidence/validation-splits/report.json),
which includes the split and the full validation report.
