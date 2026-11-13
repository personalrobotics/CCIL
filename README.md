# CCIL

[![PyPI](https://img.shields.io/pypi/v/prlccil.svg?v=1)](https://pypi.org/project/prlccil/)
[![Python](https://img.shields.io/pypi/pyversions/prlccil.svg?v=1)](https://pypi.org/project/prlccil/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2310.12972-b31b1b.svg)](https://arxiv.org/abs/2310.12972)

Behavior cloning learns from the states a demonstrator visited. At deployment, small
policy errors can move the system away from those states, where the policy has no
corrective supervision. Collecting corrections there normally requires another expert
or environment interaction.

[CCIL](https://arxiv.org/abs/2310.12972) uses local continuity to construct corrective
labels from sequential demonstrations instead. Given a demonstrated transition

```text
(state, action) -> next state
```

CCIL perturbs the action and uses a dynamics model to find a nearby state from which
the perturbed action returns to the same demonstrated next state. The resulting
state-action pair is ordinary supervision that can be consumed by a downstream
imitation learner.

The `ccil` package owns that supervision workflow. It fits or loads a dynamics model,
diagnoses whether the proposed construction has adequate reach and model support,
generates immutable candidates with quality evidence, and writes separate selections
of rows to use. It does not train or evaluate policies. Those experiments live in
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks).

CCIL 0.1 supports vector and keypoint demonstrations. Historical image CCIL is not
part of this release.

## How the workflow fits together

**Demonstrations → dynamics model → diagnosis → candidates and evidence →
selection → your trainer**

1. **Demonstrations and dynamics.** Episodes provide ordered observations and actions,
   with transition timing when it is available. CCIL can fit a linear or MLP dynamics
   model, or load a compatible checkpoint.
2. **Diagnosis.** Before emitting labels, CCIL measures construction yield, correction
   reach, demonstration spacing, and held-out model error. A failed diagnosis produces
   a valid empty result with an explanation rather than weak labels.
3. **Generation and evidence.** `ccil generate` writes candidates and the evidence
   available for judging their transition error. Candidates are immutable.
4. **Curation.** `ccil curate` chooses which immutable candidates should enter
   training. It can keep every eligible row, rows below an error tolerance, or a
   chosen lowest-error fraction. The command prints the path to an immutable
   `curate-<id>.json`, which records the policy and counts and points to a selection
   file containing a mask and per-row weights. It does not copy or modify the
   candidate labels, so several selections can be compared without regenerating
   them.
5. **Consumption.** `ccil inspect` on a curate result prints its selected and rejected
   counts and the exact Python needed to load it.
   `ccil.supervision.build_supervision` then returns arrays containing the training
   demonstrations, selected synthetic rows, and their weights. Alternatively,
   `ccil export lerobot` materializes the same selection as a LeRobot v3.0 dataset
   for an unchanged LeRobot trainer.

On Diagnose and Generate results, `ccil inspect` instead explains the resolved
configuration, provenance, evidence, and rejection reasons.

### The choices that matter

For a first Generate, point `data.path` at the demonstrations, choose the observation
modalities that form the state, then either choose a transition-model kind or provide
a compatible checkpoint. The main generation choices are:

- **`generation.method`: how to propose corrections.** The usual starting point is
  `noisy_action`: perturb a demonstrated action, then solve for a nearby state from
  which that action reaches the same demonstrated successor.
- **`generation.action_noise_std`: how far to explore in action space.** With
  `action_noise_scale: absolute`, it has the dataset's action units. With
  `training_std`, `1.0` means one training-set standard deviation in each action
  dimension. Larger values propose more ambitious corrections but ask the dynamics
  model to extrapolate further.
- **`generation.labels_per_transition`: how densely to explore.** This is a count of
  action proposals per demonstrated transition, not a quality threshold. More rows
  improve coverage and increase downstream synthetic exposure.
- **`generation.recovery_steps`: how long a correction may take to return.** Its unit
  is recorded action intervals, not simulator integration steps. If data arrive at
  10 Hz, one interval is 0.1 s. `auto` asks Diagnose to choose the shortest trusted
  length; release 0.1 currently implements only one-step labels.
- **`certificate.profile`: what evidence to compute.** A profile converts model
  error, solver error, and displacement into a per-row score or bound. Bound profiles
  require declared assumptions about the dynamics; their guarantees are conditional
  on those assumptions.

Curate interprets that evidence rather than changing the labels. A
**`curate.tolerance`** is a maximum transition-error score in *scaled state units*:
each state coordinate is divided by its declared scale before taking the norm. Thus
`0.1` means one tenth in that normalized state geometry, not automatically 0.1 m or
0.1 rad. `ccil inspect` prints the descriptor, units, and error quantiles to choose
from. A **`curate.quantile`** is instead a unitless data budget: `0.25` keeps the
lowest-error quarter of eligible rows, but makes no absolute quality claim. Selection
weights are dimensionless relative loss multipliers, not probabilities or confidence
scores.

## Start here

The core supports Python 3.10 and newer and depends only on NumPy and PyYAML.

```bash
python -m pip install prlccil
ccil --help
```

Then follow the [quickstart](https://github.com/personalrobotics/CCIL/blob/main/docs/quickstart.md), a complete CPU example from
demonstrations to curated rows and a small downstream pilot. Its commands are run from
the built wheel in a clean environment. For a robotics dataset and an unchanged
LeRobot trainer, use the [PushT walkthrough](https://github.com/personalrobotics/CCIL/blob/main/docs/pusht.md).

Optional extras provide the MLP dynamics model (`prlccil[torch]`) and the LeRobot
dataset bridge (`prlccil[lerobot]`). The installed Python package and command are
both named `ccil`.

## Correct labels are not necessarily useful labels

CCIL's evidence concerns whether a synthetic transition is trustworthy. It cannot
establish from demonstrations alone that a downstream learner will benefit. Adding
synthetic rows also changes training exposure, so policy studies should compare CCIL
against duplicated demonstrations with the same source distribution and exposure, not
only against behavior cloning.

Our matched studies found that duplication explained more than half of CCIL's measured
Pendulum gain, with the remaining difference unresolved. On PushT keypoints, no CCIL
advantage over duplication replicated across learner seeds. These outcomes are scoped
to the tested labels, learners, and synthetic doses. See the
[Pendulum](https://github.com/personalrobotics/il-supervision-benchmarks/blob/main/docs/studies/pendulum.md)
and [PushT](https://github.com/personalrobotics/il-supervision-benchmarks/blob/main/docs/studies/pusht.md)
accounts.

The practical distinction is important: CCIL generates candidate supervision and
records evidence about its transition error; the downstream pilot determines whether
that supervision is useful for a particular learner and task.

## Current scope

| Capability | Status |
| --- | --- |
| Vector and keypoint trajectory data | Supported |
| Linear and MLP dynamics models, fitted or loaded | Supported |
| Data-only diagnosis and one-step corrective labels | Supported |
| Quality evidence, calibration, and independent curation | Supported |
| LeRobot v3.0 tabular import and export | Supported |
| Multi-step recovery segments | Not implemented |
| Image, video, depth, and audio observations | Not implemented |
| Policy training and evaluation | Provided by downstream packages |

Legacy pickle datasets are never read by the package because unpickling can execute
code. The one-time, hash-checked conversion procedure is documented in
[legacy conversion](https://github.com/personalrobotics/CCIL/blob/main/docs/migration/legacy-conversion.md).

## Documentation

| Task | Document |
| --- | --- |
| Run the complete workflow | [Quickstart](https://github.com/personalrobotics/CCIL/blob/main/docs/quickstart.md) |
| Use LeRobot data and a LeRobot trainer | [PushT](https://github.com/personalrobotics/CCIL/blob/main/docs/pusht.md) and [LeRobot datasets](https://github.com/personalrobotics/CCIL/blob/main/docs/lerobot.md) |
| Understand commands, artifacts, and failure behavior | [User contract](https://github.com/personalrobotics/CCIL/blob/main/docs/contract.md) |
| Choose configuration values | [Parameter reference](https://github.com/personalrobotics/CCIL/blob/main/docs/config/parameters.md) |
| Interpret quality evidence | [Certificates](https://github.com/personalrobotics/CCIL/blob/main/docs/certificates.md) |
| Consume selected rows | [Supervision](https://github.com/personalrobotics/CCIL/blob/main/docs/supervision.md) |
| Implement a model, generator, or representation | [Core interfaces](https://github.com/personalrobotics/CCIL/blob/main/docs/interfaces.md) |
| Review schemas, run identity, or security | [Schemas](https://github.com/personalrobotics/CCIL/blob/main/docs/schemas/reference.md), [runs](https://github.com/personalrobotics/CCIL/blob/main/docs/runs.md), and [security](https://github.com/personalrobotics/CCIL/blob/main/docs/security.md) |
