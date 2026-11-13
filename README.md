# CCIL

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

```text
demonstrations -> dynamics model -> diagnosis -> candidates + evidence -> selection -> your trainer
```

1. **Demonstrations and dynamics.** Episodes provide ordered observations and actions,
   with transition timing when it is available. CCIL can fit a linear or MLP dynamics
   model, or load a compatible checkpoint.
2. **Diagnosis.** Before emitting labels, CCIL measures construction yield, correction
   reach, demonstration spacing, and held-out model error. A failed diagnosis produces
   a valid empty result with an explanation rather than weak labels.
3. **Generation and evidence.** `ccil generate` writes candidates and the evidence
   available for judging their transition error. Candidates are immutable.
4. **Curation.** `ccil curate` writes a mask and weights over those candidates. Several
   selections can be compared without regenerating labels.
5. **Consumption.** `ccil.supervision.build_supervision` returns the training
   demonstrations plus selected labels. `ccil export lerobot` writes the same
   supervision as a LeRobot v3.0 dataset.

`ccil inspect` explains any result, including its resolved configuration, provenance,
row counts, evidence, rejections, and the code needed to load selected rows.

## Start here

The core supports Python 3.10 and newer and depends only on NumPy and PyYAML.

```bash
git clone https://github.com/personalrobotics/CCIL.git
cd CCIL
uv sync
uv run ccil --help
```

Then follow the [quickstart](docs/quickstart.md), a complete CPU example from
demonstrations to curated rows and a small downstream pilot. Its commands are run from
the built wheel in a clean environment. For a robotics dataset and an unchanged
LeRobot trainer, use the [PushT walkthrough](docs/pusht.md).

Optional extras provide the MLP dynamics model (`ccil[torch]`) and the LeRobot dataset
bridge (`ccil[lerobot]`).

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
[legacy conversion](docs/migration/legacy-conversion.md).

## Documentation

| Task | Document |
| --- | --- |
| Run the complete workflow | [Quickstart](docs/quickstart.md) |
| Use LeRobot data and a LeRobot trainer | [PushT](docs/pusht.md) and [LeRobot datasets](docs/lerobot.md) |
| Understand commands, artifacts, and failure behavior | [User contract](docs/contract.md) |
| Choose configuration values | [Parameter reference](docs/config/parameters.md) |
| Interpret quality evidence | [Certificates](docs/certificates.md) |
| Consume selected rows | [Supervision](docs/supervision.md) |
| Implement a model, generator, or representation | [Core interfaces](docs/interfaces.md) |
| Review schemas, run identity, or security | [Schemas](docs/schemas/reference.md), [runs](docs/runs.md), and [security](docs/security.md) |
