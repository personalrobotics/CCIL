# CCIL

A policy trained by behavior cloning sees only the states its demonstrations visited.
Once it is deployed, its own small errors move it to nearby states that no
demonstration covers. Those errors then compound, because the training data say
nothing about how to return. Collecting corrective demonstrations at those states
helps, but it needs an expert, an environment, or both.

CCIL (Ke et al., [ICLR 2024](https://arxiv.org/abs/2310.12972)) approaches this from
the demonstrations themselves. Many systems are locally continuous: a small change of
state or action produces a correspondingly small change in the next state. A dynamics
model fitted to the demonstrations, and regularized to respect that continuity, can
then propose states near the demonstrated ones together with actions that would lead
back to them. Each such pair is a synthetic corrective label that an ordinary
imitation learner can train on.

The `ccil` package implements that workflow as a set of inspectable artifacts. It
takes sequential demonstrations, fits or loads a dynamics model, decides from the data
whether labels are worth generating, generates candidate labels together with the
evidence for trusting each one, and writes the selections a user chooses to consume.
The result is ordinary state-action supervision with recorded provenance.

The scope is intentionally narrow. CCIL is not a policy trainer, simulator, benchmark
harness, or dataset library. It needs no environment, reward, or expert policy, and it
imports no downstream learner. Comparisons of policies trained with and without its
labels belong to
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks).

## Demonstrations and dynamics

The input is a dataset of episodes of observations and actions, written as a
`ccil.trajectories` file. A LeRobot v3.0 dataset converts to the same format through
the optional `ccil[lerobot]` bridge, which records the disposition of every feature
and the hash of every source file. CCIL relies on sequence semantics that the format
alone cannot express. Episode boundaries must be known, frames must be ordered, and
each recorded transition must be governed by the single recorded action. An interval
in which unrecorded actions may have acted is refused, because a label built across it
would describe a transition the system never made.

The dynamics source is either fitted to the demonstrations (a linear model, or an MLP
with local Lipschitz regularization) or loaded from a checkpoint that a previous fit
wrote. A loaded model carries a record of the representation, dimensions, direction,
and action period it was trained for, and it is refused unless that record matches the
current data and configuration. In either case the model is evaluated on held-out
episodes it never saw, and that evaluation is part of the result. When the true
dynamics are available, as for an analytic system, they may be supplied separately as
reference dynamics for certification. They are never required.

## Diagnosis before generation

A dynamics model can always be asked for labels, but its answers are not always worth
training on. Labels that stay on top of the demonstrations add little information, and
labels the model cannot support add error. `ccil diagnose` therefore asks a sequence of
distinct questions before any label is used. Did generation work? How far do the
labels move from their source states, measured against the demonstrations' own
spacing? Do they reach states the dataset does not already cover? How accurate is the
model on held-out data? Is there any evidence, beyond the data, that the model is
credible at the proposed states?

The answers determine the admitted recovery length, the number of recorded action
intervals over which a synthetic state rejoins a demonstration. In this release the
only admissible length is one step, the case studied in the paper. When no length
passes, the command still succeeds. It writes an empty but schema-valid result and a
diagnosis that names the failing check and the action that could change it, such as
collecting recovery data or improving the model. No additional rows is a legitimate
outcome. The diagnosis uses no policy return, task reward, or downstream result, so it
cannot say whether the labels will help a policy.

## Candidates, evidence, and selections

Generation and the decision of what to train on are kept separate. `ccil generate`
writes an immutable candidate set, and for each candidate a record of the quality
evidence that bears on it. That evidence is labelled by what it rests on. A bound that
follows from exact reference dynamics is **certified**. A bound that holds only if the
user's declared continuity constants are true is **conditional**, and those constants
are recorded as assumptions rather than measurements. An empirical score, which bounds
nothing, is never presented as a bound.

`ccil curate` applies a declared policy, such as a largest acceptable label error or a
quantile of the evidence, and writes a selection: a mask and weights over the
unchanged candidates. Any number of selections can be made from one generation without
regenerating anything, which makes it inexpensive to compare several correction bands
in a downstream pilot. `ccil inspect` prints the resolved configuration with each
value's source, the provenance and hashes, the distribution of evidence, the reasons
for rejection, and, for a selection, the Python that loads its rows.

Every stage writes a content-addressed run directory. Repeating a stage with the same
data, dynamics, and configuration reuses its result. Curation and inspection refuse a
result whose recorded files have changed, before writing anything.

## Consuming the supervision

A selection becomes training data through `ccil.supervision.build_supervision`. It
returns the demonstrations of the training episodes together with the selected labels,
with their weights and classifications. Only usable rows are returned, so a rejected or
unselected candidate cannot reach a loss through a zero weight. The held-out episodes
contribute no rows. The trainer is the user's own, and CCIL trains nothing.

For a dataset that came from LeRobot, `ccil export lerobot` writes one augmented
LeRobot v3.0 dataset: the training episodes, plus one single-frame episode per selected
label, with provenance columns and an export manifest. An unchanged LeRobot training
command can consume it. CCIL depends only on LeRobot's documented storage format and
never imports the `lerobot` package.

## Whether the labels help

Nothing in the data alone can establish that synthetic labels improve a policy. That
requires a learner, an evaluation, and a comparison that controls for the extra rows.
A method that adds rows also adds optimizer exposure, so the informative comparison is
against duplicated demonstrations at matched exposure, not only against behavior
cloning.

The studies in il-supervision-benchmarks make that comparison. On continuous Pendulum,
CCIL improved on behavior cloning, but in a matched recovery study duplication
reproduced more than half of that gain, and the remaining difference was not resolved.
On PushT keypoints, one-step CCIL showed no advantage over duplication that replicated
across learner seeds, and at a high synthetic fraction it lowered coverage. These
results are scoped to their learners, label recipes, and synthetic doses. They are
documented in the
[Pendulum](https://github.com/personalrobotics/il-supervision-benchmarks/blob/main/docs/studies/pendulum.md)
and [PushT](https://github.com/personalrobotics/il-supervision-benchmarks/blob/main/docs/studies/pusht.md)
study reports.
The diagnosis and curation tools exist so that each user can run the same kind of
pilot on their own task and learner.

## From demonstrations to curated rows

From a clone, [uv](https://docs.astral.sh/uv/) installs the core, which needs only
NumPy and PyYAML and supports Python 3.10 and newer:

```sh
git clone https://github.com/personalrobotics/CCIL
cd CCIL
uv sync
```

The extras `ccil[torch]` (the MLP dynamics model) and `ccil[lerobot]` (the LeRobot
bridge) add optional dependencies. The MLP model and Hub access require Python 3.11 or
newer.

With a dataset and a configuration in place, the workflow is three commands:

```sh
ccil generate --config ccil.yaml
ccil curate runs/generate-result.json --set curate.tolerance=0.1
ccil inspect runs/generate-result.json
```

`generate` splits the episodes, fits the model, runs the diagnosis, generates
candidates at the admitted recovery length, and attaches their evidence. It ends with
a summary of the artifacts, row counts, rejections, and evidence classes, and the
suggested next command. The [quickstart](docs/quickstart.md) builds the demonstrations
and configuration for a small double integrator and follows this path to a pilot that
compares behavior cloning with each selection. Its code blocks are executed from the
built wheel in a clean environment by `tools/golden_path/check.py`. The
[PushT walkthrough](docs/pusht.md) follows the same path from a LeRobot dataset to an
unchanged LeRobot trainer.

## Current capabilities

| Capability | Status | Details |
| --- | --- | --- |
| Vector and keypoint demonstrations in `ccil.trajectories` | Implemented | [Schemas](docs/schemas/reference.md) |
| LeRobot v3.0 tabular datasets, imported and exported | Implemented; image, video, depth, and audio features are refused | [LeRobot](docs/lerobot.md) |
| Linear and MLP dynamics models, fitted or loaded, with held-out evaluation | Implemented | [Contract](docs/contract.md#inputs) |
| Diagnosis and one-step corrective labels | Implemented | [Contract](docs/contract.md#recovery-length) |
| Multi-step recovery segments | Not admitted in this release | [Contract](docs/contract.md#recovery-length) |
| Certified, conditional, and calibrated label evidence, and curation | Implemented | [Certificates](docs/certificates.md) |
| Supervision rows for any downstream trainer | Implemented | [Supervision](docs/supervision.md) |
| Image CCIL with a learned representation | Planned for M5 and M6 | [Contract](docs/contract.md#what-ccil-is) |

Policy training, NoiseBC, evaluation, and benchmark studies are provided by
il-supervision-benchmarks rather than by this package.

## Legacy inputs

The datasets of the original implementation are Python pickles, which can execute code
when loaded, so `ccil` never reads them. A one-time conversion tool verifies each
pickle's SHA-256 against [the artifact manifest](docs/artifacts/manifest.json) before
unpickling it, and then writes the canonical format. The procedure is documented in
[legacy conversion](docs/migration/legacy-conversion.md). The original `correct_il`
package, its scripts, and its configurations are preserved at the `surface-archive-v1`
tag.

## Related packages

| Package | Owns |
| --- | --- |
| CCIL | Dynamics fitting, diagnosis, corrective-label generation, quality evidence, curation, and supervision artifacts |
| [il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks) | Matched studies of supervision methods, policy training, evaluation, and reports |
| [LeRobot](https://github.com/huggingface/lerobot) | Dataset format, policies, training, and evaluation that consume an exported dataset |

## Documentation

| Task | Document |
| --- | --- |
| Run the workflow end to end on a small system | [Quickstart](docs/quickstart.md) |
| Use a LeRobot dataset and trainer | [PushT](docs/pusht.md) and [LeRobot datasets](docs/lerobot.md) |
| Understand the commands, entry points, exit codes, and artifacts precisely | [User contract](docs/contract.md) and [worked examples](docs/contract-examples.md) |
| Choose configuration values | [Parameter reference](docs/config/parameters.md) |
| Understand label evidence, profiles, and calibration | [Certificates](docs/certificates.md) |
| Consume the selected rows in a trainer | [Supervision](docs/supervision.md) |
| Read the artifact formats | [Schemas](docs/schemas/reference.md) |
| Understand stages and run directories | [Pipelines](docs/pipelines.md) and [run directories](docs/runs.md) |
| Implement a dynamics model, generator, or representation | [Core interfaces](docs/interfaces.md) |
| Review dependency and checkpoint policy | [Security](docs/security.md) |

The README introduces the design. The documents above contain the operational and
normative details.
