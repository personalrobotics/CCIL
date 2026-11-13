# The CCIL user contract

This is the normative specification of CCIL's public workflow and its
artifact vocabulary (#166). Other documents link here rather than restating it.
Field encodings belong to the schemas ([`schemas/`](schemas/README.md), and
#167 for the quality bundle). [contract-examples.md](contract-examples.md)
works through three concrete cases.

> **Status.** Implemented by #170. The former commands map to these as listed
> in [Dispositions](#dispositions-of-the-current-surface).

## What CCIL is

Sequential demonstrations in, more ordinary state-action supervision out.
CCIL generates synthetic recovery labels from a dynamics model, records the
evidence for trusting each one, and lets the user choose which to consume. It
needs no environment, reward, expert policy, downstream trainer or
continuous-time model.

| Level | What it is | Where |
|---|---|---|
| **1. Generate** | Demonstrations plus a dynamics source produce immutable candidates and their quality evidence. | Core |
| **2. Curate** | A declared tolerance or selection policy produces a separate selection. The candidates do not change. | Core |
| **3. Auto-fit** | An adapter trains and compares BC and CCIL on held-out validation. BC with no synthetic rows may win. | Optional, outside the core |
| **4. Integrated image CCIL** | Representation, dynamics, labels and policy training coupled algorithmically. | A declared exception (M5) |

Levels 1 and 2, with Diagnose and Inspect, are the default product. Nothing in
them imports a policy trainer, benchmark task, simulator or statistics package.

## Inputs

**Dataset.** One of:
- a `ccil.trajectories` file ([schema](schemas/reference.md));
- a LeRobot v3.0 dataset, converted by the `prlccil[lerobot]` bridge into
  `ccil.trajectories` plus a `ccil.lerobot.conversion` manifest that records
  every feature's disposition and every source file's hash
  ([lerobot.md](lerobot.md)). Image, video, depth and audio features are refused
  until level 4, unless the caller ignores them by name.

**Required sequence semantics.**
- Episode boundaries, and ordered frames within each episode.
- Each recorded transition x_t → x_(t+1) is governed by the single action a_t.
  An interval in which unrecorded actions may have acted is not a transition:
  it is refused, unless an explicit aggregate-action contract is declared.
- A representation in which distances are meaningful: `identity` on a state
  modality, or a declared representation.

**Optional metadata.**
- **Timing:** timestamps, or the source's FPS. They give the action period
  `dt_ctrl`. Without them, CCIL reports recovery length M in dataset action
  intervals only, leaves the physical horizon H unknown, and makes no
  deployment-rate claim.
- **Action bounds** (`data.action_low`, `data.action_high`): labels outside them
  are rejected.
- **Endings:** whether an episode terminated or was only truncated. Unknown is
  recorded as unknown.

**Dynamics source.**
- **Fit** (default): a transition model fitted to the demonstrations. A
  held-out evaluation on unseen episodes is always reported.
- **Load** (`transition_model.checkpoint`): a transition-model checkpoint
  that `transition.fit` wrote, which records the contract below; an older one
  records none and is refused. CCIL never unpickles. Its SHA-256 is recorded
  in every artifact built from it. Before the model is used, its recorded
  contract must match this dataset and configuration, or the command fails
  with `CheckpointError` (exit 2):
  - component, kind and version are known;
  - representation kind, modality and, for a fitted representation (PCA),
    its checkpoint SHA-256: the same kind and shape with another fit is
    another coordinate system;
  - state and action dimensions;
  - direction (forward or backward);
  - action period: equal when both the checkpoint and the data declare
    one; both unknown is accepted and limits the run to M without H; one known
    and one unknown is refused.

  A model's own input normalization is stored inside its checkpoint, and
  loads with it.
- **Exact dynamics** (optional, `certificate.reference_dynamics`): see
  [Reference dynamics](#reference-dynamics).

## Commands

| Command | Reads | Writes |
|---|---|---|
| `ccil diagnose` | dataset, dynamics source, configuration | the prepared dataset and split, the dynamics checkpoint and its held-out evaluation, and the **diagnosis**: the recovery-length frontier and the evidence class. No candidates. |
| `ccil generate` | dataset, dynamics source, configuration | everything Diagnose writes, plus immutable **candidates** and their **quality evidence** |
| `ccil curate` | one `generate` result, a curation policy | a **selection**: a usable-row mask and weights. Any number of selections can be made from one candidate set. |
| `ccil export lerobot` | one curate result of a dataset converted from LeRobot, its conversion manifest | one LeRobot v3.0 dataset an unchanged trainer consumes: the training episodes plus one one-frame episode per selected label, with provenance columns and `ccil/export.json` ([lerobot.md](lerobot.md#an-augmented-dataset-for-an-unchanged-trainer-30)) |
| `ccil inspect` | any artifact or run directory | nothing; it prints a summary: resolved configuration with each value's source, provenance and hashes, quality distributions, rejection counts, usable rows, and the next consumption step |

**Durable outputs and reuse.** Every stage is a content-addressed run
directory ([runs.md](runs.md)). A stage whose run directory is already
complete with the same identity is reused. So `ccil generate` after `ccil
diagnose` with the same data, dynamics and configuration reuses the prepared
dataset, the checkpoint and the diagnosis, and generates only at the admitted
recovery length. Generate ends by printing the artifacts, row counts,
rejection classes, evidence classes, and the next step.

**Result files.** Generate and Curate each write an immutable result in
`run.output_dir`, named by the SHA-256 of its content (`generate-<id>.json`,
`curate-<id>.json`) and never replaced; `generate-result.json` is a copy of
the latest Generate there. A curate result records the immutable generate
result it read, with its hash. Every path a result records is relative and
must resolve, through any symlink, inside its directory. Curate and Inspect
refuse a result (exit 2) before writing anything when a path leaves the
directory or any recorded file has changed.

**Empty results are complete.** When no recovery length is admitted, Generate
still writes schema-valid zero-row candidates, zero-row quality evidence, and
a complete manifest recording the reason. Curate on them writes a zero-row
selection. A consumer never branches on a missing file.

**Exit codes.**

| Code | Meaning |
|---|---|
| 0 | Success, including an empty result |
| 1 | An unexpected program or infrastructure failure, reported with its traceback. It is never presented as an input error. |
| 2 | An invalid configuration, dataset, checkpoint, artifact or combination: a `ccil.InputError` |
| 3 | The code changed while the command ran (`CodeStateChanged`) |

**Python entry points.** The CLI is a thin wrapper over these functions in
the `ccil` package:

```python
ccil.resolve_config(recipe=None, config=None, overrides: dict | None = None) -> ccil.Config
ccil.diagnose(config: ccil.Config) -> ccil.DiagnoseResult
ccil.generate(config: ccil.Config) -> ccil.GenerateResult
ccil.curate(generated: ccil.GenerateResult | str | Path, policy: ccil.CurationPolicy) -> ccil.CurateResult
ccil.inspect(target: str | Path) -> ccil.Inspection
```

- **`Config`** is the resolved, validated configuration, with each value's
  source. `overrides` maps dotted keys to typed values, as `--set` does to
  strings. Resolution is the only place configuration is parsed.
- **Results** are immutable records:
  - **`DiagnoseResult`:** the run directory of each stage, the diagnosis path,
    and the admitted M, or `None` with a reason.
  - **`GenerateResult`:** the same, plus the candidates and quality-evidence
    paths, the row counts, the immutable result file (`path`) and
    `generate-result.json` (`latest`).
  - **`CurateResult`:** the selection path, its counts and its immutable
    result file.
  - **`Inspection`:** the printed summary as data.
- **Exceptions.** Every input failure raises a subclass of `ccil.InputError`,
  which itself subclasses `ValueError`. That covers configuration, schema,
  checkpoint, dataset-format, split and artifact-mismatch errors, and each one
  lists every problem found. Errors are classified by cause: settings or
  artifacts a component refuses raise `ccil.core.RequestError` (an
  `InputError`), while malformed output from a component implementation
  raises `ccil.core.InterfaceError`, a defect. `CodeStateChanged` is separate.
  Any other exception is a defect and propagates unchanged.

## Recovery length

The public object is the **recovery segment** (#171): starting from a
synthetic state, M actions that rejoin a demonstration at x_(t+M). One-step
CCIL is the case M = 1.

- `generation.recovery_steps` is `auto` (default) or an integer.
- `auto` probes a fixed ladder up to a declared cap and admits the **smallest**
  M that passes the data-only gates. **The current ladder is {1}.** Raising the
  cap (#171) changes no field's meaning.
- **The gates use no reference dynamics.** Their inputs are:
  - the demonstrations and their split;
  - the learned or loaded transition model and its held-out evaluation;
  - the user's declared constants, which are recorded as assumptions.

  They are distinct questions ([pipelines.md](pipelines.md#diagnosis-labelsdiagnose)):
  - yield: did generation work?
  - displacement: how far labels move from their sources, relative to τ_demo,
    the demonstration-support scale computed from the data;
  - novelty: whether they reach states the dataset does not cover, from
    d_demo, the distance to the nearest demonstration state, relative to τ_demo;
  - fit: the held-out M-step model error ε_M, at demonstration states;
  - trust: off-data evidence that the model is credible at the proposed
    states. Data alone cannot give it, so it comes only from a bound
    certificate's declared or model-computed constants, recorded as
    conditional.
- No policy return, task reward or downstream outcome enters the selection. An
  integer `recovery_steps` skips the ladder but not the gates' report.
- **When no M passes,** the command succeeds with an **empty synthetic view**
  and a diagnosis that names the failing gate and the action that could change
  it: collect recovery data, improve the dynamics model, or accept that CCIL
  has no admissible labels here. No extra rows is preferable to untrustworthy
  rows.
- **What a diagnosis can and cannot say.** It states which correction bands
  are populated and recommends nested views for a downstream pilot. Whether
  those labels improve a policy needs a learner and a validation signal: that
  is Auto-fit's question, and the diagnosis never answers it.

**Clocks.** `dt_ctrl` is the physical action period, from the data.
`recovery_horizon_s`, when timing exists, is derived: H = M · dt_ctrl, or the
sum of the action durations. The internal integration step h_int is numerical
provenance only: refining it may change work and numerical error, never M, H,
or a label's meaning. Resampling a dataset changes `dt_ctrl`, which is a
changed system, not a harmless reformatting.

### Reference dynamics

Exact reference dynamics (`certificate.reference_dynamics`) never enter the
default decision. In the default `data_only` evidence mode they may only
annotate an already-decided result, in a separate audit record. They cannot
change:
- the admitted M;
- the candidate set;
- the default quality evidence;
- any selection.

A run that lets reference dynamics shape admission or evidence must declare
`generation.evidence_mode: reference_assisted`. That mode is recorded in every
artifact it produces, shown by Inspect, and propagated into every selection
made from those artifacts.

Two artifacts carry reference dynamics as declared evidence
([certificates.md](certificates.md#calibration-and-the-reference-audit)):
- **The reference audit** (`audit.reference_dynamics`, `labels.audit`)
  annotates a decided selection with the label error under the reference, in
  any evidence mode. Nothing reads it back, so an audited `data_only` result
  stays `data_only`.
- **A calibration** (`calibration.level`, `labels.calibrate`) bounds a score
  profile's label error with one episode-level conformal bound from the
  error observed on the held-out episodes. It is always `reference_assisted`,
  and only through it can a row become `calibrated`.

## Artifacts and views

| Artifact | Contents | Schema |
|---|---|---|
| Prepared dataset and split | the validated input, and the episode split | `ccil.trajectories`, `ccil.split` |
| Dynamics checkpoint | the fitted or loaded model, and its held-out error | transition-model checkpoint |
| Diagnosis | the per-M gates, the admitted M or the empty-result reason, the evidence class | `ccil.diagnosis` 1.0 |
| Candidates | every generated row, accepted and rejected, with source identity, generator identity, draw identity, measurements and rejection reasons | `ccil.candidates` 2.x |
| Quality evidence | per-row error terms and classification (certified, conditional, empirical, unsupported), with every constant's provenance | `ccil.certificates` 2.0 |
| Calibration (optional, `reference_assisted`) | one conformal bound on the observed label error, from the held-out episodes' per-episode largest errors | `ccil.calibration` 1.0 |
| Selection | one named curation policy, its tolerance, a mask and weights, and with a calibration the rows it calibrated | `ccil.selection` 1.0, or 1.1 with a calibration |
| Reference audit (optional) | reference-dynamics annotations of a decided result; never an input to admission, evidence or selection | `ccil.reference-audit` 1.0 |

**Immutability.** No artifact is modified after it is written. Curate never
changes candidates or quality evidence, and never silently promotes an
empirical score to a bound: only a recorded, `reference_assisted` calibration
makes a row `calibrated`, and the selection marks each such row. Every artifact records the SHA-256 of each input it was built from.

**Views a consumer reads:**
- **Pointwise (default):** the demonstration rows plus the selected synthetic
  rows, each with its weight, source, evidence class and segment identity and
  offset ([supervision.md](supervision.md), `ccil.supervision.build_supervision`).
  A claim about a whole recovery segment, such as reaching its target after M
  actions, is never attached to one isolated row.
- **Chunk (optional):** a recovery segment's actions as one chunk, for a
  consumer that declares it will consume chunks. Otherwise chunked supervision
  is refused, never silently reduced to its first action.
- **LeRobot overlay:** the pointwise view stored beside an unchanged copy of
  the source dataset ([lerobot.md](lerobot.md#ccil-supervision-as-an-overlay-28)).
  Synthetic rows are never written as LeRobot episodes.

## What a user can inspect before consuming a label

For every candidate row:
- whether it was accepted and, if not, every reason it was rejected;
- its evidence class, and whether its error figure is a mathematical bound or
  an empirical score;
- its error-to-tolerance ratio, with the raw error and its units beside it;
- its novelty, d_demo / τ_demo, with the raw distance;
- its source transition, generator, draw, and recovery segment;
- the identities of the dataset, model and configuration it came from.

For the run: the diagnosis, the distributions of these quantities, and the
counts each curation policy would keep.

## Configuration

A configuration is resolved from defaults, an optional recipe, an optional
file, and `--set` overrides, in that order. `ccil config show` prints every
value with its source ([parameters.md](config/parameters.md)).

| Group | Owner | Meaning |
|---|---|---|
| `run`, `compute` | CCIL | seeds, output location, device |
| `data` | CCIL | dataset, split, action bounds |
| `representation` | CCIL | the space labels live in |
| `transition_model` | CCIL | dynamics fit or load |
| `generation` | CCIL | generator, recovery length, support and validity rules |
| `certificate` | CCIL | quality-evidence profile and its declared constants |
| `curate` | CCIL | curation tolerance, quantile and population |
| `policy`, `evaluation`, `environment` | downstream | the benchmark harness (personalrobotics/il-supervision-benchmarks) |

CCIL commands refuse downstream keys once #168 moves them. An unknown key, a
value outside its domain, or a broken cross-field rule fails before any model
is constructed.

## Trust boundaries and failure behavior

- **Datasets** are validated at ingress: shapes, dtypes, finiteness, episode
  structure. Legacy pickles cross only through the offline converter
  (`tools/legacy_conversion/convert_legacy_dataset.py`); CCIL itself never unpickles
  ([security.md](security.md)).
- **Checkpoints** load only by known kind and version, are bound by hash, and
  must match the dataset's representation, dimensions, normalization, direction
  and transition interval ([Inputs](#inputs)).
- **Configuration**: unknown keys and unsupported combinations are errors, not
  warnings.
- **Artifact reuse** requires the same content identity and code state
  ([runs.md](runs.md)). A changed input, a corrupted artifact, or artifacts
  that do not belong together fail with every reason listed.
- **Ordering.** All of these checks run before any model is constructed or
  any artifact is written. A failed command leaves no completed run directory.
- **Writes are atomic:** an interrupted command never leaves an artifact that
  passes as complete.

## Dispositions of the current surface

| Current | Disposition |
|---|---|
| `ccil labels generate` | **Renamed** to `ccil generate` (#170). Kept for one minor release as a deprecated alias with its former outputs: it also curates, and `--result` writes `ccil.labels-result` 1.0. |
| `labels.certify` stage | **Keep** inside Generate, as its quality evidence. Its selection keys (`certificate.max_total_error`, `score_threshold`, `selection_quantile`, `selection_population`) **move** to `curate.*`; the old keys are converted with a recorded note. |
| — | **New:** `ccil diagnose`, `ccil curate`, `ccil inspect`, `generation.recovery_steps`, `generation.evidence_mode`, and the `curate` group. |
| `certificate.reference_dynamics` | **Keep** as an audit input. A profile that uses it for admission or selection, as today's `labels.certify` can, runs only under `evidence_mode: reference_assisted`; a configuration converted from an older version gets that mode, with a recorded note. |
| `ccil data prepare` | **Keep**, as the first stage Generate runs. |
| `ccil data convert`, `ccil config convert` | **Keep**, as one-way legacy migration. |
| `ccil schema validate`, `ccil run validate`, `ccil config show` | **Keep.** |
| `ccil schema docs`, `ccil config docs`, `ccil config audit` | **Keep** as maintainer tools that regenerate references. |
| `ccil policy train`, `ccil.training`, `ccil.experiments` | **Moved** to il-supervision-benchmarks (#168): `ilsb policy train CONFIG`, `ilsb.learners.vector` (same trainer names, versions and training identities) and `ilsb run SPEC`. |
| `ccil pipeline integrated` | **Keep** as the level-4 reference, outside the default product: the one CCIL command that trains a policy, with its own reference trainers (`integrated.trainer`). |
| `ccil.supervision.build_supervision`, `ccil.schema` readers, `ccil.lerobot` | **Keep** as the consumer interfaces. The supervision sources are `demonstrations` (formerly `bc`, and `noise_bc` with `perturbation="noise_bc"`) and `demonstrations_plus_selection` (formerly `ccil`). |
| `policy.*`, `evaluation.*`, `environment.*` | **Moved** to the benchmark harness (#168). The keys the integrated pipeline reads are renamed (below). `config_version` 4 reads an older file's renamed keys with a note, drops its other `policy.*`, `evaluation.*` and `environment.*` keys with a note, and refuses `policy.method noise_bc`; a version 4 file that sets any of them is refused. |

**Configuration migration (#168).** A former key in a version 1 to 3 file is converted, with a note.

| Former | Replacement |
|---|---|
| `policy.method: bc`, `ccil` | `supervision.source: demonstrations`, `demonstrations_plus_selection` |
| `policy.method: noise_bc` | `ilsb policy train` with `policy.noise`; CCIL's supervision for it is `supervision.source: demonstrations` |
| `policy.demonstration_weight`, `policy.exclude_validation_episodes` | `supervision.demonstration_weight`, `supervision.exclude_validation_episodes` |
| `policy.trainer` | `integrated.trainer` |
| other `policy.*`, `evaluation.*`, `environment.id` | the `ilsb.policy-train` configuration's `policy` section |
