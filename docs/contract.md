# The CCIL user contract

This is the normative specification of CCIL's public workflow and its
artifact vocabulary (#166). Other documents link here rather than restating it.
Field encodings belong to the schemas ([`schemas/`](schemas/README.md), and
#167 for the quality bundle). [contract-examples.md](contract-examples.md)
works through three concrete cases.

> **Status.** This contract is the target of M4.5. The commands below are
> implemented by #170; until then, the current commands map to them as listed
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
- a LeRobot v3.0 dataset, converted by the `ccil[lerobot]` bridge into
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
- **Load:** a transition-model checkpoint, accepted only when its recorded
  component, kind and version are known and its SHA-256 is recorded in every
  artifact built from it. CCIL never unpickles.
- **Exact dynamics** (optional, `certificate.reference_dynamics`): used as
  evidence for certificates and audits. It never selects a recovery length or
  a label.

## Commands

Every command is also a Python function with the same name and arguments in
the `ccil` package. Exit codes: 0 success, including an empty result; 2 an
invalid configuration, dataset, artifact or combination; 3 the code changed
while the command ran. A stage whose run directory is already complete with
the same identity is reused ([runs.md](runs.md)).

| Command | Reads | Writes |
|---|---|---|
| `ccil diagnose` | dataset, dynamics source, configuration | a **diagnosis**: the recovery-length frontier and the strongest available evidence class. No labels. |
| `ccil generate` | dataset, dynamics source, configuration | the prepared dataset and split, the dynamics checkpoint, the diagnosis, immutable **candidates**, and their **quality evidence** |
| `ccil curate` | one `generate` result, a curation policy | a **selection**: a usable-row mask and weights. Any number of selections can be made from one candidate set. |
| `ccil inspect` | any artifact or run directory | a compact summary: resolved configuration with each value's source, provenance and hashes, quality distributions, rejection counts, usable rows, and the next consumption step |

`ccil generate` runs Diagnose first and generates only at an admitted recovery
length. It ends by printing the artifacts, row counts, rejection classes,
evidence classes, and the next step.

## Recovery length

The public object is the **recovery segment** (#171): starting from a
synthetic state, M actions that rejoin a demonstration at x_(t+M). One-step
CCIL is the case M = 1.

- `generation.recovery_steps` is `auto` (default) or an integer.
- `auto` probes a fixed ladder up to a declared cap and admits the **smallest**
  M that passes the data-only gates. **The current ladder is {1}.** Raising the
  cap (#171) changes no field's meaning.
- **The gates** use only the demonstrations and the dynamics evidence:
  - the held-out M-step model error ε_M;
  - reach: η_reach(M) = r_reach(M) / τ_demo, where τ_demo is the declared
    demonstration-support scale;
  - novelty: each synthetic state's distance to the nearest demonstration
    state, d_demo, relative to τ_demo;
  - the strongest evidence class available for the labels.
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

## Artifacts and views

| Artifact | Contents | Schema |
|---|---|---|
| Prepared dataset and split | the validated input, and the episode split | `ccil.trajectories`, `ccil.split` |
| Dynamics checkpoint | the fitted or loaded model, and its held-out error | transition-model checkpoint |
| Diagnosis | the per-M gates, the admitted M or the empty-result reason, the evidence class | #167 |
| Candidates | every generated row, accepted and rejected, with source identity, generator identity, draw identity, measurements and rejection reasons | `ccil.candidates` 2.x |
| Quality evidence | per-row error terms and classification (certified, conditional, empirical, unsupported), with every constant's provenance | `ccil.certificates` today; #167's bundle |
| Selection | one named curation policy, its tolerance, a mask and weights | #167 |

**Immutability.** No artifact is modified after it is written. Curate never
changes candidates or quality evidence, and never promotes an empirical score
to a bound. Every artifact records the SHA-256 of each input it was built from.

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
| `curate` | CCIL | curation policy and tolerance (#167) |
| `policy`, `evaluation`, `environment` | downstream | the benchmark harness (personalrobotics/il-supervision-benchmarks) |

CCIL commands refuse downstream keys once #168 moves them. An unknown key, a
value outside its domain, or a broken cross-field rule fails before any model
is constructed.

## Trust boundaries and failure behavior

- **Datasets** are validated at ingress: shapes, dtypes, finiteness, episode
  structure. Legacy pickles cross only through the offline converter
  (`correct_il/convert_legacy_dataset.py`); CCIL itself never unpickles
  ([security.md](security.md)).
- **Checkpoints** load only by known kind and version, and are bound by hash.
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
| `ccil labels generate` | **Rename** to `ccil generate`. Kept as a deprecated alias for one minor release. |
| `labels.certify` stage | **Keep** inside Generate, as its quality evidence. Its selection keys (`certificate.max_total_error`, `score_threshold`, `selection_quantile`, `selection_population`) **move** to `curate.*`; the old keys are converted with a recorded note. |
| — | **New:** `ccil diagnose`, `ccil curate`, `ccil inspect`, `generation.recovery_steps`, and the `curate` group. |
| `ccil data prepare` | **Keep**, as the first stage Generate runs. |
| `ccil data convert`, `ccil config convert` | **Keep**, as one-way legacy migration. |
| `ccil schema validate`, `ccil run validate`, `ccil config show` | **Keep.** |
| `ccil schema docs`, `ccil config docs`, `ccil config audit` | **Keep** as maintainer tools that regenerate references. |
| `ccil policy train`, `ccil.experiments`, `ccil.training` | **Deprecate** and move to the benchmark harness (#168). |
| `ccil pipeline integrated` | **Keep** as the level-4 reference, outside the default product. |
| `ccil.supervision.build_supervision`, `ccil.schema` readers, `ccil.lerobot` | **Keep** as the consumer interfaces. |
| `policy.*`, `evaluation.*`, `environment.*` | **Move** to the benchmark harness (#168). |
