# LeRobot datasets

> **`ccil[lerobot]` installs CCIL's LeRobot Dataset v3.0 format bridge. It does
> not install the LeRobot runtime.**

CCIL's product is a versioned supervision artifact. LeRobot is one ecosystem
that consumes it; robomimic, your own PyTorch code, or anything else that reads
the artifact can consume it too. So CCIL depends on LeRobot's documented
storage format, not on the `lerobot` package. CCIL never imports `lerobot`, and
it does not inherit LeRobot's training stack, Python floor or import paths.

```bash
pip install 'ccil[lerobot]'      # pyarrow, and huggingface_hub on Python 3.11+; not lerobot, torch, datasets or video decoders
```

On Python 3.10 the extra installs pyarrow only. The bridge works fully on local directories there, but `fetch`
(Hub access) needs Python 3.11+, because huggingface_hub depends on filelock, whose fixed releases need 3.11
(as for the torch extra; see [`security.md`](security.md)).

## What the bridge supports

The bridge reads and writes the **tabular subset of LeRobotDataset v3.0**:
states, actions, timestamps, task indices, episode metadata, statistics and
schema metadata.

| LeRobot | ccil.trajectories |
|---|---|
| `observation.state` | `observations["state"]` |
| `observation.<name>`, a float vector | `observations["state.<name>"]` |
| `action` | `actions` |
| `timestamp` | `timestamps` |
| an episode's task (`task_index` → `meta/tasks.parquet`) | `task` |
| `index`, `frame_index`, `episode_index`, `task_index` | checked, not copied |
| — | `termination_known=False`, `terminated=False`: an episode boundary says that recording stopped, not why (see below) |

**How it reads and writes:**
- **The path template.** It follows the templates and feature declarations in
  `meta/info.json`, and never assumes fixed shard paths.
- **Streaming.** It reads and writes with PyArrow directly, one data shard at
  a time. `iter_episodes` streams; `read_dataset` builds the in-memory
  `ccil.trajectories` dataset.
- **Writing.** `write_dataset` streams episodes into shards of at most
  `data_files_size_in_mb`. It builds the directory under a temporary name and
  moves it into place, and it refuses an existing destination.
- **The Hub.** `fetch(repo_id, revision)` is an I/O convenience: it resolves
  the revision to a commit and downloads `meta/` and `data/`. Everything else
  works on local directories.

**Statistics.** CCIL writes:
- per-episode and dataset count, min, max, mean and std, computed exactly
  (std is pooled across episodes);
- quantiles q01, q10, q50, q90 and q99: exact per episode, and aggregated as
  their count-weighted mean, the aggregation LeRobot uses.

LeRobot's own quantiles come from an approximate histogram estimator, so the
two differ in quantiles only. The compatibility workflow reports the
difference.

### Ingestion: endings, dispositions and conversion manifests (#27)

**Endings.** No `next.*` feature is ever interpreted by default. Every
episode's ending is unknown: `termination_known=False`, `terminated=False`
(trajectories schema 1.1). A 1.0 reader sees the same episodes as
non-terminal, the conservative reading.

A caller may declare `terminal_feature`, an exhaustive boolean terminal-state
indicator:
- The feature must exist and be stored as a boolean.
- It may be true only on an episode's final frame.
- A final true means terminated, and a final false means truncated.
- A true value before the final frame refuses the episode.

`next.done` gets no special treatment. In `lerobot/pusht_keypoints` it is true
on the last two frames of every episode, so declaring it as the terminal
feature is refused.

**Dispositions.** Every source feature gets an explicit disposition, recorded
in the dataset's meta and the conversion manifest:
- **mapped** to a CCIL field;
- **checked_index**, for the index columns;
- **terminal_indicator**;
- **ignored**, when the caller names it in `ignore`;
- **preserved_as_source_metadata**, when the caller names it in `preserve`.
  The manifest records its schema, a summary and a SHA-256 of its values, but
  never copies the values into JSON.

A feature with no disposition is refused.

**Diagnostics.** Every refusal names the shard, row, episode and frame.
- **Refused:** duplicate (episode, frame) identities and non-monotonic
  indices.
- **Not refused:** consecutive frames with identical values, which are
  legitimate data.

**Conversion.** `convert(source, out.npz, manifest.json, ...)` writes the
trajectories file and the manifest atomically. The manifest holds the source
file hashes, every feature's disposition, the episodes' ending classification,
and the output hash. It is dataset metadata, kept apart from any algorithm
parameter. The source is only read, and its hashes are checked before and
after.

### What it refuses

Every refusal names the problem, and nothing partial is returned or written.

- **Unsupported modalities.** Images, video, depth and audio are not supported
  until image CCIL (M5). They are refused, never silently dropped.
- **Unknown format versions.** Anything other than `codebase_version` v3.0
  (for example v2.1, written by LeRobot 0.3) fails closed.
- **Unsupported encodings.** A string, integer, boolean or multi-dimensional
  feature, or a feature with no CCIL slot, is refused. A caller can name a
  feature in `ignore` to read the dataset without it; the read records that
  choice.
- **Inconsistent data.** Malformed `info.json`, a missing shard, a path
  template that leaves the dataset, episode offsets that disagree with the
  data, an episode with several tasks, and writing timestamps that do not
  advance by 1 / fps (LeRobot checks this when it loads).

## CCIL supervision as an overlay (#28)

A LeRobot episode means temporally related frames, while a CCIL label is an
independent counterfactual supervision record. So labels are never written as
LeRobot episodes. Instead `export_overlay` writes one immutable bundle:

```
destination/
  data/  meta/  videos/          # an unchanged copy of the source LeRobot dataset
  ccil/meta.json                 # schema, source files, run ID, input hashes, feature map, generator, shard hashes
  ccil/labels/chunk-000/file-000.parquet ...
```

- **The LeRobot portion** is a byte-for-byte copy of the source, so stock LeRobot readers see exactly the original
  episodes.
- **The sidecar** holds exactly the synthetic rows `build_supervision` returns.
  - Each row has its observation (in the LeRobot feature its modality came from), its action or action chunk and
    mask, its weight, its source episode, frame and global index, its candidate row, its measurements and its
    classification.
  - Rejected, inadmissible and unselected candidates stay in the run's artifacts.
  - The run ID is recorded once, in `ccil/meta.json`. Version 1 holds one supervision run per overlay.
- **Provenance is checked before anything is written:**
  - the source's files must hash to the conversion manifest's;
  - the supervision's dataset must be the conversion's output, directly or through the `data.prepare` run that read
    it;
  - only identity representations are accepted.
- **Writes are atomic.** An existing destination is refused, and a failed export leaves nothing. `read_overlay`
  verifies every shard's hash and that the LeRobot portion is unchanged.

**Training (`ccil.lerobot.training`, needs torch).** `CCILTrainingDataset(LeRobotDataset(...))` is CCIL's
supervision as one dataset. It wraps a LeRobot dataset object that you construct; CCIL never imports LeRobot.
- **Its rows:** the demonstration frames (t ≤ T−2 of the overlay's training episodes) and every usable label, each
  with `ccil.weight`.
- **Sampling:** `sampler()` draws rows by weight, optionally balancing demonstrations against labels.
- **Stock trainers:** a stock trainer that ignores `ccil.weight` is not CCIL-correct.

**Temporal windows fail closed.**
- **Demonstration rows** are LeRobot's own items, with its ordinary windows.
- **An observation history** for a synthetic label is refused, until M5 defines a valid construction.
- **Future actions** for a synthetic label come only from its stored action chunk, never from neighbouring labels.
  Without a stored chunk, a future-action window is refused.
- **Single-frame policies** consume every label directly.

```bash
uv run --group evidence --extra lerobot python tools/m4_overlay/check.py --out <new-dir>
```

For each supported release, the evidence checks three things: a stock load that sees exactly the source, a
CCIL-aware load that sees every label and weight, and the temporal windows.

## The round trip (#29)

```bash
uv run --group evidence --extra lerobot python tools/m4_roundtrip/check.py --out <new-dir>
```

**The fixture,** [`fixtures/lerobot-roundtrip/`](fixtures/lerobot-roundtrip/), was written by LeRobot 0.6.1.
- **Contents:** a double integrator with a 2-d state, an 8×8 RGB image and a 1-d action, in four episodes of 12, 9,
  15 and 6 frames.
- **Safe to redistribute:** it is synthetic.
- **Checked by hash:** LeRobot's own writes are not byte-deterministic (a random fingerprint in the Parquet
  metadata, sampled image statistics), so the fixture is committed and checked by hash.

**The pipeline:** the fixture is converted (with the image named in `ignore`), labelled with 5-step chunks,
resolved and exported. It runs twice, and both runs must be identical. Then it runs once more with a support radius
that accepts nothing.

**Published** under [`fixtures/evidence/m4-roundtrip/`](fixtures/evidence/m4-roundtrip/):
- the report;
- the accepted and rejected tables, with every candidate's source frame, weight, reasons and measurements;
- the quality report;
- the transcripts of an unchanged LeRobot 0.6.1 consumer, which iterates the bundle and decodes its images.

CCIL's training dataset sees every label with state-only keys, and refuses a policy that reads the image.

## Compatibility: an evidence-backed matrix

A LeRobot release is listed as supported only when it passes **both
directions** of the compatibility workflow, and the list names exact patch
releases. Each release runs in its own Python 3.12 environment:

1. CCIL writes a state-only v3.0 dataset: several data shards, two state
   modalities, two tasks.
2. The release loads it and iterates it through its public dataset and
   DataLoader interface.
3. The release writes a fixture of known values, and CCIL reads it.
4. Both sides agree on:
   - episode boundaries, ordering, feature values, dtypes, shapes, timestamps
     and task mappings;
   - the stored statistics, exactly;
   - count, min, max, mean and std, against LeRobot's own computation on the
     same frames.
5. A small policy trains on CCIL's dataset through LeRobot's dataset, without
   rewriting it.

The workflow also checks refusals, and the install boundary: the extra
installs no LeRobot runtime, the bridge gives install guidance without the
extra, and the declared minimum pyarrow and huggingface_hub work on Python 3.10
and 3.12.

| Release | CCIL → LeRobot | LeRobot → CCIL | Training | Supported |
|---|---|---|---|---|
| 0.4.0 | pass | pass | pass | **yes** |
| 0.4.1 | pass | pass | pass | **yes** |
| 0.4.2 | pass | pass | pass | **yes** |
| 0.4.3 | pass | pass | pass | **yes** |
| 0.4.4 | pass | pass | pass | **yes** |
| 0.5.0 | pass | pass | pass | **yes** |
| 0.5.1 | pass | pass | pass | **yes** |
| 0.6.0 | pass | pass | pass | **yes** |
| 0.6.1 | pass | pass | pass | **yes** |
| 0.3.3 | fail | fail (writes v2.1, refused) | fail | no |

The same matrix on two hosts, each release in its own Python 3.12 environment with the versions it resolved
(recorded in the reports): macOS arm64 from commit `31f29ad`
([`report.json`](fixtures/evidence/m4-lerobot/report.json)) and Linux x86_64 from `fc78a4a`
([`report-siddh.json`](fixtures/evidence/m4-lerobot/report-siddh.json)).

The range grows only when a new release passes both directions. If a release
fails, the bridge is fixed or the release stays unlisted. We do not promise
compatibility with untested LeRobot commits, and we do not depend on private
LeRobot APIs.

## Evidence

```bash
uv run --extra lerobot python tools/m4_ingest/check.py --out <new-dir> [--hub]
```

[`fixtures/evidence/m4-ingest/report.json`](fixtures/evidence/m4-ingest/report.json) covers ingestion (#27):
- **Endings:** unknown by default; an explicit terminal and an explicit
  truncation; an early terminal flag, a non-boolean feature and a missing
  feature, each refused; the Push-T pattern.
- **Dispositions, identity and conversion:** every feature's disposition; the
  identity refusals and the identical-frames case; the conversion manifest,
  with the source unchanged and nothing written on refusal.
- **Compatibility:** a 1.1 file read by the 1.0 reader.
- **With `--hub`:** the real Push-T data.

The compatibility workflow for #26:

```bash
uv run --extra lerobot python tools/m4_lerobot/check.py --out <new-dir> [--hub]
```

[`fixtures/evidence/m4-lerobot/report.json`](fixtures/evidence/m4-lerobot/report.json)
records the matrix and the installed versions of each environment. With
`--hub`, it also fetches `lerobot/pusht_keypoints` at a recorded revision.
That dataset is refused as it is, because `next.reward`, `next.done` and
`next.success` have no CCIL slot. It reads once the caller names them in
`ignore`. Mapping terminal information is #27's job.
