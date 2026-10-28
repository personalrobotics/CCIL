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
| — | `terminated`: v3.0 records no terminal flag, so every episode reads as truncated, and the dataset's meta says so |

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
| *filled from the evidence report* | | | | |

The range grows only when a new release passes both directions. If a release
fails, the bridge is fixed or the release stays unlisted. We do not promise
compatibility with untested LeRobot commits, and we do not depend on private
LeRobot APIs.

## Evidence

```bash
uv run --extra lerobot python tools/m4_lerobot/check.py --out <new-dir> [--hub]
```

[`fixtures/evidence/m4-lerobot/report.json`](fixtures/evidence/m4-lerobot/report.json)
records the matrix and the installed versions of each environment. With
`--hub`, it also fetches and reads `lerobot/pusht_keypoints` at a recorded
revision.
