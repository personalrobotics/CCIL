# Run directories

Every `ccil` workflow writes one run directory (#18). The implementation is in
[`src/ccil/runs/`](../src/ccil/runs/).

## Layout

| Path | Contents |
|---|---|
| `manifest.json` | `ccil.run-manifest` 2.0. See below. |
| `config.json` | The resolved configuration (`ccil.config`), with every value's source. |
| `metrics.json` | `ccil.metrics` 1.0: `{"values": {...}}`. |
| `data/dataset.npz`, `data/split.json` | `ccil.trajectories` and `ccil.split`. |
| `labels/candidates.npz` | `ccil.candidates`: accepted and rejected rows in one file, sharing `run_id` and `(episode_id, frame)` source IDs. Accepted labels are the `accepted` rows. |
| `checkpoints/<name>.pt` | Model checkpoints. |
| `plots/<name>.png` | Figures. |
| `logs/run.log` | Plain log lines, with no timestamps. |

A workflow writes only the components it produces.

## Manifest

| Field | Contents |
|---|---|
| `run_id`, `workflow`, `status` | `status` is `complete` once the run has been written and validated. |
| `files` | Every durable component by relative path: SHA-256, size, and the `schema`/`version` it declares. |
| `inputs` | External inputs (datasets, checkpoints): path and SHA-256. |
| `seeds` | Every `*.seed` value from the configuration. |
| `config` | Pointer to `config.json` and its `config_hash`. |
| `code` | The ccil version, plus the git commit and dirty flag when the source is a checkout (otherwise `null`). |
| `environment` | Python, platform, and numpy/pyyaml/torch versions. Devices are listed only when the run itself imported torch. |

The manifest has no timestamps, so identical inputs reproduce a run directory
byte for byte on the same environment.

## Atomic writes

`RunWriter` writes components into `.<run_id>.tmp-<pid>`, next to the
destination.

- **Success:** it adds the manifest, validates the staged run, and renames it
  into place.
- **Replacing a run:** an existing run is replaced only with `run.overwrite=true`.
  The old run is renamed aside first, then removed.
- **Exception:** the staging directory is removed, and any existing run is
  untouched.
- **Hard kill:** a writer killed outright can't clean up. The destination is
  still untouched, the staging directory validates as `incomplete`, and the next
  writer for that run ID removes staging directories whose process is gone.

## Validation

```bash
uv run ccil run validate RUN_DIR    # exit code 0 only when complete
```

| Status | Meaning |
|---|---|
| `complete` | Manifest version 2.x, `status: complete`, every listed file present with a matching size, SHA-256 and declared schema, no unlisted files, and a candidates `run_id` equal to the manifest's. |
| `incomplete` | The manifest is missing or not `complete`, a listed file is missing, or the directory is a staging directory. |
| `corrupt` | An unreadable manifest, a size or hash mismatch, a schema mismatch, an unlisted file, or a candidates `run_id` mismatch. |
| `incompatible` | A manifest major version other than 2, or a component schema major version newer than this ccil reads. |

When several problems apply, the reported status is the first of
incompatible, corrupt, incomplete. All problems are listed.

## Evidence

```bash
uv run --group evidence python tools/m2_runs/check.py --out <new-dir>
```

[`fixtures/evidence/m2-runs/report.json`](fixtures/evidence/m2-runs/report.json)
has 12 cases:

- **Analytic run:** written twice and byte-identical; accepted and rejected
  rows trace to the stored dataset.
- **`data prepare`** validates as complete, and a 1.0 manifest as incompatible.
- **Interruption:** an exception mid-write, and a SIGKILL mid-write.
- **Seven damaged copies,** each classified correctly.
