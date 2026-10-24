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
  The staging directory and the destination are exchanged atomically, using
  `renameat2(RENAME_EXCHANGE)` on Linux and `renamex_np(RENAME_SWAP)` on macOS,
  so a valid run is at the destination at every instant. The old run, now under
  the staging name, is then removed. On a filesystem that can't exchange, the
  writer renames the old run to `.<run_id>.old-<pid>` and then moves staging
  into place. If the second rename fails, the old run is renamed back.
- **Exception:** the staging directory is removed, and any existing run is
  untouched.
- **Hard kill:** a writer killed outright can't clean up. Before any writer or
  pipeline stage looks at a destination, it recovers what dead writers left:
  - an `.old-<pid>` directory is restored if the destination is missing, and
    removed otherwise;
  - a staging directory, which validates as `incomplete`, is removed.

  A kill between the two fallback renames is the only point at which the
  destination is briefly absent, until this recovery runs.

## Validation

```bash
uv run ccil run validate RUN_DIR    # exit code 0 only when complete
```

| Status | Meaning |
|---|---|
| `complete` | Manifest version 2.x, `status: complete`, every listed file present with a matching size, SHA-256 and declared schema, no unlisted files, no symlinks, and a candidates `run_id` equal to the manifest's. |
| `incomplete` | The manifest is missing or not `complete`, a listed file is missing, or the directory is a staging directory. |
| `corrupt` | An unreadable or mistyped manifest or file entry, a listed path that isn't a canonical relative path inside the run, any symlink in the run, a size or hash mismatch, a schema mismatch, an unlisted file, or a candidates `run_id` mismatch. |
| `incompatible` | A manifest major version other than 2, or a component schema major version newer than this ccil reads. |

When several problems apply, the reported status is the first of
incompatible, corrupt, incomplete. All problems are listed.

Listed paths are parsed before any file is opened. A path must be relative,
`/`-separated, and free of empty, `.` and `..` segments, and no component on
the way to it may be a symlink. A manifest therefore can't make the validator
read or vouch for a file outside the run.

## Evidence

```bash
uv run --group evidence python tools/m2_runs/check.py --out <new-dir>
```

[`fixtures/evidence/m2-runs/report.json`](fixtures/evidence/m2-runs/report.json)
has 24 cases:

- **Analytic run:** written twice and byte-identical; accepted and rejected
  rows trace to the stored dataset.
- **`data prepare`** validates as complete, and a 1.0 manifest as incompatible.
- **Interruption:** an exception mid-write, and a SIGKILL mid-write.
- **Commit boundaries,** each in its own process:
  - a failed second rename leaves the old run in place;
  - a SIGKILL after the first fallback rename is restored by recovery;
  - a SIGKILL right after the atomic exchange leaves the new run complete.

  The next writer also recovers before checking whether the destination exists.
- **Fifteen damaged copies,** each classified correctly, including:
  - entries that traverse with `..`, are absolute, or are non-canonical;
  - a symlinked file or directory that points outside the run;
  - mistyped `files` mappings and entries, which give a diagnostic rather than a
    traceback.
