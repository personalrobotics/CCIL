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
| `stage.json` | Pipeline stages only: the identity the stage's run ID hashes, which decides reuse. See [`pipelines.md`](pipelines.md). |

A workflow writes only the components it produces.

## Manifest

| Field | Contents |
|---|---|
| `run_id`, `workflow`, `status` | `status` is `complete` once the run has been written and validated. |
| `files` | Every durable component by relative path: SHA-256, size, and the `schema`/`version` it declares. |
| `inputs` | External inputs (datasets, checkpoints): path and SHA-256. |
| `seeds` | Every `*.seed` value from the configuration. |
| `config` | Pointer to `config.json` and its `config_hash`. |
| `code` | The ccil version, plus the git commit, dirty flag and a hash of `git status` when the source is a checkout (otherwise `null`). Captured once per process. |
| `environment` | Python, platform, and numpy/pyyaml/torch versions. Devices are listed only when the run itself imported torch. |

The manifest has no timestamps, so identical inputs reproduce a run directory
byte for byte on the same environment.

## Code state and content hashes

A process captures its code state once, before the work it covers: the
`code` fields above, and the hash of every ccil source file that decides
stage reuse. The CLI captures the git state when a command starts, and
`RunWriter` captures both when it is entered, so a direct writer workflow is
covered too. The state is then checked again when work is finalized:
- **Before every commit,** `RunWriter` rehashes the ccil source. If it changed,
  the run is not committed.
- **When a command ends,** the CLI rechecks the git state as well.

On any difference, the command exits 3 and its unit is not finalized. An
already-dirty tree is covered too: the source hash sees an edit that the dirty
flag cannot.

Each hash of a pipeline-owned artifact is computed once per process. Hashes
are cached only for files that `RunWriter` committed, or that matched a
complete run's manifest when it was validated. A cached hash is used only
while the file's resolved path, device, inode, size and nanosecond mtime are
unchanged. User inputs, such as datasets and checkpoints passed in by path,
are hashed in full every time.

The cache's threat model:
- **Covered:** an atomic replacement, which changes the inode, and an in-place
  edit that changes the mtime. Both are rehashed.
- **Not covered:** rewriting a pipeline-owned artifact in place while keeping
  its size, inode and nanosecond mtime. These artifacts are write-once and are
  only ever replaced atomically.
- **Strict mode:** `CCIL_PROVENANCE_STRICT=1` turns the cache off. Use it for
  adversarial or archival verification. It produces the same records.

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
uv run --group evidence python tools/m3_provenance/check.py --out <new-dir>
```

Reports: [`m2-runs/report.json`](fixtures/evidence/m2-runs/report.json), [`m3-provenance/report.json`](fixtures/evidence/m3-provenance/report.json).
Each case, and the contract it protects, is listed in the docstrings of [`tools/m2_runs`](../tools/m2_runs/), [`tools/m3_provenance`](../tools/m3_provenance/).
