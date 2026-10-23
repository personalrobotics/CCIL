# Legacy serialization trust boundary

This covers how CCIL handles pickle files (#15). **Loading a pickle can execute
arbitrary code.** Checking a file's structure *after* loading it says nothing
about whether it was safe to load.

## Trust model

| Path | Reads | Pickle? |
|---|---|---|
| Training and label generation (`utils.load_trajectories`) | `ccil-trajectories/1` `.npz` via `np.load(allow_pickle=False)` | **Never.** A `.pkl` path is refused with instructions to convert it. |
| Policy training (`load_demo_for_policy`) | `aug_data.npz`, written by label generation | Never |
| Evaluation and tabulation | `rewards*.npy`, `success*.npy` | Never |
| Dynamics checkpoints (#12) | `torch.load(weights_only=True)` | Never |
| **`correct_il/convert_legacy_dataset.py`** | legacy `.pkl` | **Only here**, and only after a trust decision |

A legacy pickle is unpickled only when one of these holds:
1. **`--shipped` or `--expect-sha256 HEX`:** the file's SHA-256 matches a known
   value, checked **before** unpickling. The shipped datasets are verified
   against [`docs/artifacts/manifest.json`](../artifacts/manifest.json), so a
   tampered file is refused unopened.
2. **`--trust-legacy-pickle`:** the caller explicitly acknowledges the file is
   trusted.

After loading, the data must pass every #14 invariant before anything is
written. Writes are atomic, so a failed conversion leaves any existing
destination untouched.

## Format `ccil-trajectories/1`

This is a single `.npz` file:

| Array | Contents |
|---|---|
| `observations` | (Σ T_i, state_dim) |
| `actions` | (Σ T_i, act_dim) |
| `episode_lengths` | (n_episodes,) int64 |
| `meta` | A JSON string: `format`, `converter_version`, `source` (name, sha256, bytes) and `trust` (`sha256` or `acknowledged`) |

The loader rejects foreign files and lengths that don't add up, and it
re-validates every invariant on each load. The arrays keep their original
dtype.

## Usage

```bash
python correct_il/convert_legacy_dataset.py --shipped   # writes data/*.npz (gitignored)
```

The converted files are derived and reproducible, so they aren't committed.
The source pickles and their hashes are the recorded artifacts.

## Not supported

- **Legacy `dynamics.pkl` checkpoints.** They're pickled whole objects that
  carry CUDA tensors and closures, and they lack the model configuration
  needed to rebuild them reliably. Retrain instead; dynamics training is cheap.
- **The retired F1 tooling** in `correct_il/envs/f1/` still uses pickle. It is
  out of scope under the migration contract, section 3.

## Evidence

```bash
uv run --no-project --python 3.11 --with-requirements tools/legacy_serialization/requirements.txt \
  python tools/legacy_serialization/check.py --out <new-dir>
```

[`evidence/legacy-serialization/report.json`](evidence/legacy-serialization/report.json)
has 8 cases:

1. **Shipped datasets.** All 15 convert after SHA-256 verification and
   round-trip **exactly** (values and dtypes), with their provenance recorded.
2. **Tampered pickle.** A malicious pickle whose hash doesn't match is refused
   before unpickling; its marker file is never created.
3. **No hash and no acknowledgement.** The file is refused.
4. **Acknowledged but malformed pickle.** It fails `length_mismatch`, and the
   existing destination stays byte-identical.
5. **Foreign or inconsistent `.npz`.** It is rejected.
6. **Normal entry point.** `train_dynamics_model.py` refuses a `.pkl` path, and
   the malicious pickle is never executed.
7. **Converted data end to end.** Pendulum data trains and labels end to end,
   writing `aug_data.npz` and no pickle files.
8. **Static check.** The only `pickle.load` on input data in `correct_il/`,
   apart from the retired F1 code, is in `trajectory_io.convert_legacy_pickle`.

As a positive control, a plain `pickle.load` of the same malicious payload
does create the marker, so cases 2, 3 and 6 show real prevention.
