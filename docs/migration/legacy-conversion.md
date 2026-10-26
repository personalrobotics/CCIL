# Converting legacy inputs

Legacy `correct_il` inputs enter the canonical schemas through one visible,
validated, one-way boundary (#23). Conversion is required only for the
**retained** inputs: the Pendulum datasets and configurations
([migration contract](../migration-contract.md) §§3 and 7). The D4RL,
Meta-World, drone and F1 pickles are superseded or retired, and are not
converted.

## Retained Pendulum inputs: one command

```bash
uv run python tools/legacy_conversion/convert_pendulum.py --out converted/
```

It writes:

| Path | Contents |
|---|---|
| `converted/data/pendulum_{cont_100,disc_500}.npz` | Canonical `ccil.trajectories` 1.0 |
| `converted/config/pendulum_{cont,disc}.{corrected,historical}.yaml` | Canonical configurations |
| `converted/manifests/*.json` | One manifest per dataset and per configuration |
| `converted/conversion.json` | Every SHA-256: pickle, M1, canonical dataset, legacy and canonical configs |

## Datasets: two steps across the trust boundary

Legacy datasets are pickles, and unpickling can execute code.

1. **The trusted-input boundary, outside the `ccil` package.**
   `correct_il/convert_legacy_dataset.py` (`trajectory_io.convert_legacy_pickle`)
   unpickles a file only after checking its SHA-256 against
   [`docs/artifacts/manifest.json`](../artifacts/manifest.json) (`--shipped`
   or `--expect-sha256`), or after an explicit `--trust-legacy-pickle`
   acknowledgement. It writes the M1 `ccil-trajectories/1` npz. A tampered or
   unacknowledged pickle is refused before it is unpickled. The dependency
   gate proves that `ccil` itself never unpickles.
2. **`ccil data convert M1.npz OUT.npz --manifest M.json`** writes canonical
   `ccil.trajectories` 1.0 atomically. It reads the file back and refuses
   unless the result is exact.
   - **Exit codes:** 0 when written. 2 when the input is not M1 or is
     unreadable; nothing is written.
   - **Provenance** in the file's `meta.dataset` (`converted_from`): the M1
     file's SHA-256, and from the M1 meta the legacy pickle's name and
     SHA-256, the trust mode and the converter version.

**Field map:**

| Legacy | Canonical |
|---|---|
| `observations` | `obs/state`, dtype kept |
| `actions` | `actions`, dtype kept |
| Episode *i* | `episode_id` `"i"`, `terminated = False` (legacy recorded no termination), no timestamps |

The dataset manifest records:
- the source and output SHA-256;
- the legacy pickle and trust mode;
- the field map;
- the episode, frame and transition counts;
- the dtypes and dimensions.

`data.prepare` still reads M1 files directly. The explicit conversion exists
so the canonical artifact can be inspected and hashed before any model is
trained.

## Configurations: `ccil config convert`

```bash
uv run ccil config convert config/pendulum_disc.yml --out pendulum_disc.yaml --manifest pendulum_disc.json \
    [--fidelity corrected|historical] [--dataset-map data/old.pkl=data/new.npz]
```

**Dispositions.** The manifest gives a disposition for every legacy key:

| Disposition | Meaning |
|---|---|
| `renamed` | The value moves unchanged to its canonical key, for example `aug.num_labels` → `generation.labels_per_transition`. |
| `derived` | Computed from legacy keys: `policy.naive` + `policy.noise_bc` → `policy.method` (+ `policy.noise_scale`); `no_gpu` → `compute.device`; `eval.noise` → `evaluation.noise_levels`; `data.pkl` → `data.path`. |
| `dropped` | The key never affected behavior, for example `data.num_traj` (a `'...'` placeholder) or `output.folder_suffix`. The reason is recorded. |
| `unsupported`, `unknown` | Conversion fails. |

**Never substitute a supplied value.** Every supplied value must appear
unchanged in the resolved canonical configuration, or conversion fails.

**Path conventions.**
- `data.pkl: X.pkl` becomes `data.path: X.npz`, the converted dataset, unless
  `--dataset-map` names it.
- A `data.pkl` that is not a `.pkl` and has no map is ambiguous, and fails.
- `output.location` becomes `run.output_dir`.

**Failures.** Unknown, duplicate, ambiguous (`policy.naive: false` with
`noise_bc > 0`), incompatible (`data.data_frac`, `state_encoder`) and
mistyped settings exit 2. Each diagnostic names the legacy key, and nothing
is written.

**Output.**
- A `config_version: 1` YAML, with a header recording the source SHA-256 and
  the fidelity.
- It is written only after it resolves.
- The manifest holds the resolved configuration and its hash.

### Fidelity: what legacy did without saying so

Two legacy behaviors are written in no legacy configuration, and the
canonical defaults differ:

| Key | Legacy behavior | Canonical default |
|---|---|---|
| `policy.exclude_validation_episodes` | `false`: BC trained on every expert transition. `dynamics.validation_fraction` (#13) only held out episodes for evaluating the dynamics model. | `true`: the harness also holds the split's validation episodes out of policy training. |
| `policy.noise_model` | `legacy_uniform_half_batch`: `noise_bc · U[0, 1)`, always positive | `symmetric_uniform_half_batch` (#22) |

- **`historical`** writes the legacy behavior.
- **`corrected`** (the default) keeps the canonical defaults.
- **The manifest** (`implicit_legacy_behavior`) records both values, the one
  written, and why, for either fidelity.

**Non-configurable corrections** from #21 and #22 are listed too:
- eval-mode prediction;
- the Lipschitz units and Jacobian;
- slack pairing;
- counter-keyed NoiseBC;
- the chunk alignment.

No flag restores those. #24's faithful-port arm (commit `50cfc535`)
isolates them.

The shipped recipes `pendulum_cont` and `pendulum_disc` resolve exactly as
the `corrected` conversions do. The evidence checks this.

## Legacy entry points

Each legacy script took the configuration path as its only positional
argument:

| Legacy | Now |
|---|---|
| `python correct_il/train_dynamics_model.py CONFIG` | `ccil labels generate --config CANON` (stage `transition.fit`) |
| `python correct_il/gen_aug_label.py CONFIG` | `ccil labels generate --config CANON` (`labels.generate`, `labels.certify`) |
| `python correct_il/train_bc_policy.py CONFIG` | `ccil policy train --config CANON` (experimental harness, [policy-harness.md](../policy-harness.md)) |
| `python correct_il/eval_bc_policy.py CONFIG` | Simulator evaluation: #24 and M7 |

## Evidence

```bash
uv run --group evidence python tools/legacy_conversion/check.py --out <new-dir>
```

The report is
[`../fixtures/evidence/legacy-conversion/report.json`](../fixtures/evidence/legacy-conversion/report.json),
with `report-siddh.json` from the CUDA host. The manifests and converted
configurations are under `artifacts/`; datasets are recorded by SHA-256 only.
The report shows:
- **Datasets.** Both datasets equal the unpickled legacy arrays exactly, with
  their hash chain, and a second conversion is byte-identical.
- **Configurations.** Every legacy key has a disposition, and supplied values
  are preserved, as checked independently from the legacy files.
- **Fidelities.** The two fidelities differ exactly in the implicit
  behaviors.
- **Recipes.** The recipes match the corrected conversions.
- **Malformed inputs.** Twelve cases fail, each naming its problem. Nothing
  is written, and no pickle executes.
- **Inspection.** The converted artifacts prepare complete runs.
- **Hold-out flag.** It controls whether the harness holds validation
  episodes out of policy training.
