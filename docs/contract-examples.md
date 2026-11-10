# Contract examples

Three worked cases of [the user contract](contract.md): direct NumPy input, a
LeRobot dataset, and an external consumer. Each gives the input, the command,
the resolved configuration, and the output inventory.

The inventories come from running the CLI on CPU.

## 1. Direct NumPy input

**Input.** Eight episodes of a double integrator, x = (position, velocity),
under a linear feedback controller, with 41 frames each at a 0.05 s action
period. Any array source works; only the `ccil.trajectories` layout matters.

```python
import numpy as np
from ccil.schema import Episode, TrajectoryDataset, write_trajectories

rng, dt, episodes = np.random.default_rng(0), 0.05, []
for i in range(8):
    x = np.zeros((41, 2), np.float32); x[0] = rng.normal(0, 1, 2)
    u = np.zeros((41, 1), np.float32)          # u[t] governs x[t] -> x[t+1]; the last action is unused
    for t in range(40):
        u[t] = -1.0 * x[t, 0] - 1.5 * x[t, 1]
        x[t + 1] = x[t] + dt * np.array([x[t, 1], u[t, 0]])
    episodes.append(Episode(f"ep{i}", {"state": x}, u, timestamps=np.arange(41) * dt, termination_known=True))
write_trajectories(TrajectoryDataset(episodes), "demos.npz")
```

**Command.**

```bash
ccil generate --recipe analytic_double_integrator --set data.path=demos.npz --set compute.device=cpu
ccil curate runs/generate-result.json     # without a certificate.profile: every accepted row
```

In Python: `ccil.generate(ccil.resolve_config("analytic_double_integrator", overrides={"data.path": "demos.npz", "compute.device": "cpu"}))`.

**Resolved configuration** (`ccil config show`, values that are not defaults):

| Key | Value | Source |
|---|---|---|
| `data.path` | `demos.npz` | cli |
| `compute.device` | `cpu` | cli |
| `run.seed` | 20261023 | recipe |
| `data.validation_fraction` | 0.5 | recipe |
| `representation.kind`, `.modality` | `identity`, `state` | recipe |
| `transition_model.kind` | `linear` | recipe |
| `generation.method` | `backward_euler_fast` | recipe |
| `generation.support_radius` | 0.01 | recipe |
| `generation.recovery_steps` | `1` | the recipe, written before `config_version` 3; a new configuration defaults to `auto` |

**Output inventory.**

```
runs/
  data-prepare-<id>/data/dataset.npz        ccil.trajectories
  data-prepare-<id>/data/split.json         ccil.split            4 train, 4 validation episodes
  transition-fit-<id>/checkpoints/          transition model      held-out max one-step error 6.7e-8
  labels-diagnose-<id>/diagnosis/diagnosis.json  ccil.diagnosis 1.0  per-M gates, admitted M, dt_ctrl 0.05 s, H 0.05 s
  labels-generate-<id>/labels/candidates.npz  ccil.candidates 2.x 320 rows, 46 accepted, 274 out_of_support
  labels-certify-<id>/labels/certificates.npz  ccil.certificates 2.0  with certificate.profile
  generate-result.json                      ccil.generate-result 1.0  what curate reads
  labels-curate-<id>/labels/selection.npz   ccil.selection 1.0    (ccil curate) the accepted rows, without certificates
  curate-<id>.json                          ccil.curate-result 1.0
  each run directory: manifest.json, config.json, metrics.json, stage.json, logs/run.log
```

Pointwise view (`build_supervision`; `ccil inspect` on the curate result): 160 training demonstration rows
plus 46 synthetic rows, 206 in all. The 160 held-out demonstration rows and
any labels from held-out episodes are excluded and counted.

## 2. A LeRobot dataset

**Input.** The LeRobot v3.0 fixture
[`fixtures/lerobot-roundtrip`](fixtures/lerobot-roundtrip/): 4 episodes, 42
frames at 10 FPS, a 2-D `observation.state`, a 1-D `action`, and an 8×8 RGB
`observation.image`.

**Conversion** (`ccil[lerobot]`):

```python
from ccil.lerobot import v3
v3.convert("docs/fixtures/lerobot-roundtrip", "demos.npz", "conversion.json", ignore=("observation.image",))
```

Without `ignore`, the conversion is refused: image features are not supported
until level 4. The manifest `conversion.json` (`ccil.lerobot.conversion` 1.0)
records each feature's disposition (`action` → `actions`, `observation.state`
→ `observations[state]`, `timestamp` → `timestamps`, `observation.image`
ignored by the caller), the indices it checked, all 4 endings as unknown, the
source format and FPS, and the SHA-256 of every source file and of the output.

**Command and configuration.** The same as example 1, with
`data.path=demos.npz`. `dt_ctrl` comes from the timestamps (0.1 s).

**Output inventory.** As in example 1: 38 candidate rows, 1 accepted,
37 `out_of_support`. Pointwise view: 16 training demonstration rows and 1
synthetic row. Exported as an overlay (`ccil.lerobot.overlay.export_overlay`):

```
overlay/
  data/  meta/                              an unchanged copy of the source dataset
  ccil/meta.json                            ccil.lerobot.overlay 1.0: inputs, feature map, run ID, shard hashes
  ccil/labels/chunk-000/file-000.parquet    the synthetic rows, with weight and evidence class
```

A stock LeRobot reader sees exactly the source episodes. A CCIL-aware reader
sees every usable label and its weight.

## 3. An external consumer

A trainer that is not CCIL reads the artifacts with numpy and `ccil.schema`,
and nothing else. It needs no CCIL trainer, configuration or benchmark code.

```python
import json
import numpy as np
from ccil.schema import read_candidates

c = read_candidates("labels/candidates.npz")
train = json.loads(open("data/split.json").read())["train"]
usable = c.accepted & np.isin(c.source_episode_id, train)   # with a selection: & selection.mask
rows = np.flatnonzero(usable)
obs, act, w = c.observations["state"][rows], c.actions[rows], c.weight[rows]
```

On example 1 this gives 46 rows with observations of shape (46, 2), actions
(46, 1), and a weight total of 46.0, the same rows `build_supervision` returns.

**With a selection** (`labels/selection.npz`): the consumer also reads the
selection file, checks that its recorded candidates SHA-256 is this candidates
file's, and multiplies by its mask and weights. Before relying on the rows, it
checks the evidence class it needs: an empirical score is never a bound.

**What it may not do:**
- train on rejected or unselected rows, whatever their weight;
- use labels from held-out episodes;
- take only the first action of a chunked label without declaring it;
- treat a recovery-segment claim as a claim about one row.
