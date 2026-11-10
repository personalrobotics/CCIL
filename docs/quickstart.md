# Quickstart

From demonstrations to curated, consumable supervision, on CPU, in about a
minute. What each command and artifact means is defined in
[the user contract](contract.md); this page only walks the path. Save each
code block below under the file name given above it, in one empty directory.

## Install

```bash
pip install "ccil @ git+https://github.com/personalrobotics/CCIL"
```

The core needs only numpy and PyYAML. Policy training is yours (or
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks)):
CCIL hands you rows.

## 1. Demonstrations

A dataset is a list of episodes of observations and actions, written as a
`ccil.trajectories` file. Here, six episodes of a double integrator under a
noisy feedback controller. Save as `demos.py`:

```python quickstart demos.py
import numpy as np
from ccil.schema import Episode, TrajectoryDataset, write_trajectories

rng = np.random.default_rng(0)
A, B = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([0.0, 0.1])   # a double integrator, 0.1 s steps
episodes = []
for i in range(6):
    s, states, actions = rng.uniform(-1, 1, 2), [], []
    for t in range(40):
        a = -np.array([1.0, 1.5]) @ s + 0.1 * rng.normal()           # the demonstrator's feedback
        states.append(s); actions.append([a]); s = A @ s + B * a
    episodes.append(Episode(f"ep{i}", {"state": np.array(states, np.float32)}, np.array(actions, np.float32)))
write_trajectories(TrajectoryDataset(episodes), "demos.npz")
```

A [LeRobot](lerobot.md) dataset converts to the same file:
`ccil.lerobot.v3.convert(source, "demos.npz", "conversion.json")`.

## 2. Configuration

Save as `ccil.yaml`:

```yaml quickstart ccil.yaml
config_version: 4
data: {path: demos.npz}
run: {output_dir: runs}
compute: {device: cpu}
transition_model: {kind: linear}             # learned from the demonstrations
generation: {method: noisy_action, labels_per_transition: 4, action_noise_std: 1.0}
certificate:                                 # bound each label's error from declared assumptions
  profile: disturbed_action
  true_state_lipschitz: 1.0
  true_action_lipschitz: 0.2
  demonstration_residual_bound: 0.0
```

Every key is in the [parameter reference](config/parameters.md). The
constants are your assumptions, not measurements, so the bounds they give are
*conditional* on them. To use a model you trained elsewhere, set
`transition_model.checkpoint`.

## 3. Generate

```bash quickstart
python demos.py
ccil generate --config ccil.yaml
```

It splits the episodes into training and held-out ones (one of the six is
held out, `data.validation_fraction`), fits the dynamics model, diagnoses
whether labels are worth generating (from the data alone), generates
candidate labels from the training episodes, and bounds each one's error.
It ends with a summary:

```
candidates     runs/labels-generate-<id>/labels/candidates.npz: 936 rows, 936 accepted
rejected       none
evidence       conditional 936
admitted       recovery length 1
result         runs/generate-result.json
next           ccil curate runs/generate-result.json --set curate.tolerance=<largest acceptable label error>
```

- **Recovery length 1** means one-step labels: a perturbed state and the
  action that returns it to the demonstration in one step.
- **Conditional** evidence is an error bound that holds if your declared
  constants are true; with exact reference dynamics it would be *certified*.
- If nothing is admitted, the summary says which check failed and what to
  do: move labels further, collect recovery data, improve the model, or
  accept that CCIL has nothing to add here. That is a result, not an error.
- `runs/generate-result.json` always describes the latest Generate in
  `runs/`; each Generate also leaves an immutable `runs/generate-<id>.json`.
  To keep experiments apart, give each its own `run.output_dir`.

## 4. Inspect and curate

```bash quickstart
ccil inspect runs/generate-result.json
ccil curate runs/generate-result.json --set curate.tolerance=0.1
ccil curate runs/generate-result.json --set curate.tolerance=0.3
```

`inspect` shows the configuration with each value's source, provenance and
hashes, the diagnosis, the distribution of error bounds, and the rejections.
A label's error bound is in the units of `curate.tolerance` (scaled state:
see the [parameter reference](config/parameters.md)); `curate.tolerance=t`
keeps the labels whose bound is at most t, so pick tolerances from the bound
quantiles `inspect` prints, or use `curate.quantile` to keep a fraction.
Without either, every admissible label is selected, and the summary says so.
Each `curate` writes one immutable selection of the same candidates: try as
many as you like, nothing is regenerated. `ccil inspect
runs/curate-<id>.json` prints the Python that loads its rows.

## 5. Use the rows

`build_supervision` returns the usable rows of a selection: the training
demonstrations plus the selected labels of the training episodes, with their
weights (the held-out episode contributes none). A small pilot compares BC on
the demonstrations alone with BC on each selection of the latest Generate,
using your learner and your evaluation (here, a linear policy rolled out on
the true system). Save as `pilot.py`:

```python quickstart pilot.py
import glob, json
from pathlib import Path
import numpy as np
from ccil.core import IdentityRepresentation
from ccil.supervision import build_supervision

def fit(sup):                                                       # your learner: here, a linear policy
    obs, act, w = sup.arrays()
    X = np.hstack([obs, np.ones((len(obs), 1))]) * np.sqrt(w)[:, None]
    return np.linalg.lstsq(X, act * np.sqrt(w)[:, None], rcond=None)[0]

def rollout(W):                                                     # your evaluation: here, the true system
    A, B, rng = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([0.0, 0.1]), np.random.default_rng(1)
    s = rng.uniform(-2, 2, (200, 2))                                # starts beyond the demonstrations
    for _ in range(40):
        s = s @ A.T + np.outer(np.hstack([s, np.ones((len(s), 1))]) @ W, B)
    return float(np.linalg.norm(s, axis=1).mean())

rep = IdentityRepresentation("state")
latest = json.loads(Path("runs/generate-result.json").read_text())["artifacts"]["candidates"]["sha256"]
for f in sorted(glob.glob("runs/curate-*.json")):
    cur = json.loads(Path(f).read_text())
    gen = json.loads(Path("runs", cur["generated"]["path"]).read_text())
    if gen["artifacts"]["candidates"]["sha256"] != latest:
        continue                                                    # a selection of an earlier Generate
    p = {k: f"runs/{v['path']}" for k, v in {**gen["artifacts"], **cur["artifacts"]}.items()}
    bc = build_supervision(p["dataset"], "demonstrations", representation=rep, split=p["split"])
    cc = build_supervision(p["dataset"], "demonstrations_plus_selection", representation=rep, split=p["split"],
                           candidates=p["candidates"], certificates=p["certificates"], selection=p["selection"])
    policy = {k: v for k, v in cur["counts"]["policy"].items() if k in ("tolerance", "quantile") and v is not None}
    band = ", ".join(f"{k} {v}" for k, v in policy.items()) or "every admissible label"
    print(f"{band}: BC {rollout(fit(bc)):.4f} "
          f"({len(bc.arrays()[0])} rows), CCIL {rollout(fit(cc)):.4f} ({len(cc.arrays()[0])} rows)")
```

```bash quickstart
python pilot.py
```

Read the result as a development pilot, not a verdict. If one tolerance helps,
use that correction band; if none does, the next step is the diagnosis's:
better recovery data, a better dynamics model, a different learner, or no
evidence that CCIL helps this task. The data alone never certify that labels
improve a policy.

## Next

These pages are in the repository's `docs/` directory:

- [contract.md](contract.md): commands, Python entry points (`ccil.generate`,
  `ccil.curate`, `ccil.inspect`), exit codes, and every artifact.
- [schemas/reference.md](schemas/reference.md): the artifact formats.
- [supervision.md](supervision.md): which rows a consumer may use, and their weights.
- [certificates.md](certificates.md): profiles, constants, and calibration.
