# PushT: from a LeRobot dataset to an unchanged LeRobot trainer

One-step CCIL on `lerobot/pusht_keypoints`, handed to a stock LeRobot
training run. CCIL writes one augmented LeRobot dataset; training and
evaluation are LeRobot's own commands, unchanged. What each command and
artifact means is in [the user contract](contract.md); the general path is
in [the quickstart](quickstart.md). Whether the labels help a policy is a
separate, matched study in
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks).

## 1. The dataset

```bash
pip install "ccil[lerobot,torch] @ git+https://github.com/personalrobotics/CCIL"
python -c '
from ccil.lerobot import convert, fetch
path, sha = fetch("lerobot/pusht_keypoints", revision="ace8c161a68bc025c21a5f29f85b86a9a2c5e64b", local_dir="hub")
convert(path, "pusht.npz", "conversion.json", preserve=["next.reward", "next.done", "next.success"])'
```

The state is two LeRobot features: `observation.state` (the agent, 2) and
`observation.environment_state` (the block's keypoints, 16). The `next.*`
columns are kept as source metadata, never as labels or endings
([lerobot.md](lerobot.md)).

## 2. Configuration

Save as `pusht.yaml`:

```yaml
config_version: 4
data: {path: pusht.npz, action_low: [0.0, 0.0], action_high: [512.0, 512.0]}
run: {output_dir: runs}
compute: {device: cpu}
representation: {modality: state+state.environment_state}   # both features, concatenated in this order
transition_model:                      # a residual MLP with spectrally normalized layers
  kind: mlp
  fit_episodes: train                  # never fit on the held-out episodes: their error is the held-out error
  trainer_version: "1"
  hidden_layers: [256, 256]
  activation: relu
  lipschitz_type: spectral_normalization
  lipschitz_constraint: 3.0
  epochs: 50
  batch_size: 256
generation:
  method: noisy_action
  labels_per_transition: 4
  action_noise_scale: training_std     # action_noise_std in training-action standard deviations
  action_noise_std: 0.1
  recovery_steps: "1"                  # one-step labels
```

`state+state.environment_state` is an ordered list: the model sees the 18
coordinates in that order, and its checkpoint refuses another order. The
noise is resolved per action dimension from the training transitions and
recorded with the labels.

## 3. Generate, inspect, curate

```bash
ccil generate --config pusht.yaml
ccil inspect runs/generate-result.json
ccil curate runs/generate-result.json
```

- **The model.** It is fitted on the training episodes only
  (`fit_episodes: train`), so `ccil inspect runs/transition-fit-<id>` shows a
  true held-out error, and the held-out episodes, which a downstream study
  may reserve for checkpoint selection, never inform the labels. It also
  shows `lipschitz_measured`: the weight-based Lipschitz bound of the
  learned residual map, recomputed from the saved checkpoint with its
  normalization, output scaling and mask. The spectral constraint is a
  training method; this number is a measurement, not a certificate.
- **The labels.** The summary and `inspect` report the diagnosis, the
  rejections, and each label's distance from its source. Labels need not
  leave the demonstrations' global support (τ_demo); how far they move is
  reported, not required.
- **The selection.** Without a certificate profile every accepted label is
  selected with weight 1, which is what an unchanged trainer can consume.

## 4. Export for LeRobot

```bash
ccil export lerobot runs/curate-<id>.json --conversion conversion.json --out pusht-ccil \
    --revision ace8c161a68bc025c21a5f29f85b86a9a2c5e64b
ccil inspect pusht-ccil
```

`pusht-ccil` is one LeRobot v3.0 dataset:
- the training episodes, frame for frame as converted; the held-out episodes
  are not written;
- one one-frame episode per selected label, its observation split back into
  the two features and its action;
- on every frame, `ccil.synthetic`, `ccil.source_episode`,
  `ccil.source_frame`, `ccil.source_index` and `ccil.candidate_row`;
- `ccil/export.json`: the inputs by hash, the episode split, the counts,
  the hash of the labels' source indices, and every file's SHA-256.

Only unit weights export: a stock trainer reads no row weight.

## 5. Train and evaluate with LeRobot, unchanged

A one-frame episode has no real history or future, so the policy must read
one observation and predict one action. With LeRobot 0.6.1:

```bash
lerobot-train --dataset.repo_id=local/pusht-ccil --dataset.root=pusht-ccil \
    --policy.type=act --policy.n_obs_steps=1 --policy.chunk_size=1 --policy.n_action_steps=1 \
    --policy.push_to_hub=false --output_dir=outputs/act-ccil
lerobot-eval --policy.path=outputs/act-ccil/checkpoints/last/pretrained_model \
    --env.type=pusht --env.obs_type=environment_state_agent_pos --eval.n_episodes=100
```

ACT then trains on each label exactly as on a demonstration frame: its
action window is the current action only, so nothing is padded into the
loss ([evidence](fixtures/evidence/m4-export/report.json)). A policy with
observation history or longer action chunks (the published diffusion
recipe, for example) would see fabricated context around every label and
is not a correct consumer of this export.
