"""Run inside a LeRobot environment (never imported by ccil): stock LeRobotDataset and ACT on an export.

    python lerobot_side.py EXPORT_DIR INDICES_JSON OUT_JSON

ACT with one observation and a one-step action chunk (n_obs_steps 1, chunk_size 1, n_action_steps 1), its delta
timestamps resolved by LeRobot from the policy config. For each requested global index, the item LeRobot returns
(observations, action, action_is_pad); then one training step of ACT on a batch of those items, through LeRobot's
own pre-processor (normalization from the dataset's stats).
"""
import json
import sys

import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

root, indices, out = sys.argv[1], json.loads(open(sys.argv[2]).read()), sys.argv[3]
torch.manual_seed(0)  # ACT's initialization, so the reported loss is reproducible
meta = LeRobotDatasetMetadata("local/ccil-export", root=root)
cfg = make_policy_config("act", n_obs_steps=1, chunk_size=1, n_action_steps=1, device="cpu", use_vae=False,
                         dim_model=32, dim_feedforward=64, n_heads=2, n_encoder_layers=1, n_decoder_layers=1,
                         push_to_hub=False)
delta = resolve_delta_timestamps(cfg, meta)
ds = LeRobotDataset("local/ccil-export", root=root, delta_timestamps=delta)
keys = ("observation.state", "observation.environment_state", "action", "action_is_pad")
items = {}
for i in indices:
    it = ds[int(i)]
    items[str(i)] = {k: it[k].tolist() for k in keys if k in it}
    items[str(i)].update({k: int(it[k]) for k in ("episode_index", "frame_index", "ccil.synthetic", "ccil.candidate_row")})
policy = make_policy(cfg, ds_meta=meta)
pre, _ = make_pre_post_processors(cfg, dataset_stats=meta.stats)
batch = torch.utils.data.default_collate([ds[int(i)] for i in indices])
policy.train()
loss, _ = policy.forward(pre(batch))
loss.backward()
grads = [p.grad for p in policy.parameters() if p.grad is not None]
json.dump({"delta_timestamps": delta, "frames": ds.num_frames, "episodes": ds.num_episodes, "items": items,
           "act": {"loss": float(loss.detach()), "finite_gradients": bool(all(torch.isfinite(g).all() for g in grads)),
                   "parameters_with_gradients": len(grads)},
           "input_features": sorted(policy.config.input_features), "output_features": sorted(policy.config.output_features)},
          open(out, "w"), indent=1)
