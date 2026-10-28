"""The LeRobot side of the compatibility workflow (#26). Runs inside one LeRobot release's own environment and
never imports ccil; tools/m4_lerobot/check.py drives it.

    python lerobot_side.py load DIR OUT.npz     # LeRobot loads DIR and iterates it with a DataLoader
    python lerobot_side.py write DIR            # LeRobot writes the deterministic fixture
    python lerobot_side.py train DIR OUT.json   # a small policy trained on DIR through LeRobot's dataset
"""
import json
import sys

import numpy as np
import torch
from lerobot.datasets import compute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FRAME_KEYS = ("index", "episode_index", "frame_index", "task_index", "timestamp")


def load(root, out):
    ds = LeRobotDataset("ccil/compat", root=root)
    loader = torch.utils.data.DataLoader(ds, batch_size=7, shuffle=False, num_workers=0)
    cols, tasks = {}, []
    for batch in loader:
        for k, v in batch.items():
            if k == "task":
                tasks += list(v)
            elif isinstance(v, torch.Tensor):
                cols.setdefault(k, []).append(v.numpy())
    arrays = {f"frame/{k}": np.concatenate(v) for k, v in cols.items()}
    arrays["frame/task"] = np.array(tasks)
    eps = ds.meta.episodes
    arrays["episodes/from"] = np.array([eps[i]["dataset_from_index"] for i in range(ds.num_episodes)])
    arrays["episodes/to"] = np.array([eps[i]["dataset_to_index"] for i in range(ds.num_episodes)])
    for feat, stats in ds.meta.stats.items():  # as LeRobot loaded them from the dataset
        for name, v in stats.items():
            arrays[f"stored_stats/{feat}/{name}"] = np.asarray(v)
    # LeRobot's own statistics of the same frames, per episode and aggregated as LeRobot aggregates them.
    feats = {k: v for k, v in ds.meta.features.items() if v["dtype"] not in ("string",)}
    per = []
    for i in range(ds.num_episodes):
        lo, hi = arrays["episodes/from"][i], arrays["episodes/to"][i]
        data = {k: arrays[f"frame/{k}"][lo:hi] for k in feats if f"frame/{k}" in arrays}
        per.append(compute_stats.compute_episode_stats(data, {k: feats[k] for k in data}))
    for feat, stats in compute_stats.aggregate_stats(per).items():
        for name, v in stats.items():
            arrays[f"lerobot_stats/{feat}/{name}"] = np.asarray(v)
    np.savez(out, **arrays)
    print(json.dumps({"frames": len(ds), "episodes": ds.num_episodes}))


def write(root):
    """3 episodes (5, 7, 4 frames) of known values, two tasks, at 10 fps: the same content in every release."""
    features = {"observation.state": {"dtype": "float32", "shape": (3,), "names": None},
                "observation.environment_state": {"dtype": "float32", "shape": (4,), "names": None},
                "action": {"dtype": "float32", "shape": (2,), "names": None}}
    ds = LeRobotDataset.create(repo_id="ccil/fixture", fps=10, root=root, features=features, use_videos=False)
    for e, length in enumerate((5, 7, 4)):
        for t in range(length):
            v = np.float32(e * 100 + t)
            ds.add_frame({"observation.state": np.array([v, v + 0.25, -v], np.float32),
                          "observation.environment_state": np.full(4, v / 7, np.float32),
                          "action": np.array([v / 3, 1 - v], np.float32), "task": f"task {e % 2}"})
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()


def train(root, out):
    """A small MLP policy, trained through LeRobot's dataset and DataLoader with mean/std normalization taken from
    the dataset's stored statistics: CCIL's output is used as written."""
    torch.manual_seed(0)
    ds = LeRobotDataset("ccil/compat", root=root)
    obs_keys = sorted(k for k in ds.meta.features if k.startswith("observation."))
    st = {k: (torch.tensor(ds.meta.stats[k]["mean"], dtype=torch.float32),
              torch.tensor(ds.meta.stats[k]["std"], dtype=torch.float32).clamp_min(1e-6))
          for k in (*obs_keys, "action")}
    dim = sum(ds.meta.features[k]["shape"][0] for k in obs_keys)
    adim = ds.meta.features["action"]["shape"][0]
    net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, adim))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    losses = []
    for _ in range(20):
        for batch in loader:
            x = torch.cat([(batch[k].reshape(len(batch[k]), -1) - st[k][0]) / st[k][1] for k in obs_keys], -1)
            y = (batch["action"].reshape(len(x), -1) - st["action"][0]) / st["action"][1]
            loss = torch.nn.functional.mse_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    json.dump({"steps": len(losses), "loss_first": first, "loss_last": last,
               "finite": bool(np.all(np.isfinite(losses)))}, open(out, "w"))


if __name__ == "__main__":
    mode = sys.argv[1]
    {"load": lambda: load(sys.argv[2], sys.argv[3]), "write": lambda: write(sys.argv[2]),
     "train": lambda: train(sys.argv[2], sys.argv[3])}[mode]()
