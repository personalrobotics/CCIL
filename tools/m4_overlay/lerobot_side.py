"""The LeRobot side of the overlay evidence (#28): runs in one LeRobot release's environment, with ccil installed.

    python lerobot_side.py stock DEST SRC OUT.npz      # stock LeRobot on the bundle and on the source
    python lerobot_side.py aware DEST OUT.npz          # CCIL's training dataset over the bundle
    python lerobot_side.py temporal DEST OUT.json K    # temporal windows: refusals, and stored chunks (K > 0)
"""
import json
import sys

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ccil.lerobot.training import CCILTrainingDataset, TemporalWindowRefused


def dump(ds):
    cols = {}
    for batch in torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                cols.setdefault(k, []).append(v.numpy())
    return {k: np.concatenate(v) for k, v in cols.items()}


def stock(dest, src, out):
    a, b = LeRobotDataset("ccil/bundle", root=dest), LeRobotDataset("ccil/source", root=src)
    da, db = dump(a), dump(b)
    np.savez(out, **{f"bundle/{k}": v for k, v in da.items()}, **{f"source/{k}": v for k, v in db.items()},
             episodes=np.array([a.num_episodes, b.num_episodes]))


def aware(dest, out):
    ds = CCILTrainingDataset(LeRobotDataset("ccil/bundle", root=dest))
    n_demo = len(ds.demo)
    syn = {}
    for i in range(n_demo, len(ds)):
        for k, v in ds[i].items():
            syn.setdefault(k, []).append(v.numpy())
    demo_first = ds[0]
    w = ds.weights()
    wb = ds.weights(balance=True)
    np.savez(out, **{f"synthetic/{k}": np.stack(v) for k, v in syn.items()}, demo_rows=np.array(n_demo),
             demo_index=ds.demo, demo_first_weight=demo_first["ccil.weight"].numpy(),
             demo_first_synthetic=demo_first["ccil.synthetic"].numpy(), weights=w, weights_balanced=wb,
             sampled=np.array(list(ds.sampler(num_samples=2000, generator=torch.Generator().manual_seed(0)))))


def temporal(dest, out, k):
    k = int(k)
    res = {}
    fps = LeRobotDataset("ccil/bundle", root=dest).fps
    cases = {"observation_history": {"observation.state": [-1 / fps, 0.0]},
             "future_actions": {"action": [i / fps for i in range(max(k, 2))]}}
    for name, deltas in cases.items():
        try:
            ds = CCILTrainingDataset(LeRobotDataset("ccil/bundle", root=dest, delta_timestamps=deltas))
            res[name] = {"refused": False}
            if name == "future_actions" and k:
                L = ds.labels
                K = L["action_chunk_mask"].shape[1]
                own = all(np.array_equal(ds[len(ds.demo) + j]["action"].numpy(),
                                         L["action_chunk"][j].reshape(K, -1)[:k])
                          and np.array_equal(ds[len(ds.demo) + j]["action_is_pad"].numpy(), ~L["action_chunk_mask"][j][:k])
                          for j in range(len(L["candidate_row"])))
                # Not vacuous: neighbouring labels' chunks differ, so walking into a neighbour would be seen.
                chunks = L["action_chunk"]
                distinct = all(not np.array_equal(chunks[j], chunks[j + 1]) for j in range(len(chunks) - 1))
                # A demonstration row keeps LeRobot's own window over real neighbouring frames.
                demo = ds[0]
                res[name].update(windows_are_each_labels_own_chunk=own and distinct and len(chunks) > 1,
                                 neighbouring_chunks_differ=distinct, labels=len(L["candidate_row"]),
                                 demonstration_window_shape=list(demo["action"].shape))
        except TemporalWindowRefused as e:
            res[name] = {"refused": True, "message": str(e)}
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    {"stock": lambda: stock(*sys.argv[2:5]), "aware": lambda: aware(*sys.argv[2:4]),
     "temporal": lambda: temporal(*sys.argv[2:5])}[sys.argv[1]]()
