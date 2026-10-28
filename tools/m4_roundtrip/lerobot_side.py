"""The LeRobot side of the round trip (#29). Runs in a LeRobot release's environment, with ccil installed.

    python lerobot_side.py fixture DIR             # LeRobot writes the fixture: state, image, four episodes
    python lerobot_side.py consume BUNDLE OUT.json # an unchanged LeRobot consumer, then CCIL's training dataset
"""
import hashlib
import json
import sys

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LENGTHS, FPS, H = (12, 9, 15, 6), 10, 0.1


def render(pos: float) -> np.ndarray:
    """An 8x8 RGB frame with a bright column at the position (clipped to [-1, 1])."""
    img = np.zeros((8, 8, 3), np.uint8)
    col = int(round((np.clip(pos, -1, 1) + 1) * 3.5))
    img[:, col] = (255, 128, 0)
    return img


def fixture(root):
    """A double integrator, x' = x + h v, v' = v + h a, with smooth actions: four episodes, deterministic."""
    features = {"observation.state": {"dtype": "float32", "shape": (2,), "names": ["position", "velocity"]},
                "observation.image": {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
                "action": {"dtype": "float32", "shape": (1,), "names": ["acceleration"]}}
    ds = LeRobotDataset.create(repo_id="ccil/roundtrip", fps=FPS, root=root, features=features, use_videos=False)
    for e, T in enumerate(LENGTHS):
        x, v = -0.6 + 0.3 * e, 0.0
        for t in range(T):
            a = np.float32(np.sin(0.5 * t + e))
            ds.add_frame({"observation.state": np.array([x, v], np.float32), "observation.image": render(x),
                          "action": np.array([a], np.float32), "task": f"reach {'left' if e % 2 else 'right'}"})
            x, v = np.float32(x + H * v), np.float32(v + H * a)
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()


def consume(bundle, out):
    from ccil.lerobot.training import CCILTrainingDataset
    res = {}
    ds = LeRobotDataset("ccil/roundtrip", root=bundle)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    h, frames, images = hashlib.sha256(), 0, 0
    for batch in loader:  # the unchanged consumer: LeRobot's own dataset and a stock DataLoader
        for k in sorted(batch):
            if isinstance(batch[k], torch.Tensor):
                h.update(k.encode() + batch[k].numpy().tobytes())
        frames += len(batch["index"])
        images += int(batch["observation.image"].shape[0])
    res["stock"] = {"frames": frames, "episodes": ds.num_episodes, "images_decoded": images,
                    "image_shape": list(ds[0]["observation.image"].shape), "content_sha256": h.hexdigest()}
    try:
        CCILTrainingDataset(ds)
        res["image_policy_refused"] = False
    except ValueError as e:
        res["image_policy_refused"] = "observation.image" in str(e)
    cd = CCILTrainingDataset(ds, keys=["observation.state"])
    syn = [cd[i] for i in range(len(cd.demo), len(cd))]
    res["ccil"] = {"rows": len(cd), "demonstration_rows": len(cd.demo), "labels": len(syn),
                   "label_candidate_rows": [int(s["ccil.candidate_row"]) for s in syn],
                   "label_weights": [float(s["ccil.weight"]) for s in syn]}
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    {"fixture": lambda: fixture(sys.argv[2]), "consume": lambda: consume(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
