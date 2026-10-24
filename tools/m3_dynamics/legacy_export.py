#!/usr/bin/env python3
"""Export legacy WorldModel weights and reference predictions for the #21 equivalence check.

Runs in the pinned legacy harness environment, never in the project:

    uv run --no-project --python 3.11 --with-requirements tools/dynamics_checkpoint/requirements.txt \\
        python tools/m3_dynamics/legacy_export.py --out DIR

For each Lipschitz type it trains a small legacy WorldModel
(correct_il/models/nn_dynamics.py) on a deterministic double-integrator
dataset, so spectral-norm buffers and slack networks hold trained values, then
writes DIR/<lipschitz_type>.npz with plain numpy arrays (no pickle):
- state/<key>: the dynamics network state_dict, and lambda/<key> for slack;
- norm/<name>: the normalization tensors, which legacy keeps outside state_dict;
- z, a: fixed probe inputs;
- f_eval: WorldModel.f(z, a) in eval mode (stored power-iteration vectors);
- f_train, u_after/<layer>, v_after/<layer>: one training-mode call, which runs
  a power iteration and updates the buffers, and the buffers it leaves.
Nothing here is loaded back with torch.load: the export is numpy only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import correct_il.models.nn_dynamics as nd  # noqa: E402

TYPES = ("none", "spectral_normalization", "slack", "spectral_normalization_slack", "soft_sampling",
         "soft_sampling_slack", "soft_jac")
SETTINGS = {"hidden": (32, 32), "lipschitz_constraint": 1.5, "soft_lipschitz_penalty_weight": 0.01,
            "soft_lipschitz_sampling_eps": 0.0003, "soft_lipschitz_n_samples": 5, "slack_weight": 1.0,
            "lr": 0.001, "weight_decay": 1e-05, "epochs": 5, "batch_size": 64, "seed": 7}


def dataset(seed):
    """400 double-integrator transitions (h = 0.1) with uniform states and actions."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, size=(400, 2)).astype(np.float32)
    a = rng.uniform(-1, 1, size=(400, 1)).astype(np.float32)
    sp = s + np.stack([0.1 * s[:, 1], 0.1 * a[:, 0]], axis=1).astype(np.float32)
    return s, a, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    s, a, sp = dataset(SETTINGS["seed"])
    probe = np.random.default_rng(99)
    z = probe.uniform(-1.2, 1.2, size=(64, 2)).astype(np.float32)
    act = probe.uniform(-1.2, 1.2, size=(64, 1)).astype(np.float32)
    for lt in TYPES:
        torch.manual_seed(SETTINGS["seed"])
        np.random.seed(SETTINGS["seed"])
        d_config = nd._Config({"lipschitz_type": lt, "lipschitz_constraint": SETTINGS["lipschitz_constraint"],
                               "soft_lipschitz_penalty_weight": SETTINGS["soft_lipschitz_penalty_weight"],
                               "soft_lipschitz_sampling_eps": SETTINGS["soft_lipschitz_sampling_eps"],
                               "soft_lipschitz_n_samples": SETTINGS["soft_lipschitz_n_samples"],
                               "slack_weight": SETTINGS["slack_weight"]})
        model = nd.WorldModel(2, 1, d_config=d_config, hidden_size=SETTINGS["hidden"], fit_lr=SETTINGS["lr"],
                              fit_wd=SETTINGS["weight_decay"], device="cpu")
        model.fit_dynamics(s, a, sp, batch_size=SETTINGS["batch_size"], train_epochs=SETTINGS["epochs"])
        net = model.dynamics_net
        arrays = {f"state/{k}": v.detach().numpy().copy() for k, v in net.state_dict().items()}
        if hasattr(model, "learnable_lambda"):
            arrays.update({f"lambda/{k}": v.detach().numpy().copy() for k, v in model.learnable_lambda.state_dict().items()})
        arrays.update({f"norm/{k}": v.detach().numpy().copy() for k, v in net.transformations.items()})
        zt, at = torch.as_tensor(z), torch.as_tensor(act)
        with torch.no_grad():
            net.eval()
            arrays["f_eval"] = model.f(zt, at).numpy()
            net.train()
            arrays["f_train"] = model.f(zt, at).numpy()
        for name, buf in net.named_buffers():
            if name.endswith(("weight_u", "weight_v")):
                arrays[f"after/{name}"] = buf.numpy().copy()
        arrays["z"], arrays["a"] = z, act
        arrays["meta"] = np.array(json.dumps({"lipschitz_type": lt, "settings": SETTINGS, "torch": torch.__version__,
                                              "numpy": np.__version__, "source": "correct_il/models/nn_dynamics.py"}))
        np.savez(out / f"{lt}.npz", **arrays)
        print(f"exported {lt}")


if __name__ == "__main__":
    main()
