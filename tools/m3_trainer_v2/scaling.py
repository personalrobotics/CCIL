"""Scaling of trainer v2 (#108): time per step and peak memory as models, rows, dimension and width grow.

Synthetic BC supervision (one shared table; M models with distinct seeds), one epoch each, compiled on
CUDA. Time per step is the group's train time divided by its steps; memory is torch's peak allocated
during training. Rows grow from 10^4 to 10^7 resident and 10^7 to 10^8 streaming.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ccil.supervision import Supervision
from ccil.training import vector_bc_v2 as v2


def _sup(n, d):
    r = np.random.default_rng(0)
    obs = r.standard_normal((n, d), dtype=np.float32)
    act = np.tanh(obs[:, :1])
    src = np.full(n, "demonstration")
    ep = (np.arange(n) // 500).astype(str)
    return Supervision(obs, act, np.ones(n, np.float32), src, ep, np.arange(n) % 500, np.full(n, -1), src,
                       {"method": "bc", "inputs": {}})


def _measure(n, d, M, width, mode, device):
    sup = _sup(n, d)
    settings = {"hidden_layers": [width, width], "learning_rate": 1e-3, "batch_size": 512, "epochs": 1,
                "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = v2.train_group([v2.Job(sup, s) for s in range(1, M + 1)], settings, device=device, data_mode=mode)
    wall = time.time() - t0
    g = out[0][1]["group"]
    steps = out[0][1]["steps_per_epoch"]
    return {"rows": n, "dim": d, "models": M, "width": width, "mode": g["data_mode"], "steps": steps,
            "ms_per_step": 1000 * g["train_seconds"] / steps, "train_seconds": g["train_seconds"],
            "setup_seconds": g["setup_seconds"], "wall_seconds": wall,
            "peak_device_gb": torch.cuda.max_memory_allocated() / 2**30, "table_gb": g["table_bytes"] / 2**30}


def run(device) -> dict:
    rows = []
    for n, mode in ((10**4, "resident"), (10**5, "resident"), (10**6, "resident"), (10**7, "resident"),
                    (10**7, "streaming"), (10**8, "streaming")):
        rows.append(_measure(n, 3, 4, 256, mode, device))
    for M in (1, 4, 16, 64):
        rows.append(_measure(10**6, 3, M, 256, "resident", device))
    for d in (64, 512):
        rows.append(_measure(10**6, d, 16, 256, "resident", device))
    rows.append(_measure(10**6, 3, 16, 1024, "resident", device))
    stream = [r for r in rows if r["mode"] == "streaming"]
    bounded = max(r["peak_device_gb"] for r in stream) < 4 * min(r["peak_device_gb"] for r in stream) + 0.5
    return {"pass": bool(bounded), "criterion": "streaming peak device memory does not grow with rows "
                                                "(10^7 to 10^8 within 4x + 0.5 GB)", "measurements": rows}
