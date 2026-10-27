"""Scaling of trainer v2 (#108): time per step and peak memory as models, rows, dimension and width grow.

Synthetic BC supervision (one shared table; M models with distinct seeds), two epochs each, compiled on
CUDA. The steady-state time per step is the second epoch's time divided by its steps; the first epoch's
excess over it is the one-time cost (compilation, warm-up), reported separately, as is host setup.
Memory is torch's peak allocated during training. Rows grow from 10^4 to 10^7 resident and 10^7 to 10^8
streaming. A repeated group of the same shapes shows whether the compiled step is reused.
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
    settings = {"hidden_layers": [width, width], "learning_rate": 1e-3, "batch_size": 512, "epochs": 2,
                "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = v2.train_group([v2.Job(sup, s) for s in range(1, M + 1)], settings, device=device, data_mode=mode)
    wall = time.time() - t0
    g = out[0][1]["group"]
    steps = out[0][1]["steps_per_epoch"]
    e1, e2 = g["epoch_seconds"]
    return {"rows": n, "dim": d, "models": M, "width": width, "mode": g["data_mode"], "steps_per_epoch": steps,
            "steady_ms_per_step": 1000 * e2 / steps, "one_time_seconds": max(0.0, e1 - e2),
            "setup_seconds": g["setup_seconds"], "wall_seconds": wall, "chunk_steps": g["chunk_steps"],
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
    rows.append(_measure(10**6, 3, 16, 1024, "resident", device))  # same shapes again: is the compiled step reused?
    rows.append(_measure(10**6, 512, 16, 256, "streaming", device))
    stream = {r["rows"]: r["peak_device_gb"] for r in rows if r["mode"] == "streaming" and r["dim"] == 3}
    bounded = stream[10**8] <= 1.1 * stream[10**7] + 0.02  # 10x the rows: at most 10% + 20 MB more device memory
    big_d = [r for r in rows if r["dim"] == 512]
    return {"pass": bool(bounded), "criterion": "streaming peak device memory at 10^8 rows is at most 1.1x that at "
                                                "10^7 rows + 0.02 GB",
            "compile_reused": rows[-2]["one_time_seconds"] < 0.5 * max(rows[-3]["one_time_seconds"], 1e-9),
            "dim512_peak_gb": {r["mode"]: r["peak_device_gb"] for r in big_d}, "measurements": rows}
