#!/usr/bin/env python3
"""The policy-step workload of the #134 check, run against one source tree (the check runs it twice).

    python tools/m3_compile/workload.py --device DEVICE --compile {0,1} > result.json

Imports ccil from PYTHONPATH when set (the check points it at the pre-change source), so it uses only interfaces
that exist before and after #134: train_group, Job, the engine's __call__, counter_uniforms_device. Prints, per
group: every step's row indices and NoiseBC draws (SHA-256 over the steps), the returned parameters (SHA-256 per
policy), the per-epoch losses, and the engine record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import torch

import ccil
from ccil.supervision import Supervision
from ccil.training import vector_bc_v2 as v2
from ccil.training.noise_bc import NoiseBC

D, A = 3, 1
BASE = {"hidden_layers": [32, 32], "learning_rate": 1e-3, "batch_size": 32, "epochs": 2,
        "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None}
MODERN = {"optimizer": "adamw", "weight_decay": 1e-4, "lr_schedule": "cosine", "warmup_steps": 5, "step_budget": 37,
          "grad_clip_norm": 1.0, "ema_decay": 0.99}
# Groups of one bucket (3 jobs, padded to 4) whose tables, learning rates, NoiseBC and recipes differ. chunk_steps 5
# leaves a shorter last chunk in every epoch (the stride that recompiled before #134).
GROUPS = [
    {"name": "bc_a", "rows": 300, "lr": 1e-3},
    {"name": "bc_b_other_table_and_lr", "rows": 450, "lr": 3e-4},
    {"name": "bc_c_other_table_and_lr", "rows": 610, "lr": 1e-2},
    {"name": "noise_bc", "rows": 300, "lr": 1e-3, "noise": True},
    {"name": "modern_a", "rows": 400, "lr": 1e-4, "recipe": MODERN},
    {"name": "modern_b_other_table_and_lr", "rows": 520, "lr": 3e-3, "recipe": MODERN},
    {"name": "bc_streaming", "rows": 450, "lr": 3e-4, "data_mode": "streaming"},
]


def supervision(n: int, seed: int, method: str) -> Supervision:
    r = np.random.default_rng(seed)
    obs = r.normal(size=(n, D))
    act = np.tanh(obs @ r.normal(size=(D, A)))
    src = np.array(["demonstration"] * n)
    return Supervision(obs, act, np.ones(n), src, np.array([str(i // 50) for i in range(n)]), np.arange(n) % 50,
                       np.full(n, -1), src, {"method": method, "inputs": {}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--chunk-steps", type=int, default=5)
    a = ap.parse_args()
    trace: list = []
    real_call = v2._Engine.__call__

    def traced(self, *args, **kwargs):  # every step's rows and NoiseBC draws, before the step runs
        if len(args) == 13:  # resident: (params, m1, m2, t, ema, X, Y, W, WORDS, idx, active, gstep, c)
            idx, gstep, c = args[9], args[11], args[12]
            rows, words = idx, (args[8][idx] if kwargs["noisy"] else None)
        else:  # streaming: (params, m1, m2, t, ema, x, y, w, words, active, gstep, c): the gathered rows
            rows, words, gstep, c = args[5], args[8], args[10], args[11]
        h = hashlib.sha256(rows.detach().cpu().contiguous().numpy().tobytes())
        if words is not None:
            u = v2.counter_uniforms_device(c["n_seed"], gstep, words, 0, 1 + D)
            h.update(u.detach().cpu().contiguous().numpy().tobytes())
        trace.append(h.digest())
        return real_call(self, *args, **kwargs)
    v2._Engine.__call__ = traced
    out = {"ccil": ccil.__file__, "torch": torch.__version__, "device": a.device, "compile": bool(a.compile), "chunk_steps": a.chunk_steps,
           "groups": []}
    for g in GROUPS:
        trace.clear()
        noisy = g.get("noise", False)
        method = "noise_bc" if noisy else "bc"
        jobs = [v2.Job(supervision(g["rows"] - 17 * i, 100 + i, method), 7 + i,
                       NoiseBC("symmetric_uniform_half_batch", 0.1, seed=7 + i) if noisy else None) for i in range(3)]
        settings = {**BASE, "learning_rate": g["lr"], **g.get("recipe", {})}
        res = v2.train_group(jobs, settings, device=a.device, compile=bool(a.compile), chunk_steps=a.chunk_steps,
                             data_mode=g.get("data_mode", "resident"))
        ex = res[0][1]["execution"]
        out["groups"].append({
            "name": g["name"], "steps": len(trace),
            "rows_and_draws_sha256": hashlib.sha256(b"".join(trace)).hexdigest(),
            "parameters_sha256": [r["parameters_sha256"] for _, r in res],
            "epoch_losses": [[e["mean_loss"] for e in r["epochs"]] for _, r in res],
            "engine": ex["engine"], "compilation": ex.get("compilation")})
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
