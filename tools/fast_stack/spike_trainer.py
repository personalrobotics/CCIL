#!/usr/bin/env python3
"""#108 spike: stacked-parameter ensembles versus torch.func vmap, eager versus torch.compile.

    uv run --group evidence python tools/fast_stack/spike_trainer.py --out DIR [--gpu 0]

M legacy-protocol vector policies ([256, 256] ReLU MLP, tanh, batch 512, Adam 1e-3)
train together. One shared row table of the continuous CCIL unit's shape (1,047,900
rows, 3 observation and 1 action dimensions) is stored once on the device, and each
model reads it through its own index stream. Rows are never replicated per model.
No step syncs with the host.

For each engine and M in {1, 4, 20, 64}, it reports:
- ms per optimizer step for the group, and per model, measured in steady state after
  warm-up;
- the time of the first K steps, which includes compilation;
- peak device memory;
- agreement with trainer v1's update (torch.optim.Adam on an nn.Sequential) after K
  steps, from the same initial weights and the same index stream: max |Δθ| / max |θ|;
- batch invariance: model 0's parameters after K steps at M = 1 against M = 20.

The data is random; speed does not depend on the values. Every engine uses the same
data, and agreement is measured on it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROWS, OBS, ACT, HIDDEN, BS, LR = 1_047_900, 3, 1, (256, 256), 512, 1e-3
BETAS, EPS = (0.9, 0.999), 1e-8
K_AGREE, WARM, TIMED = 200, 50, 1000
GROUP_SIZES = (1, 4, 20, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    import torch.nn as nn
    from torch.func import grad, vmap

    a.out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def v1_model(seed):
        torch.manual_seed(seed)
        sizes = (OBS, *HIDDEN)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(sizes[-1], ACT), nn.Tanh())

    def stacked_params(seeds):
        lins = [[m for m in v1_model(s) if isinstance(m, nn.Linear)] for s in seeds]
        out = []
        for li in range(len(lins[0])):
            out.append(torch.stack([ls[li].weight.detach().T for ls in lins]).to(dev).contiguous())
            out.append(torch.stack([ls[li].bias.detach() for ls in lins]).to(dev).contiguous())
        return out

    def forward(ps, x):  # ps: [W [M,in,out], b [M,out], ...], x [M, B, in]
        h, n = x, len(ps) // 2
        for i in range(n):
            h = torch.baddbmm(ps[2 * i + 1].unsqueeze(1), h, ps[2 * i])
            h = torch.relu(h) if i < n - 1 else torch.tanh(h)
        return h

    def adam_update(params, grads, m, v, step_t):
        step_t.add_(1)
        bc1, bc2 = 1 - BETAS[0] ** step_t, 1 - BETAS[1] ** step_t
        for p, g, mm, vv in zip(params, grads, m, v):
            mm.mul_(BETAS[0]).add_(g, alpha=1 - BETAS[0])
            vv.mul_(BETAS[1]).addcmul_(g, g, value=1 - BETAS[1])
            p.add_(mm / ((vv.sqrt() / bc2.sqrt()).add_(EPS)) * (-LR / bc1))  # torch.optim.Adam's order

    # One shared table on the device; per-model index streams [M, B] into it.
    g = torch.Generator(device="cpu").manual_seed(0)
    X = (torch.rand(ROWS, OBS, generator=g) * 2 - 1).to(dev)
    Y = (torch.rand(ROWS, ACT, generator=g) * 2 - 1).to(dev)
    Mmax = max(GROUP_SIZES)
    streams = [torch.randint(0, ROWS, (Mmax, BS), generator=g).to(dev) for _ in range(K_AGREE + WARM)]

    def make_step(kind, params, m, v, step_t):
        if kind == "stacked":
            def step(idx):
                x, y = X[idx], Y[idx]  # gathers from the shared table: [M, B, ·]
                with torch.enable_grad():
                    ps = [p.detach().requires_grad_() for p in params]
                    loss = ((forward(ps, x) - y) ** 2).mean(dim=(1, 2))
                    grads = torch.autograd.grad(loss.sum(), ps)
                adam_update(params, grads, m, v, step_t)
            return step

        def loss_one(ps, x, y):
            return ((forward([q.unsqueeze(0) for q in ps], x.unsqueeze(0)) - y.unsqueeze(0)) ** 2).mean()
        g_all = vmap(grad(loss_one))

        def step(idx):
            grads = g_all(tuple(params), X[idx], Y[idx])
            adam_update(params, grads, m, v, step_t)
        return step

    def run(kind, compiled, M):
        params = stacked_params(range(1, M + 1))
        m, v = [torch.zeros_like(p) for p in params], [torch.zeros_like(p) for p in params]
        step_t = torch.zeros((), device=dev)
        fn = make_step(kind, params, m, v, step_t)
        if compiled:
            fn = torch.compile(fn, mode="reduce-overhead")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for k in range(K_AGREE):
            fn(streams[k][:M])
        torch.cuda.synchronize()
        first = time.perf_counter() - t0
        snap = [p[0].detach().cpu().clone() for p in params]
        for k in range(WARM):
            fn(streams[K_AGREE + k][:M])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(TIMED):
            fn(streams[k % len(streams)][:M])
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / TIMED
        return {"engine": kind, "compiled": compiled, "M": M, "ms_per_step_group": ms, "ms_per_step_per_model": ms / M,
                "first_K_steps_seconds_incl_compile": first,
                "peak_device_mb": torch.cuda.max_memory_allocated() / 2**20}, snap

    def v1_reference():
        model = v1_model(1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(K_AGREE):
            idx = streams[k][0]
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / K_AGREE
        out = []
        for lin in (mm for mm in model if isinstance(mm, nn.Linear)):
            out += [lin.weight.detach().T.cpu(), lin.bias.detach().cpu()]
        return out, ms

    ref, ref_ms = v1_reference()
    scale = max(float(p.abs().max()) for p in ref)
    rows, snaps = [{"engine": "v1 torch.optim.Adam on nn.Sequential (no host syncs)", "M": 1,
                    "ms_per_step_group": ref_ms}], {}
    print(json.dumps(rows[0]), flush=True)
    for kind in ("stacked", "vmap"):
        for compiled in (False, True):
            for M in GROUP_SIZES:
                try:
                    r, snap = run(kind, compiled, M)
                    r["agreement_vs_v1_max_abs_over_max"] = max(float((p - q).abs().max()) for p, q in zip(snap, ref)) / scale
                    snaps[(kind, compiled, M)] = snap
                except Exception as e:  # a spike reports what fails rather than stopping
                    r = {"engine": kind, "compiled": compiled, "M": M, "error": f"{type(e).__name__}: {e}"[:600]}
                rows.append(r)
                print(json.dumps(r), flush=True)
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    invariance = {f"{k}{'+compile' if c else ''}": max(float((p - q).abs().max())
                                                       for p, q in zip(snaps[(k, c, 1)], snaps[(k, c, 20)]))
                  for k in ("stacked", "vmap") for c in (False, True) if (k, c, 1) in snaps and (k, c, 20) in snaps}
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "tf32": False,
              "workload": {"rows": ROWS, "obs": OBS, "act": ACT, "hidden": HIDDEN, "batch": BS,
                           "data": "one shared table on the device, per-model index streams"},
              "steps": {"agreement": K_AGREE, "warmup": WARM, "timed": TIMED}, "results": rows,
              "batch_invariance_M1_vs_M20_max_abs": invariance}
    (a.out / "spike.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"batch_invariance": invariance}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
