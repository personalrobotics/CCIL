"""Batched transition-model fits: many MLPTransitionModel fits trained as one program on one device (#110).

The same models, losses and optimizer as MLPTransitionModel.fit (the legacy WorldModel protocol): the
normalized residual MLP with per-fit input and output normalizers, the declared Lipschitz regularizer, and Adam
with L2 weight decay, for the configured epochs and batch size. A group of fits with one architecture and
regularizer trains as a single vmap'd, compiled step, the way policy trainer v2 (training/vector_bc_v2.py) trains
policies, and shares its machinery: content-deduplicated shared rows with per-fit segments, counter-based
permutations, fixed group buckets with masked padding, a guarded compiled engine, enforced float32 precision,
and per-epoch host synchronization only.

Randomness is per fit and independent of the group. Initialization is MLPTransitionModel's, under
torch.manual_seed(seed). Minibatches come from the counter-based permutation keyed by (seed, epoch). The
soft-sampling perturbations are counter-keyed standard normals (splitmix64 uniforms through Box-Muller) keyed
by (seed, optimizer step, row of the minibatch, perturbation, component), where the per-fit loop drew
torch.randn.

Supported regularizers: none, slack, soft_sampling, soft_sampling_slack. The spectral-normalization and
soft_jac variants keep MLPTransitionModel.fit (fit_group refuses them).

Not bit-compatible with MLPTransitionModel.fit (the shuffle, the perturbation draws and the arithmetic order
differ); the evidence checks engine correctness and the fits' held-out error.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import grad, stack_module_state, vmap

from torch.utils._pytree import tree_map

from ..training.vector_bc_v2 import (CUDA_GRAPHS, _M64, _block_key, _bucket, _Engine, _lsr, _mix, _Precision, _s64,
                                     _Streams)
from .mlp import DynamicsNet, MLPTransitionModel, one_step_metrics

SUPPORTED = ("none", "slack", "soft_sampling", "soft_sampling_slack")
_PERTURB = _s64(0x50455254 << 32)  # domain separation of the perturbation draws


@dataclass
class FitJob:
    """One fit of a group: transitions (z, a, z_next) and the seed."""

    z: np.ndarray
    a: np.ndarray
    z_next: np.ndarray
    seed: int


def _normals(seed: torch.Tensor, step: torch.Tensor, shape: tuple, device) -> torch.Tensor:
    """Counter-keyed standard normals (M, *shape): Box-Muller over splitmix64 uniforms keyed by (seed, step, index)."""
    M = seed.shape[0]
    n = int(np.prod(shape))
    idx = torch.arange(n, dtype=torch.int64, device=device)
    h = _mix(_mix(_mix(seed[:, None] ^ _PERTURB) ^ step) ^ idx[None, :])
    u1 = (_lsr(h, 11).to(torch.float64) + 0.5) * 2.0 ** -53            # (0, 1): log is finite
    u2 = _lsr(_mix(h), 11).to(torch.float64) * 2.0 ** -53
    z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * np.pi * u2)
    return z.to(torch.float32).view(M, *shape)


def _forward(p: dict, s, a, layers: int, relu: bool):
    """DynamicsNet.forward from its state-dict names, one model."""
    out = torch.cat([(s - p["s_shift"]) / (p["s_scale"] + 1e-8), (a - p["a_shift"]) / (p["a_scale"] + 1e-8)], -1)
    for i in range(layers):
        out = out @ p[f"fc_layers.{i}.weight"].T + p[f"fc_layers.{i}.bias"]
        if i < layers - 1:
            out = torch.relu(out) if relu else torch.tanh(out)
    return out


def _predict(p: dict, s, a, layers: int, relu: bool):
    """DynamicsNet.predict: the raw residual."""
    return (_forward(p, s, a, layers, relu) * (p["out_scale"] + 1e-8) + p["out_shift"]) * (p["out_scale"] >= 1e-8)


def _loss(trainable: dict, buffers: dict, s, a, sp, noise, k: dict):
    """One fit's loss and its logged terms, as MLPTransitionModel._loss_fn: (loss, terms)."""
    p = {**trainable["net"], **buffers["net"]}
    y = (sp - s - p["out_shift"]) / (p["out_scale"] + 1e-8)
    y_pred = _forward(p, s, a, k["layers"], k["relu"])
    lt = k["lipschitz_type"]
    mse = ((y_pred - y) ** 2).mean()
    terms = {"mse_loss": mse}
    if lt == "none":
        return mse, {"mean_error": (y_pred - y).norm(dim=-1).mean()}
    if "slack" in lt:
        lam = {**trainable["lam"], **buffers["lam"]}
        lam_out = _predict(lam, s, a, k["layers"], k["relu"])
    if lt == "slack":
        slack = torch.sigmoid(lam_out)
        per = (y_pred - y) ** 2
        weighted = torch.linalg.norm(slack * per, dim=-1).mean()
        avg = slack.mean()
        return weighted - k["slack_weight"] * avg, {"mse_loss": per.norm(dim=-1).mean(), "weighted_mse_loss": weighted,
                                                     "slack_penalty": avg, "slack_std": slack.std()}
    # soft_sampling and soft_sampling_slack: the finite-difference Lipschitz hinge over perturbations.
    n = k["soft_samples"]
    base = _predict(p, s, a, k["layers"], k["relu"]).repeat_interleave(n, dim=0)
    ss, aa = s.repeat_interleave(n, dim=0), a.repeat_interleave(n, dim=0)
    ds, da = noise[:, :ss.shape[1]] * k["soft_sampling_eps"], noise[:, ss.shape[1]:] * k["soft_sampling_eps"]
    step_norm = torch.linalg.norm(torch.cat([ds, da], dim=-1), dim=-1)
    ratio = torch.linalg.norm(_predict(p, ss + ds, aa + da, k["layers"], k["relu"]) - base, dim=-1) / step_norm
    hinge = torch.relu(ratio - k["lipschitz_constraint"])
    if lt == "soft_sampling":
        penalty = k["soft_penalty_weight"] * hinge.mean()
    else:
        slack = lam_out.reshape(-1)
        t = slack * hinge.reshape(len(slack), n).mean(dim=1)
        penalty = k["soft_penalty_weight"] * (t.mean() + 1 - torch.exp(-0.1 * torch.abs(slack).mean()))
    terms["lipschitz_penalty"] = penalty
    return mse + penalty, terms


def _step(trainable, m1, m2, t, buffers, s, a, sp, noise_seed, gstep, active, k: dict):
    """One masked Adam step (L2 weight decay, torch.optim.Adam's order) for every fit of the group."""
    M = active.shape[0]
    if k["needs_noise"]:
        noise = _normals(noise_seed, gstep, (s.shape[1] * k["soft_samples"], s.shape[2] + a.shape[2]), s.device)
    else:
        noise = torch.zeros((M, 1, 1), device=s.device)

    def one(tr, buf, s_, a_, sp_, nz):
        loss, terms = _loss(tr, buf, s_, a_, sp_, nz, k)
        return loss, terms
    (loss, terms), g = vmap(lambda tr, buf, s_, a_, sp_, nz: (one(tr, buf, s_, a_, sp_, nz),
                                                               grad(lambda q: one(q, buf, s_, a_, sp_, nz)[0])(tr)))(
        trainable, buffers, s, a, sp, noise)
    b1, b2, eps, lr, wd = 0.9, 0.999, 1e-8, k["learning_rate"], k["weight_decay"]
    t = t + active.to(t.dtype)
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    new_tr, new_m1, new_m2 = {}, {}, {}
    for net in trainable:
        new_tr[net], new_m1[net], new_m2[net] = {}, {}, {}
        for name, p in trainable[net].items():
            shape = (M,) + (1,) * (p.dim() - 1)
            u = active.view(shape)
            gg = g[net][name] + wd * p                               # torch.optim.Adam's L2 weight decay
            mk = torch.where(u, m1[net][name] * b1 + gg * (1 - b1), m1[net][name])
            vk = torch.where(u, m2[net][name] * b2 + gg * gg * (1 - b2), m2[net][name])
            denom = (vk.sqrt() / bc2.clamp_min(1e-30).sqrt().view(shape)) + eps
            stepped = p - (lr / bc1.clamp_min(1e-30)).view(shape) * mk / denom
            new_tr[net][name], new_m1[net][name], new_m2[net][name] = torch.where(u, stepped, p), mk, vk
    return new_tr, new_m1, new_m2, t, loss, terms


class _restore_determinism:
    """Restore torch's deterministic-algorithms flag on exit, whatever happened."""

    def __init__(self, previous: bool):
        self.previous = previous

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.previous)
        return False


def fit_identity(z, a, z_next, settings: dict, seed: int) -> str:
    """Content identity of one fit: its exact transition rows, settings and seed, and the engine.

    A stage that receives a model fitted elsewhere (fit_transition_stage(prefit=...)) recomputes this from its
    own inputs and refuses a mismatch, so a grouped result can never be written under another unit's stage."""
    h = hashlib.blake2b(digest_size=32)
    for x in (z, a, z_next):
        x = np.ascontiguousarray(x)
        h.update(f"{x.dtype.str}{x.shape}".encode() + x.tobytes())
    h.update(json.dumps({"settings": settings, "seed": int(seed), "engine": "mlp_batched/1"}, sort_keys=True,
                        default=str).encode())
    return h.hexdigest()


def fit_group(jobs: list[FitJob], settings: dict, *, device: str = "cpu", compile: bool | None = None,
              chunk_steps: int = 256, deterministic: bool = False):
    """Fit every job's transition model together; return [(MLPTransitionModel, fit record)] in job order.

    settings are MLPTransitionModel's (hidden_layers, activation, lipschitz_type, lipschitz_constraint,
    soft_penalty_weight, soft_sampling_eps, soft_samples, slack_weight, learning_rate, weight_decay, epochs,
    batch_size), shared by the group. Device-resident data (the transition tables of Pendulum-scale fits);
    the streaming regime of the policy trainer is the next step for larger data.
    """
    lt = settings.get("lipschitz_type") or "none"
    if lt not in SUPPORTED:
        raise ValueError(f"fit_group supports {SUPPORTED}; {lt!r} keeps MLPTransitionModel.fit")
    if not jobs:
        raise ValueError("a group needs at least one fit")
    device = torch.device(device)
    compile = (device.type == "cuda") if compile is None else bool(compile)
    bs, epochs = int(settings["batch_size"]), int(settings["epochs"])
    t_setup = time.time()
    n_real = len(jobs)
    M = _bucket(n_real)
    padded = list(jobs) + [jobs[0]] * (M - n_real)
    real = torch.tensor([i < n_real for i in range(M)], device=device)

    # Shared transition table: each distinct (z, a, z_next) block once; a fit is one segment of it.
    blocks, starts, parts = {}, [], []
    for j in padded:
        key = _block_key(np.asarray(j.z, np.float32), np.asarray(j.a, np.float32), np.asarray(j.z_next, np.float32))
        if key not in blocks:
            blocks[key] = (sum(len(p[0]) for p in parts), len(j.z))
            parts.append((np.asarray(j.z, np.float32), np.asarray(j.a, np.float32), np.asarray(j.z_next, np.float32)))
        starts.append(blocks[key])
    for j in jobs:
        if len(j.z) < bs:
            raise ValueError(f"{len(j.z)} transitions is fewer than one minibatch (transition_model.batch_size={bs})")
    Z = torch.as_tensor(np.concatenate([p[0] for p in parts]), device=device)
    Aa = torch.as_tensor(np.concatenate([p[1] for p in parts]), device=device)
    ZN = torch.as_tensor(np.concatenate([p[2] for p in parts]), device=device)
    streams = _Streams([[s] for s in starts], [j.seed for j in padded], bs, device)
    steps = streams.steps.tolist()

    # Models: MLPTransitionModel's construction per seed, and its normalizers per fit (float32 on the device).
    models = []
    for j, (st, ln) in zip(padded, starts):
        m = MLPTransitionModel(**{k: settings[k] for k in settings}, device=str(device))
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(int(j.seed))
            m._build(Z.shape[1], Aa.shape[1])
        s, act, sp = Z[st:st + ln], Aa[st:st + ln], ZN[st:st + ln]
        s_shift, a_shift, out_shift = s.mean(0), act.mean(0), (sp - s).mean(0)
        m.net.set_transformations(s_shift=s_shift, s_scale=(s - s_shift).abs().mean(0), a_shift=a_shift,
                                  a_scale=(act - a_shift).abs().mean(0), out_shift=out_shift,
                                  out_scale=(sp - s - out_shift).abs().mean(0))
        models.append(m)
    nets = {"net": [m.net for m in models]} | ({"lam": [m.lam for m in models]} if models[0].lam is not None else {})
    trainable, buffers = {}, {}
    for name, mods in nets.items():
        params, bufs = stack_module_state(mods)
        trainable[name] = {k: v.detach().to(device) for k, v in params.items()}
        buffers[name] = {k: v.detach().to(device) for k, v in bufs.items()}
    k = {"layers": len(settings["hidden_layers"]) + 1, "relu": settings.get("activation", "relu") != "tanh",
         "lipschitz_type": lt, "lipschitz_constraint": settings.get("lipschitz_constraint"),
         "soft_penalty_weight": settings.get("soft_penalty_weight"),
         "soft_sampling_eps": settings.get("soft_sampling_eps"), "soft_samples": int(settings.get("soft_samples") or 1),
         "slack_weight": settings.get("slack_weight"), "learning_rate": float(settings["learning_rate"]),
         "weight_decay": float(settings["weight_decay"]), "needs_noise": lt.startswith("soft_sampling")}
    noise_seed = torch.tensor([_s64(int(j.seed) & _M64) for j in padded], device=device)
    engine = _Engine(_step, compile)
    m1 = {n_: {q: torch.zeros_like(v) for q, v in d.items()} for n_, d in trainable.items()}
    m2 = {n_: {q: torch.zeros_like(v) for q, v in d.items()} for n_, d in trainable.items()}
    t = torch.zeros(M, device=device)
    S_dev = torch.tensor(steps, device=device)
    history = [{} for _ in jobs]
    epoch_seconds = []
    t_train = time.time()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(bool(deterministic) or previous)  # compute.deterministic, as v1
    with _Precision() as precision, _restore_determinism(previous):
        for epoch in range(epochs):
            t_epoch = time.time()
            totals: dict = {}
            max_steps = max(steps)
            for s0 in range(0, max_steps, chunk_steps):
                idx = streams.chunk(epoch, s0, min(chunk_steps, max_steps - s0))
                for j_ in range(idx.shape[1]):
                    b = s0 + j_
                    active = real & (b < S_dev)
                    rows = idx[:, j_]
                    gstep = (epoch * S_dev + b)[:, None]
                    out = engine(trainable, m1, m2, t, buffers, Z[rows], Aa[rows], ZN[rows], noise_seed, gstep,
                                 active, k)
                    trainable, m1, m2, t, loss, terms = (tree_map(lambda v: v.clone(), out)  # a CUDA-graph replay
                                                         if CUDA_GRAPHS and engine.state.startswith("compiled")
                                                         else out)  # reuses its outputs
                    for name, v in {"loss": loss, **terms}.items():
                        totals[name] = totals.get(name, 0.0) + torch.where(active, v, 0.0)
            means = {name: (v / S_dev).tolist() for name, v in totals.items()}  # one host read per epoch
            epoch_seconds.append(time.time() - t_epoch)
            for i in range(n_real):
                for name, vals in means.items():
                    history[i].setdefault(name, []).append(vals[i])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_done = time.time()
    execution = {"members": [{"seed": int(j.seed), "rows": int(len(j.z))} for j in jobs], "bucket": M,
                 "padding": M - n_real, "engine": engine.state, "device": str(device), "torch": torch.__version__,
                 "precision": precision.effective, "data_mode": "resident", "deterministic": bool(deterministic)}
    execution_sha = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
    out = []
    for i, (j, m) in enumerate(zip(jobs, models)):
        for name, mod in nets.items():
            target = m.net if name == "net" else m.lam
            target.load_state_dict({**{q: v[i] for q, v in trainable[name].items()},
                                    **{q: v[i] for q, v in buffers[name].items()}})
        record = {"method": "adam", "trainer": "batched", "epochs": epochs, "steps_per_epoch": steps[i],
                  "fit_identity_sha256": fit_identity(j.z, j.a, j.z_next, settings, j.seed),
                  "final_epoch_losses": {q: v[-1] for q, v in history[i].items()}, "epoch_losses": history[i],
                  "execution": {"execution_sha256": execution_sha, "position": i, "bucket": M, "engine": engine.state},
                  "group": {**execution, "setup_seconds": t_train - t_setup, "train_seconds": t_done - t_train,
                            "epoch_seconds": epoch_seconds},
                  "shuffle": "counter-based permutation (policy trainer v2's), keyed by (seed, epoch)",
                  "perturbations": "counter-keyed normals (seed, step, minibatch row, sample, component)"
                  if k["needs_noise"] else None,
                  **one_step_metrics(m, j.z, j.a, j.z_next)}
        out.append((m, json.loads(json.dumps(record, default=float))))
    return out
