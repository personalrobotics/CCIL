"""Policy trainer v2: many vector policies trained together on one device (#108).

Experimental reference infrastructure, like v1 (vector_bc.py): it trains the
same VectorBCPolicy (the d3rlpy-default MLP with tanh, standard and min_max
scalers, the weighted loss of ccil.supervision) under the paper's protocol
(Adam, batch size, epochs, balance_data), but trains a group of M models as one
batched program:

- One shared row table. Each job's demonstration and synthetic blocks are
  deduplicated by content, so demonstrations shared across seeds are stored
  once; each model reads the table through its own index stream, and
  balancing repeats indices, never rows.
- Device-side shuffling. Model m's epoch e permutation comes from a torch
  Generator seeded by SeedSequence([seed_m, e]); models with fewer steps are
  masked, and so are minibatches whose weights sum to 0 (no optimizer step,
  as in v1).
- One compiled step. torch.func vmap over the stacked parameters
  (stack_module_state), the weighted loss, its gradient, and a functional Adam
  in torch.optim.Adam's update order with per-model step counts, compiled with
  torch.compile on CUDA (eager elsewhere, or when compile is off); the engine
  used is recorded.
- NoiseBC on the device: the counter-keyed splitmix64 draws of noise_bc.py,
  bit-identical for the same key (int64 wrapping arithmetic, logical shifts by
  masking), from each row's identity words computed once on the host.
- No per-step host synchronization: losses, row counts and a NoiseBC
  fingerprint accumulate on the device and are read once per epoch.

v2 is not bit-compatible with v1 (the shuffle and the arithmetic order differ;
#108 replaces bit identity with engine-correctness checks and statistical
equivalence). Checkpoints are ordinary inference checkpoints of VectorBCPolicy;
v2 has no resume: a group trains in seconds to minutes.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import functional_call, grad, stack_module_state, vmap

from ..core.identity import identity_words64
from .vector_bc import ADAM, OBS_EPS, ACT_EPS, TrainingError, VectorBCPolicy, training_identity, training_rows

TRAINER_VERSION = "2"
_M64 = (1 << 64) - 1


def _s64(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c


_GAMMA, _MIX1, _MIX2 = _s64(0x9E3779B97F4A7C15), _s64(0xBF58476D1CE4E5B9), _s64(0x94D049BB133111EB)


def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift of int64 bit patterns."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _mix(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer of x + gamma on int64 bit patterns (wrapping), as noise_bc._mix on uint64."""
    z = x + _GAMMA
    z = (z ^ _lsr(z, 30)) * _MIX1
    z = (z ^ _lsr(z, 27)) * _MIX2
    return z ^ _lsr(z, 31)


def counter_uniforms_device(seed: torch.Tensor, step: torch.Tensor, words: torch.Tensor, draw: int,
                            components: int) -> torch.Tensor:
    """noise_bc.counter_uniforms on the device: seed (M,), step (M, 1), words (M, B, 2) int64 -> (M, B, components) float64."""
    h = _mix(seed[:, None].expand(words.shape[0], words.shape[1]))
    h = _mix(h ^ step)
    h = _mix(h ^ words[..., 0])
    h = _mix(h ^ words[..., 1])
    h = _mix(h ^ _s64(draw & _M64))
    h = _mix(h[..., None] ^ torch.arange(components, dtype=torch.int64, device=words.device))
    return _lsr(h, 11).to(torch.float64) * 2.0 ** -53


@dataclass
class Job:
    """One model of a group: its supervision, seed, and NoiseBC perturbation (noise_bc arm only)."""

    supervision: object
    seed: int
    noise: object = None


def _block_key(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes() if a.dtype != object else "\0".join(map(str, a.tolist())).encode())
    return h.hexdigest()


def _table(jobs):
    """The shared row table and each job's map from its own row index to a table row."""
    blocks, order, maps = {}, [], []
    for job in jobs:
        sup = job.supervision
        obs, act, w = sup.arrays()
        local_to_table = np.empty(len(w), np.int64)
        for part in ("demonstration", "synthetic"):
            rows = np.flatnonzero(sup.source == part)
            if not len(rows):
                continue
            key = _block_key(obs[rows], act[rows], w[rows], sup.episode_id[rows].astype(str), sup.frame[rows])
            if key not in blocks:
                start = sum(len(b[0]) for b in (blocks[k] for k in order))
                blocks[key] = (np.asarray(obs[rows], np.float32), np.asarray(act[rows], np.float32),
                               np.asarray(w[rows], np.float32), sup.episode_id[rows].astype(str),
                               np.asarray(sup.frame[rows], np.int64), part, start)
                order.append(key)
            local_to_table[rows] = blocks[key][6] + np.arange(len(rows))
        maps.append(local_to_table)
    parts = [blocks[k] for k in order]
    table = {"obs": np.concatenate([p[0] for p in parts]), "act": np.concatenate([p[1] for p in parts]),
             "w": np.concatenate([p[2] for p in parts]), "episode_id": np.concatenate([p[3] for p in parts]),
             "frame": np.concatenate([p[4] for p in parts]),
             "synthetic": np.concatenate([np.full(len(p[0]), p[5] == "synthetic") for p in parts]),
             "blocks": [{"part": p[5], "rows": len(p[0]), "start": p[6]} for p in parts]}
    return table, maps


def _epoch_generator(seed: int, epoch: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.SeedSequence([int(seed), int(epoch)]).generate_state(1, np.uint64)[0] & ((1 << 63) - 1)))
    return g


def train_group(jobs: list[Job], settings: dict, *, device: str = "cpu", compile: bool | None = None,
                memory_budget_fraction: float = 0.5):
    """Train every job's policy together; return [(policy, record)] in job order.

    settings are v1's (hidden_layers, learning_rate, batch_size, epochs, observation_scaler, action_scaler,
    balance_labels) and are shared by the group. compile defaults to True on CUDA and False elsewhere.
    """
    if not jobs:
        raise TrainingError("a group needs at least one job")
    device = torch.device(device)
    compile = (device.type == "cuda") if compile is None else bool(compile)
    for job in jobs:
        method = job.supervision.manifest()["method"]
        if (job.noise is not None) != (method == "noise_bc"):
            raise TrainingError("NoiseBC perturbation applies to the noise_bc arm, and the noise_bc arm needs one")
    balance = settings.get("balance_labels")
    bs = int(settings["batch_size"])
    t_setup = time.time()

    table, maps = _table(jobs)
    rows, epochs, steps = [], [], []
    for job, m in zip(jobs, maps):
        if balance and job.supervision.manifest()["method"] != "ccil":
            raise TrainingError("balance_labels repeats demonstrations against labels; it applies to ccil only")
        r = m[training_rows(job.supervision, balance)]
        e = int(settings["epochs"]) // int(balance) if balance else int(settings["epochs"])
        if e < 1 or len(r) // bs < 1:
            raise TrainingError(f"job (seed {job.seed}): {len(r)} rows and {e} epochs leave no optimizer step")
        rows.append(r)
        epochs.append(e)
        steps.append(len(r) // bs)
    M, R = len(jobs), len(table["w"])
    need = R * 4 * (table["obs"].shape[1] + table["act"].shape[1] + 1) + sum(len(r) for r in rows) * 8
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if need > memory_budget_fraction * free:
            raise TrainingError(f"resident mode needs {need / 2**30:.1f} GiB of a {memory_budget_fraction:.0%} budget of "
                                f"{free / 2**30:.1f} GiB free; streaming mode is not implemented yet (#108)")

    # Policies: v1's initialization per seed, v1's scalers per job, then stacked.
    policies = []
    for job, r in zip(jobs, rows):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(job.seed))
            p = VectorBCPolicy(table["obs"].shape[1], table["act"].shape[1], settings["hidden_layers"],
                               settings["observation_scaler"], settings["action_scaler"])
        p.fit_scalers(table["obs"][r], table["act"][r])
        policies.append(p)
    params, _ = stack_module_state([p.net for p in policies])
    params = {k: v.detach().to(device) for k, v in params.items()}
    template = policies[0].net.to("meta")
    D, A = table["obs"].shape[1], table["act"].shape[1]

    def stat(name, default):
        return torch.tensor(np.stack([getattr(p, name) if getattr(p, name) is not None else default for p in policies]),
                            dtype=torch.float32, device=device)
    obs_mean, obs_std = stat("obs_mean", np.zeros(D)), stat("obs_std", np.ones(D) - OBS_EPS)
    act_min, act_max = stat("act_min", -np.ones(A)), stat("act_max", np.ones(A))
    scale_obs = settings["observation_scaler"] != "none"
    scale_act = settings["action_scaler"] != "none"

    X = torch.as_tensor(table["obs"], device=device)
    Y = torch.as_tensor(table["act"], device=device)
    W = torch.as_tensor(table["w"], device=device)
    SYN = torch.as_tensor(table["synthetic"], device=device)
    noisy = [j.noise is not None for j in jobs]
    if any(noisy):
        words = torch.as_tensor(np.array([identity_words64(e, f) for e, f in zip(table["episode_id"], table["frame"])],
                                         dtype=np.uint64).view(np.int64), device=device)
        n_seed = torch.tensor([_s64(int(j.noise.seed) & _M64) if j.noise else 0 for j in jobs], device=device)
        n_scale = torch.tensor([j.noise.scale if j.noise else 0.0 for j in jobs], dtype=torch.float64, device=device)
        n_thresh = torch.tensor([1.0 - j.noise.mask_probability if j.noise else 2.0 for j in jobs],
                                dtype=torch.float64, device=device)
        n_sym = torch.tensor([j.noise is not None and j.noise.mode == "symmetric_uniform_half_batch" for j in jobs],
                             device=device)
    b1, b2 = ADAM["betas"]
    eps, lr = ADAM["eps"], float(settings["learning_rate"])

    def loss_one(p, x, y, w):
        pred = functional_call(template, p, (x,))
        per_row = ((pred - y) ** 2).mean(dim=1)
        return (w * per_row).sum() / w.sum().clamp_min(1e-30)
    grad_all = vmap(grad(loss_one, has_aux=False))
    loss_all = vmap(loss_one)

    def step(params, m1, m2, t, idx, active, gstep):
        x, y, w = X[idx], Y[idx], W[idx]                      # (M, B, ·)
        if scale_obs:
            x = (x - obs_mean[:, None]) / (obs_std[:, None] + OBS_EPS)
        if any(noisy):
            u = counter_uniforms_device(n_seed, gstep, words[idx], 0, 1 + D)
            mask = u[..., 0] > n_thresh[:, None]
            unit = torch.where(n_sym[:, None, None], 2.0 * u[..., 1:] - 1.0, u[..., 1:])
            x = x + torch.where(mask[..., None], n_scale[:, None, None] * unit, 0.0).to(x.dtype)
        if scale_act:
            y = ((y - act_min[:, None]) / (act_max[:, None] - act_min[:, None] + ACT_EPS)) * 2.0 - 1.0
        upd = active & (w.sum(dim=1) > 0)                       # zero-weight minibatches take no step
        g = grad_all(params, x, y, w)
        loss = loss_all(params, x, y, w)
        t = t + upd.to(t.dtype)
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        new_p, new_m1, new_m2 = {}, {}, {}
        for k in params:
            shape = (M,) + (1,) * (params[k].dim() - 1)
            u_ = upd.view(shape)
            mk = torch.where(u_, m1[k] * b1 + g[k] * (1 - b1), m1[k])
            vk = torch.where(u_, m2[k] * b2 + g[k] * g[k] * (1 - b2), m2[k])
            denom = (vk.sqrt() / bc2.clamp_min(1e-30).sqrt().view(shape)) + eps
            stepped = params[k] - (lr / bc1.clamp_min(1e-30)).view(shape) * mk / denom
            new_p[k], new_m1[k], new_m2[k] = torch.where(u_, stepped, params[k]), mk, vk
        return new_p, new_m1, new_m2, t, torch.where(upd, loss, 0.0), upd
    engine = "eager"
    if compile:
        try:
            step = torch.compile(step)
            engine = "compiled"
        except Exception as e:  # recorded, never silent
            engine = f"eager (compile unavailable: {type(e).__name__})"

    m1 = {k: torch.zeros_like(v) for k, v in params.items()}
    m2 = {k: torch.zeros_like(v) for k, v in params.items()}
    t = torch.zeros(M, device=device)
    rows_dev = [torch.as_tensor(r, device=device) for r in rows]
    S = torch.tensor(steps, device=device)
    max_steps, max_epochs = max(steps), max(epochs)
    epoch_records = [[] for _ in jobs]
    t_train = time.time()
    for epoch in range(max_epochs):
        live = torch.tensor([epoch < e for e in epochs], device=device)
        perm = torch.zeros(M, max_steps * bs, dtype=torch.int64, device=device)
        for i, r in enumerate(rows_dev):
            if epoch < epochs[i]:
                p = r[torch.randperm(len(r), generator=_epoch_generator(jobs[i].seed, epoch, device), device=device)]
                perm[i, :steps[i] * bs] = p[:steps[i] * bs]
        loss_sum = torch.zeros(M, device=device)
        n_steps = torch.zeros(M, device=device)
        for b in range(max_steps):
            idx = perm[:, b * bs:(b + 1) * bs]
            active = live & (b < S)
            gstep = (epoch * S + b)[:, None]  # model i's optimizer step, as v1 counts it; a tensor, so compile never specializes
            params, m1, m2, t, loss, upd = step(params, m1, m2, t, idx, active, gstep)
            loss_sum += loss
            n_steps += upd.to(n_steps.dtype)
        seen_syn = torch.stack([SYN[perm[i, :steps[i] * bs]].sum() for i in range(M)])
        ls, ns, sy = loss_sum.tolist(), n_steps.tolist(), seen_syn.tolist()  # one host read per epoch
        for i in range(M):
            if epoch < epochs[i]:
                used = steps[i] * bs
                epoch_records[i].append({"epoch": epoch, "steps": steps[i],
                                         "mean_loss": ls[i] / ns[i] if ns[i] else None,
                                         "rows_seen": {"demonstration": int(used - sy[i]), "synthetic": int(sy[i])},
                                         "zero_weight_steps": int(steps[i] - ns[i])})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_done = time.time()

    out = []
    group = {"size": M, "table_rows": R, "table_blocks": table["blocks"], "engine": engine, "device": str(device),
             "torch": torch.__version__, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
             "data_mode": "resident", "setup_seconds": t_train - t_setup, "train_seconds": t_done - t_train}
    for i, (job, policy) in enumerate(zip(jobs, policies)):
        policy.net.to_empty(device="cpu")
        policy.net.load_state_dict({k: v[i].detach().cpu() for k, v in params.items()})
        policy.net.eval()
        sup = job.supervision
        source = np.asarray(sup.source)
        local = training_rows(sup, balance)
        ident = training_identity(sup, settings, job.seed, job.noise)
        record = {"trainer": "vector_mlp", "version": policy.version, "trainer_version": TRAINER_VERSION,
                  "training_identity_sha256": ident["sha256"], "seed": int(job.seed),
                  "settings": {**settings, "hidden_layers": list(settings["hidden_layers"])},
                  "device": str(device), "rows": int(len(rows[i])), "steps_per_epoch": steps[i],
                  "rows_dropped_per_epoch": int(len(rows[i]) % bs), "epochs_run": epochs[i],
                  "balance_labels": balance or None,
                  "row_sources": {"demonstration": int((source[local] == "demonstration").sum()),
                                  "synthetic": int((source[local] == "synthetic").sum())},
                  "epochs": epoch_records[i], "group": group,
                  "loss": "sum_i w_i mean_j (pi(s_i) - a_i)_j^2 / sum_i w_i per minibatch, scaled action space",
                  "optimizer": {"name": "adam", "learning_rate": lr, "betas": list(ADAM["betas"]), "eps": eps,
                                "weight_decay": ADAM["weight_decay"], "implementation": "functional, per-model masked"},
                  "shuffle": "per-epoch torch.randperm, Generator seeded by SeedSequence([seed, epoch]), on the device",
                  "noise_bc": None if job.noise is None else job.noise.descriptor()}
        out.append((policy, json.loads(json.dumps(record, default=str))))
    return out
