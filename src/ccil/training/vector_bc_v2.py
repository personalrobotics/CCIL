"""Policy trainer v2: many vector policies trained together on one device (#108).

Experimental reference infrastructure, like v1 (vector_bc.py): it trains the
same VectorBCPolicy (the d3rlpy-default MLP with tanh, standard and min_max
scalers, the weighted loss of ccil.supervision) under the paper's protocol
(Adam, batch size, epochs, balance_data), but trains a group of M models as one
batched program.

Data. One shared row table: each job's demonstration and synthetic blocks are
deduplicated by content, so the demonstrations shared by a group's seeds are
stored once. A model never holds an index array: its training rows are a few
segments of the table (balancing repeats the demonstration segment), and its
minibatches come from a counter-based permutation of [0, n_m): a keyed
unbalanced Feistel network over the next power of two with cycle-walking,
keyed by (seed, epoch). Positions map to table rows through the segments.
Working memory is therefore O(M * chunk * B + M * P) besides the table, where
chunk is the number of steps whose indices are computed together.

Two data modes, the same batches. The index function is ordinary torch code
that runs identically on the GPU and on the CPU:
- resident: the table lives on the device and minibatches are device gathers;
- streaming: the table stays in pinned host memory; a producer thread computes
  each chunk's indices on the device (they need only the segments), gathers
  the rows on the host, and copies them to the device on a side stream while
  the previous chunk trains (double-buffered).
  Device memory is independent of the table size.
The mode is chosen from the table size and a memory budget, or forced, and is
recorded; both modes give the same rows in the same order.

Step. torch.func vmap over the stacked parameters (stack_module_state): the
weighted loss, its gradient, and a functional Adam in torch.optim.Adam's update
order with per-model step counts and masks (models past their last step, and
minibatches whose weights sum to 0, take no step), compiled with torch.compile
on CUDA (eager elsewhere, or when compile is off); the engine is recorded.

NoiseBC runs on the device: noise_bc.py's counter-keyed splitmix64 draws,
bit-identical for the same key (int64 wrapping arithmetic, logical shifts by
masking), from each row's identity words computed once on the host.

Host synchronization happens once per chunk (to finish cycle-walking) and once
per epoch (losses, row counts), never per step.

v2 is not bit-compatible with v1 (the shuffle and the arithmetic order differ;
#108 replaces bit identity with engine-correctness checks and statistical
equivalence). Checkpoints are ordinary inference checkpoints of VectorBCPolicy;
v2 has no resume.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import functional_call, grad, stack_module_state, vmap

from ..core.identity import identity_words64
from .vector_bc import ACT_EPS, ADAM, OBS_EPS, TrainingError, VectorBCPolicy, training_identity

TRAINER_VERSION = "2"
CHUNK_STEPS = 256
FEISTEL_ROUNDS = 6
_M64 = (1 << 64) - 1


def _s64(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c


_GAMMA, _MIX1, _MIX2 = _s64(0x9E3779B97F4A7C15), _s64(0xBF58476D1CE4E5B9), _s64(0x94D049BB133111EB)
_SHUFFLE = _s64(0x5348554646 << 24)  # domain separation of the shuffle keys from NoiseBC's


def _lsr(x: torch.Tensor, k) -> torch.Tensor:
    """Logical right shift of int64 bit patterns by k (an int or a broadcastable int64 tensor)."""
    if isinstance(k, int):
        return (x >> k) & ((1 << (64 - k)) - 1)
    return (x >> k) & ((torch.ones_like(x) << (64 - k)) - 1)


def _mix(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer of x + gamma on int64 bit patterns (wrapping), as noise_bc._mix on uint64."""
    z = x + _GAMMA
    z = (z ^ _lsr(z, 30)) * _MIX1
    z = (z ^ _lsr(z, 27)) * _MIX2
    return z ^ _lsr(z, 31)


def counter_uniforms_device(seed: torch.Tensor, step: torch.Tensor, words: torch.Tensor, draw: int,
                            components: int) -> torch.Tensor:
    """noise_bc.counter_uniforms on the device: seed (M,), step (M, 1), words (M, B, 2) -> (M, B, components) float64."""
    h = _mix(seed[:, None].expand(words.shape[0], words.shape[1]))
    h = _mix(h ^ step)
    h = _mix(h ^ words[..., 0])
    h = _mix(h ^ words[..., 1])
    h = _mix(h ^ _s64(draw & _M64))
    h = _mix(h[..., None] ^ torch.arange(components, dtype=torch.int64, device=words.device))
    return _lsr(h, 11).to(torch.float64) * 2.0 ** -53


def _feistel(x, key, a, b):
    """A keyed bijection of [0, 2^(a+b)): unbalanced Feistel rounds (an even count restores the widths)."""
    hi, lo, wh, wl = _lsr(x, b), x & ((1 << b) - 1), a, b
    for r in range(FEISTEL_ROUNDS):
        f = _mix(lo ^ _mix(key ^ r))
        hi, lo, wh, wl = lo, hi ^ (f & ((1 << wh) - 1)), wl, wh
    return (hi << wl) | lo


def permute(pos: torch.Tensor, key: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """A keyed pseudo-random permutation of [0, n_m) applied to pos (M, K): Feistel over 2^ceil(log2 n) with
    cycle-walking (re-applied only to positions still outside [0, n)); the walk ends when all are in range."""
    k = torch.clamp(torch.ceil(torch.log2(n.double())).long(), min=2)[:, None]
    a, b = k // 2, k - k // 2
    n_, key_ = n[:, None], key[:, None]
    x = _feistel(pos, key_, a, b)
    out = x >= n_
    while bool(out.any()):  # one host check per walk iteration per chunk; the domain is < 2n, so few iterations
        x = torch.where(out, _feistel(x, key_, a, b), x)
        out = x >= n_
    return x


def shuffle_key(seed: int, epoch: int) -> int:
    """The permutation key of (seed, epoch): splitmix64 over the pair, domain-separated."""
    h = _mix(torch.tensor([_s64(int(seed) & _M64)]) ^ _SHUFFLE)
    return int(_mix(h ^ int(epoch))[0])


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
    """The shared row table, and each job's {part: (table start, rows)} blocks."""
    blocks, order, job_blocks = {}, [], []
    for job in jobs:
        sup = job.supervision
        obs, act, w = sup.arrays()
        mine = {}
        for part in ("demonstration", "synthetic"):
            rows = np.flatnonzero(sup.source == part)
            if not len(rows):
                continue
            if not np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
                raise TrainingError("supervision rows must be demonstrations, then synthetic rows (the contract)")
            key = _block_key(obs[rows], act[rows], w[rows], sup.episode_id[rows].astype(str), sup.frame[rows])
            if key not in blocks:
                start = sum(len(blocks[k][0]) for k in order)
                blocks[key] = (np.asarray(obs[rows], np.float32), np.asarray(act[rows], np.float32),
                               np.asarray(w[rows], np.float32), sup.episode_id[rows].astype(str),
                               np.asarray(sup.frame[rows], np.int64), part, start)
                order.append(key)
            mine[part] = (blocks[key][6], len(rows))
        job_blocks.append(mine)
    parts = [blocks[k] for k in order]
    table = {"obs": np.concatenate([p[0] for p in parts]), "act": np.concatenate([p[1] for p in parts]),
             "w": np.concatenate([p[2] for p in parts]), "episode_id": np.concatenate([p[3] for p in parts]),
             "frame": np.concatenate([p[4] for p in parts]),
             "synthetic": np.concatenate([np.full(len(p[0]), p[5] == "synthetic") for p in parts]),
             "blocks": [{"part": p[5], "rows": len(p[0]), "start": p[6]} for p in parts]}
    return table, job_blocks


def _segments(blocks: dict, balance) -> list[tuple[int, int]]:
    """A model's training rows as (table start, length) segments, in v1's training_rows order."""
    demo, syn = blocks.get("demonstration"), blocks.get("synthetic")
    if not balance:
        return [s for s in (demo, syn) if s]
    return ([syn] if syn else []) + ([demo] * (int(balance) + 1) if demo else [])


class _Streams:
    """Per-model index streams: segments, permutation keys, and chunked position-to-row maps."""

    def __init__(self, segments, seeds, bs, device):
        S = max(len(s) for s in segments)
        starts = np.zeros((len(segments), S), np.int64)
        cum = np.full((len(segments), S + 1), np.iinfo(np.int64).max, np.int64)
        for i, segs in enumerate(segments):
            cum[i, 0] = 0
            for j, (st, ln) in enumerate(segs):
                starts[i, j], cum[i, j + 1] = st, cum[i, j] + ln
        self.n = torch.tensor([sum(ln for _, ln in s) for s in segments], device=device)
        self.steps = (self.n // bs)
        self.starts, self.cum = torch.as_tensor(starts, device=device), torch.as_tensor(cum, device=device)
        self.seeds, self.bs, self.device = seeds, bs, device

    def chunk(self, epoch: int, step0: int, n_steps: int) -> torch.Tensor:
        """Table rows (M, n_steps, B) for steps [step0, step0 + n_steps) of epoch; positions past a model's last
        full minibatch are clamped (those steps are masked by the caller)."""
        M, bs = len(self.seeds), self.bs
        key = torch.tensor([shuffle_key(s, epoch) for s in self.seeds], device=self.device)
        pos = step0 * bs + torch.arange(n_steps * bs, device=self.device)
        pos = torch.minimum(pos[None, :].expand(M, -1), (self.steps * bs - 1)[:, None])
        perm = permute(pos, key, self.n)
        seg = torch.searchsorted(self.cum, perm, right=True) - 1
        return (self.starts.gather(1, seg) + perm - self.cum.gather(1, seg)).view(M, n_steps, bs)


def training_identity_v2(supervision, settings: dict, seed: int, noise) -> dict:
    """v1's training identity with the trainer version, so a v2 run is never mistaken for a v1 run."""
    ident = {k: v for k, v in training_identity(supervision, settings, seed, noise).items() if k != "sha256"}
    ident["trainer_version"] = TRAINER_VERSION
    return {**ident, "sha256": hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()}


def train_group(jobs: list[Job], settings: dict, *, device: str = "cpu", compile: bool | None = None,
                data_mode: str = "auto", memory_budget_fraction: float = 0.5, chunk_steps: int = CHUNK_STEPS,
                deterministic: bool = False):
    """Train every job's policy together; return [(policy, record)] in job order.

    settings are v1's (hidden_layers, learning_rate, batch_size, epochs, observation_scaler, action_scaler,
    balance_labels) and are shared by the group. compile defaults to True on CUDA and False elsewhere.
    data_mode: "auto" (resident when the table fits the budget, else streaming), "resident" or "streaming".
    deterministic: torch.use_deterministic_algorithms during training (recorded).
    """
    if not jobs:
        raise TrainingError("a group needs at least one job")
    if data_mode not in ("auto", "resident", "streaming"):
        raise TrainingError(f"data_mode {data_mode!r} is not auto, resident or streaming")
    device = torch.device(device)
    compile = (device.type == "cuda") if compile is None else bool(compile)
    for job in jobs:
        method = job.supervision.manifest()["method"]
        if (job.noise is not None) != (method == "noise_bc"):
            raise TrainingError("NoiseBC perturbation applies to the noise_bc arm, and the noise_bc arm needs one")
        if settings.get("balance_labels") and method != "ccil":
            raise TrainingError("balance_labels repeats demonstrations against labels; it applies to ccil only")
    balance = settings.get("balance_labels")
    bs = int(settings["batch_size"])
    t_setup = time.time()

    table, job_blocks = _table(jobs)
    segments = [_segments(b, balance) for b in job_blocks]
    epochs = [int(settings["epochs"]) // int(balance) if balance else int(settings["epochs"]) for _ in jobs]
    M, R = len(jobs), len(table["w"])
    D, A = table["obs"].shape[1], table["act"].shape[1]
    noisy = any(j.noise is not None for j in jobs)
    table_bytes = R * 4 * (D + A + 1) + (R * 16 if noisy else 0)
    if data_mode == "auto":
        free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else table_bytes * 4
        data_mode = "resident" if table_bytes <= memory_budget_fraction * free else "streaming"
    streams = _Streams(segments, [j.seed for j in jobs], bs, device)  # index math needs only the segments, never the table
    steps = streams.steps.tolist()
    for job, e, s in zip(jobs, epochs, steps):
        if e < 1 or s < 1:
            raise TrainingError(f"job (seed {job.seed}): {s} minibatches and {e} epochs leave no optimizer step")

    # Policies: v1's initialization per seed and v1's scalers (fitted on the model's training rows), then stacked.
    policies = []
    for job, segs in zip(jobs, segments):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(job.seed))
            p = VectorBCPolicy(D, A, settings["hidden_layers"], settings["observation_scaler"], settings["action_scaler"])
        rows = np.concatenate([np.arange(st, st + ln) for st, ln in segs])
        p.fit_scalers(table["obs"][rows], table["act"][rows])
        policies.append(p)
    params, _ = stack_module_state([p.net for p in policies])
    params = {k: v.detach().to(device) for k, v in params.items()}
    template = policies[0].net.to("meta")

    def stat(name, default):
        return torch.tensor(np.stack([getattr(p, name) if getattr(p, name) is not None else default for p in policies]),
                            dtype=torch.float32, device=device)
    obs_mean, obs_std = stat("obs_mean", np.zeros(D)), stat("obs_std", np.ones(D) - OBS_EPS)
    act_min, act_max = stat("act_min", -np.ones(A)), stat("act_max", np.ones(A))
    scale_obs, scale_act = settings["observation_scaler"] != "none", settings["action_scaler"] != "none"

    host = {"x": torch.as_tensor(table["obs"]), "y": torch.as_tensor(table["act"]), "w": torch.as_tensor(table["w"]),
            "syn": torch.as_tensor(table["synthetic"])}
    if noisy:
        host["words"] = torch.as_tensor(np.array([identity_words64(e, f) for e, f in zip(table["episode_id"], table["frame"])],
                                                 dtype=np.uint64).reshape(-1, 2).view(np.int64))
        n_seed = torch.tensor([_s64(int(j.noise.seed) & _M64) if j.noise else 0 for j in jobs], device=device)
        n_scale = torch.tensor([j.noise.scale if j.noise else 0.0 for j in jobs], dtype=torch.float64, device=device)
        n_thresh = torch.tensor([1.0 - j.noise.mask_probability if j.noise else 2.0 for j in jobs],
                                dtype=torch.float64, device=device)
        n_sym = torch.tensor([j.noise is not None and j.noise.mode == "symmetric_uniform_half_batch" for j in jobs],
                             device=device)
    if data_mode == "resident":
        dev_table = {k: v.to(device) for k, v in host.items()}
    else:
        host = {k: v.pin_memory() if device.type == "cuda" else v for k, v in host.items()}

    b1, b2 = ADAM["betas"]
    eps, lr = ADAM["eps"], float(settings["learning_rate"])

    def loss_one(p, x, y, w):
        per_row = ((functional_call(template, p, (x,)) - y) ** 2).mean(dim=1)
        return (w * per_row).sum() / w.sum().clamp_min(1e-30)
    grad_all, loss_all = vmap(grad(loss_one)), vmap(loss_one)

    def step(params, m1, m2, t, x, y, w, words, active, gstep):
        """One masked Adam step for all models on a gathered minibatch (M, B, ·)."""
        if scale_obs:
            x = (x - obs_mean[:, None]) / (obs_std[:, None] + OBS_EPS)
        if noisy:
            u = counter_uniforms_device(n_seed, gstep, words, 0, 1 + D)
            mask = u[..., 0] > n_thresh[:, None]
            unit = torch.where(n_sym[:, None, None], 2.0 * u[..., 1:] - 1.0, u[..., 1:])
            x = x + torch.where(mask[..., None], n_scale[:, None, None] * unit, 0.0).to(x.dtype)
        if scale_act:
            y = ((y - act_min[:, None]) / (act_max[:, None] - act_min[:, None] + ACT_EPS)) * 2.0 - 1.0
        upd = active & (w.sum(dim=1) > 0)  # zero-weight minibatches take no step, as in v1
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

    def gather(src, idx):
        return {k: src[k][idx] for k in src}

    def chunks(epoch):
        """(first step, gathered device batch dict for the chunk) for every chunk of the epoch."""
        max_steps = max(s for s, e in zip(steps, epochs) if epoch < e)
        bounds = [(s0, min(chunk_steps, max_steps - s0)) for s0 in range(0, max_steps, chunk_steps)]
        if data_mode == "resident":
            for s0, n in bounds:
                yield s0, gather(dev_table, streams.chunk(epoch, s0, n))
            return
        q: queue.Queue = queue.Queue(maxsize=2)  # double buffering: one chunk training, one in flight
        side = torch.cuda.Stream(device) if device.type == "cuda" else None

        def produce():
            for s0, n in bounds:
                if side is not None:  # indices on the device (the same code as resident mode), rows from the host
                    with torch.cuda.stream(side):
                        idx = streams.chunk(epoch, s0, n).cpu()
                    cpu = {k: v.pin_memory() for k, v in gather(host, idx).items()}
                    with torch.cuda.stream(side):
                        dev = {k: v.to(device, non_blocking=True) for k, v in cpu.items()}
                    ev = torch.cuda.Event()
                    ev.record(side)
                    q.put((s0, dev, ev, cpu))
                else:
                    cpu = gather(host, streams.chunk(epoch, s0, n))
                    q.put((s0, cpu, None, cpu))
            q.put(None)
        th = threading.Thread(target=produce, daemon=True)
        th.start()
        while (item := q.get()) is not None:
            s0, dev, ev, _keepalive = item
            if ev is not None:
                main = torch.cuda.current_stream(device)
                main.wait_event(ev)
                for v in dev.values():  # allocated on the side stream, used on this one: keep the allocator from
                    v.record_stream(main)  # reusing the memory before this stream's kernels have read it
            yield s0, dev
        th.join()

    m1 = {k: torch.zeros_like(v) for k, v in params.items()}
    m2 = {k: torch.zeros_like(v) for k, v in params.items()}
    t = torch.zeros(M, device=device)
    S_dev = torch.tensor(steps, device=device)
    epochs_dev = torch.tensor(epochs, device=device)
    epoch_records = [[] for _ in jobs]
    t_train = time.time()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(bool(deterministic) or previous)
    try:
        for epoch in range(max(epochs)):
            live = epoch < epochs_dev
            loss_sum = torch.zeros(M, device=device)
            n_steps = torch.zeros(M, device=device)
            seen_syn = torch.zeros(M, device=device)
            for s0, batch in chunks(epoch):
                for j in range(batch["x"].shape[1]):
                    b = s0 + j
                    active = live & (b < S_dev)
                    gstep = (epoch * S_dev + b)[:, None]  # model m's optimizer step as v1 counts it (NoiseBC key)
                    params, m1, m2, t, loss, upd = step(params, m1, m2, t, batch["x"][:, j], batch["y"][:, j],
                                                        batch["w"][:, j], batch["words"][:, j] if noisy else None,
                                                        active, gstep)
                    loss_sum += loss
                    n_steps += upd.to(n_steps.dtype)
                    seen_syn += torch.where(active, batch["syn"][:, j].sum(dim=1).to(seen_syn.dtype), 0.0)
            ls, ns, sy = loss_sum.tolist(), n_steps.tolist(), seen_syn.tolist()  # one host read per epoch
            for i in range(M):
                if epoch < epochs[i]:
                    epoch_records[i].append({"epoch": epoch, "steps": steps[i],
                                             "mean_loss": ls[i] / ns[i] if ns[i] else None,
                                             "rows_seen": {"demonstration": int(steps[i] * bs - sy[i]),
                                                           "synthetic": int(sy[i])},
                                             "zero_weight_steps": int(steps[i] - ns[i])})
    finally:
        torch.use_deterministic_algorithms(previous)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_done = time.time()

    group = {"size": M, "table_rows": R, "table_bytes": table_bytes, "table_blocks": table["blocks"],
             "data_mode": data_mode, "chunk_steps": chunk_steps, "engine": engine, "device": str(device),
             "deterministic": bool(deterministic),
             "torch": torch.__version__, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
             "shuffle": "counter-based permutation of [0, n): 6-round unbalanced Feistel (splitmix64 rounds) over "
                        "2^ceil(log2 n) with cycle-walking, keyed by splitmix64(seed, epoch)",
             "setup_seconds": t_train - t_setup, "train_seconds": t_done - t_train}
    out = []
    for i, (job, policy) in enumerate(zip(jobs, policies)):
        policy.net.to_empty(device="cpu")
        policy.net.load_state_dict({k: v[i].detach().cpu() for k, v in params.items()})
        policy.net.eval()
        n_demo = sum(ln for (st, ln) in segments[i] if job_blocks[i].get("demonstration") == (st, ln))
        ident = training_identity_v2(job.supervision, settings, job.seed, job.noise)
        record = {"trainer": "vector_mlp", "version": policy.version, "trainer_version": TRAINER_VERSION,
                  "training_identity_sha256": ident["sha256"], "seed": int(job.seed),
                  "settings": {**settings, "hidden_layers": list(settings["hidden_layers"])},
                  "device": str(device), "rows": int(streams.n[i]), "steps_per_epoch": steps[i],
                  "rows_dropped_per_epoch": int(streams.n[i]) % bs, "epochs_run": epochs[i],
                  "balance_labels": balance or None,
                  "row_sources": {"demonstration": int(n_demo), "synthetic": int(streams.n[i]) - int(n_demo)},
                  "epochs": epoch_records[i], "group": group,
                  "loss": "sum_i w_i mean_j (pi(s_i) - a_i)_j^2 / sum_i w_i per minibatch, scaled action space",
                  "optimizer": {"name": "adam", "learning_rate": lr, "betas": list(ADAM["betas"]), "eps": eps,
                                "weight_decay": ADAM["weight_decay"], "implementation": "functional, per-model masked"},
                  "noise_bc": None if job.noise is None else job.noise.descriptor()}
        out.append((policy, json.loads(json.dumps(record, default=str))))
    return out
