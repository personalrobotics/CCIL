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
Working memory is therefore O(M * chunk * B) indices + O(M * P) parameters
besides the table: resident mode gathers each step's rows inside the step, and
streaming mode sizes a chunk of gathered rows by a byte budget, so neither
grows with the observation width or the table.

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
minibatches whose weights sum to 0, take no step). The step is a module-level
function of its tensors, compiled with torch.compile once per process on CUDA
(eager elsewhere, or when compile is off), so later groups of the same shapes
reuse the compiled code; the engine is recorded.

Setup is linear in the distinct data: blocks are keyed by hashing raw bytes,
scalers come from per-block statistics combined with their repetitions (each
distinct row set once), and NoiseBC identity words are hashed once per distinct
(episode, frame).

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
from torch.func import grad, stack_module_state, vmap

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
    rows, cols = torch.nonzero(x >= n_, as_tuple=True)
    while len(rows):  # walk only the positions still outside [0, n): the domain is < 2n, so the set halves each pass
        y = _feistel(x[rows, cols], key[rows], a[rows, 0], b[rows, 0])
        x[rows, cols] = y
        keep = y >= n[rows]
        rows, cols = rows[keep], cols[keep]
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
    validation: object = None  # held-out supervision, for checkpoint_selection "validation"


def _bytes_of(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a)
    if a.dtype == object:
        a = a.astype(str)  # only for object arrays; supervision ids are already fixed-width unicode
    return str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()


def _block_key(*arrays) -> str:
    """Content identity of a block: BLAKE2b over the raw bytes, with no per-element conversion."""
    h = hashlib.blake2b(digest_size=32)
    for a in arrays:
        h.update(_bytes_of(a))
    return h.hexdigest()


def _table(jobs):
    """The shared row table, each job's {part: (table start, rows)} blocks, and the table's block list.

    Blocks are deduplicated by content, so seeds that share demonstrations (or labels) store them once; a
    supervision object seen twice is keyed once.
    """
    blocks, order, job_blocks, seen = {}, [], [], {}
    for job in jobs:
        sup = job.supervision
        if id(sup) in seen:
            job_blocks.append(seen[id(sup)])
            continue
        obs, act, w = sup.arrays()
        mine = {}
        for part in ("demonstration", "synthetic"):
            rows = np.flatnonzero(sup.source == part)
            if not len(rows):
                continue
            lo, hi = int(rows[0]), int(rows[0]) + len(rows)
            if rows[-1] != hi - 1:
                raise TrainingError("supervision rows must be demonstrations, then synthetic rows (the contract)")
            key = _block_key(np.array(part), obs[lo:hi], act[lo:hi], w[lo:hi], sup.episode_id[lo:hi], sup.frame[lo:hi])
            if key not in blocks:
                start = sum(blocks[k]["rows"] for k in order)
                blocks[key] = {"part": part, "rows": hi - lo, "start": start, "sup": sup, "lo": lo, "hi": hi}
                order.append(key)
            mine[part] = (blocks[key]["start"], hi - lo)
        seen[id(sup)] = mine
        job_blocks.append(mine)
    parts = [blocks[k] for k in order]

    def cat(fn, dtype):
        return np.concatenate([np.asarray(fn(p["sup"])[p["lo"]:p["hi"]], dtype) for p in parts])
    table = {"obs": cat(lambda u: u.observations, np.float32), "act": cat(lambda u: u.actions, np.float32),
             "w": cat(lambda u: u.weights, np.float32),
             "synthetic": np.concatenate([np.full(p["rows"], p["part"] == "synthetic") for p in parts]),
             "blocks": [{"part": p["part"], "rows": p["rows"], "start": p["start"]} for p in parts],
             "_parts": parts}
    return table, job_blocks


def _identity_words(table) -> np.ndarray:
    """(R, 2) int64 identity words of every table row, hashing each distinct (episode_id, frame) once."""
    out = np.empty((sum(p["rows"] for p in table["_parts"]), 2), np.int64)
    for p in table["_parts"]:
        ep = np.asarray(p["sup"].episode_id[p["lo"]:p["hi"]])
        fr = np.asarray(p["sup"].frame[p["lo"]:p["hi"]], np.int64)
        keys, inverse = np.unique(np.stack([ep.astype(str), fr.astype(str)], 1), axis=0, return_inverse=True)
        words = np.array([identity_words64(e, int(f)) for e, f in keys], np.uint64).reshape(-1, 2).view(np.int64)
        out[p["start"]:p["start"] + p["rows"]] = words[inverse.reshape(-1)]
    return out


def _scalers(table, segments, observation_scaler, action_scaler):
    """v1's scalers for a model's training rows (with balancing repetitions) from per-block float64 statistics.

    The observation mean and population std combine blocks by Chan's parallel update, so repeated
    demonstrations count with their multiplicity without materializing rows; each distinct block is
    reduced once. Same quantities as v1's fit_scalers, not bit-identical arithmetic.
    """
    stats = {}

    def block(start, length):
        if (start, length) not in stats:
            o = np.asarray(table["obs"][start:start + length], np.float64)
            a = table["act"][start:start + length]
            mean = o.mean(axis=0)
            stats[(start, length)] = (length, mean, ((o - mean) ** 2).sum(axis=0), a.min(axis=0), a.max(axis=0),
                                      float(np.abs(a).max()))
        return stats[(start, length)]
    n, mean, m2 = 0, 0.0, 0.0
    amin, amax, amax_abs = None, None, 0.0
    for st, ln in segments:
        k, mu, q, lo, hi, ab = block(st, ln)
        delta = mu - mean
        tot = n + k
        mean = mean + delta * (k / tot)
        m2 = m2 + q + delta ** 2 * (n * k / tot)
        n = tot
        amin = lo if amin is None else np.minimum(amin, lo)
        amax = hi if amax is None else np.maximum(amax, hi)
        amax_abs = max(amax_abs, ab)
    out = {"obs_mean": None, "obs_std": None, "act_min": None, "act_max": None}
    if observation_scaler == "standard":
        out["obs_mean"], out["obs_std"] = mean, np.sqrt(m2 / n)
    if action_scaler == "min_max":
        out["act_min"], out["act_max"] = amin, amax
    elif amax_abs > 1:
        raise TrainingError("action_scaler none needs actions in [-1, 1]: the policy output is tanh")
    return out


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


def parameters_sha256(policy) -> str:
    """SHA-256 of a policy's returned network parameters, binding a training record to the weights it describes."""
    h = hashlib.sha256()
    for name, value in sorted(policy.net.state_dict().items()):
        h.update(name.encode() + value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def training_identity_v2(supervision, settings: dict, seed: int, noise, validation=None) -> dict:
    """v1's training identity with the trainer version, so a v2 run is never mistaken for a v1 run, and the
    held-out validation supervision that selected the checkpoint (#115), on which the returned policy depends."""
    ident = {k: v for k, v in training_identity(supervision, settings, seed, noise).items() if k != "sha256"}
    ident["trainer_version"] = TRAINER_VERSION
    if validation is not None:  # by content: the rows that judge the checkpoints, not only their provenance
        o, act, w = validation.arrays()
        ident["validation_sha256"] = _block_key(o, act, w, np.asarray(validation.episode_id).astype(str),
                                                np.asarray(validation.frame))
        ident["validation_inputs"] = validation.manifest()["inputs"]
    return {**ident, "sha256": hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()}


def _mlp(p: dict, x: torch.Tensor, layers: int) -> torch.Tensor:
    """VectorBCPolicy.net's forward from its state-dict names (Sequential: Linear, ReLU, ..., Linear, Tanh)."""
    for i in range(layers):
        x = x @ p[f"{2 * i}.weight"].T + p[f"{2 * i}.bias"]
        x = torch.relu(x) if i < layers - 1 else torch.tanh(x)
    return x


def _lr_at(t: torch.Tensor, lr: float, r: dict, total: torch.Tensor) -> torch.Tensor:
    """The learning rate at each model's step count t (after this step): constant, or linear warm-up to lr then
    cosine decay to 0 at the model's total steps."""
    if r["lr_schedule"] == "constant":
        return torch.full_like(t, lr)
    warm = float(max(1, int(r["warmup_steps"])))
    progress = ((t - warm) / (total - warm).clamp_min(1.0)).clamp(0.0, 1.0)
    return lr * torch.where(t <= warm, t / warm, 0.5 * (1.0 + torch.cos(np.pi * progress)))


def _step(params, m1, m2, t, ema, x, y, w, words, active, gstep, c, layers: int, noisy: bool, scale_obs: bool,
          scale_act: bool, lr: float, recipe: tuple):
    """One masked optimizer step for all models on a gathered minibatch (M, B, ·). c holds the per-model constants.

    recipe (a sorted tuple of items, so it is a compile-time constant) selects the optimizer (adam: v1's, with
    L2 weight decay 0; adamw: decoupled weight decay), the learning-rate schedule, global-norm gradient clipping
    per model, and an EMA of the weights; the defaults are v1's protocol.
    """
    r = dict(recipe)
    M = active.shape[0]
    if scale_obs:
        x = (x - c["obs_mean"][:, None]) / (c["obs_std"][:, None] + OBS_EPS)
    if noisy:
        u = counter_uniforms_device(c["n_seed"], gstep, words, 0, 1 + x.shape[-1])
        mask = u[..., 0] > c["n_thresh"][:, None]
        unit = torch.where(c["n_sym"][:, None, None], 2.0 * u[..., 1:] - 1.0, u[..., 1:])
        x = x + torch.where(mask[..., None], c["n_scale"][:, None, None] * unit, 0.0).to(x.dtype)
    if scale_act:
        y = ((y - c["act_min"][:, None]) / (c["act_max"][:, None] - c["act_min"][:, None] + ACT_EPS)) * 2.0 - 1.0

    def loss_one(p, x_, y_, w_):
        per_row = ((_mlp(p, x_, layers) - y_) ** 2).mean(dim=1)
        return (w_ * per_row).sum() / w_.sum().clamp_min(1e-30)
    g = vmap(grad(loss_one))(params, x, y, w)
    loss = vmap(loss_one)(params, x, y, w)
    upd = active & (w.sum(dim=1) > 0)  # zero-weight minibatches take no step, as in v1
    if r["grad_clip_norm"]:  # global norm over each model's parameters
        norm = torch.sqrt(sum((g[k] ** 2).reshape(M, -1).sum(dim=1) for k in g))
        coef = (r["grad_clip_norm"] / (norm + 1e-6)).clamp(max=1.0)
        g = {k: v * coef.view((M,) + (1,) * (v.dim() - 1)) for k, v in g.items()}
    b1, b2 = ADAM["betas"]
    t = t + upd.to(t.dtype)
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    lr_t = _lr_at(t, lr, r, c["total_steps"])
    new_p, new_m1, new_m2, new_ema = {}, {}, {}, ({} if ema is not None else None)
    for k in params:
        shape = (M,) + (1,) * (params[k].dim() - 1)
        u_ = upd.view(shape)
        p = params[k]
        if r["optimizer"] == "adamw" and r["weight_decay"]:  # decoupled: p <- p (1 - lr wd), then the Adam step
            p = p * (1 - lr_t * r["weight_decay"]).view(shape)
        mk = torch.where(u_, m1[k] * b1 + g[k] * (1 - b1), m1[k])
        vk = torch.where(u_, m2[k] * b2 + g[k] * g[k] * (1 - b2), m2[k])
        denom = (vk.sqrt() / bc2.clamp_min(1e-30).sqrt().view(shape)) + ADAM["eps"]
        stepped = p - (lr_t / bc1.clamp_min(1e-30)).view(shape) * mk / denom
        new_p[k], new_m1[k], new_m2[k] = torch.where(u_, stepped, params[k]), mk, vk
        if ema is not None:
            new_ema[k] = torch.where(u_, ema[k] * r["ema_decay"] + new_p[k] * (1 - r["ema_decay"]), ema[k])
    return new_p, new_m1, new_m2, t, new_ema, torch.where(upd, loss, 0.0), upd


def _step_resident(params, m1, m2, t, ema, X, Y, W, WORDS, idx, active, gstep, c, layers: int, noisy: bool,
                   scale_obs: bool, scale_act: bool, lr: float, recipe: tuple):
    """_step with the minibatch gathered from the device-resident table inside the step (no chunk of rows)."""
    return _step(params, m1, m2, t, ema, X[idx], Y[idx], W[idx], WORDS[idx] if noisy else None, active, gstep, c,
                 layers, noisy, scale_obs, scale_act, lr, recipe)


_COMPILED: dict = {}
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)  # group sizes are padded to these, so compilation specializes per bucket
# Plain compilation (Inductor). CUDA graphs ("reduce-overhead") were measured slower here: a replay copies every
# input that is not static into its own buffers, which for the resident table is the whole table on every step
# (3.4 against 1.2 ms per step at D = 512, 2.6 against 1.0 at 64 models; #117).
COMPILE_MODE = "default"
CUDA_GRAPHS = COMPILE_MODE in ("reduce-overhead", "max-autotune")


def _compiled(fn):
    """fn compiled once per process with COMPILE_MODE; torch.compile then specializes per shape bucket.

    Keyed by module and qualified name: the policy trainer and the batched fits (models/mlp_batched.py) both
    have a _step, and one process (a study runner) compiles both. dynamic=False specializes every setting, so a
    recompile never turns a float hyperparameter into a symbolic one (Inductor refused one, "x must be Tensor or
    scalar, got SymFloat", for the slack loss after a recompile)."""
    key = f"{fn.__module__}.{fn.__qualname__}"
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(fn, mode=COMPILE_MODE, dynamic=False)
    return _COMPILED[key]


class _Engine:
    """The step, compiled or eager. Compilation is lazy, so the first compiled call is guarded: a compiler
    failure before any update falls back to eager for the whole run (the step is functional, so a failed call
    changed nothing), and the outcome is recorded."""

    def __init__(self, fn, compile: bool):
        self.fn, self.eager = (_compiled(fn), fn) if compile else (fn, fn)
        self.state = f"compiled ({COMPILE_MODE})" if compile else "eager"
        self.first = compile

    def __call__(self, *args, **kwargs):
        if not self.first:
            return self.fn(*args, **kwargs)
        self.first = False
        try:
            return self.fn(*args, **kwargs)
        except Exception as e:  # a compiler failure on the first call: recorded, never silent
            self.fn, self.state = self.eager, f"eager (compilation failed on the first step: {type(e).__name__}: {e})"[:300]
            return self.fn(*args, **kwargs)


def _bucket(m: int) -> int:
    for b in BUCKETS:
        if m <= b:
            return b
    raise TrainingError(f"a group has at most {BUCKETS[-1]} models; split {m} jobs into several groups")


class _Precision:
    """float32 with TF32 off for the training scope (the #108 protocol), restoring the caller's settings."""

    def __enter__(self):
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                      torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.effective = {"dtype": "float32", "tf32_matmul": False, "tf32_cudnn": False,
                          "float32_matmul_precision": "highest"}
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, prec = self.saved
        torch.set_float32_matmul_precision(prec)
        return False


def _own(out):
    """Clone a compiled step's outputs: a CUDA-graph replay overwrites its previous outputs."""
    return tuple(v.clone() if isinstance(v, torch.Tensor) else None if v is None else {k: x.clone() for k, x in v.items()}
                 for v in out)  # None: no EMA


def _validation_rows(padded, D, A, chunk_bytes: int):
    """The models' held-out rows as one deduplicated host table and per-model segments.

    Identical validation sets (for example the arms of one seed) are stored once. Each model's set is a segment
    (start, n) of the table, and the loss reads it in chunks of at most `rows` rows per model, so device memory
    is bounded by chunk_bytes whatever the number and size of the validation sets. A set with zero total weight
    is refused: it cannot select a checkpoint.
    """
    blocks, segments, parts = {}, [], []
    for job in padded:
        o, a, w = job.validation.arrays()
        if not np.isfinite(w).all() or float(np.sum(w)) <= 0:
            raise TrainingError(f"job (seed {job.seed}): the validation supervision has zero total weight")
        key = _block_key(o, a, w)
        if key not in blocks:
            blocks[key] = (sum(len(p[2]) for p in parts), len(w))
            parts.append((np.asarray(o, np.float32), np.asarray(a, np.float32), np.asarray(w, np.float32)))
        segments.append(blocks[key])
    table = {k: torch.as_tensor(np.concatenate([p[i] for p in parts])) for i, k in enumerate(("x", "y", "w"))}
    rows = max(1, int(chunk_bytes) // (len(padded) * 4 * (D + A + 1)))
    return {**table, "segments": segments, "n": [n for _, n in segments], "rows": rows, "blocks": len(parts)}


def _validation_loss(params, v, c, layers, scale_obs, scale_act, device):
    """The weighted training loss of every model on its held-out rows (M,).

    Chunk j gathers rows [j * rows, (j + 1) * rows) of each model's segment from the host table (padding rows
    carry zero weight), so at most M * rows rows are on the device at once."""
    M, rows = len(v["segments"]), v["rows"]
    start = torch.tensor([st for st, _ in v["segments"]])
    n = torch.tensor(v["n"])
    num = torch.zeros(M, device=device, dtype=torch.float64)
    den = torch.zeros_like(num)
    with torch.no_grad():
        for j in range(0, int(n.max()), rows):
            pos = j + torch.arange(rows)
            valid = pos[None, :] < n[:, None]                            # (M, rows)
            idx = torch.where(valid, start[:, None] + pos[None, :], 0)
            x, y = v["x"][idx].to(device), v["y"][idx].to(device)
            w = torch.where(valid, v["w"][idx], 0.0).to(device)
            if scale_obs:
                x = (x - c["obs_mean"][:, None]) / (c["obs_std"][:, None] + OBS_EPS)
            if scale_act:
                y = ((y - c["act_min"][:, None]) / (c["act_max"][:, None] - c["act_min"][:, None] + ACT_EPS)) * 2.0 - 1.0
            per = ((vmap(lambda p, x_: _mlp(p, x_, layers))(params, x) - y) ** 2).mean(dim=2)
            num += (w * per).sum(dim=1).double()
            den += w.sum(dim=1).double()
    return (num / den).float()


def _put(q: queue.Queue, item, stop: threading.Event) -> bool:
    """Put unless the consumer has stopped; never blocks forever."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


RECIPE_DEFAULTS = {"optimizer": "adam", "weight_decay": 0.0, "lr_schedule": "constant", "warmup_steps": 0,
                   "step_budget": None, "grad_clip_norm": None, "ema_decay": None, "checkpoint_selection": "last",
                   "validation_interval_steps": None}


def recipe_of(settings: dict) -> dict:
    """The training recipe (#115) in settings, with v1's protocol as the default; refuses inconsistent recipes."""
    r = {k: settings.get(k, v) for k, v in RECIPE_DEFAULTS.items()}
    r = {k: (RECIPE_DEFAULTS[k] if v is None and RECIPE_DEFAULTS[k] is not None else v) for k, v in r.items()}
    if r["optimizer"] not in ("adam", "adamw"):
        raise TrainingError(f"optimizer {r['optimizer']!r} is not adam or adamw")
    if r["optimizer"] == "adam" and r["weight_decay"]:
        raise TrainingError("weight decay with adam would be L2 in the gradient, not the declared protocol; use adamw")
    if r["lr_schedule"] not in ("constant", "cosine"):
        raise TrainingError(f"lr_schedule {r['lr_schedule']!r} is not constant or cosine")
    if r["checkpoint_selection"] not in ("last", "validation"):
        raise TrainingError(f"checkpoint_selection {r['checkpoint_selection']!r} is not last or validation")
    if r["ema_decay"] is not None and not 0 < float(r["ema_decay"]) < 1:
        raise TrainingError("ema_decay must be in (0, 1)")
    if r["validation_interval_steps"] is not None and (r["checkpoint_selection"] != "validation"
                                                       or int(r["validation_interval_steps"]) < 1):
        raise TrainingError("validation_interval_steps is a positive step count, for checkpoint_selection validation")
    return r


def train_group(jobs: list[Job], settings: dict, *, device: str = "cpu", compile: bool | None = None,
                data_mode: str = "auto", memory_budget_fraction: float = 0.5, chunk_steps: int = CHUNK_STEPS,
                stream_chunk_bytes: int = 256 * 2**20, deterministic: bool = False):
    """Train every job's policy together; return [(policy, record)] in job order.

    settings are v1's (hidden_layers, learning_rate, batch_size, epochs, observation_scaler, action_scaler,
    balance_labels) and are shared by the group. compile defaults to True on CUDA and False elsewhere.
    data_mode: "auto" (resident when the table fits the budget, else streaming), "resident" or "streaming".
    Streaming chunks are sized so one chunk of gathered rows is at most stream_chunk_bytes.
    deterministic: torch.use_deterministic_algorithms during training (recorded).

    The group is padded to the next size in BUCKETS with masked copies of its first job (they never update and
    are not returned), so a group's compiled shapes, and therefore its float32 rounding, depend only on its
    bucket; each record carries the group's execution identity (ordered members, its position, bucket, engine,
    mode), because results can depend on it through rounding.
    """
    if not jobs:
        raise TrainingError("a group needs at least one job")
    if data_mode not in ("auto", "resident", "streaming"):
        raise TrainingError(f"data_mode {data_mode!r} is not auto, resident or streaming")
    device = torch.device(device)
    compile = (device.type == "cuda") if compile is None else bool(compile)
    balance = settings.get("balance_labels")
    for job in jobs:
        method = job.supervision.manifest()["method"]
        if (job.noise is not None) != (method == "noise_bc"):
            raise TrainingError("NoiseBC perturbation applies to the noise_bc arm, and the noise_bc arm needs one")
        if balance and method != "ccil":
            raise TrainingError("balance_labels repeats demonstrations against labels; it applies to ccil only")
    bs = int(settings["batch_size"])
    recipe = recipe_of(settings)
    if recipe["checkpoint_selection"] == "validation" and any(j.validation is None for j in jobs):
        raise TrainingError("checkpoint_selection validation needs every job's held-out validation supervision")
    t_setup = time.time()

    n_real = len(jobs)
    M = _bucket(n_real)
    padded = list(jobs) + [jobs[0]] * (M - n_real)  # masked pads: they share job 0's data and never update
    real = torch.tensor([i < n_real for i in range(M)], device=device)

    table, job_blocks = _table(padded)
    segments = [_segments(b, balance) for b in job_blocks]
    epochs = [int(settings["epochs"]) // int(balance) if balance else int(settings["epochs"]) for _ in padded]
    R = len(table["w"])
    D, A = table["obs"].shape[1], table["act"].shape[1]
    noisy = any(j.noise is not None for j in jobs)
    row_bytes = 4 * (D + A + 1) + 1 + (16 if noisy else 0)  # features, weight, source flag, identity words
    table_bytes = R * row_bytes
    if data_mode == "auto":
        free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else table_bytes * 4
        data_mode = "resident" if table_bytes <= memory_budget_fraction * free else "streaming"
    if data_mode == "streaming":  # bound a chunk of gathered rows, whatever the row width
        chunk_steps = max(1, min(chunk_steps, stream_chunk_bytes // (M * bs * row_bytes)))
    streams = _Streams(segments, [j.seed for j in padded], bs, device)  # index math needs only the segments
    steps = streams.steps.tolist()
    budget = recipe["step_budget"]
    if budget:  # a step budget replaces epochs: as many epochs as it takes, the last one cut at the budget
        epochs = [-(-int(budget) // n) for n in steps]
    total_steps = [int(budget) if budget else e * n for e, n in zip(epochs, steps)]
    for job, e, n, total in zip(jobs, epochs, steps, total_steps):
        if e < 1 or n < 1:
            raise TrainingError(f"job (seed {job.seed}): {n} minibatches and {e} epochs leave no optimizer step")
        if recipe["lr_schedule"] == "cosine" and int(recipe["warmup_steps"]) >= total:
            raise TrainingError(f"job (seed {job.seed}): warmup_steps {recipe['warmup_steps']} is not shorter than its "
                                f"{total} optimizer steps, so the cosine decay never starts")

    # Policies: v1's initialization per seed, v1's scalers (once per distinct row set), then stacked.
    policies, scaler_cache = [], {}
    for job, segs in zip(padded, segments):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(job.seed))
            p = VectorBCPolicy(D, A, settings["hidden_layers"], settings["observation_scaler"], settings["action_scaler"])
        key = tuple(segs)
        if key not in scaler_cache:
            scaler_cache[key] = _scalers(table, segs, settings["observation_scaler"], settings["action_scaler"])
        for name, value in scaler_cache[key].items():
            setattr(p, name, value)
        policies.append(p)
    params, _ = stack_module_state([p.net for p in policies])
    params = {k: v.detach().to(device) for k, v in params.items()}
    layers = len(settings["hidden_layers"]) + 1

    def stat(name, default):
        return torch.tensor(np.stack([getattr(p, name) if getattr(p, name) is not None else default for p in policies]),
                            dtype=torch.float32, device=device)
    c = {"obs_mean": stat("obs_mean", np.zeros(D)), "obs_std": stat("obs_std", np.ones(D) - OBS_EPS),
         "act_min": stat("act_min", -np.ones(A)), "act_max": stat("act_max", np.ones(A)),
         "total_steps": torch.tensor(total_steps, dtype=torch.float32, device=device)}
    scale_obs, scale_act = settings["observation_scaler"] != "none", settings["action_scaler"] != "none"
    host = {"x": torch.as_tensor(table["obs"]), "y": torch.as_tensor(table["act"]), "w": torch.as_tensor(table["w"]),
            "syn": torch.as_tensor(table["synthetic"])}
    if noisy:
        host["words"] = torch.as_tensor(_identity_words(table))
        c["n_seed"] = torch.tensor([_s64(int(j.noise.seed) & _M64) if j.noise else 0 for j in padded], device=device)
        c["n_scale"] = torch.tensor([j.noise.scale if j.noise else 0.0 for j in padded], dtype=torch.float64,
                                    device=device)
        c["n_thresh"] = torch.tensor([1.0 - j.noise.mask_probability if j.noise else 2.0 for j in padded],
                                     dtype=torch.float64, device=device)
        c["n_sym"] = torch.tensor([j.noise is not None and j.noise.mode == "symmetric_uniform_half_batch"
                                   for j in padded], device=device)
    if data_mode == "resident":
        dev = {k: v.to(device) for k, v in host.items()}
        if not noisy:
            dev["words"] = torch.zeros((1, 2), dtype=torch.int64, device=device)
    elif device.type == "cuda":
        host = {k: v.pin_memory() for k, v in host.items()}
    lr = float(settings["learning_rate"])
    recipe_key = tuple(sorted((k, v) for k, v in recipe.items() if k not in ("step_budget", "checkpoint_selection")))
    statics = {"layers": layers, "noisy": noisy, "scale_obs": scale_obs, "scale_act": scale_act, "lr": lr,
               "recipe": recipe_key}
    validation = (_validation_rows(padded, D, A, stream_chunk_bytes) if recipe["checkpoint_selection"] == "validation"
                  else None)
    step_fn = _Engine(_step_resident if data_mode == "resident" else _step, compile)

    def epoch_steps(epoch: int, i: int) -> int:
        """Model i's optimizer steps in this epoch: a full epoch, or what remains of its budget."""
        return max(0, min(steps[i], total_steps[i] - epoch * steps[i])) if epoch < epochs[i] else 0

    def chunks(epoch):
        """(first step, indices (M, n, B) on the device, gathered rows or None) for every chunk of the epoch.

        The epoch runs only as many steps as its longest remaining budget, so a step budget bounds the work
        whatever the dataset size."""
        max_steps = max(epoch_steps(epoch, i) for i in range(M))
        bounds = [(s0, min(chunk_steps, max_steps - s0)) for s0 in range(0, max_steps, chunk_steps)]
        if data_mode == "resident":
            for s0, n in bounds:
                yield s0, streams.chunk(epoch, s0, n), None
            return
        q: queue.Queue = queue.Queue(maxsize=2)  # double buffering: one chunk training, one in flight
        stop = threading.Event()
        side = torch.cuda.Stream(device) if device.type == "cuda" else None

        def produce():
            try:
                for s0, n in bounds:
                    if stop.is_set():
                        return
                    if side is not None:  # indices on the device (the same code as resident mode), rows from the host
                        with torch.cuda.stream(side):
                            idx = streams.chunk(epoch, s0, n)
                            idx_cpu = idx.cpu()
                        cpu = {k: v[idx_cpu].pin_memory() for k, v in host.items()}
                        with torch.cuda.stream(side):
                            rows = {k: v.to(device, non_blocking=True) for k, v in cpu.items()}
                        ev = torch.cuda.Event()
                        ev.record(side)
                        item = (s0, idx, rows, ev, cpu)
                    else:
                        idx = streams.chunk(epoch, s0, n)
                        item = (s0, idx, {k: v[idx] for k, v in host.items()}, None, None)
                    if not _put(q, item, stop):
                        return
            except BaseException as e:  # reaches the consumer as an error, never as a silent hang
                _put(q, ("error", e), stop)
            finally:
                _put(q, None, stop)
        th = threading.Thread(target=produce, daemon=True, name="ccil-v2-producer")
        th.start()
        try:
            while (item := q.get()) is not None:
                if item[0] == "error":
                    raise TrainingError(f"the streaming producer failed: {type(item[1]).__name__}: {item[1]}") from item[1]
                s0, idx, rows, ev, _keepalive = item
                if ev is not None:
                    main = torch.cuda.current_stream(device)
                    main.wait_event(ev)
                    for v in (idx, *rows.values()):  # allocated on the side stream, used on this one: keep the
                        v.record_stream(main)        # allocator from reusing the memory before this stream reads it
                yield s0, idx, rows
        finally:  # the consumer finished or failed: release and join the producer either way
            stop.set()
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            th.join(timeout=30)

    m1 = {k: torch.zeros_like(v) for k, v in params.items()}
    m2 = {k: torch.zeros_like(v) for k, v in params.items()}
    ema = {k: v.clone() for k, v in params.items()} if recipe["ema_decay"] is not None else None
    best, best_loss, best_at, val_history = None, None, [None] * M, [[] for _ in jobs]
    interval = recipe["validation_interval_steps"]

    def select(due: list, epoch: int, at: list):
        """Validate the models in due (host booleans) and keep each one's best weights so far: the EMA weights
        when EMA is on (they are the policy), else the parameters."""
        nonlocal best, best_loss
        weights = ema if ema is not None else params
        vl = _validation_loss(weights, validation, c, layers, scale_obs, scale_act, device)
        due_t = torch.tensor(due + [False] * (M - len(due)), device=device)
        better = due_t & (vl < best_loss) if best_loss is not None else due_t.clone()
        best = ({k: v.clone() for k, v in weights.items()} if best is None else
                {k: torch.where(better.view((M,) + (1,) * (v.dim() - 1)), weights[k], v) for k, v in best.items()})
        best_loss = torch.where(better, vl, best_loss) if best_loss is not None else torch.where(due_t, vl, float("inf"))
        for i, (bt, v) in enumerate(zip(better.tolist(), vl.tolist())):
            if i < n_real and due[i]:
                val_history[i].append({"step": at[i], "epoch": epoch, "loss": v})
                best_at[i] = len(val_history[i]) - 1 if bt else best_at[i]
    t = torch.zeros(M, device=device)
    S_dev = torch.tensor(steps, device=device)
    budget_dev = torch.tensor(total_steps, device=device)
    epochs_dev = torch.tensor(epochs, device=device)
    epoch_records, epoch_seconds = [[] for _ in jobs], []
    t_train = time.time()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(bool(deterministic) or previous)
    try:
        with _Precision() as precision:
            for epoch in range(max(epochs)):
                t_epoch = time.time()
                live = (epoch < epochs_dev) & real
                loss_sum = torch.zeros(M, device=device)
                n_steps = torch.zeros(M, device=device)
                seen_syn = torch.zeros(M, device=device)
                for s0, idx, rows in chunks(epoch):
                    for j in range(idx.shape[1]):
                        b = s0 + j
                        gstep = (epoch * S_dev + b)[:, None]  # model m's optimizer step as v1 counts it (NoiseBC key)
                        active = live & (b < S_dev) & (gstep[:, 0] < budget_dev)
                        if rows is None:
                            out = step_fn(params, m1, m2, t, ema, dev["x"], dev["y"], dev["w"], dev["words"],
                                          idx[:, j], active, gstep, c, **statics)
                            syn_j = dev["syn"][idx[:, j]]
                        else:
                            out = step_fn(params, m1, m2, t, ema, rows["x"][:, j], rows["y"][:, j], rows["w"][:, j],
                                          rows["words"][:, j] if noisy else None, active, gstep, c, **statics)
                            syn_j = rows["syn"][:, j]
                        params, m1, m2, t, ema, loss, upd = (_own(out) if CUDA_GRAPHS and step_fn.state.startswith("compiled")
                                                             else out)
                        loss_sum += loss
                        n_steps += upd.to(n_steps.dtype)
                        seen_syn += torch.where(active, syn_j.sum(dim=1).to(seen_syn.dtype), 0.0)
                        if validation is not None and interval:  # a fixed step cadence, plus each model's last step
                            at = [epoch * steps[i] + b + 1 for i in range(n_real)]
                            due = [epoch < epochs[i] and b < steps[i] and at[i] <= total_steps[i]
                                   and (at[i] % int(interval) == 0 or at[i] == total_steps[i]) for i in range(n_real)]
                            if any(due):
                                select(due, epoch, at)
                ls, ns, sy = loss_sum.tolist(), n_steps.tolist(), seen_syn.tolist()  # one host read per epoch
                if validation is not None and not interval:  # once per epoch, after its last step
                    select([epoch < epochs[i] for i in range(n_real)], epoch,
                           [min(total_steps[i], (epoch + 1) * steps[i]) for i in range(n_real)])
                epoch_seconds.append(time.time() - t_epoch)
                for i in range(n_real):
                    if epoch < epochs[i]:
                        k = epoch_steps(epoch, i)
                        epoch_records[i].append({"epoch": epoch, "steps": k, "optimizer_steps": int(ns[i]),
                                                 "mean_loss": ls[i] / ns[i] if ns[i] else None,
                                                 "rows_seen": {"demonstration": int(k * bs - sy[i]),
                                                               "synthetic": int(sy[i])},
                                                 "zero_weight_steps": int(k - ns[i])})
    finally:
        torch.use_deterministic_algorithms(previous)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_done = time.time()

    members = [{"seed": int(j.seed), "method": j.supervision.manifest()["method"],
                "supervision_sha256": training_identity_v2(j.supervision, settings, j.seed, j.noise)["supervision_sha256"],
                "noise_bc": None if j.noise is None else j.noise.descriptor()} for j in jobs]
    execution = {"members": members, "bucket": M, "padding": M - n_real, "engine": step_fn.state,
                 "compile_mode": COMPILE_MODE if compile else None, "data_mode": data_mode, "chunk_steps": chunk_steps,
                 "batch_size": bs, "device": str(device), "torch": torch.__version__, "precision": precision.effective,
                 "deterministic": bool(deterministic)}
    execution_sha = hashlib.sha256(json.dumps(execution, sort_keys=True, default=str).encode()).hexdigest()
    group = {**execution, "size": n_real, "execution_sha256": execution_sha, "table_rows": R, "table_bytes": table_bytes,
             "table_blocks": table["blocks"],
             "shuffle": "counter-based permutation of [0, n): 6-round unbalanced Feistel (splitmix64 rounds) over "
                        "2^ceil(log2 n) with cycle-walking, keyed by splitmix64(seed, epoch)",
             "setup_seconds": t_train - t_setup, "train_seconds": t_done - t_train, "epoch_seconds": epoch_seconds,
             "validation": None if validation is None else {"distinct_sets": validation["blocks"],
                                                             "rows_per_chunk": validation["rows"]}}
    final = best if best is not None else (ema if ema is not None else params)
    out = []
    for i, (job, policy) in enumerate(zip(jobs, policies)):
        policy.net.load_state_dict({k: v[i].detach().cpu() for k, v in final.items()})
        policy.net.eval()
        n_demo = sum(ln for (st, ln) in segments[i] if job_blocks[i].get("demonstration") == (st, ln))
        ident = training_identity_v2(job.supervision, settings, job.seed, job.noise, job.validation)
        record = {"trainer": "vector_mlp", "version": policy.version, "trainer_version": TRAINER_VERSION,
                  "training_identity_sha256": ident["sha256"], "parameters_sha256": parameters_sha256(policy),
                  "seed": int(job.seed),
                  "execution": {"execution_sha256": execution_sha, "position": i, "bucket": M,
                                "engine": step_fn.state, "data_mode": data_mode},
                  "settings": {**settings, "hidden_layers": list(settings["hidden_layers"])},
                  "device": str(device), "rows": int(streams.n[i]), "steps_per_epoch": steps[i],
                  "rows_dropped_per_epoch": int(streams.n[i]) % bs, "epochs_run": epochs[i],
                  "balance_labels": balance or None,
                  "row_sources": {"demonstration": int(n_demo), "synthetic": int(streams.n[i]) - int(n_demo)},
                  "epochs": epoch_records[i], "group": group,
                  "loss": "sum_i w_i mean_j (pi(s_i) - a_i)_j^2 / sum_i w_i per minibatch, scaled action space",
                  "optimizer": {"name": recipe["optimizer"], "learning_rate": lr, "betas": list(ADAM["betas"]),
                                "eps": ADAM["eps"], "weight_decay": recipe["weight_decay"],
                                "implementation": "functional, per-model masked"},
                  "recipe": {**recipe, "total_steps": total_steps[i],
                             "weights_returned": ("best validation epoch" if best is not None else
                                                  "EMA" if ema is not None else "last")},
                  "validation": None if validation is None else {
                      "cadence": f"every {interval} steps and the last" if interval else "every epoch",
                      "loss": "sum_i w_i mean_j (pi(s_i) - a_i)_j^2 / sum_i w_i over the held-out demonstrations, "
                              "scaled action space, of the weights returned (EMA when on)",
                      "evaluations": val_history[i], "selected": best_at[i],
                      "selected_step": None if best_at[i] is None else val_history[i][best_at[i]]["step"],
                      "selected_loss": None if best_at[i] is None else val_history[i][best_at[i]]["loss"],
                      "rows": int(validation["n"][i])},
                  "noise_bc": None if job.noise is None else job.noise.descriptor()}
        out.append((policy, json.loads(json.dumps(record, default=str))))
    return out
