"""The batched engine shared by CCIL's batched transition fits (mlp_batched.py) and the policy trainer v2.

Many models of one shape trained as one vmap'd program: counter-based
splitmix64 hashing on int64 bit patterns, the keyed Feistel permutation that
orders minibatches, content keys of data blocks, the per-model index streams,
the group-size buckets, the compiled-or-eager step with its compilation
record, and the float32 precision scope. Moved here from
training/vector_bc_v2.py unchanged (#168), so the dynamics fit does not import
the policy trainer.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


class EngineError(ValueError):
    """A batched group that cannot run as declared."""


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


def _feistel(x, key, a, b):
    """A keyed bijection of [0, 2^(a+b)): unbalanced Feistel rounds (an even count restores the widths)."""
    hi, lo, wh, wl = _lsr(x, b), x & ((1 << b) - 1), a, b
    for r in range(FEISTEL_ROUNDS):
        f = _mix(lo ^ _mix(key ^ r))
        hi, lo, wh, wl = lo, hi ^ (f & ((1 << wh) - 1)), wl, wh
    return (hi << wl) | lo


def permute(pos: torch.Tensor, key: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """A keyed pseudo-random permutation of [0, n_m) applied to pos (M, K): Feistel over 2^ceil(log2 n) with
    cycle-walking (re-applied only to positions still outside [0, n)); the walk ends when all are in range.
    key is one key per model (M,) or one per position (M, K)."""
    k = torch.clamp(torch.ceil(torch.log2(n.double())).long(), min=2)[:, None]
    a, b = k // 2, k - k // 2
    n_ = n[:, None]
    key_ = key[:, None].expand_as(pos) if key.dim() == 1 else key
    x = _feistel(pos, key_, a, b)
    rows, cols = torch.nonzero(x >= n_, as_tuple=True)
    while len(rows):  # walk only the positions still outside [0, n): the domain is < 2n, so the set halves each pass
        y = _feistel(x[rows, cols], key_[rows, cols], a[rows, 0], b[rows, 0])
        x[rows, cols] = y
        keep = y >= n[rows]
        rows, cols = rows[keep], cols[keep]
    return x


def shuffle_key(seed: int, epoch: int) -> int:
    """The permutation key of (seed, epoch): splitmix64 over the pair, domain-separated."""
    h = _mix(torch.tensor([_s64(int(seed) & _M64)]) ^ _SHUFFLE)
    return int(_mix(h ^ int(epoch))[0])


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


class CompilationRefused(EngineError):
    """A compiled step would have run uncompiled: refused before that step ran."""


# Whether a compiler failure on the first step falls back to eager (recorded). A study process turns it off: its
# units must all run the engine they request, so the failure is refused instead.
EAGER_FALLBACK = True


def _dynamo():
    """torch._dynamo, imported when a step is compiled (it adds half a second to importing torch)."""
    import torch._dynamo
    return torch._dynamo


def _specializations(fn) -> int:
    """torch.compile's specializations (cache entries) of fn's code in the current compiler context."""
    from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list
    return len(_debug_get_cache_entry_list(fn.__code__))


def _fresh_context(fn) -> bool:
    """Start a fresh compiler context (torch._dynamo.reset: every compiled function recompiles on its next use)
    when fn already holds the recompile limit's specializations; return whether it did.

    A group's shapes and settings are fixed, so a group adds at most one specialization of its step, and a group
    that starts below the limit stays within it. A process that runs many groups (a study phase: tasks, settings
    and buckets) would otherwise exceed it and be refused. The reset happens only between groups, and never
    changes a compiled step: only how many are kept."""
    if _specializations(fn) < _dynamo().config.recompile_limit:
        return False
    _dynamo().reset()
    return True


class _Engine:
    """The step, compiled or eager, and what ran.

    Compilation is lazy, so the first compiled call is guarded: a compiler failure before any update falls back
    to eager for the whole run (the step is functional, so a failed call changed nothing) and is recorded, or is
    refused when EAGER_FALLBACK is off. Past torch.compile's recompile limit a compiled function would run eager
    with only a warning; a group starts a fresh compiler context when its step is at the limit (_fresh_context),
    and a step that would still exceed it is refused (fail_on_recompile_limit_hit, process-wide) before it runs,
    so an engine that says compiled ran compiled on every step. compilation() is the record."""

    def __init__(self, fn, compile: bool, refusal: type = CompilationRefused):
        self.refusal = refusal  # the caller's CompilationRefused subclass, so its own error class catches it
        self.requested = f"compiled ({COMPILE_MODE})" if compile else "eager"
        self.fn, self.eager, self.state, self.first = (_compiled(fn) if compile else fn), fn, self.requested, compile
        self.limit_hit, self.before, self.fresh = (), None, None
        if compile:
            _dynamo().config.fail_on_recompile_limit_hit = True
            self.fresh = _fresh_context(fn)
            self.limit_hit, self.before = _dynamo().exc.FailOnRecompileLimitHit, _specializations(fn)

    def __call__(self, *args, **kwargs):
        first, self.first = self.first, False
        try:
            return self.fn(*args, **kwargs)
        except self.limit_hit as e:
            raise self.refusal(
                f"{self.eager.__module__}.{self.eager.__qualname__} needs more than torch.compile's "
                f"{_dynamo().config.recompile_limit} specializations in this process, so this step would run "
                "uncompiled; refused before it ran") from e
        except Exception as e:  # a compiler failure on the first call: recorded or refused, never silent
            if not first:
                raise
            if not EAGER_FALLBACK:
                raise self.refusal(f"compilation failed on the first step: {type(e).__name__}: {e}"[:300]) from e
            self.fn, self.state = self.eager, f"eager (compilation failed on the first step: {type(e).__name__}: {e})"[:300]
        return self.fn(*args, **kwargs)

    def compilation(self) -> dict:
        """The requested and effective engine, and the step's specializations in the compiler context: in all,
        and new in this run (0 when it reused compiled code), and whether this run started a fresh context; None
        when eager was requested."""
        n = _specializations(self.eager) if self.before is not None else None
        return {"requested": self.requested, "effective": self.state, "specializations": n,
                "new_specializations": None if n is None else n - self.before, "fresh_context": self.fresh}


def _bucket(m: int) -> int:
    for b in BUCKETS:
        if m <= b:
            return b
    raise EngineError(f"a group has at most {BUCKETS[-1]} models; split {m} jobs into several groups")


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
