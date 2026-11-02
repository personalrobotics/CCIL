"""The discrete Bellman (DP) oracle of the continuous Pendulum (docs/experiments/pendulum-oracle.md, #145).

Everything here is task semantics and the oracle's numerics: the exact map in (e, omega) coordinates, the cost, the
local LQR terminal set, the periodic grid, value iteration, greedy rollouts and the canonical scorer. check.py runs
the predeclared checks on top of it. State arrays in the map's own representation, (sin theta, cos theta, omega)
with theta = pi upright, are called s3; e = wrap(theta - pi) is the angle from upright.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

REPO = Path(__file__).resolve().parents[2]
for _p in ("vector_env", "pendulum_eval"):
    sys.path.insert(0, str(REPO / "tools" / _p))
import expert  # noqa: E402  (tools/pendulum_eval: the corrected demonstrator, primary variant)
import pendulum as pm  # noqa: E402  (tools/vector_env: the evaluator's exact map)

G, DT, U_MAX = 9.81, 0.02, 3.0
RHO = U_MAX / G
OMEGA0 = float(np.sqrt(G))  # the physical metric's velocity scale, rad/s
BETAS = (1e-3, 1e-2, 1e-1, 1.0)
BETA0_TERMINAL = 1e-3  # beta = 0 (descriptive) exits to this beta's terminal set with zero terminal value
BIG = 1e6  # a cost no capture reaches; values >= BIG / 2 are reported as unreachable
LEVELS = {"L1": (200, 241, 31), "L2": (400, 481, 61), "L3": (800, 961, 121)}
OMEGA_MAX, OMEGA_MAX_CHECK, OMEGA_REPORT = 12.0, 14.0, 6.5
OMEGA_SAFE = OMEGA_MAX - 2.0  # every reported trajectory stays within |omega| <= 10: a 2 rad/s margin
CONTINUATION_STEPS = 2000  # the demonstrator's horizon when it continues a transition that leaves the grid
UPRIGHT_COS = float(np.cos(np.deg2rad(8.0)))
CHUNK = 50_000
VI_ROWS = 20_000  # value iteration works in blocks of nodes: bounded temporaries, the same arithmetic per node
PROGRESS_SECONDS = 300  # long stages log flushed progress at least this often


def lam(beta: float) -> float:
    return beta / RHO ** 2


def wrap(x):
    return (np.asarray(x, np.float64) + np.pi) % (2 * np.pi) - np.pi


def s3_of(e, w) -> np.ndarray:
    th = np.asarray(e, np.float64) + np.pi
    return np.stack([np.sin(th), np.cos(th), np.broadcast_to(np.asarray(w, np.float64), th.shape)], axis=-1)


def ew_of(s3: np.ndarray):
    return wrap(np.arctan2(s3[..., 0], s3[..., 1]) - np.pi), s3[..., 2].copy()


def step3(s3: np.ndarray, u) -> np.ndarray:
    u = np.clip(np.broadcast_to(np.asarray(u, np.float64), (len(s3),)), -U_MAX, U_MAX)
    return pm.step(np.asarray(s3, np.float64), np.ascontiguousarray(u), False)[0]


def cost(e, u, beta: float):
    return DT * ((1 - np.cos(e)) / 2 + lam(beta) * (np.asarray(u) / G) ** 2)


def energy(e, w):
    """H, relative to upright at rest: 0 there, -2g hanging at rest."""
    return 0.5 * np.asarray(w) ** 2 - G * (1 - np.cos(e))


def quad(P: np.ndarray, e, w):
    return P[0, 0] * e * e + 2 * P[0, 1] * e * w + P[1, 1] * w * w


def demonstrator(s3: np.ndarray) -> np.ndarray:
    return expert.act(s3)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- parallel helpers (fork: workers inherit the module globals set before the pool starts) ----

_SHARED: dict = {}


def pmap(fn, jobs: list, workers: int | None = None, label: str | None = None) -> list:
    """fn over jobs in forked worker processes, results in job order. A worker that dies (e.g. killed for memory)
    raises BrokenProcessPool rather than hanging; while jobs run, progress is logged every PROGRESS_SECONDS."""
    t0 = time.time()
    with ProcessPoolExecutor(min(len(jobs), workers or os.cpu_count()), mp_context=mp.get_context("fork")) as ex:
        futures = [ex.submit(fn, j) for j in jobs]
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=PROGRESS_SECONDS, return_when=FIRST_EXCEPTION)
            if any(f.exception() for f in done):
                break
            if pending:
                log(label or fn.__name__, f"{len(jobs) - len(pending)}/{len(jobs)} jobs done,",
                    f"{time.time() - t0:.0f} s")
        return [f.result() for f in futures]


def _map_rows(args):
    s3, u = args
    return step3(s3, u)


def step_many(s3: np.ndarray, u: np.ndarray, workers: int | None = None) -> np.ndarray:
    """step3 over many rows, in parallel chunks; the result is independent of the chunking (rows are independent)."""
    if len(s3) <= CHUNK:
        return step3(s3, u)
    parts = [(s3[i:i + CHUNK], u[i:i + CHUNK]) for i in range(0, len(s3), CHUNK)]
    return np.concatenate(pmap(_map_rows, parts, workers, "exact map"))


# ---- local LQR terminal set ----

def linearize(h: float = 1e-6):
    """Central-difference Jacobians (A, B) of the exact one-step map at upright, in x = (e, omega)."""
    pts = np.array([[h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]])
    e1, w1 = ew_of(step3(s3_of(pts[:, 0], pts[:, 1]), pts[:, 2]))
    x1 = np.stack([e1, w1], axis=1)
    A = np.stack([(x1[0] - x1[1]) / (2 * h), (x1[2] - x1[3]) / (2 * h)], axis=1)
    B = ((x1[4] - x1[5]) / (2 * h)).reshape(2, 1)
    return A, B


def lqr(beta: float, A, B):
    Q = DT * np.diag([0.25, 0.0])
    R = np.array([[DT * lam(beta) / G ** 2]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


ACCURACY_TOL, ACCURACY_SHELLS, ACCURACY_ANGLES = 0.05, (0.25, 0.5, 0.75, 1.0), 64
INNER_FRACTION, ACCURACY_STEPS = 1e-4, 2_000


def _largest(ok, hi: float, rel: float) -> float:
    """The largest c in (0, hi] passing ok, by bisection (hi itself first); 0.0 when none is found."""
    if ok(hi):
        return hi
    lo, floor = 0.0, hi * 1e-12
    while (hi - lo) / hi > rel:
        if hi < floor:  # nothing above a negligible c passes
            return 0.0
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def lqr_cost_ratio(P, K, beta: float, x0: np.ndarray, c: float) -> np.ndarray:
    """(exact closed-loop cost under u = -Kx until x^T P x <= INNER_FRACTION * c, plus x^T P x there) / x0^T P x0.

    A row that does not enter the inner ellipsoid within ACCURACY_STEPS gets ratio +inf.
    """
    s = s3_of(x0[:, 0], x0[:, 1])
    acc, total = np.zeros(len(x0)), np.full(len(x0), np.inf)
    active = np.ones(len(x0), bool)
    for _ in range(ACCURACY_STEPS):
        a = np.flatnonzero(active)
        if not len(a):
            break
        e, w = ew_of(s[a])
        inner = quad(P, e, w) <= INNER_FRACTION * c
        total[a[inner]] = acc[a[inner]] + quad(P, e[inner], w[inner])
        active[a[inner]] = False
        a, e, w = a[~inner], e[~inner], w[~inner]
        u = np.clip(-(np.stack([e, w], 1) @ K.T)[:, 0], -U_MAX, U_MAX)
        acc[a] += cost(e, u, beta)
        s[a] = step3(s[a], u)
    return total / quad(P, x0[:, 0], x0[:, 1])


def terminal_set(P, K, beta: float, seed: int = 1451, n_in: int = 20_000, n_edge: int = 2_000,
                 rel: float = 1e-3) -> dict:
    """c = min(c_input, c_decrease, c_accuracy) (version 4 of the predeclaration).

    - c_input = 9 / (K P^-1 K^T): |Kx| <= 3 on all of Omega_c.
    - c_decrease: the largest c (bisection) for which a seeded sample of Omega_c satisfies V(F_cl(x)) < V(x).
    - c_accuracy: the largest c (bisection) for which, at deterministic points on the radial shells
      ACCURACY_SHELLS (x^T P x = s^2 c) and ACCURACY_ANGLES angles, the exact closed-loop cost under u = -Kx (until
      the inner ellipsoid x^T P x <= INNER_FRACTION * c, plus x^T P x there) is within ACCURACY_TOL of x0^T P x0.
    A beta for which no positive c passes raises: it fails, and is never dropped silently.
    """
    rng = np.random.default_rng(seed)
    r, a = np.sqrt(rng.uniform(size=n_in)), rng.uniform(0, 2 * np.pi, n_in)
    ab = rng.uniform(0, 2 * np.pi, n_edge)
    z = np.concatenate([np.stack([r * np.cos(a), r * np.sin(a)], 1), np.stack([np.cos(ab), np.sin(ab)], 1)])
    Lt = np.linalg.cholesky(P).T  # P = L L^T, so x = sqrt(c) L^-T z has x^T P x = c |z|^2
    unit = np.linalg.solve(Lt, z.T).T
    ang = 2 * np.pi * np.arange(ACCURACY_ANGLES) / ACCURACY_ANGLES
    zs = np.concatenate([s * np.stack([np.cos(ang), np.sin(ang)], 1) for s in ACCURACY_SHELLS])
    shells = np.linalg.solve(Lt, zs.T).T
    c_input = float(U_MAX ** 2 / (K @ np.linalg.solve(P, K.T))[0, 0])

    def decreases(c):
        x = np.sqrt(c) * unit
        u = -(x @ K.T)[:, 0]
        if np.abs(u).max() > U_MAX:
            return False
        e1, w1 = ew_of(step3(s3_of(x[:, 0], x[:, 1]), u))
        v0, v1 = quad(P, x[:, 0], x[:, 1]), quad(P, e1, w1)
        return bool((v1[v0 > 0] < v0[v0 > 0]).all())

    def accurate(c):
        ratio = lqr_cost_ratio(P, K, beta, np.sqrt(c) * shells, c)
        return bool((np.abs(ratio - 1) <= ACCURACY_TOL).all())

    c_dec = _largest(decreases, c_input, rel)
    c_acc = _largest(accurate, c_input, rel)
    c = min(c_input, c_dec, c_acc)
    if not (c > 0 and decreases(c) and accurate(c)):
        raise RuntimeError(f"beta {beta}: no positive c passes the terminal-set rule "
                           f"(c_input {c_input}, c_decrease {c_dec}, c_accuracy {c_acc})")
    ratio = lqr_cost_ratio(P, K, beta, np.sqrt(c) * shells, c)
    binding = [k for k, v in (("input", c_input), ("decrease", c_dec), ("accuracy", c_acc)) if v == c]
    return {"c": c, "c_input": c_input, "c_decrease": c_dec, "c_accuracy": c_acc, "binding": binding,
            "accuracy_ratio_min": float(ratio.min()), "accuracy_ratio_max": float(ratio.max()),
            "inner_fraction": INNER_FRACTION, "saturation_limited": c == c_input}


def lqr_bundle(beta: float, A, B) -> dict:
    """P, K and c for beta > 0; beta = 0 uses beta = BETA0_TERMINAL's set with zero terminal value."""
    if beta == 0:
        out = dict(lqr_bundle(BETA0_TERMINAL, A, B))
        return out | {"beta": 0.0, "terminal_value": False}
    P, K = lqr(beta, A, B)
    return {"beta": beta, "P": P, "K": K, **terminal_set(P, K, beta), "terminal_value": True}


def in_terminal(L: dict, e, w):
    return quad(L["P"], e, w) <= L["c"]


def terminal_value(L: dict, e, w):
    return quad(L["P"], e, w) if L["terminal_value"] else np.zeros_like(np.asarray(e, np.float64))


# ---- the grid ----

class Grid:
    """Periodic in e (N_e nodes, one at e = 0 and one at e = -pi), [-omega_max, omega_max] in omega, actions on [-3, 3]."""

    def __init__(self, ne: int, nw: int, nu: int, wmax: float = OMEGA_MAX, dw: float | None = None):
        self.ne, self.de = ne, 2 * np.pi / ne
        self.e = -np.pi + self.de * np.arange(ne)
        if dw is not None:  # the boundary check: same spacing, wider range
            nw = int(round(2 * wmax / dw)) + 1
        self.nw, self.wmax = nw, wmax
        self.w = np.linspace(-wmax, wmax, nw)
        self.dw = 2 * wmax / (nw - 1)
        self.u = np.linspace(-U_MAX, U_MAX, nu)
        E, W = np.meshgrid(self.e, self.w, indexing="ij")
        self.E, self.W = E.ravel(), W.ravel()  # node k = i * nw + j

    @classmethod
    def level(cls, name: str, wmax: float = OMEGA_MAX):
        ne, nw, nu = LEVELS[name]
        base = cls(ne, nw, nu)
        return base if wmax == OMEGA_MAX else cls(ne, nw, nu, wmax=wmax, dw=base.dw)

    def node(self, i, j):
        return np.asarray(i) % self.ne * self.nw + np.asarray(j)

    def weights(self, e, w):
        """Bilinear (indices (..., 4), weights (..., 4), inside) for continuous (e, w); periodic in e."""
        fe = (wrap(e) + np.pi) / self.de
        i0 = np.floor(fe)
        te = fe - i0
        i0 = i0.astype(np.int64) % self.ne
        i1 = (i0 + 1) % self.ne
        fw = (np.asarray(w) + self.wmax) / self.dw
        j0 = np.clip(np.floor(fw), 0, self.nw - 2)
        tw = np.clip(fw - j0, 0.0, 1.0)
        j0 = j0.astype(np.int64)
        inside = (np.asarray(w) >= -self.wmax) & (np.asarray(w) <= self.wmax)
        idx = np.stack([i0 * self.nw + j0, i0 * self.nw + j0 + 1, i1 * self.nw + j0, i1 * self.nw + j0 + 1], -1)
        wts = np.stack([(1 - te) * (1 - tw), (1 - te) * tw, te * (1 - tw), te * tw], -1)
        return idx, wts, inside

    def interp(self, V, e, w):
        idx, wts, inside = self.weights(e, w)
        return blend(V, idx, wts, inside)

    def reported(self):
        return np.abs(self.W) <= OMEGA_REPORT


def blend(V, idx, wts, inside):
    """Bilinear V, with +inf arithmetic: touching an unreachable (BIG) corner is unreachable. Outside the grid it is
    BIG here; callers replace that by the boundary continuation (continuation, boundary_values)."""
    corner = V[idx]
    unreachable = ~inside | ((corner >= BIG / 2) & (wts > 0)).any(-1)
    return np.where(unreachable, BIG, (np.minimum(corner, BIG) * wts).sum(-1))


def transitions(grid: Grid, workers: int | None = None) -> dict:
    """(e', w') for every node and grid action, (N, N_u), through the exact map: computed once, shared by every beta."""
    n, nu = len(grid.E), len(grid.u)
    s = np.repeat(s3_of(grid.E, grid.W), nu, axis=0)
    u = np.tile(grid.u, n)
    e1, w1 = ew_of(step_many(s, u, workers))
    return {"e": e1.reshape(n, nu), "w": w1.reshape(n, nu)}


# ---- the boundary continuation ----

def continuation(e, w, L: dict, beta: float) -> np.ndarray:
    """The corrected demonstrator's first-exit cost from continuous states, through the exact map, for up to
    CONTINUATION_STEPS steps: a finite admissible upper bound where it reaches Omega_c, else BIG (unsupported)."""
    if not len(e):
        return np.zeros(0)
    _, J, _, _ = entry_rollout(s3_of(e, w), L, beta, CONTINUATION_STEPS, demonstrator)
    return J


def lookahead_continuation(e, w, L: dict, beta: float, actions: np.ndarray) -> np.ndarray:
    """min over the given first actions of c(x, u) plus the terminal value or the demonstrator's continuation from the
    exact successor: the cost of a feasible policy, so an admissible upper bound (BIG if no first action recovers)."""
    n, nu = len(e), len(actions)
    s1 = step3(np.repeat(s3_of(e, w), nu, axis=0), np.tile(actions, n))
    e1, w1 = ew_of(s1)
    hit = in_terminal(L, e1, w1)
    nxt = np.where(hit, terminal_value(L, e1, w1), 0.0)
    nxt[~hit] = continuation(e1[~hit], w1[~hit], L, beta)
    q = cost(np.repeat(e, nu), np.tile(actions, n), beta) + np.minimum(nxt, BIG)
    return np.minimum(q.reshape(n, nu).min(1), BIG)


def boundary_values(grid: Grid, T: dict, L: dict, beta: float, workers: int | None = None):
    """The continuation value of every transition that leaves the grid without entering Omega_c (N, N_u; NaN for
    the others), and where the demonstrator fails to recover (unsupported, BIG)."""
    _, _, inside = grid.weights(T["e"], T["w"])
    exits = ~inside & ~in_terminal(L, T["e"], T["w"])
    cont = np.full(exits.shape, np.nan)
    if exits.any():
        _, J, _, _ = first_entry(T["e"][exits], T["w"][exits], L, beta, CONTINUATION_STEPS, "demonstrator",
                                 workers=workers)
        cont[exits] = J
    return cont


# ---- value iteration ----

def bellman_parts(grid: Grid, T: dict, L: dict, beta: float, cont: np.ndarray, weights=None) -> dict:
    """What a Bellman application needs. A transition into Omega_c takes the terminal value, one leaving the grid its
    boundary continuation cont (boundary_values), a fixed number either way; one inside the grid takes bilinear V.
    weights, grid.weights of the transitions, is shared by every beta (pass it to avoid recomputing it)."""
    idx, wts, inside = weights if weights is not None else grid.weights(T["e"], T["w"])
    hit = in_terminal(L, T["e"], T["w"])
    term = in_terminal(L, grid.E, grid.W)
    return {"idx": idx, "wts": wts, "inside": inside, "interp": ~hit & inside,
            "fixed": np.where(hit, terminal_value(L, T["e"], T["w"]), cont),
            "c": cost(grid.E[:, None], grid.u[None, :], beta), "term": term,
            "term_value": terminal_value(L, grid.E[term], grid.W[term])}


def bellman_q(p: dict, V: np.ndarray, rows: slice) -> np.ndarray:
    """Q at the nodes in rows: stage cost plus the successor's value."""
    nxt = np.where(p["interp"][rows], blend(V, p["idx"][rows], p["wts"][rows], p["inside"][rows]), p["fixed"][rows])
    return p["c"][rows] + nxt


def bellman_finish(p: dict, V: np.ndarray):
    """Q at every node and the Bellman residual of V."""
    Q = np.empty(p["c"].shape)
    for i in range(0, len(V), VI_ROWS):
        Q[i:i + VI_ROWS] = bellman_q(p, V, slice(i, i + VI_ROWS))
    resid = np.abs(np.minimum(Q.min(axis=1), BIG) - V)
    resid[p["term"]] = 0.0
    return Q, resid


def value_iteration(grid: Grid, T: dict, L: dict, beta: float, V0: np.ndarray, cont: np.ndarray,
                    tol: float = 1e-10, cap: int = 20_000, check_monotone: bool = False, weights=None,
                    label: str | None = None) -> dict:
    """Jacobi value iteration of the first-exit problem from V0, capped at cap iterations (bellman_parts).

    Terminal nodes hold the terminal value. Nodes are updated in blocks of VI_ROWS from the previous iterate, so the
    result does not depend on the blocking. With a label, progress is logged every PROGRESS_SECONDS.
    """
    p = bellman_parts(grid, T, L, beta, cont, weights)
    term = p["term"]
    V = np.minimum(np.asarray(V0, np.float64).copy(), BIG)
    V[term] = p["term_value"]
    monotone, worst_drop, it, change = True, 0.0, 0, np.inf
    t0 = last = time.time()
    for it in range(1, cap + 1):
        Vn = np.empty_like(V)
        for i in range(0, len(V), VI_ROWS):
            Vn[i:i + VI_ROWS] = np.minimum(bellman_q(p, V, slice(i, i + VI_ROWS)).min(axis=1), BIG)
        Vn[term] = V[term]
        d = Vn - V
        if check_monotone:
            drop = float(d.min())
            worst_drop = min(worst_drop, drop)
            monotone &= drop >= -1e-12
        fin = (Vn < BIG / 2) | (V < BIG / 2)
        change = float(np.abs(d[fin]).max()) if fin.any() else 0.0
        V = Vn
        if change <= tol:
            break
        if label and time.time() - last >= PROGRESS_SECONDS:
            log(label, "iteration", it, "change", f"{change:.3e}", f"{time.time() - t0:.0f} s")
            last = time.time()
    Q, resid = bellman_finish(p, V)
    return {"V": V, "Q": Q, "iterations": it, "converged": change <= tol, "last_change": change,
            "monotone": bool(monotone), "worst_drop": worst_drop, "residual": resid}


# ---- policies and rollouts ----

def next_value(grid: Grid, V: np.ndarray, L: dict, beta: float, e1, w1) -> np.ndarray:
    """V at continuous successors: the terminal value in Omega_c, bilinear V on the grid, and the boundary
    continuation (computed here, from the exact successor) off it."""
    hit = in_terminal(L, e1, w1)
    out = np.where(hit, terminal_value(L, e1, w1), grid.interp(V, e1, w1))
    off = ~hit & (np.abs(w1) > grid.wmax)
    out[off] = continuation(e1[off], w1[off], L, beta)
    return out


def oracle_policy(grid: Grid, V: np.ndarray, L: dict, beta: float):
    """Greedy over the action grid with interpolated V (exact terminal value on entry); -Kx inside Omega_c."""
    nu = len(grid.u)

    def act(s3: np.ndarray) -> np.ndarray:
        e, w = ew_of(s3)
        out = np.empty(len(s3))
        inside = in_terminal(L, e, w)
        out[inside] = np.clip(-(np.stack([e[inside], w[inside]], 1) @ L["K"].T)[:, 0], -U_MAX, U_MAX)
        rest = np.flatnonzero(~inside)
        if len(rest):
            e1, w1 = ew_of(step3(np.repeat(s3[rest], nu, axis=0), np.tile(grid.u, len(rest))))
            nxt = next_value(grid, V, L, beta, e1, w1)
            q = (cost(np.repeat(e[rest], nu), np.tile(grid.u, len(rest)), beta) + nxt).reshape(len(rest), nu)
            out[rest] = grid.u[q.argmin(axis=1)]
        return out
    return act


def q_continuous(grid: Grid, V: np.ndarray, L: dict, beta: float, s3: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Q*(z, a) at the given continuous actions, through the exact map; never snapped to the action grid."""
    e, _ = ew_of(s3)
    e1, w1 = ew_of(step3(s3, a))
    return cost(e, np.clip(a, -U_MAX, U_MAX), beta) + next_value(grid, V, L, beta, e1, w1)


def rollout(policy, s0: np.ndarray, steps: int):
    """States (N, steps + 1, 3) and applied actions (N, steps) of a deterministic policy through the exact map."""
    s = np.empty((len(s0), steps + 1, 3))
    u = np.empty((len(s0), steps))
    s[:, 0] = s0
    for t in range(steps):
        u[:, t] = np.clip(policy(s[:, t]), -U_MAX, U_MAX)
        s[:, t + 1] = step3(s[:, t], u[:, t])
    return s, u


def entry_rollout(s: np.ndarray, L: dict, beta: float, steps: int, policy, history: bool = False):
    """From s3 rows, until first entry into Omega_c or steps: (entry step or -1, first-exit cost or BIG, the largest
    |omega| on the way, and for rows that never enter, the least physical-metric distance from the final state to an
    earlier one (a recurrence diagnostic; NaN for rows that enter, or without history))."""
    s = np.array(s, np.float64)
    n = len(s)
    entry = np.full(n, -1, np.int64)
    acc, final = np.zeros(n), np.zeros(n)
    active = np.ones(n, bool)
    e, w = ew_of(s)
    wmax = np.abs(w)
    hit = in_terminal(L, e, w)
    entry[hit], final[hit], active[hit] = 0, terminal_value(L, e[hit], w[hit]), False
    path = np.full((n, steps + 1, 2), np.nan) if history else None
    if history:
        path[:, 0] = np.stack([e, w], 1)
    for t in range(steps):
        a = np.flatnonzero(active)
        if not len(a):
            break
        u = np.clip(policy(s[a]), -U_MAX, U_MAX)
        acc[a] += cost(ew_of(s[a])[0], u, beta)
        s[a] = step3(s[a], u)
        e, w = ew_of(s[a])
        wmax[a] = np.maximum(wmax[a], np.abs(w))
        if history:
            path[a, t + 1] = np.stack([e, w], 1)
        hit = in_terminal(L, e, w)
        h = a[hit]
        entry[h], final[h], active[h] = t + 1, terminal_value(L, e[hit], w[hit]), False
    recur = np.full(n, np.nan)
    if history:
        for i in np.flatnonzero(entry < 0):
            de = wrap(path[i, :-2, 0] - path[i, -1, 0])
            dw = (path[i, :-2, 1] - path[i, -1, 1]) / OMEGA0
            recur[i] = float(np.sqrt(de * de + dw * dw).min())
    return entry, np.where(entry >= 0, acc + final, BIG), wmax, recur


def _first_entry_chunk(args):
    lo, hi, steps, kind, beta = args
    g = _SHARED
    policy = demonstrator if kind == "demonstrator" else g["policy"]
    return entry_rollout(s3_of(g["E"][lo:hi], g["W"][lo:hi]), g["L"], beta, steps, policy,
                         history=kind == "oracle")


def first_entry(E, W, L: dict, beta: float, steps: int, kind: str, policy=None, workers: int | None = None):
    """entry_rollout from every (E, W) under the demonstrator or a given ("oracle") policy, in parallel chunks."""
    _SHARED.update(E=np.asarray(E), W=np.asarray(W), L=L, policy=policy)
    n = len(E)
    size = max(1, min(CHUNK // 10, -(-n // (os.cpu_count() or 1))))
    jobs = [(lo, min(n, lo + size), steps, kind, beta) for lo in range(0, n, size)]
    parts = pmap(_first_entry_chunk, jobs, workers, f"{kind} rollouts beta {beta}")
    return tuple(np.concatenate([part[k] for part in parts]) for k in range(4))


# ---- the canonical scorer ----

def score(s: np.ndarray, u: np.ndarray, L: dict, beta: float) -> dict:
    """Per-trajectory reference scores over the given horizon (arrays of length N)."""
    e, w = ew_of(s)
    T = u.shape[1]
    height = (DT * (1 - np.cos(e[:, :T])) / 2).sum(1)
    effort = (DT * lam(beta) * (u / G) ** 2).sum(1)
    inside = in_terminal(L, e, w)
    entered = inside.any(axis=1)
    first = np.where(entered, inside.argmax(axis=1), -1)
    up = np.cos(e) >= UPRIGHT_COS
    swings, direction, sat = np.zeros(len(s)), np.zeros(len(s)), np.full(len(s), np.nan)
    for i in range(len(s)):
        stop = first[i] if entered[i] else T
        sg = np.sign(w[i, :stop + 1])
        sg = sg[sg != 0]
        swings[i] = int((sg[1:] != sg[:-1]).sum())
        direction[i] = np.sign(w[i, first[i]]) if entered[i] else 0.0
        if stop > 0:
            sat[i] = float((np.abs(u[i, :stop]) >= U_MAX - 1e-9).mean())
    return {"running_cost": height + effort, "height_cost": height, "effort_cost": effort,
            "physical_effort": (u ** 2).sum(1) * DT, "normalized_effort": ((u / U_MAX) ** 2).mean(1),
            "action_variation": np.abs(np.diff(u, axis=1)).sum(1), "upright_occupancy": up[:, :T].mean(1),
            "final_hold": up[:, T - 99:T + 1].all(1), "time_to_terminal": np.where(entered, first * DT, np.inf),
            "captured": entered, "swing_count": swings, "direction": direction, "saturation_before_capture": sat}
