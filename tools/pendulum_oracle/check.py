#!/usr/bin/env python3
"""Run the predeclared Pendulum DP-oracle checks (docs/experiments/pendulum-oracle.md, version 6.1).

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_oracle/check.py"
    $R --data pendulum_cont_corrected.npz --out DIR [--workers N] [--resume]

DIR must not exist. It receives fields.npz (the oracle level's fields per beta, the release asset) and oracle.json
(every check, pass or fail, with the reference scores, the beta table, hashes and the code state). Levels run in
the declared order: L1 and L2, and L3 only if the L1-L2 convergence rule fails. Exit 1 unless every check passes.

Until oracle.json is written the run is incomplete. Each value iteration's result is checkpointed, atomically, in
DIR/checkpoints; --resume continues an incomplete DIR from them at the same code commit (anything else is refused)
and recomputes everything else, which is deterministic. Progress is logged at least every O.PROGRESS_SECONDS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle as O  # noqa: E402

DATA_SHA256 = "59c332bc829db9c9da40d9c1bd9ebbf808965aca6a167be48dc0f2ade52cf43c"
ALL_BETAS = (0.0,) + O.BETAS
STARTS = {"S1": (np.pi, 0.0), "S2": (np.pi - 0.3, 0.0), "S3": (np.pi / 2, 0.0), "S4": (-np.pi / 2, 0.0),
          "S5": (np.pi, 2.0), "S6": (2.0, -3.0), "S7": (0.6, 0.0), "S8": (0.0, 4.0)}
HORIZON, DEMO_STEPS, CAPTURE_STEPS = 500, 2000, 25
GAP_TOL, RESID_TOL, BOUNDARY_REL, TRAJ_REL = 1e-6, 1e-9, 1e-3, 1e-2


log = O.log


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=O.REPO, capture_output=True, text=True, check=True).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_s3(e, w):
    """The named start as a map state; hanging (e = +-pi) is the exact fixed point (0, 1, w)."""
    s = O.s3_of(np.array([e]), np.array([w]))
    if abs(abs(e) - np.pi) < 1e-15:
        s[:, 0], s[:, 1] = 0.0, 1.0
    return s


# ---- the two gates (module-level so a smoke test can force each path) ----

def level_failed(checks: dict) -> bool:
    return not checks["pass"]


def rule_passes(name: str, rule: dict) -> bool:
    return bool(rule["pass"])


# ---- one level ----

VI_SCALARS = ("iterations", "converged", "last_change", "monotone", "worst_drop")


def _vi_job(args):
    """One value iteration, or its checkpoint: V is stored and Q and the residual are recomputed from it."""
    kind, beta = args
    g = O._SHARED
    grid, T, wgt, cont = ((g["gridb"], g["Tb"], g["wgtb"], g["contb"][beta]) if kind == "boundary"
                          else (g["grid"], g["T"], g["wgt"], g["cont"][beta]))
    L = g["L"][beta]
    path = g["checkpoints"] / f"{g['name']}-{kind}-beta{beta:g}.npz"
    key = json.dumps({"commit": g["commit"], "level": g["name"], "kind": kind, "beta": beta,
                      "nodes": len(grid.E), "actions": len(grid.u), "wmax": grid.wmax})
    if path.exists():
        ck = np.load(path)
        if str(ck["key"]) != key:
            raise RuntimeError(f"checkpoint {path} is from a different run: {ck['key']}")
        Q, resid = O.bellman_finish(O.bellman_parts(grid, T, L, beta, cont, wgt), ck["V"])
        log(g["name"], kind, "beta", beta, "value iteration resumed from", path.name)
        return kind, beta, {"V": ck["V"], "Q": Q, "residual": resid, **json.loads(str(ck["scalars"]))}
    V0 = g["U0"][beta] if kind == "upper" else np.zeros(len(grid.E))
    r = O.value_iteration(grid, T, L, beta, V0, cont, check_monotone=kind == "lower", weights=wgt,
                          label=f"{g['name']} {kind} beta {beta}")
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, V=r["V"], key=key, scalars=json.dumps({k: r[k] for k in VI_SCALARS}))
    os.replace(tmp, path)
    return kind, beta, r


def continuation_report(grid, T, cont) -> dict:
    """Transitions that leave the grid, and those whose demonstrator continuation does not recover (unsupported)."""
    exits = ~np.isnan(cont)
    bad = exits & (np.nan_to_num(cont, nan=0.0) >= O.BIG / 2)
    return {"exiting_transitions": int(exits.sum()), "unsupported_transitions": int(bad.sum()),
            "nodes_with_exits": int(exits.any(1).sum()), "nodes_all_unsupported": int((bad.sum(1) == exits.shape[1]).sum()),
            "unsupported_min_abs_omega": float(np.abs(grid.W[bad.any(1)]).min()) if bad.any() else None}


def action_mismatch(Qa, Qb, V, keep) -> int:
    """Nodes where either field's greedy action is outside the other's near-optimal set (BOUNDARY_REL)."""
    eps = BOUNDARY_REL * np.maximum(1.0, V)
    ia, ib = Qa.argmin(1), Qb.argmin(1)
    r = np.arange(len(V))
    bad = (Qb[r, ia] > Qb.min(1) + eps) | (Qa[r, ib] > Qa.min(1) + eps)
    return int((bad & keep).sum())


def solve_level(name: str, Ls: dict, workers: int | None, checkpoints: Path) -> dict:
    """One level: boundary continuations, the bracketed value iterations, and the behavioral properness gates."""
    grid, gridb = O.Grid.level(name), O.Grid.level(name, wmax=O.OMEGA_MAX_CHECK)
    log(name, "transitions", len(grid.E), "nodes x", len(grid.u), "actions; boundary grid", len(gridb.E))
    T, Tb = O.transitions(grid, workers), O.transitions(gridb, workers)
    cont = {b: O.boundary_values(grid, T, Ls[b], b, workers) for b in ALL_BETAS}
    contb = {b: O.boundary_values(gridb, Tb, Ls[b], b, workers) for b in ALL_BETAS}
    crep = {str(b): {"omega_max_12": continuation_report(grid, T, cont[b]),
                     "omega_max_14": continuation_report(gridb, Tb, contb[b])} for b in ALL_BETAS}
    log(name, "boundary continuation", json.dumps(crep))
    out = {"grid": grid, "continuation": crep, "betas": {}}
    R = grid.reported()
    rs = np.flatnonzero(R)
    U0, demo = {}, {}
    for beta in ALL_BETAS:  # the demonstrator's own cost-to-go: the upper initialization
        _, U0[beta], dmax, _ = O.first_entry(grid.E, grid.W, Ls[beta], beta, DEMO_STEPS, "demonstrator",
                                             workers=workers)
        miss = U0[beta] >= O.BIG / 2  # e.g. exact hanging rest, where energy shaping applies u = 0
        U0[beta][miss] = O.lookahead_continuation(grid.E[miss], grid.W[miss], Ls[beta], beta, grid.u)
        demo[beta] = {"unreached_R": int((R & miss).sum()), "lookahead_recovered_R": int((R & miss & (U0[beta] < O.BIG / 2)).sum()),
                      "margin_violations_R": int((R & (dmax > O.OMEGA_SAFE)).sum()),
                      "max_abs_omega_R": float(dmax[R].max())}
        log(name, "beta", beta, "demonstrator from R", json.dumps(demo[beta]))
    O._SHARED.update(grid=grid, gridb=gridb, T=T, Tb=Tb, L=Ls, U0=U0, cont=cont, contb=contb, name=name,
                     wgt=grid.weights(T["e"], T["w"]), wgtb=gridb.weights(Tb["e"], Tb["w"]),
                     checkpoints=checkpoints, commit=git("rev-parse", "HEAD"))
    jobs = [(k, b) for b in ALL_BETAS for k in ("lower", "upper", "boundary")]
    res = {(k, b): r for k, b, r in O.pmap(_vi_job, jobs, workers, f"{name} value iterations")}
    for k in ("wgt", "wgtb"):
        del O._SHARED[k]
    offset = (gridb.nw - grid.nw) // 2
    ii, jj = np.divmod(np.arange(len(grid.E)), grid.nw)
    inner = ii * gridb.nw + jj + offset  # each grid node in the boundary grid
    for beta in ALL_BETAS:
        L = Ls[beta]
        lo, up, bd = res[("lower", beta)], res[("upper", beta)], res[("boundary", beta)]
        V = lo["V"]
        both = (V < O.BIG / 2) & (up["V"] < O.BIG / 2)
        one = (V < O.BIG / 2) != (up["V"] < O.BIG / 2)
        gap = float(np.abs(V - up["V"])[R & both].max()) if (R & both).any() else 0.0
        if (R & one).any():
            gap = float("inf")
        Vb = bd["V"][inner]
        fin = (V < O.BIG / 2) & (Vb < O.BIG / 2)
        brel = np.abs(V - Vb)[R & fin] / np.maximum(1.0, V[R & fin])
        bmis = int((R & ((V < O.BIG / 2) != (Vb < O.BIG / 2))).sum())
        off = R & ~O.in_terminal(L, grid.E, grid.W)
        amis = action_mismatch(lo["Q"], bd["Q"][inner], V, off & fin)
        # behavioral properness: greedy rollouts through the exact map from every node of R, under each field
        e12, J12, m12, rec12 = O.first_entry(grid.E[rs], grid.W[rs], L, beta, HORIZON, "oracle",
                                             policy=O.oracle_policy(grid, V, L, beta), workers=workers)
        e14, J14, m14, _ = O.first_entry(grid.E[rs], grid.W[rs], L, beta, HORIZON, "oracle",
                                         policy=O.oracle_policy(gridb, bd["V"], L, beta), workers=workers)
        entry = np.full(len(grid.E), -2, np.int64)  # -2: not rolled out (outside R)
        entry[rs] = e12
        arrived = (e12 >= 0) & (e14 >= 0)
        jdiff = np.abs(J12 - J14)[arrived] > TRAJ_REL * np.maximum(1.0, J12[arrived])
        miss = e12 < 0
        checks = {
            "lower": {"iterations": lo["iterations"], "stopped": lo["converged"], "last_change": lo["last_change"],
                      "monotone": lo["monotone"], "worst_drop": lo["worst_drop"]},
            "upper": {"iterations": up["iterations"], "stopped": up["converged"], "last_change": up["last_change"]},
            "bellman_residual_R": float(lo["residual"][R].max()),
            "initialization_gap_R": gap,
            "unreachable_R": int((R & (V >= O.BIG / 2)).sum()),
            "greedy_unreached_R": int(miss.sum()),
            "greedy_recurrence_min_distance": float(np.nanmin(rec12)) if miss.any() else None,
            "greedy_margin_violations_R": int((m12 > O.OMEGA_SAFE).sum()),
            "greedy_max_abs_omega_R": float(m12.max()),
            "demonstrator": demo[beta],
            "boundary_max_rel_R": float(brel.max()) if len(brel) else 0.0,
            "boundary_reachability_mismatch_R": bmis,
            "boundary_action_mismatch_R": amis,
            "boundary_greedy_unreached_R": int((e14 < 0).sum()),
            "boundary_greedy_margin_violations_R": int((m14 > O.OMEGA_SAFE).sum()),
            "boundary_trajectory_cost_mismatch_R": int(jdiff.sum()),
            "boundary_trajectory_cost_max_rel_R": float((np.abs(J12 - J14) / np.maximum(1.0, J12))[arrived].max())
            if arrived.any() else None,
        }
        checks["pass"] = bool(lo["converged"] and up["converged"] and lo["monotone"]
                              and checks["bellman_residual_R"] <= RESID_TOL and gap <= GAP_TOL
                              and checks["greedy_unreached_R"] == 0 and checks["greedy_margin_violations_R"] == 0
                              and demo[beta]["margin_violations_R"] == 0
                              and checks["boundary_max_rel_R"] <= BOUNDARY_REL and bmis == 0 and amis == 0
                              and checks["boundary_greedy_unreached_R"] == 0
                              and checks["boundary_greedy_margin_violations_R"] == 0
                              and checks["boundary_trajectory_cost_mismatch_R"] == 0)
        log(name, "beta", beta, json.dumps(checks))
        out["betas"][beta] = {"V": V, "Q": lo["Q"], "entry": entry, "checks": checks}
    return out


# ---- comparisons between and within levels ----

def greedy_u(grid, Q):
    return grid.u[Q.argmin(axis=1)]


def action_sets(Q, eps):
    return Q <= Q.min(axis=1, keepdims=True) + eps


def ties(grid, sets):
    neg, pos = (sets & (grid.u <= -0.3)).any(1), (sets & (grid.u >= 0.3)).any(1)
    s = sets.astype(np.int8)
    runs = (np.diff(np.pad(s, ((0, 0), (1, 0))), axis=1) == 1).sum(1)  # contiguous runs of the set
    return neg & pos & (runs > 1)


def switching(grid, u):
    U = u.reshape(grid.ne, grid.nw)
    de = np.abs(np.roll(U, -1, axis=0) - U) >= 3  # edge (i, j)-(i+1, j), periodic
    dw = np.zeros_like(de)
    dw[:, :-1] = np.abs(U[:, 1:] - U[:, :-1]) >= 3  # edge (i, j)-(i, j+1)
    return de, dw


def near_switch(grid, de, dw, cells):
    m = (de | np.roll(de, 1, axis=0) | dw | np.pad(dw, ((0, 0), (1, 0)))[:, :-1]).astype(bool)
    out = m.copy()
    for di in range(-cells, cells + 1):
        for dj in range(-cells, cells + 1):
            sh = np.roll(m, di, axis=0)
            if dj > 0:
                sh = np.pad(sh, ((0, 0), (dj, 0)))[:, :-dj]
            elif dj < 0:
                sh = np.pad(sh, ((0, 0), (0, -dj)))[:, -dj:]
            out |= sh
    return out.ravel()


def coarse_view(coarse, fine, arr):
    """fine-node values at the coarse grid's nodes (every coarse node is a fine node)."""
    fi, fj = fine.ne // coarse.ne, (fine.nw - 1) // (coarse.nw - 1)
    ii, jj = np.divmod(np.arange(len(coarse.E)), coarse.nw)
    return arr.reshape(fine.ne, fine.nw)[ii * fi, jj * fj] if arr.ndim == 1 else arr[(ii * fi) * fine.nw + jj * fj]


def convergence(lc: dict, lf: dict, Ls: dict) -> dict:
    gc, gf = lc["grid"], lf["grid"]
    R = gc.reported()
    out = {}
    for beta in O.BETAS:
        bc, bf = lc["betas"][beta], lf["betas"][beta]
        L = Ls[beta]
        Vc, Vf = bc["V"], coarse_view(gc, gf, bf["V"])
        mask = R & ~O.in_terminal(L, gc.E, gc.W) & (Vc < O.BIG / 2) & (Vf < O.BIG / 2)
        dV = np.abs(Vc - Vf)
        if not mask.any():  # nothing finite to compare: the rule fails for this beta
            out[str(beta)] = {"comparable_nodes": 0, "pass": False}
            continue
        eps = max(1e-6, float(np.quantile(dV[mask], 0.95)))
        named = {}
        for k, (e, w) in STARTS.items():
            v1, v2 = float(gc.interp(Vc, np.array([e]), np.array([w]))[0]), float(gf.interp(bf["V"], np.array([e]), np.array([w]))[0])
            named[k] = {"coarse": v1, "fine": v2, "pass": abs(v1 - v2) <= 0.02 * v2 + 0.01}
        rel95 = float(np.quantile(dV[mask] / (Vc[mask] + 0.01), 0.95))
        uc, uf = greedy_u(gc, bc["Q"]), coarse_view(gc, gf, greedy_u(gf, bf["Q"]))
        tc = ties(gc, action_sets(bc["Q"], eps))
        tf = coarse_view(gc, gf, ties(gf, action_sets(bf["Q"], eps)))
        sc = near_switch(gc, *switching(gc, uc), 2)
        sf = coarse_view(gc, gf, near_switch(gf, *switching(gf, greedy_u(gf, bf["Q"])),
                                              2 * (gf.ne // gc.ne)))
        keep = mask & ~tc & ~tf & ~sc & ~sf
        agree = float((np.abs(uc - uf)[keep] <= 0.3).mean()) if keep.any() else 0.0
        out[str(beta)] = {"comparable_nodes": int(mask.sum()), "eps_pair_95": eps, "named": named, "rel_95": rel95, "action_agreement": agree,
                          "action_nodes": int(keep.sum()),
                          "pass": all(v["pass"] for v in named.values()) and rel95 <= 0.05 and agree >= 0.9}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def eps_tie(lc: dict, lo: dict, beta: float, L: dict) -> float:
    gc, go = lc["grid"], lo["grid"]
    R = gc.reported()
    Vc, Vo = lc["betas"][beta]["V"], coarse_view(gc, go, lo["betas"][beta]["V"])
    mask = R & ~O.in_terminal(L, gc.E, gc.W) & (Vc < O.BIG / 2) & (Vo < O.BIG / 2)
    return max(1e-6, float(np.quantile(np.abs(Vc - Vo)[mask], 0.95))) if mask.any() else 1e-6


# ---- checks at the oracle level ----

def lqr_consistency(grid, b, L, eps):
    x2 = O.quad(L["P"], grid.E, grid.W)
    shell = ~O.in_terminal(L, grid.E, grid.W) & (x2 <= 2 * L["c"])
    V = b["V"]
    vals = np.abs(V - x2)[shell] <= 0.05 * x2[shell] + eps
    Kx = (np.stack([grid.E, grid.W], 1) @ L["K"].T)[:, 0]
    tie = ties(grid, action_sets(b["Q"], eps))
    act_mask = shell & ~tie & (np.abs(Kx) <= 3)
    acts = np.abs(greedy_u(grid, b["Q"]) + Kx)[act_mask] <= 0.3
    n_v, n_a = int(shell.sum()), int(act_mask.sum())
    return {"value_nodes": n_v, "value_pass_fraction": float(vals.mean()) if n_v else None,
            "action_nodes": n_a, "action_agreement": float(acts.mean()) if n_a else None,
            "testable": n_v > 0 and n_a > 0,
            "pass": bool(n_v > 0 and vals.all() and n_a > 0 and acts.mean() >= 0.9)}


def symmetry(grid, V, eps):
    U = V.reshape(grid.ne, grid.nw)
    M = U[(-np.arange(grid.ne)) % grid.ne][:, ::-1]
    R = grid.reported().reshape(grid.ne, grid.nw)
    fin = (U < O.BIG / 2) & (M < O.BIG / 2)
    if not (R & fin).any():
        return {"max_abs": None, "eps_tie": eps, "pass": False}
    d = float(np.abs(U - M)[R & fin].max())
    return {"max_abs": d, "eps_tie": eps, "pass": d <= eps and not (R & ((U < O.BIG / 2) != (M < O.BIG / 2))).any()}


SEG = 20


def shoot_costs(params, s0, L, beta):
    """First-exit costs of every row of params (M, HORIZON / SEG piecewise-constant actions) from s0."""
    u = np.repeat(np.clip(params, -3, 3), SEG, axis=1)
    s = np.repeat(s0, len(params), axis=0)
    acc, done, total = np.zeros(len(params)), np.zeros(len(params), bool), np.full(len(params), np.nan)
    e, w = O.ew_of(s)
    hit = O.in_terminal(L, e, w)
    total[hit], done[hit] = O.terminal_value(L, e[hit], w[hit]), True
    for t in range(HORIZON):
        a = np.flatnonzero(~done)
        if not len(a):
            break
        acc[a] += O.cost(O.ew_of(s[a])[0], u[a, t], beta)
        s[a] = O.step3(s[a], u[a, t])
        e, w = O.ew_of(s[a])
        hit = O.in_terminal(L, e, w)
        total[a[hit]], done[a[hit]] = acc[a[hit]] + O.terminal_value(L, e[hit], w[hit]), True
    return np.where(done, total, acc + 1e3)


def _shoot_job(args):
    s0, p0, L, beta = args
    nseg = HORIZON // SEG
    eye = np.eye(nseg, dtype=bool)

    def fun(p):
        h = 1e-4
        lo, hi = np.maximum(p - h, -3), np.minimum(p + h, 3)
        c = shoot_costs(np.vstack([p, np.where(eye, hi, p), np.where(eye, lo, p)]), s0, L, beta)
        return c[0], (c[1:nseg + 1] - c[nseg + 1:]) / (hi - lo)
    r = minimize(fun, np.clip(p0, -3, 3), jac=True, method="L-BFGS-B", bounds=[(-3, 3)] * nseg,
                 options={"maxiter": 200})
    return float(r.fun)


def shooting(grid, V, L, beta, eps, workers=None):
    """An upper-bound consistency check: the best of three bounded L-BFGS-B solves from each named start."""
    pol = O.oracle_policy(grid, V, L, beta)
    nseg = HORIZON // SEG
    jobs, meta = [], {}
    for k, (e0, w0) in STARTS.items():
        s0 = start_s3(e0, w0)
        _, ug = O.rollout(pol, s0, HORIZON)
        seg_mean = ug.reshape(nseg, SEG).mean(1)
        meta[k] = (float(grid.interp(V, np.array([e0]), np.array([w0]))[0]),
                   float(shoot_costs(seg_mean[None], s0, L, beta)[0]))
        for p0 in (seg_mean, np.zeros(nseg), np.random.default_rng(1452).uniform(-3, 3, nseg)):
            jobs.append((s0, p0, L, beta))
    best = O.pmap(_shoot_job, jobs, workers, f"beta {beta} shooting")
    out = {}
    for n, (k, (v, greedy_cost)) in enumerate(meta.items()):
        j = min(best[3 * n:3 * n + 3])
        out[k] = {"V_dp": v, "J_shoot": j, "J_greedy_segmented": greedy_cost,
                  "lower_pass": j >= v - (eps + 0.02 * v), "gap_pass": j <= 1.10 * v + 0.01}
    out["pass"] = all(x["lower_pass"] and x["gap_pass"] for x in out.values() if isinstance(x, dict))
    return out


def references(Ls) -> dict:
    hang = np.array([[0.0, 1.0, 0.0]])
    up = O.s3_of(np.array([0.0]), np.array([0.0]))
    zero = lambda s: np.zeros(len(s))  # noqa: E731
    out = {}
    s, u = O.rollout(zero, hang, HORIZON)
    h = float(O.score(s, u, Ls[O.BETAS[0]], 0.0)["height_cost"][0])
    out["hanging_rest"] = {"height_total": h, "pass": abs(h - 10.0) <= 1e-9}
    s, u = O.rollout(zero, up, HORIZON)
    h = float(O.score(s, u, Ls[O.BETAS[0]], 0.0)["height_cost"][0])
    out["upright_rest"] = {"height_total": h, "pass": h <= 1e-4}
    sp, _ = O.rollout(lambda s: np.full(len(s), 3.0), hang, HORIZON)
    sm, _ = O.rollout(lambda s: np.full(len(s), -3.0), hang, HORIZON)
    th = np.unwrap(np.arctan2(sp[0, :, 0], sp[0, :, 1]))
    inv = 0.5 * sp[0, :, 2] ** 2 - O.G * np.cos(th) - 3.0 * th
    star = brentq(lambda x: O.G * (1 - np.cos(x)) - 3.0 * x, 0.1, np.pi)
    mirror = float(np.abs(sm[0] - sp[0] * np.array([-1, 1, -1])).max())
    out["constant_plus3"] = {"invariant_drift": float(np.abs(inv - inv[0]).max()), "theta_max": float(th.max()),
                             "theta_star": star, "pass": bool(float(np.abs(inv - inv[0]).max()) <= 1e-2
                                                           and abs(th.max() - star) <= 1e-2)}
    out["constant_minus3_mirror"] = {"max_abs": mirror, "pass": mirror <= 1e-12}
    for beta in O.BETAS:
        L = Ls[beta]
        ang = np.arange(8) * np.pi / 4
        z = np.stack([np.cos(ang), np.sin(ang)], 1)
        x0 = 0.9 * np.sqrt(L["c"]) * np.linalg.solve(np.linalg.cholesky(L["P"]).T, z.T).T
        s, u = O.rollout(lambda s, L=L: -(np.stack(O.ew_of(s), 1) @ L["K"].T)[:, 0], O.s3_of(x0[:, 0], x0[:, 1]),
                         HORIZON)
        e, w = O.ew_of(s)
        v = O.quad(L["P"], e, w)
        floor = 1e-12 * L["c"]
        dec = all(bool((np.diff(v[i])[v[i, :-1] > floor] < 0).all()) for i in range(8))
        sc = O.score(s, u, L, beta)
        ratio = (sc["running_cost"] + v[:, -1]) / O.quad(L["P"], x0[:, 0], x0[:, 1])
        out[f"lqr_beta{beta}"] = {"decreasing_above_floor": dec, "decrease_floor": floor,
                                  "final_over_c_max": float((v[:, -1] / L["c"]).max()),
                                  "cost_ratio_min": float(ratio.min()), "cost_ratio_max": float(ratio.max()),
                                  "pass": dec and float((v[:, -1] / L["c"]).max()) <= 1e-6
                                  and bool((np.abs(ratio - 1) <= 0.10).all())}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def summarize(sc: dict) -> dict:
    return {k: (float(np.nanmedian(np.where(np.isfinite(v), v, np.nan))) if np.asarray(v).dtype != bool
                else float(np.mean(v))) for k, v in sc.items()}


def beta_selection(level: dict, Ls: dict, eps: dict, data: Path) -> dict:
    d = np.load(data)
    st = d["obs/state"].astype(np.float64)
    first = np.r_[0, np.cumsum(d["episode_lengths"])[:-1]]
    s0 = st[first]
    grid = level["grid"]
    table = {}
    for beta in O.BETAS:
        L, V, Q = Ls[beta], level["betas"][beta]["V"], level["betas"][beta]["Q"]
        sd, ud = O.rollout(O.demonstrator, s0, HORIZON)
        so, uo = O.rollout(O.oracle_policy(grid, V, L, beta), s0, HORIZON)
        dsc, osc = O.score(sd, ud, L, beta), O.score(so, uo, L, beta)
        cap = dsc["captured"]
        admissible = bool((osc["captured"] & osc["final_hold"])[cap].all())
        both = cap & osc["captured"]
        # a start is tied when its continuous Q over the action grid has a clockwise/counterclockwise tie
        e0, _ = O.ew_of(s0)
        qs = np.stack([O.q_continuous(grid, V, L, beta, s0, np.full(len(s0), a)) for a in grid.u], 1)
        tied = ties(grid, action_sets(qs, eps[beta]))
        agree = float(((osc["direction"] == dsc["direction"]) | tied)[both].mean()) if both.any() else 0.0
        med = lambda k: float(np.nanmedian(np.abs(osc[k] - dsc[k])[both])) if both.any() else float("inf")  # noqa: E731
        scales = {"swing_count": 1.0, "saturation_before_capture": 0.1, "time_to_terminal": 1.0,
                  "physical_effort": max(1.0, 0.1 * float(np.median(dsc["physical_effort"][cap]))),
                  "action_variation": max(0.1, 0.1 * float(np.median(dsc["action_variation"][cap])))}
        disc = {k: med(k) for k in scales}
        compatible = admissible and agree >= 0.9 and all(disc[k] <= scales[k] for k in scales)
        total = sum(disc[k] / scales[k] for k in scales) + (1 - agree) / 0.1
        table[str(beta)] = {"admissible": admissible, "direction_agreement": agree, "tied_starts": int(tied.sum()),
                            "median_discrepancy": disc, "scales": scales, "compatible": compatible,
                            "scaled_sum": total, "demonstrator_captured": int(cap.sum()),
                            "oracle_captured_and_held": int((osc["captured"] & osc["final_hold"]).sum()),
                            "demonstrator": summarize(dsc), "oracle": summarize(osc)}
    ok = [b for b in O.BETAS if table[str(b)]["compatible"]]
    sel = min(ok, key=lambda b: (table[str(b)]["scaled_sum"], b)) if ok else None
    return {"table": table, "selected_beta": sel,
            "outcome": "demonstrator-compatible DP objective" if sel is not None else "no compatible beta"}


def regions(level: dict, L: dict, beta: float) -> dict:
    g = level["grid"]
    term = O.in_terminal(L, g.E, g.W)
    entry = level["betas"][beta]["entry"]
    capture = ~term & (entry >= 0) & (entry <= CAPTURE_STEPS)
    H = O.energy(g.E, g.W)
    rest = ~term & ~capture
    return {"omega_c": term, "capture": capture, "pumping": rest & (H < 0), "overshoot": rest & (H >= 0)}


def provenance(A, B, t0) -> dict:
    return {"predeclaration": "docs/experiments/pendulum-oracle.md (version 6.1)",
            "environment": {"code_commit": git("rev-parse", "HEAD"),
                            "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                            "python": sys.version.split()[0]},
            "data_sha256": DATA_SHA256, "linearization": {"A": A.tolist(), "B": B.tolist()},
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.out.exists() and not (a.resume and (a.out / "checkpoints").is_dir() and not (a.out / "oracle.json").exists()):
        raise SystemExit(f"refused: {a.out} exists" + ("" if a.resume else " (an incomplete run takes --resume)"))
    if a.resume and not a.out.exists():
        raise SystemExit(f"refused: nothing to resume at {a.out}")
    if sha(a.data) != DATA_SHA256:
        raise SystemExit("refused: the dataset is not the canonical continuous dataset")
    (a.out / "checkpoints").mkdir(parents=True, exist_ok=a.resume)
    t0 = time.time()
    A, B = O.linearize()
    Ls, levels, rules = {}, {}, {}

    def stop(reason: str) -> int:
        """The failure artifact: every diagnostic computed so far; no fields, no beta selection, no figure."""
        doc = {**provenance(A, B, t0), "status": "stopped", "reason": reason, "all_pass": False,
               "levels_run": list(levels), "convergence": rules,
               "levels": {n: {"continuation": lv["continuation"], "stop": lv.get("stop"),
                              "checks": {str(b): v["checks"] for b, v in lv["betas"].items()}}
                          for n, lv in levels.items()},
               "terminal_sets": {str(b): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in L.items()}
                                 for b, L in Ls.items()},
               "references": references(Ls) if len(Ls) == len(ALL_BETAS) else None}
        (a.out / "oracle.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
        log("stopped:", reason)
        return 1

    def failure(name: str):
        lv = levels[name]
        if "stop" in lv:
            return lv["stop"]
        bad = [b for b in O.BETAS if level_failed(lv["betas"][b]["checks"])]
        return f"{name}: level checks fail for beta {bad}" if bad else None

    for beta in ALL_BETAS:
        try:
            Ls[beta] = O.lqr_bundle(beta, A, B)
        except RuntimeError as err:  # a beta whose terminal set has no positive c fails the run
            return stop(f"terminal set: {err}")
    log("terminal sets", {b: {k: Ls[b][k] for k in ("c", "binding", "accuracy_ratio_min", "accuracy_ratio_max")}
                          for b in O.BETAS})
    for name in ("L1", "L2"):
        levels[name] = solve_level(name, Ls, a.workers, a.out / "checkpoints")
        if why := failure(name):
            return stop(why)
    rules["L1-L2"] = convergence(levels["L1"], levels["L2"], Ls)
    log("convergence L1-L2", rules["L1-L2"]["pass"])
    oracle_level, coarser = "L2", "L1"
    if not rule_passes("L1-L2", rules["L1-L2"]):
        levels["L3"] = solve_level("L3", Ls, a.workers, a.out / "checkpoints")
        if why := failure("L3"):
            return stop(why)
        rules["L2-L3"] = convergence(levels["L2"], levels["L3"], Ls)
        log("convergence L2-L3", rules["L2-L3"]["pass"])
        if not rule_passes("L2-L3", rules["L2-L3"]):
            return stop("not converged: the L2-L3 rule fails")
        oracle_level, coarser = "L3", "L2"
    converged = True
    lv = levels[oracle_level]
    g = lv["grid"]
    eps = {b: eps_tie(levels[coarser], lv, b, Ls[b]) for b in ALL_BETAS}
    per_beta, fields = {}, {"e": g.e, "w": g.w, "u": g.u}
    for beta in ALL_BETAS:
        b = lv["betas"][beta]
        L = Ls[beta]
        sets = action_sets(b["Q"], eps[beta])
        u = greedy_u(g, b["Q"])
        de, dw = switching(g, u)
        rec = {"level_checks": {lvl: levels[lvl]["betas"][beta]["checks"] for lvl in levels}, "eps_tie": eps[beta],
               "terminal_set": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in L.items()},
               "tied_nodes": int(ties(g, sets).sum()), "saturated_nodes": int((np.abs(u) >= 3).sum())}
        if beta > 0:
            rec["symmetry"] = symmetry(g, b["V"], eps[beta])
            rec["lqr_consistency"] = lqr_consistency(g, b, L, eps[beta])
            log("beta", beta, "shooting")
            rec["shooting"] = shooting(g, b["V"], L, beta, eps[beta], a.workers)
        per_beta[str(beta)] = rec
        tag = f"beta{beta:g}"
        fields |= {f"{tag}/V": b["V"], f"{tag}/action_set": sets, f"{tag}/tie": ties(g, sets), f"{tag}/greedy_u": u,
                   f"{tag}/switch_e": de, f"{tag}/switch_w": dw, f"{tag}/entry": b["entry"],
                   f"{tag}/P": L["P"], f"{tag}/K": L["K"], f"{tag}/c": np.array(L["c"])}
    refs = references(Ls)
    log("references", refs["pass"])
    sel = beta_selection(lv, Ls, eps, a.data)
    log("beta selection", sel["outcome"], sel["selected_beta"])
    rb = sel["selected_beta"] if sel["selected_beta"] is not None else 1e-2
    reg = regions(lv, Ls[rb], rb)
    fields |= {f"region/{k}": v for k, v in reg.items()} | {"region/beta": np.array(rb)}
    np.savez_compressed(a.out / "fields.npz", **fields)
    level_pass = all(levels[oracle_level]["betas"][b]["checks"]["pass"] for b in O.BETAS)
    beta_checks = all(per_beta[str(b)]["symmetry"]["pass"] and per_beta[str(b)]["lqr_consistency"]["pass"]
                      and per_beta[str(b)]["shooting"]["pass"] for b in O.BETAS)
    doc = {
        **provenance(A, B, t0), "status": "completed",
        "continuation": {n: lv["continuation"] for n, lv in levels.items()},
        "levels_run": list(levels), "oracle_level": oracle_level, "convergence": rules, "converged": converged,
        "per_beta": per_beta, "references": refs, "beta_selection": sel,
        "regions": {"beta": rb, "selected": sel["selected_beta"] is not None,
                    "nodes": {k: int(v.sum()) for k, v in reg.items()}},
        "fields_sha256": sha(a.out / "fields.npz"),
    }
    doc["all_pass"] = bool(converged and level_pass and beta_checks and refs["pass"])
    (a.out / "oracle.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    log("all_pass", doc["all_pass"], "fields", doc["fields_sha256"])
    return 0 if doc["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
