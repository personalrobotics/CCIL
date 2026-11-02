#!/usr/bin/env python3
"""Recovery-segment diagnostic, phase 1: data-only generation (docs/experiments/pendulum-recovery-diagnostic.md, v1).

    uv run --group evidence python tools/pendulum_recovery/generate.py --data DATASET.npz --out DIR [--device cuda]
        [--resume]

Inputs are the demonstrations, the declared representation (sin theta, cos theta, theta_dot) with its chart
(theta, omega), and the declared metric. Nothing here imports the exact map, the demonstrator, the harness or a capture
set. Stages, each written atomically under DIR and skipped on --resume: models (primary and check per seed, one fit
group), probe-s{seed}.json (the per-M ladder record and the selected M), segments-s{seed}.npz and
pointwise-s{seed}.npz (generation at the selected M). Phase 1 is complete only when manifest.json exists; it holds the
sha256 of every artifact, and phase 2 (audit.py) refuses to run without it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ccil.data.split import make_split
from ccil.schema import read_trajectories

REPO = Path(__file__).resolve().parents[2]
DATA_SHA256 = "59c332bc829db9c9da40d9c1bd9ebbf808965aca6a167be48dc0f2ade52cf43c"
SEEDS = tuple(range(3311, 3321))
R, VALIDATION_FRACTION = 0.1, 0.2
CHECK_SEED_OFFSET = 100_000
LADDER = (1, 2, 4, 8, 16, 32)
K, MAX_ITERATIONS, TOLERANCE, FD_STEP = 20, 30, 1e-4, 1e-3
SUPPORT_RADIUS, RESIDUAL_MAX = 1.0, 1e-3
BOUND, DT, G = 3.0, 0.02, 9.81  # the action half-range and interval are dataset declarations; g scales the metric
OFFSET_STD = BOUND / 3  # one third of the action half-range
PROBE_STRIDE = 10
DEGENERACY_MAX, ETA_MIN, TRUST_FRACTION, ACCEPTANCE_MIN = 0.5, 1.0, 0.25, 0.5
# the il-supervision-benchmarks study's dynamics block, unchanged (studies/pendulum-j-r-k/study.yaml)
DYNAMICS = {"hidden_layers": [512, 512], "activation": "relu", "lipschitz_type": "soft_sampling",
            "lipschitz_constraint": 2.0, "lipschitz_coordinates": "model_normalized", "soft_penalty_weight": 0.001,
            "soft_sampling_eps": 0.0003, "soft_samples": 5, "learning_rate": 0.001, "weight_decay": 1.0e-05,
            "epochs": 100, "batch_size": 512}
CLASSES = ("returned_at_initialization", "zero_displacement", "converged", "failed")
HEARTBEAT = 300


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Heartbeat:
    """Logs that a long stage is alive at least every HEARTBEAT seconds."""

    def __init__(self, label):
        self.label, self.t0, self.stop = label, time.time(), threading.Event()

    def __enter__(self):
        def run():
            while not self.stop.wait(HEARTBEAT):
                log(self.label, f"running, {time.time() - self.t0:.0f} s")
        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        log(self.label, f"done, {time.time() - self.t0:.0f} s")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ---- the declared chart and metric ----

def chart(z):
    """(theta, omega) of representation states (..., 3)."""
    return np.arctan2(z[..., 0], z[..., 1]), z[..., 2]


def encode(theta, omega):
    return np.stack([np.sin(theta), np.cos(theta), omega], -1)


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def dist(z, y):
    """d(x, y) = sqrt(wrap(e_x - e_y)^2 + ((omega_x - omega_y) / sqrt(g))^2); e - e' = theta - theta' up to wrap."""
    (tz, wz), (ty, wy) = chart(z), chart(y)
    return np.sqrt(wrap(tz - ty) ** 2 + ((wz - wy) / np.sqrt(G)) ** 2)


def residual(z, y):
    """The 2-vector whose norm is dist(z, y)."""
    (tz, wz), (ty, wy) = chart(z), chart(y)
    return np.stack([wrap(tz - ty), (wz - wy) / np.sqrt(G)], -1)


class Tree:
    """Nearest-neighbor distance under d: periodic in theta, omega scaled by 1/sqrt(g) in a box that never wraps."""

    def __init__(self, states):
        self.t = cKDTree(self._p(states), boxsize=[2 * np.pi, 1e9])

    @staticmethod
    def _p(z):
        th, w = chart(z)
        return np.stack([np.mod(th, 2 * np.pi), w / np.sqrt(G) + 1e3], -1)

    def query(self, z):
        return self.t.query(self._p(z))[0]


# ---- the model, composed with re-encoding ----

def step(model, z, a):
    """One model step re-encoded through the chart: (next states, un-projected circle drift)."""
    nxt = z + model.delta(z, a.reshape(-1, 1))
    return encode(*chart(nxt)), np.abs(nxt[:, 0] ** 2 + nxt[:, 1] ** 2 - 1)


def compose(model, z0, U):
    """States (N, M + 1, 3) of U (N, M) from z0 (N, 3), and the largest un-projected drift per row."""
    zs, drift, z = [z0], np.zeros(len(z0)), z0
    for i in range(U.shape[1]):
        z, d = step(model, z, U[:, i])
        zs.append(z)
        drift = np.maximum(drift, d)
    return np.stack(zs, 1), drift


# ---- data ----

def units(data: Path) -> dict:
    """Per seed: the positive control's subset and split (il-supervision-benchmarks tools/capture_control.py)."""
    ds = read_trajectories(data)
    n = len(ds.episodes)
    out = {}
    for s in SEEDS:
        keep = np.random.default_rng(np.random.SeedSequence([s, 9])).permutation(n)[:round(R * n)]
        eps = {ds.episodes[i].episode_id: ds.episodes[i] for i in keep}
        split = make_split(list(eps), VALIDATION_FRACTION, s)
        S = {e: eps[e].observations["state"].astype(np.float64) for e in eps}
        A = {e: eps[e].actions[:, 0].astype(np.float64) for e in eps}
        train_states = np.concatenate([S[e] for e in split["train"]])
        tree = Tree(train_states)
        tau = float(np.quantile(tree.query(np.concatenate([S[e] for e in split["validation"]])), 0.95))
        out[s] = {"split": split, "S": S, "A": A, "tree": tree, "tau_demo": tau}
    return out


def transitions(u: dict, ids):
    z = np.concatenate([u["S"][e][:-1] for e in ids])
    return z, np.concatenate([u["A"][e][:-1] for e in ids])[:, None], np.concatenate([u["S"][e][1:] for e in ids])


# ---- stages ----

def stage_models(out: Path, U: dict, device: str) -> dict:
    from ccil.models.mlp import MLPTransitionModel
    from ccil.models.mlp_batched import FitJob, fit_group
    d = out / "models"
    paths = {(s, role): d / f"s{s}-{role}.npz" for s in SEEDS for role in ("primary", "check")}
    if not all(p.exists() for p in paths.values()):
        d.mkdir(parents=True, exist_ok=True)
        jobs, keys = [], []
        for s in SEEDS:
            z, a, zn = transitions(U[s], U[s]["split"]["train"])
            for role, seed in (("primary", s), ("check", s + CHECK_SEED_OFFSET)):
                jobs.append(FitJob(z.astype(np.float32), a.astype(np.float32), zn.astype(np.float32), seed))
                keys.append((s, role))
        with Heartbeat(f"fit {len(jobs)} transition models"):
            res = fit_group(jobs, DYNAMICS, device=device, deterministic=True)
        for key, (model, rec) in zip(keys, res):
            tmp = paths[key].with_suffix(".tmp.npz")
            model.save(tmp)
            os.replace(tmp, paths[key])
            atomic_json(paths[key].with_suffix(".json"), rec)
    return {k: MLPTransitionModel.load(p, device) for k, p in paths.items()}


def held_out_error(model, u: dict, M: int) -> float:
    """epsilon_M: q95 over every validation window of d(F_M(x_t, U_t), x_{t+M}), composed with re-encoding."""
    z0, Us, tg = [], [], []
    for e in u["split"]["validation"]:
        S, A = u["S"][e], u["A"][e]
        t = np.arange(len(S) - M)
        z0.append(S[t]), Us.append(A[t[:, None] + np.arange(M)]), tg.append(S[t + M])
    zs, _ = compose(model, np.concatenate(z0), np.concatenate(Us))
    return float(np.quantile(dist(zs[:, -1], np.concatenate(tg)), 0.95))


def proposals(u: dict, M: int, stride: int):
    """Every (episode, t, k) window of the training episodes at stride, its demonstration states and actions, and
    the persistent action-bias proposal: one requested offset per segment, clipped per step."""
    rows = []
    for ei, e in enumerate(u["split"]["train"]):
        for t in range(0, len(u["S"][e]) - M, stride):
            for k in range(K):
                rows.append((ei, t, k))
    rows = np.array(rows, np.int64).reshape(-1, 3)
    train = u["split"]["train"]
    seed = u["split"]["seed"]
    req = np.array([np.random.default_rng(np.random.SeedSequence([seed, ei, t, M, k])).normal(0.0, OFFSET_STD)
                    for ei, t, k in rows])
    demoS = np.stack([u["S"][train[ei]][t:t + M + 1] for ei, t, _ in rows])
    demoU = np.stack([u["A"][train[ei]][t:t + M] for ei, t, _ in rows])
    Ut = np.clip(demoU + req[:, None], -BOUND, BOUND)
    return rows, demoS, demoU, req, Ut


def solve(model, z_src, Ut, z_tgt):
    """Newton in the chart (theta, omega) for F_M(x0, U) = target, central differences through the composed model.
    Returns (x0, initial residual, final residual, converged, moved: the chart unknowns changed)."""
    th, w = chart(z_src)
    p = np.stack([th, w], 1)
    p_init = p.copy()
    n = len(p)
    res = np.full(n, np.inf)
    done = np.zeros(n, bool)
    init = None
    for it in range(MAX_ITERATIONS + 1):
        act = np.flatnonzero(~done)
        if not len(act):
            break
        r = residual(compose(model, encode(p[act, 0], p[act, 1]), Ut[act])[0][:, -1], z_tgt[act])
        nr = np.linalg.norm(r, axis=1)
        res[act] = nr
        if it == 0:
            init = nr.copy()
        ok = nr <= TOLERANCE
        done[act[ok]] = True
        if it == MAX_ITERATIONS:
            break
        act, r = act[~ok], r[~ok]
        if not len(act):
            break
        J = np.zeros((len(act), 2, 2))
        for j in range(2):
            h = np.zeros(2)
            h[j] = FD_STEP
            fp = residual(compose(model, encode(*(p[act] + h).T), Ut[act])[0][:, -1], z_tgt[act])
            fm = residual(compose(model, encode(*(p[act] - h).T), Ut[act])[0][:, -1], z_tgt[act])
            J[:, :, j] = (fp - fm) / (2 * FD_STEP)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):  # 2x2 solves per row
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            inv = np.stack([np.stack([J[:, 1, 1], -J[:, 0, 1]], 1), np.stack([-J[:, 1, 0], J[:, 0, 0]], 1)], 1) \
                / det[:, None, None]
            p[act] -= np.einsum("nij,nj->ni", inv, r)
        bad = ~np.isfinite(p[act]).all(1)
        done[act[bad]] = True  # non-finite: failed, left unconverged
    return encode(p[:, 0], p[:, 1]), init, res, res <= TOLERANCE, (p != p_init).any(1)


def segments(model, check, u: dict, M: int, stride: int) -> dict:
    """Generate, validate and measure every proposal at (M, stride)."""
    rows, demoS, demoU, req, Ut = proposals(u, M, stride)
    x0, init, res, conv, moved = solve(model, demoS[:, 0], Ut, demoS[:, -1])
    X, drift = compose(model, x0, Ut)
    finite = np.isfinite(X).all((1, 2)) & np.isfinite(Ut).all(1)
    cls = np.where(~conv | ~finite, 3, np.where(init <= TOLERANCE, 0, np.where(~moved, 1, 2))).astype(np.int8)
    final = np.where(finite, dist(np.nan_to_num(X[:, -1]), demoS[:, -1]), np.inf)
    within = np.where(finite, np.linalg.norm(np.nan_to_num(X) - demoS, axis=2).max(1), np.inf) < SUPPORT_RADIUS
    bounded = (np.abs(Ut) <= BOUND).all(1)
    accepted = finite & conv & bounded & within & (final <= RESIDUAL_MAX)
    # check model: the larger of its largest one-step and its composed disagreement with the segment
    Xs = np.nan_to_num(X)
    one = np.stack([dist(step(check, Xs[:, i], Ut[:, i])[0], Xs[:, i + 1]) for i in range(M)], 1).max(1)
    comp = dist(compose(check, Xs[:, 0], Ut)[0][:, -1], demoS[:, -1])
    dev = Ut - demoU
    return {"rows": rows, "X": X, "U": Ut, "target": demoS[:, -1], "source": demoS[:, 0], "requested_offset": req,
            "realized_offset": dev, "clipped_fraction": (np.abs(demoU + req[:, None]) > BOUND).mean(1),
            "impulse": dev.sum(1) * DT, "deviation_energy": (dev ** 2).sum(1) * DT, "initial_residual": init,
            "residual": final, "class": cls, "finite": finite, "converged": conv, "bounded": bounded,
            "within_support_radius": within, "accepted": accepted, "check_disagreement": np.maximum(one, comp),
            "drift_max": drift, "d_demo_x0": np.where(finite, u["tree"].query(Xs[:, 0]), np.inf),
            "d_source_x0": np.where(finite, dist(Xs[:, 0], demoS[:, 0]), np.inf)}


def stage_probe(out: Path, U: dict, models: dict, s: int) -> dict:
    path = out / f"probe-s{s}.json"
    if path.exists():
        return json.loads(path.read_text())
    u, tau = U[s], U[s]["tau_demo"]
    rec = {"seed": s, "tau_demo": tau, "trust_threshold": TRUST_FRACTION * tau, "ladder": {}}
    for M in LADDER:
        g = segments(models[s, "primary"], models[s, "check"], u, M, PROBE_STRIDE)
        a = g["accepted"]
        n_a = int(a.sum())
        eps = held_out_error(models[s, "primary"], u, M)
        q = (lambda v: np.quantile(v, [0.05, 0.5, 0.95]).tolist() if len(v) else None)  # noqa: E731
        m = {"segments": int(len(a)), "accepted": n_a, "acceptance": n_a / len(a),
             "degeneracy": float(np.isin(g["class"][a], (0, 1)).mean()) if n_a else None,
             "r_reach": float(np.median(g["d_demo_x0"][a])) if n_a else None,
             "epsilon_M": eps, "kappa_M": float(np.quantile(g["check_disagreement"][a], 0.95)) if n_a else None,
             "class_counts": {c: int((g["class"] == i).sum()) for i, c in enumerate(CLASSES)},
             "residual_q05_q50_q95": q(g["residual"][a]), "d_source_x0_q05_q50_q95": q(g["d_source_x0"][a]),
             "clipped_fraction_mean": float(g["clipped_fraction"].mean()), "drift_max": float(g["drift_max"][a].max())
             if n_a else None}
        m["eta_reach"] = m["r_reach"] / tau if n_a else None
        m["criteria"] = {"nondegenerate": n_a > 0 and m["degeneracy"] <= DEGENERACY_MAX,
                         "reaches_beyond_support": n_a > 0 and m["eta_reach"] >= ETA_MIN,
                         "trust_on_demonstrations": eps <= TRUST_FRACTION * tau,
                         "trust_off_demonstrations": n_a > 0 and m["kappa_M"] <= TRUST_FRACTION * tau,
                         "workable": m["acceptance"] >= ACCEPTANCE_MIN}
        m["admissible"] = all(m["criteria"].values())
        rec["ladder"][str(M)] = m
        log("probe", s, f"M={M}", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                                               if k in ("acceptance", "degeneracy", "eta_reach", "epsilon_M",
                                                        "kappa_M", "admissible")}))
    ok = [M for M in LADDER if rec["ladder"][str(M)]["admissible"]]
    rec["selected_M"] = ok[0] if ok else None
    atomic_json(path, rec)
    log("probe", s, "selected M", rec["selected_M"])
    return rec


def stage_generate(out: Path, U: dict, models: dict, s: int, M: int | None):
    seg_p, pw_p = out / f"segments-s{s}.npz", out / f"pointwise-s{s}.npz"
    if seg_p.exists() and pw_p.exists():
        return
    if M is None:  # the empty synthetic view
        atomic_npz(seg_p, M=np.array(0), empty=np.array(True))
        atomic_npz(pw_p, observations=np.zeros((0, 3)), actions=np.zeros((0, 1)), segment=np.zeros(0, np.int64),
                   offset=np.zeros(0, np.int64), M=np.array(0), evidence_class=np.array(EVIDENCE))
        log("generate", s, "no admissible M: empty synthetic view")
        return
    with Heartbeat(f"generate s{s} M={M}"):
        g = segments(models[s, "primary"], models[s, "check"], U[s], M, 1)
    atomic_npz(seg_p, M=np.array(M), empty=np.array(False), episodes=np.array(U[s]["split"]["train"]), **g)
    acc = np.flatnonzero(g["accepted"])
    seg = np.repeat(acc, M)
    off = np.tile(np.arange(M), len(acc))
    obs, act = g["X"][seg, off], g["U"][seg, off][:, None]
    # every row reconciles to exactly one accepted segment at one offset
    if len(set(zip(seg.tolist(), off.tolist()))) != len(seg) or not (
            np.array_equal(obs, g["X"][seg, off]) and g["accepted"][seg].all()):
        raise RuntimeError(f"s{s}: the pointwise view does not reconcile to its segments")
    atomic_npz(pw_p, observations=obs, actions=act, segment=seg, offset=off, M=np.array(M),
               evidence_class=np.array(EVIDENCE))
    log("generate", s, f"M={M}", f"{len(acc)} of {len(g['accepted'])} segments accepted, {len(seg)} rows")


EVIDENCE = "learned-model recovery segment (empirical, not certified)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if sha(a.data) != DATA_SHA256:
        raise SystemExit("refused: the dataset is not the canonical continuous dataset")
    if a.out.exists() and not (a.resume and not (a.out / "manifest.json").exists()):
        raise SystemExit(f"refused: {a.out} exists" + ("" if a.resume else " (an incomplete run takes --resume)"))
    a.out.mkdir(parents=True, exist_ok=a.resume)
    t0 = time.time()
    U = units(a.data)
    log("data ready", {s: round(U[s]["tau_demo"], 4) for s in SEEDS})
    models = stage_models(a.out, U, a.device)
    log("models ready", len(models))
    selected = {}
    for i, s in enumerate(SEEDS):
        with Heartbeat(f"probe s{s} ({i + 1} of {len(SEEDS)})"):
            selected[s] = stage_probe(a.out, U, models, s)["selected_M"]
    for i, s in enumerate(SEEDS):
        stage_generate(a.out, U, models, s, selected[s])
        log("generated", i + 1, "of", len(SEEDS))
    files = sorted(p for p in a.out.rglob("*") if p.is_file() and p.name != "manifest.json" and ".tmp" not in p.name)
    manifest = {"phase": 1, "predeclaration": "docs/experiments/pendulum-recovery-diagnostic.md (version 1)",
                "code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--porcelain")),
                "data_sha256": DATA_SHA256, "seeds": list(SEEDS), "selected_M": {str(s): selected[s] for s in SEEDS},
                "splits": {str(s): {k: U[s]["split"][k] for k in ("train", "validation")} for s in SEEDS},
                "tau_demo": {str(s): U[s]["tau_demo"] for s in SEEDS}, "dynamics": DYNAMICS,
                "seconds": round(time.time() - t0, 1),
                "files": {str(p.relative_to(a.out)): sha(p) for p in files}}
    atomic_json(a.out / "manifest.json", manifest)
    log("phase 1 complete; selected M", manifest["selected_M"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
