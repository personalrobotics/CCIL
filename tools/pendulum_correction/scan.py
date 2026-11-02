#!/usr/bin/env python3
"""Gate 3A: what exact-dynamics CCIL generates on the Pendulum (docs/experiments/pendulum-correction.md, version 1).

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_correction/scan.py"
    $R --data pendulum_cont_corrected.npz --out DIR [--workers N] [--resume]
    $R --data pendulum_cont_corrected.npz --out DIR --check docs/fixtures/evidence/pendulum-correction/correction.json

DIR must not exist (or, with --resume, be an incomplete scan at the same commit). Per sigma it receives
rows-a-sigma{s}.npz (every candidate's per-row fields, the release asset) and, where the gate needs solver b,
rows-b-sigma{s}.npz; then correction.json (the cell table, tau_demo, the treatment cells, the code state). The scan is
complete only when correction.json exists. --check recomputes correction.json from DIR's row files and compares.
No task regret is computed: the DP oracle did not converge (docs/experiments/pendulum-oracle.md, "Result").
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
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "pendulum_oracle"))
import oracle as O  # noqa: E402  (the exact map, coordinates, the corrected demonstrator, pmap and log)

from ccil.core import (ActionBounds, Finite, GenerationFailed, IdentityRepresentation,  # noqa: E402
                       NoisyActionPreimage, SupportRadius, generate_labels)
from ccil.data.split import make_split  # noqa: E402
from ccil.schema import TrajectoryDataset, read_trajectories  # noqa: E402

log = O.log
DATA_SHA256 = "59c332bc829db9c9da40d9c1bd9ebbf808965aca6a167be48dc0f2ade52cf43c"
SEED = 9401  # non-study, as in il-supervision-benchmarks/tools/sigma_geometry.py
FRACTIONS = (0.1, 0.25, 1.0)
VALIDATION_FRACTION = 0.2
SIGMAS = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)  # normalized; action-noise std = 3 sigma
K, MAX_ITERATIONS, TOLERANCE, SUPPORT_RADIUS = 20, 30, 1e-4, 1.0
MANIFOLD_TOL = 6.7e-7
CONSTRUCTION_TOL, CONSISTENCY_TOL, SWITCH_AMBIGUOUS = 1e-3, 0.3, 0.01
ROLLOUT, HOLD, HOLD_E, HOLD_W = 500, 100, 0.1, 0.5
OFF_MANIFOLD_TRIGGER, NEAR_IDENTITY_MAX, MIN_SUPPORTED, MIN_FRACTION, MIN_EPISODES = 0.01, 0.5, 100, 0.01, 5
GROUPS = 10  # each sigma's generation runs in this many episode groups, for progress (rows are independent)
CLASSES = ("returned_at_initialization", "zero_displacement", "converged", "failed")
REGIONS = ("lqr_window", "pumping", "overshoot")


class ExactModel:
    """The exact map as a CCIL transition model: delta = step(z, clip(a)) - z (the harness's ExactModel)."""

    def __init__(self):
        self.sha256 = hashlib.sha256(json.dumps(
            {"model": "exact", "map": "tools/vector_env/pendulum.py", "variant": "continuous", "action_bound": 3.0,
             "map_sha256": hashlib.sha256((REPO / "tools/vector_env/pendulum.py").read_bytes()).hexdigest()},
            sort_keys=True).encode()).hexdigest()

    def delta(self, z, a):
        z = np.asarray(z, np.float64)
        return O.step3(z, np.asarray(a, np.float64).reshape(len(z))) - z


MODEL = ExactModel()
VALIDATORS = [Finite(), GenerationFailed(), SupportRadius(SUPPORT_RADIUS), ActionBounds([-3.0], [3.0])]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phys(s3: np.ndarray) -> np.ndarray:
    """(e, omega / sqrt(g)): the physical metric is Euclidean in these, with e periodic."""
    e, w = O.ew_of(s3)
    return np.stack([e, w / O.OMEGA0], 1)


def dist(s, t) -> np.ndarray:
    a, b = phys(s), phys(t)
    return np.sqrt(O.wrap(a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2)


def tree(states: np.ndarray) -> cKDTree:
    """A periodic-in-e tree over physical coordinates (e shifted into [0, 2 pi), omega into a box that never wraps)."""
    p = phys(states) + [np.pi, 1e3]
    return cKDTree(np.mod(p, [2 * np.pi, 1e9]), boxsize=[2 * np.pi, 1e9])


def nearest(t: cKDTree, states: np.ndarray) -> np.ndarray:
    p = phys(states) + [np.pi, 1e3]
    return t.query(np.mod(p, [2 * np.pi, 1e9]))[0]


# ---- subsets, splits and the demonstrations' own spacing ----

def units(ds: TrajectoryDataset) -> dict:
    """Per r: the subset's training and validation episode ids, its training demonstration tree and tau_demo."""
    n = len(ds.episodes)
    perm = np.random.default_rng(np.random.SeedSequence([SEED, 9])).permutation(n)
    by_id = {e.episode_id: e for e in ds.episodes}
    out = {}
    for r in FRACTIONS:
        ids = [ds.episodes[i].episode_id for i in perm[:round(r * n)]]
        split = make_split(ids, VALIDATION_FRACTION, SEED)
        train = np.concatenate([by_id[e].observations["state"] for e in split["train"]]).astype(np.float64)
        val = np.concatenate([by_id[e].observations["state"] for e in split["validation"]]).astype(np.float64)
        t = tree(train)
        out[r] = {"train": split["train"], "validation": split["validation"], "tree": t,
                  "tau_demo": float(np.quantile(nearest(t, val), 0.95))}
    return out


# ---- per-row fields ----

def fields(z_src, z_tgt, a, z, failed, accepted, reasons) -> dict:
    """Every per-row field that does not depend on r (docs/experiments/pendulum-correction.md, "Per-row fields")."""
    a1 = a[:, 0].astype(np.float64)
    init = np.linalg.norm(z_src + MODEL.delta(z_src, a1) - z_tgt, axis=1)
    moved = (z != z_src).any(1)
    cls = np.where(failed, 3, np.where(init <= TOLERANCE, 0, np.where(~moved, 1, 2))).astype(np.int8)
    br = O.expert.branches(z)
    e, w = O.ew_of(z)
    H = O.energy(e, w)
    region = np.where(br["in_lqr"], 0, np.where(H < 0, 1, 2)).astype(np.int8)
    finite = np.isfinite(z).all(1)
    zf = np.where(finite[:, None], z, z_src)  # the exact map is evaluated only at finite states
    r_c = np.where(finite, dist(O.step3(zf, a1), z_tgt), np.inf)
    return {"accepted": accepted, "reasons": reasons, "class": cls, "initial_residual": init,
            "manifold_error": np.abs(z[:, 0] ** 2 + z[:, 1] ** 2 - 1), "construction_residual": r_c,
            "bounded": np.abs(a1) <= 3.0, "d_source": dist(zf, z_src),
            "consistency": np.abs(a1 - O.demonstrator(zf)), "switch_margin": br["switch_margin"],
            "region": region, "e": e, "w": w}


def _solver_a(args):
    """Solver a over one group of episodes: the harness's generator and validators through generate_labels."""
    sigma, ids = args
    g = O._SHARED
    ds = TrajectoryDataset([g["episodes"][i] for i in ids], g["meta"])
    gen = NoisyActionPreimage(K, 3.0 * sigma, max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE,
                              action_bounds=([-3.0], [3.0]))
    c = generate_labels(ds, IdentityRepresentation("state"), MODEL, gen, VALIDATORS, run_id="gate-3a", seed=SEED,
                        inputs={"transition_model": MODEL.sha256})
    src = {e.episode_id: e.observations["state"].astype(np.float64) for e in ds.episodes}
    ep = c.source_episode_id.astype(str)
    z_src = np.stack([src[e][f] for e, f in zip(ep, c.source_frame)])
    z_tgt = c.target_observations["state"].astype(np.float64)
    z = c.observations["state"].astype(np.float64)
    failed = np.asarray(c.reasons)[:, list(c.reason_codes).index("generation_failed")]
    out = fields(z_src, z_tgt, c.actions, z, failed, np.asarray(c.accepted), np.asarray(c.reasons))
    out |= {"episode": ep, "frame": c.source_frame, "draw": c.measurements["generator.draw_index"],
            "z": z, "a": c.actions[:, 0].astype(np.float64), "z_source": z_src, "z_target": z_tgt}
    log(f"solver a sigma {sigma:g}: episodes {ids[0]}-{ids[-1]} done")
    return out, list(c.reason_codes)


def manifold_newton(z_src, z_tgt, a):
    """Solver b: Newton in (theta, omega) from the source, residual in the physical metric; on the manifold by
    construction. Returns (z, failed, initial residual)."""
    tgt = phys(z_tgt)

    def R(th, w, j):  # the residual of rows j at (theta, omega)
        p = phys(O.step3(np.stack([np.sin(th), np.cos(th), w], 1), a[j]))
        return np.stack([O.wrap(p[:, 0] - tgt[j, 0]), p[:, 1] - tgt[j, 1]], 1)
    th, w = np.arctan2(z_src[:, 0], z_src[:, 1]), z_src[:, 2].copy()
    every = np.arange(len(th))
    init = np.linalg.norm(R(th, w, every), axis=1)
    res, broken, eps = init.copy(), np.zeros(len(th), bool), 1e-6
    for _ in range(MAX_ITERATIONS):
        i = np.flatnonzero(~broken & ~(res <= TOLERANCE))
        if not len(i):
            break
        J = np.stack([(R(th[i] + eps, w[i], i) - R(th[i] - eps, w[i], i)) / (2 * eps),
                      (R(th[i], w[i] + eps, i) - R(th[i], w[i] - eps, i)) / (2 * eps)], 2)
        ok = np.isfinite(J).all((1, 2)) & (np.abs(np.linalg.det(J)) > 1e-14)
        step = np.zeros((len(i), 2))
        step[ok] = np.linalg.solve(J[ok], R(th[i[ok]], w[i[ok]], i[ok])[..., None])[..., 0]
        ok &= np.isfinite(step).all(1)
        th[i[ok]] -= step[ok, 0]
        w[i[ok]] -= step[ok, 1]
        broken[i[~ok]] = True
        res[i[ok]] = np.linalg.norm(R(th[i[ok]], w[i[ok]], i[ok]), axis=1)
    z = np.stack([np.sin(th), np.cos(th), w], 1)
    return z, broken | ~(res <= TOLERANCE), init


def _solver_b(args):
    """Solver b on solver a's draws (the same source, target and action per row), with the harness's checks."""
    lo, hi = args
    g = O._SHARED["b"]
    z_src, z_tgt, a = g["z_source"][lo:hi], g["z_target"][lo:hi], g["a"][lo:hi]
    z, failed, init = manifold_newton(z_src, z_tgt, a)
    codes = O._SHARED["codes"]
    reasons = np.zeros((len(z), len(codes)), bool)
    reasons[:, codes.index("non_finite")] = ~np.isfinite(z).all(1)
    reasons[:, codes.index("generation_failed")] = failed
    with np.errstate(invalid="ignore"):
        reasons[:, codes.index("out_of_support")] = ~(np.linalg.norm(z - z_src, axis=1) < SUPPORT_RADIUS)
    reasons[:, codes.index("action_out_of_bounds")] = np.abs(a) > 3.0
    out = fields(z_src, z_tgt, a[:, None], z, failed, ~reasons.any(1), reasons)
    out["class"] = np.where(failed, 3, np.where(init <= TOLERANCE, 0, np.where(~(z != z_src).any(1), 1, 2))).astype(np.int8)
    out |= {"initial_residual": init, "z": z}
    return out


def _recovery(args):
    """The demonstrator from z (projected onto the manifold), through the exact map: captured and held."""
    lo, hi = args
    z = O._SHARED["recover"][lo:hi]
    e, w = O.ew_of(z)
    s, _ = O.rollout(O.demonstrator, O.s3_of(e, w), ROLLOUT)
    ee, ww = O.ew_of(s[:, -HOLD:])
    return ((np.abs(ee) <= HOLD_E) & (np.abs(ww) <= HOLD_W)).all(1)


def save_atomic(path: Path, arrays: dict):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def scan_sigma(sigma, ds, u, out: Path, workers, key):
    """Every row of one sigma under solver a, with d_demo per r and the demonstrator's recovery where the gate needs
    it; checkpointed atomically as rows-a-sigma{sigma}.npz."""
    path = out / f"rows-a-sigma{sigma:g}.npz"
    if path.exists():
        f = dict(np.load(path))
        if str(f["key"]) != key:
            raise RuntimeError(f"{path} is from a different scan: {f['key']}")
        log(f"solver a sigma {sigma:g}: resumed from {path.name}")
        return f
    n = len(ds.episodes)
    groups = np.array_split(np.arange(n), GROUPS)
    O._SHARED.update(episodes=ds.episodes, meta=ds.meta)
    parts = O.pmap(_solver_a, [(sigma, list(gr)) for gr in groups], workers, f"solver a sigma {sigma:g}")
    codes = parts[0][1]
    f = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    f |= extend(f, u, workers, f"sigma {sigma:g} solver a")
    f |= {"key": np.array(key), "reason_codes": np.array(codes), "sigma": np.array(sigma)}
    save_atomic(path, f)
    return f


def extend(f, u, workers, label) -> dict:
    """d_demo per r, and the demonstrator's recovery for every row that is otherwise supported at some r."""
    extra, cand = {}, np.zeros(len(f["z"]), bool)
    base = (f["accepted"] & (f["manifold_error"] <= MANIFOLD_TOL) & (f["construction_residual"] <= CONSTRUCTION_TOL)
            & f["bounded"] & (f["consistency"] <= CONSISTENCY_TOL) & (f["switch_margin"] >= SWITCH_AMBIGUOUS)
            & (f["region"] > 0))
    for r, x in u.items():
        extra[f"d_demo_r{r:g}"] = nearest(x["tree"], f["z"])
        cand |= base & np.isin(f["episode"], x["train"]) & (extra[f"d_demo_r{r:g}"] >= x["tau_demo"])
    rec = np.full(len(f["z"]), -1, np.int8)  # -1: not needed by the gate, not computed
    idx = np.flatnonzero(cand)
    if len(idx):
        O._SHARED["recover"] = f["z"][idx]
        size = max(1, min(2000, -(-len(idx) // (os.cpu_count() or 1))))
        parts = O.pmap(_recovery, [(lo, min(len(idx), lo + size)) for lo in range(0, len(idx), size)], workers,
                       f"{label} recovery rollouts ({len(idx)} rows)")
        rec[idx] = np.concatenate(parts).astype(np.int8)
    extra["recovered"] = rec
    return extra


def scan_sigma_b(sigma, fa, u, out: Path, workers, key):
    path = out / f"rows-b-sigma{sigma:g}.npz"
    if path.exists():
        f = dict(np.load(path))
        if str(f["key"]) != key:
            raise RuntimeError(f"{path} is from a different scan: {f['key']}")
        log(f"solver b sigma {sigma:g}: resumed from {path.name}")
        return f
    O._SHARED.update(b={"z_source": fa["z_source"], "z_target": fa["z_target"], "a": fa["a"]},
                     codes=[str(c) for c in fa["reason_codes"]])
    n = len(fa["z"])
    size = -(-n // (4 * (workers or os.cpu_count() or 1)))
    parts = O.pmap(_solver_b, [(lo, min(n, lo + size)) for lo in range(0, n, size)], workers, f"solver b sigma {sigma:g}")
    f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    f |= {k: fa[k] for k in ("episode", "frame", "draw", "a", "z_source", "z_target", "reason_codes")}
    f |= extend(f, u, workers, f"sigma {sigma:g} solver b")
    f |= {"key": np.array(key), "sigma": np.array(sigma)}
    save_atomic(path, f)
    return f


# ---- the gate ----

def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {f"q{int(p * 100):02d}": float(np.quantile(x, p)) for p in (0.05, 0.5, 0.95)} if len(x) else None


def cell(f, r, x) -> dict:
    rows = np.isin(f["episode"], x["train"])
    acc = rows & f["accepted"]
    d_demo = f[f"d_demo_r{r:g}"]
    on = f["manifold_error"] <= MANIFOLD_TOL
    supported = (acc & on & (f["construction_residual"] <= CONSTRUCTION_TOL) & f["bounded"]
                 & (f["consistency"] <= CONSISTENCY_TOL) & (f["switch_margin"] >= SWITCH_AMBIGUOUS)
                 & (d_demo >= x["tau_demo"]) & (f["region"] > 0) & (f["recovered"] == 1))
    n_acc = int(acc.sum())
    codes = [str(c) for c in f["reason_codes"]]
    near = float(np.isin(f["class"][acc], (0, 1)).mean()) if n_acc else 1.0
    need = max(MIN_SUPPORTED, int(np.ceil(MIN_FRACTION * n_acc)))
    episodes = int(len(np.unique(f["episode"][supported])))
    out = {
        "rows": int(rows.sum()), "accepted": n_acc, "accepted_fraction": n_acc / max(1, int(rows.sum())),
        "rejected_by_reason": {c: int((rows & f["reasons"][:, i]).sum()) for i, c in enumerate(codes)},
        "solver_class_fraction_of_accepted": {c: float((f["class"][acc] == i).mean()) if n_acc else None
                                              for i, c in enumerate(CLASSES)},
        "off_manifold_fraction_of_accepted": float((~on[acc]).mean()) if n_acc else None,
        "switch_ambiguous_accepted": int((acc & (f["switch_margin"] < SWITCH_AMBIGUOUS)).sum()),
        "construction_residual": q(f["construction_residual"][acc]), "d_source": q(f["d_source"][acc]),
        "d_demo": q(d_demo[acc]), "consistency": q(f["consistency"][acc]),
        "by_region": {g: {"accepted": int((acc & (f["region"] == i)).sum()),
                          "d_source": q(f["d_source"][acc & (f["region"] == i)]),
                          "d_demo": q(d_demo[acc & (f["region"] == i)]),
                          "consistency": q(f["consistency"][acc & (f["region"] == i)]),
                          "supported": int((supported & (f["region"] == i)).sum())} for i, g in enumerate(REGIONS)},
        "supported": int(supported.sum()), "supported_episodes": episodes, "supported_needed": need,
        "near_identity_fraction": near,
    }
    out["pass"] = bool(near <= NEAR_IDENTITY_MAX and out["supported"] >= need and episodes >= MIN_EPISODES)
    return out


def summarize(rows: dict, u: dict) -> dict:
    """The cell table and the treatment cells from the row files (rows[(solver, sigma)])."""
    table, treatment = {}, {}
    for r, x in u.items():
        table[f"{r:g}"] = {}
        for sigma in SIGMAS:
            ca = cell(rows[("a", sigma)], r, x)
            entry = {"a": ca}
            if ("b", sigma) in rows:
                entry["b"] = cell(rows[("b", sigma)], r, x)
            else:
                entry["b"] = {"run": False, "why": f"solver a's off-manifold fraction "
                                                   f"{ca['off_manifold_fraction_of_accepted']} <= {OFF_MANIFOLD_TRIGGER}"}
            table[f"{r:g}"][f"{sigma:g}"] = entry
        passing = [(s, sol) for s in SIGMAS for sol in ("a", "b")
                   if table[f"{r:g}"][f"{s:g}"][sol].get("pass")]
        treatment[f"{r:g}"] = ({"sigma": passing[0][0], "solver": passing[0][1]} if passing
                               else {"sigma": None, "solver": None, "outcome": "no treatment"})
    return {"cells": table, "treatment": treatment,
            "tau_demo": {f"{r:g}": x["tau_demo"] for r, x in u.items()},
            "splits": {f"{r:g}": {"train": x["train"], "validation": x["validation"]} for r, x in u.items()}}


def needs_b(fa, u) -> bool:
    for r, x in u.items():
        acc = np.isin(fa["episode"], x["train"]) & fa["accepted"]
        if acc.any() and float((fa["manifold_error"][acc] > MANIFOLD_TOL).mean()) > OFF_MANIFOLD_TRIGGER:
            return True
    return False


def provenance() -> dict:
    return {"predeclaration": "docs/experiments/pendulum-correction.md (version 1)",
            "environment": {"code_commit": git("rev-parse", "HEAD"),
                            "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                            "numpy": np.__version__, "python": sys.version.split()[0]},
            "data_sha256": DATA_SHA256, "exact_model_sha256": MODEL.sha256, "seed": SEED,
            "generator": {"name": "noisy_action_preimage", "K": K, "max_iterations": MAX_ITERATIONS,
                          "tolerance": TOLERANCE, "action_bounds": [-3.0, 3.0],
                          "validators": ["Finite", "GenerationFailed", f"SupportRadius({SUPPORT_RADIUS})",
                                         "ActionBounds(+-3)"]},
            "sigma_grid": list(SIGMAS),
            "task_optimality": "unsupported: the DP oracle did not converge (docs/experiments/pendulum-oracle.md)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", type=Path, help="a committed correction.json to compare a regeneration from DIR with")
    a = ap.parse_args()
    if sha(a.data) != DATA_SHA256:
        raise SystemExit("refused: the dataset is not the canonical continuous dataset")
    ds = read_trajectories(a.data)
    u = units(ds)
    if a.check:
        rows = {(p.name.split("-")[1], float(np.load(p)["sigma"])): dict(np.load(p))
                for p in sorted(a.out.glob("rows-*-sigma*.npz"))}
        doc = json.loads(a.check.read_text())
        same = summarize(rows, u) == {k: doc[k] for k in ("cells", "treatment", "tau_demo", "splits")}
        print("identical" if same else "differs")
        return 0 if same else 1
    if a.out.exists() and not (a.resume and not (a.out / "correction.json").exists()):
        raise SystemExit(f"refused: {a.out} exists" + ("" if a.resume else " (an incomplete scan takes --resume)"))
    a.out.mkdir(parents=True, exist_ok=a.resume)
    t0 = time.time()
    key = json.dumps({"commit": git("rev-parse", "HEAD"), "seed": SEED, "K": K})
    log("tau_demo", {f"{r:g}": round(x["tau_demo"], 6) for r, x in u.items()},
        "training episodes", {f"{r:g}": len(x["train"]) for r, x in u.items()})
    rows = {}
    for sigma in SIGMAS:
        rows[("a", sigma)] = scan_sigma(sigma, ds, u, a.out, a.workers, key)
        if needs_b(rows[("a", sigma)], u):
            rows[("b", sigma)] = scan_sigma_b(sigma, rows[("a", sigma)], u, a.out, a.workers, key)
        log(f"sigma {sigma:g} done; solver b {'run' if ('b', sigma) in rows else 'not needed'}")
    doc = {**provenance(), **summarize(rows, u),
           "row_files": {p.name: sha(p) for p in sorted(a.out.glob("rows-*-sigma*.npz"))},
           "seconds": round(time.time() - t0, 1)}
    (a.out / "correction.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    log("treatment", json.dumps(doc["treatment"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
