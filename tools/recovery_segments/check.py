#!/usr/bin/env python3
"""The recovery-segment mathematics (docs/recovery-segments.md, #171), verified on an analytic system.

    uv run python tools/recovery_segments/check.py --out DIR [--evidence docs/fixtures/evidence/recovery-segments/report.json]

The system is an inverted pendulum, theta'' = g sin(theta) - c theta' + u, recorded every DT seconds and
integrated exactly enough (RK4, N_INT substeps) to serve as the true map T; a stabilizing demonstrator with
exploration noise records the episodes; a linear model fitted on the training episodes is the learned map.
Segments are built here with numpy (the composition, gates, and bounds are this file's own), from the product's
one-step operators and random streams, so M = 1 can be compared with today's generators. It checks:
1. M = 1 is today's noisy_action and noisy_action_preimage, byte for byte.
2. The composed disturbed_action bound, with the system's own constants and per-step residuals, covers the exact
   miss of every probed segment at every M; the model rollout misses the target by at most the propagated step
   residuals.
3. The regime, on each of three data seeds: M = 1 is not admitted (it lacks reach: displacement or novelty), the
   smallest admitted M is intermediate, and a longer M fails the fit gate. Beside each M: pointwise novelty by
   offset, and the exact miss of its segments (in tau_demo), which no gate sees.
4. Refining the integration step changes neither the admitted M nor, beyond a declared tolerance, any segment.
The report holds no paths or times, so a rerun reproduces it byte for byte.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ccil.core import IdentityRepresentation, LinearTransitionModel, NoisyAction, NoisyActionPreimage  # noqa: E402
from ccil.core.diagnose import probe_rows, sampled_tau  # noqa: E402
from ccil.core.generators import newton_preimage, transition_rng  # noqa: E402
from ccil.core.quality import demo_metric, nearest_distance  # noqa: E402
from ccil.core.representation import transitions  # noqa: E402
from ccil.schema import Episode, TrajectoryDataset  # noqa: E402

# The system and data, declared before any result (docs/recovery-segments.md, "The analytic fixture").
G, C, DT, N_INT = 30.0, 0.1, 0.05, 4
GAINS = (2 * G, 2 * np.sqrt(G))
EPISODES, T, THETA0, DEMO_NOISE, DATA_SEED = 24, 60, 1.2, 1.0, 0
REGIME_SEEDS = (0, 1, 2)  # the regime must hold on every one of these data seeds
# Generation and the diagnose thresholds (the product defaults).
K, SIGMA, SEED, LADDER = 4, 1.0, 0, (1, 2, 4, 8, 16)
PROBE, TAU_Q, TAU_SAMPLE, LABEL_Q, FIT_Q = 256, 0.95, 2048, 0.9, 0.95
MIN_DISPLACEMENT, MIN_NOVELTY, FIT_RATIO = 1.0, 1.0, 0.5
NEWTON = (50, 1e-4)
REFINED_N_INT, REFINEMENT_TOL = 16, 1e-4


def vector_field(x, u):
    return np.stack([x[:, 1], G * np.sin(x[:, 0]) - C * x[:, 1] + u[:, 0]], axis=1)


def exact_step(x, u, n_int=N_INT):
    """T(x, u): RK4 over one action interval DT with n_int substeps, u held constant."""
    x, u, h = np.asarray(x, np.float64), np.asarray(u, np.float64), DT / n_int
    for _ in range(n_int):
        k1 = vector_field(x, u)
        k2 = vector_field(x + h / 2 * k1, u)
        k3 = vector_field(x + h / 2 * k2, u)
        k4 = vector_field(x + h * k3, u)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def constants():
    """K_z, K_a of the residual map f = T - I, from the vector field's Lipschitz constants (Gronwall).

    |d/dtheta (g sin theta)| <= g everywhere, so L = ||[[0, 1], [g, c]]||_2 bounds the state Jacobian globally, and
    u enters through B = (0, 1). Over one interval: ||T(x,u) - T(y,v)|| <= e^{L DT} ||x - y|| + (e^{L DT} - 1)/L ||u - v||,
    so f is Lipschitz with K_z = e^{L DT} - 1 and K_a = (e^{L DT} - 1)/L, in every region (global)."""
    L = float(np.linalg.norm(np.array([[0.0, 1.0], [G, C]]), 2))
    return float(np.expm1(L * DT)), float(np.expm1(L * DT) / L), L


def demonstrations(n_int=N_INT, data_seed=DATA_SEED):
    rng = np.random.default_rng(data_seed)
    eps = []
    for e in range(EPISODES):
        x = np.array([[rng.uniform(-THETA0, THETA0), rng.uniform(-1.0, 1.0)]])
        xs, us = [x[0]], []
        for _ in range(T - 1):
            u = np.array([[-(GAINS[0] * x[0, 0] + GAINS[1] * x[0, 1]) + DEMO_NOISE * rng.normal()]])
            x = exact_step(x, u, n_int)
            xs.append(x[0])
            us.append(u[0])
        us.append(np.zeros(1))  # the last frame's action governs no recorded interval
        eps.append(Episode(f"ep{e:02d}", {"state": np.array(xs, np.float32)}, np.array(us, np.float32),
                           timestamps=np.arange(T) * DT))
    return TrajectoryDataset(eps)


class Data:
    """The dataset, its split (the last quarter of the episodes held out), the model, and the metric."""

    def __init__(self, n_int=N_INT, data_seed=DATA_SEED):
        self.ds = demonstrations(n_int, data_seed)
        ids = [e.episode_id for e in self.ds.episodes]
        self.split = {"train": ids[: -EPISODES // 4], "validation": ids[-EPISODES // 4:]}
        self.rep = IdentityRepresentation("state")
        self.batch = transitions(self.ds, self.rep)
        self.ep = {e.episode_id: e for e in self.ds.episodes}
        train = np.isin(self.batch.episode_id.astype(str), self.split["train"])
        self.model = LinearTransitionModel()
        self.model.fit(self.batch.z[train], self.batch.a[train], self.batch.z_next[train])
        self.metric = demo_metric(self.ds, self.split, self.rep)
        self.tau, _ = sampled_tau(self.metric, TAU_Q, TAU_SAMPLE)

    def windows(self, M, episodes):
        """(episode, t) of every window of M intervals inside an episode, in transitions() order."""
        keep = [(e, int(t)) for e, t in zip(self.batch.episode_id.astype(str), self.batch.frame)
                if e in episodes and t + M <= self.ep[e].length - 1]
        return keep

    def probe(self, M):
        w = self.windows(M, set(self.split["train"]))
        rows, _ = probe_rows(np.array([e for e, _ in w]), self.split["train"], PROBE)
        return [w[i] for i in rows]


def segments(d, wins, M, op="backward_euler", k=K, sigma=SIGMA):
    """The persistent-bias backward chain of docs/recovery-segments.md, every window x k draws, in draw order."""
    x = {e: d.ep[e].observations["state"] for e, _ in wins}
    u = {e: d.ep[e].actions for e, _ in wins}
    delta = np.concatenate([transition_rng(SEED, e, t).normal(0.0, sigma, size=(k, 1)) for e, t in wins])
    src = np.repeat(np.arange(len(wins)), k)
    U = np.stack([np.stack([(u[e][t + i] + 0).astype(np.float64) for e, t in wins])[src] for i in range(M)], axis=1)
    U_tilde = (U + delta[:, None, :]).astype(np.float32)  # stored in the action dtype, as one-step draws are
    demo = np.stack([np.stack([x[e][t + i] for e, t in wins])[src] for i in range(M + 1)], axis=1)
    X = [None] * (M + 1)
    X[M] = demo[:, M]
    failed = np.zeros(len(src), bool)
    for i in range(M - 1, -1, -1):
        if op == "backward_euler":
            X[i] = X[i + 1] - d.model.delta(X[i + 1], U_tilde[:, i])
        else:
            X[i], bad, _ = newton_preimage(d.model, X[i + 1], U_tilde[:, i], demo[:, i], *NEWTON, out_dtype=np.float32)
            failed |= bad
    X = np.stack([np.asarray(v, np.float64) for v in X], axis=1)
    s = np.stack([np.linalg.norm(X[:, i] + d.model.delta(X[:, i], U_tilde[:, i]) - X[:, i + 1], axis=1)
                  for i in range(M)], axis=1)
    return {"X": X, "U": U_tilde, "demo": demo.astype(np.float64), "demo_U": U, "s": s, "failed": failed,
            "source": src, "wins": wins}


def gates(d, M):
    seg = segments(d, d.probe(M), M)
    ok = ~seg["failed"] & np.isfinite(seg["X"]).all(axis=(1, 2))
    X, demo, scale = seg["X"][ok], seg["demo"][ok], d.metric.scale
    d_source = np.linalg.norm((X[:, 0] - demo[:, 0]) / scale, axis=1)
    d_demo = nearest_distance(X[:, 0], d.metric.train_states, scale)
    held = d.windows(M, set(d.split["validation"]))
    xs = {e: d.ep[e].observations["state"].astype(np.float64) for e, _ in held}
    z = np.stack([xs[e][t] for e, t in held])
    for i in range(M):
        z = z + d.model.delta(z, np.stack([d.ep[e].actions[t + i] for e, t in held]))
    eps = float(np.quantile(np.linalg.norm((z - np.stack([xs[e][t + M] for e, t in held])) / scale, axis=1), FIT_Q))
    disp, novel, median = (float(np.quantile(d_source, q)) for q in (LABEL_Q, LABEL_Q, 0.5))
    novel = float(np.quantile(d_demo, LABEL_Q))
    flat_novel = [float((nearest_distance(X[:, i], d.metric.train_states, scale) >= d.tau).mean()) for i in range(M)]
    g = {"M": M, "horizon_s": M * DT, "yield": float(ok.mean()),
         "displacement": disp / d.tau, "novelty": novel / d.tau, "fit": eps / median, "epsilon_M": eps,
         "pointwise_novelty": {"overall": float(np.mean(flat_novel)), "by_offset": flat_novel}}
    g["pass"] = {"yield": g["yield"] >= 0.01, "displacement": g["displacement"] >= MIN_DISPLACEMENT,
                 "novelty": g["novelty"] >= MIN_NOVELTY, "fit": g["fit"] <= FIT_RATIO,
                 "trust": True}  # declared constants: the composed bound is conditional off-data evidence
    g["admissible"] = all(g["pass"].values())
    y = seg["X"][ok][:, 0]
    for i in range(M):
        y = exact_step(y, seg["U"][ok][:, i].astype(np.float64))
    miss = np.linalg.norm((y - demo[:, M]) / scale, axis=1) / d.tau
    g["exact_miss_over_tau"] = {"median": float(np.median(miss)), "q90": float(np.quantile(miss, 0.9)),
                                "note": "the true end state's distance from x_(t+M): seen by no gate"}
    return g


def rounded(x, n=6):
    if isinstance(x, float):
        return float(f"{x:.{n}g}")
    if isinstance(x, dict):
        return {k: rounded(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, n) for v in x]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--evidence", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **rounded(detail)})
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)

    d = Data()

    # 1. M = 1 is today's generators, byte for byte.
    wins = d.probe(1)
    index = {(e, int(t)): i for i, (e, t) in enumerate(zip(d.batch.episode_id.astype(str), d.batch.frame))}
    sub = d.batch.__class__(*(np.asarray(getattr(d.batch, f))[[index[w] for w in wins]]
                              for f in ("z", "a", "z_next", "episode_id", "frame")))
    same = {}
    for op, gen in (("backward_euler", NoisyAction(K, SIGMA)),
                    ("newton", NoisyActionPreimage(K, SIGMA, max_iterations=NEWTON[0], tolerance=NEWTON[1]))):
        today, seg = gen.generate(sub, d.model, SEED), segments(d, wins, 1, op)
        same[op] = {"states": np.asarray(today.z).tobytes() == np.asarray(seg["X"][:, 0], np.asarray(today.z).dtype).tobytes()
                    and np.asarray(today.z).dtype == (np.float64 if op == "backward_euler" else np.float32),
                    "actions": np.asarray(today.a).tobytes() == seg["U"][:, 0].tobytes(),
                    "targets": np.asarray(today.target).tobytes() == np.asarray(seg["X"][:, 1], np.float32).tobytes(),
                    "failed": bool((np.asarray(today.failed) == seg["failed"]).all()), "rows": int(len(today.z))}
    record("m1_is_todays_generators", all(all(v for k, v in s.items() if k != "rows") for s in same.values()), **same)

    # 2. The composed bound covers the exact miss; the model rollout misses by at most the propagated residuals.
    K_z, K_a, L = constants()
    L_hat = float(np.linalg.norm(np.eye(2) + d.model.A, 2))
    cover = {}
    for M in LADDER:
        seg = segments(d, d.probe(M), M)
        X, U, demo, dU = seg["X"], seg["U"].astype(np.float64), seg["demo"], seg["demo_U"]
        y = X[:, 0]
        rho = np.stack([np.linalg.norm(exact_step(demo[:, i], dU[:, i]) - demo[:, i + 1], axis=1) for i in range(M)], 1)
        for i in range(M):
            y = exact_step(y, U[:, i])
        miss = np.linalg.norm(y - demo[:, M], axis=1)
        da = np.linalg.norm(U - dU, axis=2)
        w = (1 + K_z) ** np.arange(M - 1, -1, -1)
        bound = (1 + K_z) ** M * np.linalg.norm(X[:, 0] - demo[:, 0], axis=1) + ((K_a * da + rho) * w).sum(axis=1)
        r = X[:, 0]
        for i in range(M):
            r = r + d.model.delta(r, U[:, i].astype(np.float32))
        model_miss = np.linalg.norm(r - X[:, M], axis=1)
        model_bound = (seg["s"] * L_hat ** np.arange(M - 1, -1, -1)).sum(axis=1)
        cover[f"M{M}"] = {"segments": int(len(miss)), "covered": bool((miss <= bound).all()),
                          "max_miss_over_bound": float((miss / bound).max()),
                          "median_bound": float(np.median(bound)), "median_miss": float(np.median(miss)),
                          "model_covered": bool((model_miss <= model_bound * (1 + 1e-9) + 1e-12).all()),
                          "median_step_residual": float(np.median(seg["s"])), "max_step_residual": float(seg["s"].max())}
    record("composed_bound_covers_the_exact_miss", all(v["covered"] and v["model_covered"] for v in cover.values()),
           constants={"K_z": K_z, "K_a": K_a, "vector_field_lipschitz": L, "model_lipschitz": L_hat}, by_M=cover)

    # 3. The regime on every data seed: M = 1 not admitted, an intermediate M admitted, a longer M fails fit.
    regime, admitted = {}, None
    for seed in REGIME_SEEDS:
        ds = d if seed == DATA_SEED else Data(data_seed=seed)
        frontier = [gates(ds, M) for M in LADDER]
        adm = next((g["M"] for g in frontier if g["admissible"]), None)
        regime[f"seed{seed}"] = {"tau_demo": ds.tau, "admitted": adm, "frontier": frontier,
                                 "m1_lacks_reach": not (frontier[0]["pass"]["displacement"] and frontier[0]["pass"]["novelty"]),
                                 "longer_fails_fit": adm is not None and any(not g["pass"]["fit"] for g in frontier
                                                                             if g["M"] > adm)}
        admitted = adm if seed == DATA_SEED else admitted
    record("intermediate_m_admitted_longer_m_fails_fit",
           all(r["m1_lacks_reach"] and r["admitted"] not in (None, 1, LADDER[-1]) and r["longer_fails_fit"]
               for r in regime.values()), by_seed=regime)

    # 4. Integration refinement: same admitted M; segments within the declared tolerance.
    fine = Data(REFINED_N_INT)
    fine_frontier = [gates(fine, M) for M in LADDER]
    fine_admitted = next((g["M"] for g in fine_frontier if g["admissible"]), None)
    diffs = {}
    for M in LADDER:
        p, q = segments(d, d.probe(M), M), segments(fine, fine.probe(M), M)
        diffs[f"M{M}"] = {"states": float(np.abs(p["X"] - q["X"]).max()),
                          "actions": float(np.abs(p["U"].astype(np.float64) - q["U"]).max())}
    record("integration_refinement_preserves_segments", fine_admitted == admitted
           and all(v["states"] <= REFINEMENT_TOL and v["actions"] <= REFINEMENT_TOL for v in diffs.values()),
           n_int=[N_INT, REFINED_N_INT], tolerance=REFINEMENT_TOL, admitted=[admitted, fine_admitted], max_abs_diff=diffs)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "fixture": {"g": G, "c": C, "dt": DT, "n_int": N_INT, "gains": list(GAINS), "episodes": EPISODES, "T": T,
                          "theta0": THETA0, "demo_noise": DEMO_NOISE, "data_seed": DATA_SEED,
                          "regime_seeds": list(REGIME_SEEDS), "k": K, "sigma": SIGMA,
                          "seed": SEED, "ladder": list(LADDER)},
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "numpy": np.__version__}}
    text = json.dumps(rounded(report), indent=1, sort_keys=True) + "\n"
    (a.out / "report.json").write_text(text)
    if a.evidence:
        a.evidence.parent.mkdir(parents=True, exist_ok=True)
        a.evidence.write_text(text)
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {a.out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
