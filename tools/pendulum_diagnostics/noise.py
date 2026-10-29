#!/usr/bin/env python3
"""Diagnostic 3 of docs/experiments/pendulum-v6-diagnostics.md: the evaluation-noise mechanism.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python tools/pendulum_diagnostics/noise.py --root RUN_ROOT --out OUT_DIR

Part A reads the stored rollouts of bc and ccil, both tasks, all study seeds, noise-free and at each level. Part B
re-evaluates the stored checkpoints with the stored noise draws applied to the observation only, to the action only,
and to both. "Both" must reproduce the stored rollout bit for bit, which validates the loop. Part B is diagnostic
only and never enters the v6 analysis. Nothing is trained, and nothing under RUN_ROOT is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "m3_pendulum", REPO / "tools" / "vector_env", REPO / "tools" / "pendulum_diagnostics"):
    sys.path.insert(0, str(p))
import env as canonical  # noqa: E402
import pendulum as vector  # noqa: E402
from behavior import CATEGORIES, episode_metrics, summarize  # noqa: E402

TASKS = ("continuous", "discontinuous")
ARMS = ("bc", "ccil")
SEEDS = tuple(range(1201, 1221))
LEVELS = (0.0, *canonical.NOISE_LEVELS)
MODES = ("both", "observation_only", "action_only")


def unit_dir(root: Path, kind: str, task: str, arm: str) -> Path:
    base = root / kind / "study" / task
    return base / arm if (base / arm).is_dir() else next(base.glob(f"{arm}@*"))


def t_interval(d: np.ndarray) -> list[float]:
    from scipy import stats
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def action_stats(z: dict) -> dict:
    pa, a = z["policy_actions"][..., 0], z["actions"][..., 0]
    o_minus_s = z["observations"] - z["states"][:, :-1]
    ratio = np.abs(pa) / 3.0
    return {"frac_abs_policy_action_over_3": float((np.abs(pa) > 3).mean()),
            "frac_abs_applied_action_over_3": float((np.abs(a) > 3).mean()),
            "frac_abs_policy_action_over_2.9": float((np.abs(pa) > 2.9).mean()),
            "abs_policy_action_over_3_quantiles": {q: float(np.quantile(ratio, q)) for q in (0.5, 0.9, 0.99)},
            "mean_abs_policy_action": float(np.abs(pa).mean()),
            "mean_abs_action_perturbation": float(np.abs(a - pa).mean()),
            "mean_abs_observation_perturbation": [float(x) for x in np.abs(o_minus_s).mean(axis=(0, 1))]}


def part_a(root: Path) -> dict:
    out, per_seed_R = {}, {}
    for task in TASKS:
        for arm in ARMS:
            d = unit_dir(root, "rollouts", task, arm)
            for n in LEVELS:
                zs = [dict(np.load(d / f"s{s}-noise{n}.npz")) for s in SEEDS]
                ms = [episode_metrics(z) for z in zs]
                st = [action_stats(z) for z in zs]
                agg = {k: float(np.mean([x[k] for x in st])) for k in st[0] if isinstance(st[0][k], float)}
                agg["mean_abs_observation_perturbation"] = list(np.mean([x["mean_abs_observation_perturbation"]
                                                                         for x in st], axis=0))
                out[f"{task}/{arm}/{n}"] = {**summarize(ms), **agg}
                per_seed_R[(task, arm, n)] = np.array([z["R"] for z in zs])
                for s, m in zip(SEEDS, ms):
                    per_seed_R[(task, arm, n, s)] = m
    paired = {}
    for task in TASKS:
        base = {s: (per_seed_R[(task, "ccil", 0.0, s)], per_seed_R[(task, "bc", 0.0, s)]) for s in SEEDS}
        for n in LEVELS:
            d = per_seed_R[(task, "ccil", n)] - per_seed_R[(task, "bc", n)]
            terms = {}
            for k in ("angle_term", "velocity_term", "action_term"):
                dk = np.array([per_seed_R[(task, "ccil", n, s)][k].mean() - per_seed_R[(task, "bc", n, s)][k].mean()
                               for s in SEEDS])
                terms[k] = {"mean": float(dk.mean()), "t_interval_95": t_interval(dk)}
            shift = {}
            for arm in ARMS:
                cats0 = np.concatenate([per_seed_R[(task, arm, 0.0, s)]["category"] for s in SEEDS])
                catsn = np.concatenate([per_seed_R[(task, arm, n, s)]["category"] for s in SEEDS])
                shift[arm] = {c0: {c1: int(((cats0 == c0) & (catsn == c1)).sum()) for c1 in CATEGORIES}
                              for c0 in CATEGORIES}
            paired[f"{task}/{n}"] = {"ccil_minus_bc_R": {"mean": float(d.mean()), "t_interval_95": t_interval(d),
                                                         "per_seed": dict(zip(map(str, SEEDS), map(float, d)))},
                                     "ccil_minus_bc_terms": terms,
                                     "category_transitions_from_noise_free": shift}
        del base
    return {"per_arm": out, "paired": paired}


def rollouts_with(items: list, task: str) -> list[dict]:
    """vector_env.pendulum.rollouts, with the stored noise draws applied to the observation, the action or both.
    The loop is rollouts()'s, step for step; only the two noise lines take a flag."""
    discontinuous = canonical.TASKS[task]
    E, T, K = canonical.EPISODES, canonical.HORIZON, len(items)
    obs_noise, act_noise = np.zeros((K, E, T, 3)), np.zeros((K, E, T, 1))
    level, on_obs, on_act = np.zeros((K, 1, 1)), np.zeros(K, bool), np.zeros(K, bool)
    states = np.full((K, E, T + 1, 3), np.nan)
    for i, (_, seed, n, mode) in enumerate(items):
        obs_noise[i], act_noise[i] = canonical.noise_draws(seed, canonical.NOISE_LEVELS.index(n))
        level[i] = n
        on_obs[i], on_act[i] = mode in ("both", "observation_only"), mode in ("both", "action_only")
        states[i, :, 0] = canonical.initial_states(seed)
    policy_actions, actions = np.full((K, E, T, 1), np.nan), np.full((K, E, T, 1), np.nan)
    rewards = np.full((K, E, T), np.nan)
    alive = np.ones((K, E), dtype=bool)
    for t in range(T):
        obs = states[:, :, t].copy()
        obs[on_obs] = obs[on_obs] * (1 - level[on_obs]) + level[on_obs] * obs_noise[on_obs, :, t]
        pa = np.full((K, E, 1), np.nan)
        for i, (policy, _, _, _) in enumerate(items):
            if alive[i].any():
                pa[i, alive[i]] = np.asarray(policy(obs[i, alive[i]]), dtype=np.float64).reshape(-1, 1)
        policy_actions[alive, t] = pa[alive]
        act = pa.copy()
        act[on_act] = pa[on_act] * (1 - level[on_act]) + level[on_act] * act_noise[on_act, :, t]
        alive &= np.isfinite(act[:, :, 0])
        if alive.any():
            actions[alive, t] = act[alive]
            rewards[alive, t] = vector.reward(states[alive, t], act[alive][:, 0])
            states[alive, t + 1], _ = vector.step(states[alive, t], act[alive][:, 0], discontinuous)
    return [{"states": states[i], "policy_actions": policy_actions[i], "actions": actions[i], "rewards": rewards[i],
             "returns": rewards[i].sum(axis=1), "R": float(rewards[i].sum(axis=1).mean())} for i in range(K)]


def part_b(root: Path, device: str) -> dict:
    from ccil.training.vector_bc import VectorBCPolicy
    out, check = {}, {"compared": 0, "bit_identical": 0, "max_abs_return_diff": 0.0}
    for task in TASKS:
        nets = {}
        for arm in ARMS:
            for s in SEEDS:
                rec = json.loads((unit_dir(root, "results", task, arm) / f"s{s}.json").read_text())
                nets[(arm, s)] = VectorBCPolicy.load(Path(rec["stages"]["policy.train"]) / "checkpoints" / "policy.npz",
                                                     device=device)[0]
        keys = [(arm, s, n, m) for arm in ARMS for s in SEEDS for n in canonical.NOISE_LEVELS for m in MODES]
        res = rollouts_with([(nets[(arm, s)].predict, s, n, m) for arm, s, n, m in keys], task)
        R = {}
        for (arm, s, n, m), r in zip(keys, res):
            R[(arm, s, n, m)] = r["R"]
            if m == "both":
                z = np.load(unit_dir(root, "rollouts", task, arm) / f"s{s}-noise{n}.npz")
                same = all(np.array_equal(r[k], z[k]) for k in ("states", "policy_actions", "actions", "rewards"))
                check["compared"] += 1
                check["bit_identical"] += int(same)
                check["max_abs_return_diff"] = max(check["max_abs_return_diff"],
                                                   float(np.abs(r["returns"] - z["returns"]).max()))
        for n in canonical.NOISE_LEVELS:
            for m in MODES:
                bc = np.array([R[("bc", s, n, m)] for s in SEEDS])
                cc = np.array([R[("ccil", s, n, m)] for s in SEEDS])
                d = cc - bc
                out[f"{task}/{n}/{m}"] = {"bc_R_mean": float(bc.mean()), "ccil_R_mean": float(cc.mean()),
                                          "ccil_minus_bc": {"mean": float(d.mean()), "t_interval_95": t_interval(d)}}
    return {"by_task_level_mode": out, "both_reproduces_stored": check}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"declared_in": "docs/experiments/pendulum-v6-diagnostics.md §3", "stored": part_a(a.root)}
    report["separated"] = part_b(a.root, a.device)
    (a.out / "noise.json").write_text(json.dumps(report, indent=1) + "\n")
    c = report["separated"]["both_reproduces_stored"]
    print("both reproduces stored:", c)
    for k, v in report["separated"]["by_task_level_mode"].items():
        print(k, round(v["ccil_minus_bc"]["mean"]), [round(x) for x in v["ccil_minus_bc"]["t_interval_95"]])
    return 0 if c["bit_identical"] == c["compared"] else 1


if __name__ == "__main__":
    sys.exit(main())
