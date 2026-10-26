"""The independent audit of a saved evaluation (predeclaration §3).

audit(path) recomputes, from the saved states and actions alone and without
env.py's reward code, every quantity a reported return depends on:
- the 500-step horizon of every episode, with no early termination;
- every per-step reward, vectorized from the predeclared formula (max |Δ| ≤ 1e-9);
- each episode's return as the exactly rounded sum of its rewards, and the
  seed's R as the mean of its 100 returns (to 1e-9);
- the initial states and noise draws, regenerated from the seed;
- the applied inputs: observations = (1 − n)·state + n·draw, and
  actions = (1 − n)·policy action + n·draw.
It also replays every saved transition through env.step and requires the
saved next state bit for bit, so a saved trajectory is the one the
environment produced from the saved actions.

An episode that the evaluator ended on a non-finite action is a recorded
failure, not a mismatch: every step before the failure is audited as above,
every value after it must be NaN, and the result says which episodes
failed. Any mismatch makes ok false. R_recomputed is the audit's own mean
of the exactly rounded episode sums (NaN when an episode failed); the
analysis uses it, never a stored R.
"""
from __future__ import annotations

import json
import math

import numpy as np

import env as pendulum

TOL = 1e-9


def _rewards(states, actions):
    sin, cos, omega = states[..., 0], states[..., 1], states[..., 2]
    theta = np.mod(np.arctan2(sin, cos), 2 * np.pi)
    return -0.5 * ((theta - np.pi) ** 2 + omega ** 2) - 0.5 * actions[..., 0] ** 2


def audit(path, replay: bool = True) -> dict:
    with np.load(path, allow_pickle=False) as z:
        r = {k: z[k] for k in z.files}
    meta = json.loads(str(r.pop("meta")))
    E, T, n, seed = meta["episodes"], meta["horizon"], meta["noise_level"], meta["seed"]
    issues = []

    def require(ok, what):
        if not ok:
            issues.append(what)

    require(T == 500 and E == 100, f"episodes {E}, horizon {T}; predeclared 100 and 500")
    require(r["states"].shape == (E, T + 1, 3) and r["actions"].shape == (E, T, 1) and r["rewards"].shape == (E, T),
            "array shapes do not match the horizon")
    if issues:
        return {"ok": False, "issues": issues, "task": meta["task"], "seed": seed, "noise_level": n}
    # Each episode's finite prefix: its first L_e steps, L_e = T unless the episode failed at step L_e.
    finite = np.isfinite(r["actions"][:, :, 0])
    L = np.where(finite.all(axis=1), T, np.argmin(finite, axis=1))
    failed = np.flatnonzero(L < T)
    require(np.array_equal(failed, r["failed_episodes"]), "failed episodes disagree with the saved actions")
    step = np.arange(T)[None, :] < L[:, None]  # (E, T): steps inside the prefix
    seen = np.arange(T)[None, :] <= L[:, None]  # observations: the failing step's observation was seen too
    require(np.isnan(r["actions"][~step]).all() and np.isnan(r["rewards"][~step]).all()
            and np.isnan(r["states"][:, 1:][~step]).all(), "a failed episode has values after its failure")
    require(np.isfinite(r["states"][:, 1:][step]).all() and np.isfinite(r["rewards"][step]).all(),
            "an episode prefix has non-finite values")

    # Initial states and noise draws, regenerated from the seed.
    draw = np.random.default_rng(np.random.SeedSequence([seed, 7])).uniform([-1.0, -0.5], [1.0, 0.5], size=(E, 2))
    init = np.column_stack([np.sin(draw[:, 0]), np.cos(draw[:, 0]), draw[:, 1]])
    require(np.array_equal(r["states"][:, 0], init) and np.array_equal(r["initial_states"], init),
            "initial states differ from the seed's draws")
    obs_in, act_in = r["states"][:, :-1], r["policy_actions"]
    if n:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8, meta["noise_index"]]))
        on = rng.uniform([-1.0, -1.0, -4 * np.pi], [1.0, 1.0, 4 * np.pi], size=(E, T, 3))
        an = rng.uniform([-3.0], [3.0], size=(E, T, 1))
        require(np.array_equal(r["obs_noise"], on) and np.array_equal(r["act_noise"], an),
                "noise draws differ from the seed's draws")
        require(n == (0.01, 0.05, 0.1)[meta["noise_index"]], "noise index does not name the noise level")
        obs_in, act_in = obs_in * (1 - n) + n * on, act_in * (1 - n) + n * an
    require(np.array_equal(r["observations"][seen], obs_in[seen]), "observations are not the predeclared noisy states")
    require(np.array_equal(r["actions"][step], act_in[step]), "applied actions are not the predeclared noisy policy actions")

    # Rewards, returns, R.
    errs = np.abs(_rewards(r["states"][:, :-1], r["actions"]) - r["rewards"])[step]
    reward_err = float(errs.max()) if errs.size else 0.0
    require(reward_err <= TOL, f"rewards differ by {reward_err}")
    ok_eps = np.setdiff1d(np.arange(E), failed)
    sums = np.array([math.fsum(row) for row in r["rewards"][ok_eps]])
    return_err = float(np.abs(sums - r["returns"][ok_eps]).max()) if len(ok_eps) else 0.0
    require(return_err <= TOL, f"returns differ from the sums of their rewards by {return_err}")
    require(np.isnan(r["returns"][failed]).all(), "a failed episode has a finite return")
    require(len(failed) > 0 or abs(math.fsum(sums) / E - float(r["R"])) <= TOL, "R is not the mean of the returns")
    require(len(failed) == 0 or not np.isfinite(r["R"]), "R is finite although an episode failed")

    # Replay: the saved next state is env.step of the saved state and action, over every prefix.
    replay_mismatches, flips = 0, 0
    if replay:
        disc = pendulum.TASKS[meta["task"]]
        for e in range(E):
            for t in range(L[e]):
                nxt, f = pendulum.step(r["states"][e, t].copy(), float(r["actions"][e, t, 0]), disc)
                replay_mismatches += int(not np.array_equal(nxt, r["states"][e, t + 1]) or f != r["flipped"][e, t])
                flips += int(f)
        require(replay_mismatches == 0, f"{replay_mismatches} transitions do not replay")
    recomputed = math.fsum(sums) / E if len(failed) == 0 else float("nan")
    return {"ok": not issues, "issues": issues, "task": meta["task"], "seed": seed, "noise_level": n,
            "failed_episodes": failed.tolist(), "R": float(r["R"]), "R_recomputed": recomputed,
            "max_reward_error": reward_err,
            "max_return_error": return_err, "replayed": bool(replay), "discontinuity_flips": flips}
