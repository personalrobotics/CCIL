"""The canonical Pendulum swing-up environment and evaluator of the #24 study.

Predeclaration (docs/experiments/pendulum-ccil-study-predeclaration.md) §3
and §7. The physics is the data generator's (correct_il/envs/pendulum_env.py,
PendulumEnv_NP in "rk4" mode):

- state (sin θ, cos θ, θ̇), integrated as three independent coordinates under
  θ̈ = −(g/l) sin θ + a with l = 1, g = 9.81;
- one step is scipy.integrate.solve_ivp RK45 over (0, dt), default
  tolerances, dense output evaluated at dt = 0.02;
- the discontinuous variant negates θ̇ after a step that lands within 1e-2
  of π/6 or π/3 (θ = atan2(sin θ, cos θ));
- the reward is computed on the pre-step state and the applied action:
  −½‖(θ, θ̇) − (π, 0)‖² − ½a², θ wrapped to [0, 2π);
- actions are not clipped; episodes are 500 steps with no termination.

rollout() evaluates a policy on one seed's 100 episodes, all of them in
lockstep so the policy sees a batch, while each episode is integrated on its
own (a joint solve_ivp would change the adaptive step control). It returns
every per-step array the audit (audit.py) needs, and save() writes them
without pickle, atomically.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import scipy
import scipy.integrate

VERSION = "m3_pendulum.env/1"
G, LENGTH, DT, HORIZON, EPISODES = 9.81, 1.0, 0.02, 500, 100
OBS_LOW, OBS_HIGH = np.array([-1.0, -1.0, -4 * np.pi]), np.array([1.0, 1.0, 4 * np.pi])
ACT_LOW, ACT_HIGH = np.array([-3.0]), np.array([3.0])
NOISE_LEVELS = (0.01, 0.05, 0.1)  # §7; level NOISE_LEVELS[k] draws from SeedSequence([seed, 8, k])
TASKS = {"continuous": False, "discontinuous": True}


def _field(s, a):
    return np.array([s[2] * s[1], s[2] * -s[0], -(G / LENGTH) * s[0] + a])


def step(state, action: float, discontinuous: bool):
    """One environment step from state (3,) under a scalar action; returns (next_state, flipped)."""
    res = scipy.integrate.solve_ivp(lambda _, s: _field(s, action), (0, DT), state, method="RK45", dense_output=True)
    nxt = res.sol(DT)
    flipped = False
    if discontinuous:
        theta = np.arctan2(nxt[0], nxt[1])
        if np.abs(theta - np.pi / 6) < 1e-2 or np.abs(theta - np.pi / 3) < 1e-2:
            nxt[2] = -nxt[2]
            flipped = True
    return nxt, flipped


def reward(state, action: float) -> float:
    theta = np.arctan2(state[0], state[1])
    if theta < 0:
        theta += 2 * np.pi
    return float(-0.5 * np.linalg.norm(np.array([theta, state[2]]) - np.array([np.pi, 0.0])) ** 2 - 0.5 * action ** 2)


def initial_states(seed: int):
    """The seed's 100 initial states (§3): θ ~ U(−1, 1), θ̇ ~ U(−0.5, 0.5)."""
    draw = np.random.default_rng(np.random.SeedSequence([seed, 7])).uniform([-1.0, -0.5], [1.0, 0.5], size=(EPISODES, 2))
    return np.stack([np.sin(draw[:, 0]), np.cos(draw[:, 0]), draw[:, 1]], axis=1)


def noise_draws(seed: int, k: int):
    """The uniform draws for noise level NOISE_LEVELS[k] (§7), shared by every arm: observation and action."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8, k]))
    return (rng.uniform(OBS_LOW, OBS_HIGH, size=(EPISODES, HORIZON, 3)),
            rng.uniform(ACT_LOW, ACT_HIGH, size=(EPISODES, HORIZON, 1)))


def rollout(policy, task: str, seed: int, noise_level: float = 0.0) -> dict:
    """Run policy (an (E, 3) float64 batch to (E, 1) actions) on the seed's 100 episodes.

    A non-finite policy action ends that episode: its later steps are NaN and
    its return is non-finite, which §9 counts as an algorithmic failure.
    """
    discontinuous = TASKS[task]
    if noise_level:
        k = NOISE_LEVELS.index(noise_level)
        obs_noise, act_noise = noise_draws(seed, k)
    else:
        k, obs_noise, act_noise = None, None, None
    states = np.full((EPISODES, HORIZON + 1, 3), np.nan)
    states[:, 0] = initial_states(seed)
    observations = np.full((EPISODES, HORIZON, 3), np.nan)
    policy_actions = np.full((EPISODES, HORIZON, 1), np.nan)
    actions = np.full((EPISODES, HORIZON, 1), np.nan)
    rewards = np.full((EPISODES, HORIZON), np.nan)
    flipped = np.zeros((EPISODES, HORIZON), dtype=bool)
    alive = np.ones(EPISODES, dtype=bool)
    for t in range(HORIZON):
        obs = states[:, t]
        if noise_level:
            obs = obs * (1 - noise_level) + noise_level * obs_noise[:, t]
        observations[alive, t] = obs[alive]
        pa = np.full((EPISODES, 1), np.nan)
        pa[alive] = np.asarray(policy(obs[alive]), dtype=np.float64).reshape(-1, 1)
        policy_actions[alive, t] = pa[alive]
        a = pa * (1 - noise_level) + noise_level * act_noise[:, t] if noise_level else pa
        alive &= np.isfinite(a[:, 0])
        for e in np.flatnonzero(alive):
            actions[e, t] = a[e]
            rewards[e, t] = reward(states[e, t], float(a[e, 0]))
            states[e, t + 1], flipped[e, t] = step(states[e, t], float(a[e, 0]), discontinuous)
    returns = rewards.sum(axis=1)
    return {"states": states, "observations": observations, "policy_actions": policy_actions, "actions": actions,
            "rewards": rewards, "flipped": flipped, "initial_states": states[:, 0].copy(), "returns": returns,
            "R": np.float64(returns.mean()), "failed_episodes": np.flatnonzero(~alive),
            **({"obs_noise": obs_noise, "act_noise": act_noise} if noise_level else {}),
            "meta": {"version": VERSION, "task": task, "seed": seed, "noise_level": noise_level, "noise_index": k,
                     "episodes": EPISODES, "horizon": HORIZON, "scipy": scipy.__version__, "numpy": np.__version__}}


def save(result: dict, path) -> None:
    """Write a rollout to path (.npz, no pickle) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    arrays = {k: v for k, v in result.items() if k != "meta"}
    np.savez(tmp, **arrays, meta=np.array(json.dumps(result["meta"], sort_keys=True)))
    os.replace(tmp, path)
