"""Versioned Pendulum evaluation profiles (#143): one definition of every episode score, for any trajectory source.

A profile scores trajectories; it never rewrites them. A rollout or dataset keeps the arrays it was written with
(states, commanded and applied actions, and for rollouts the per-step legacy rewards), and each score is reported
under the profile that produced it. No existing artifact is reinterpreted under a different profile.

- legacy_ccil_v1: the #24 study's return, sum over steps of
      -0.5 * ((wrap(theta) - pi)^2 + theta_dot^2) - 0.5 * a^2,   wrap to [0, 2 pi),
  on the pre-step state and the applied action (tools/m3_pendulum/env.py, tools/vector_env/pendulum.py). Kept as a
  named historical metric; it is not the task definition, because zero action is competitive under it (#143).
- pendulum_swingup_v2: task-aligned swing-up and balance metrics. For e = theta - pi and v = cos(e):
      upright occupancy   mean over steps of 1[v >= cos(8 deg)]     (primary; dm_control's swing-up criterion)
      normalized height   mean over steps of (1 + v) / 2            (0 hanging down, 1 upright)
      final hold          upright on every one of the last 100 steps (2 s)
  plus behavior diagnostics (first upright time, upright time, longest upright run, falls after first upright,
  RMS angular velocity, normalized effort mean((a / 3)^2)) and the Gymnasium Pendulum-v1 return as a second named
  historical metric. Episode scores are means over steps, so their scale does not depend on the horizon.

Coordinates: states are (sin theta, cos theta, theta_dot) with theta = 0 hanging down and theta = pi upright. Scores
use the T pre-action states of an episode (a rollout's states[:, :T]; a dataset episode's T stored frames), the same
states the legacy reward is computed on.

Actions: the environment applies actions unclipped (v6 semantics). The declared action range [-3, 3] is checked,
not enforced: commanded and applied out-of-range fractions are reported.
"""
from __future__ import annotations

import numpy as np

PROFILES = ("legacy_ccil_v1", "pendulum_swingup_v2")
VERSION = "pendulum_eval.profiles/1"
DT, ACTION_BOUND = 0.02, 3.0
UPRIGHT_DEG, HOLD_STEPS = 8.0, 100
UPRIGHT_COS = float(np.cos(np.deg2rad(UPRIGHT_DEG)))
GYM_MAX_TORQUE = 2.0  # Gymnasium Pendulum-v1 clips the torque before its cost


def vertical(states: np.ndarray) -> np.ndarray:
    """v = cos(theta - pi): -1 hanging down, 1 upright."""
    theta = np.arctan2(states[..., 0], states[..., 1])
    return np.cos(theta - np.pi)


def upright(states: np.ndarray) -> np.ndarray:
    return vertical(states) >= UPRIGHT_COS


def _longest(up: np.ndarray) -> np.ndarray:
    """Per episode, the longest run of consecutive upright steps."""
    longest = np.zeros(len(up), dtype=np.int64)
    for e in range(len(up)):
        run = 0
        for u in up[e]:
            run = run + 1 if u else 0
            longest[e] = max(longest[e], run)
    return longest


def _falls(up: np.ndarray, below: np.ndarray) -> np.ndarray:
    """Per episode, falls after first reaching upright: the pole dropping below horizontal (v < 0) after having been
    upright, counted once until it is upright again."""
    E, T = up.shape
    falls = np.zeros(E, dtype=np.int64)
    for e in range(E):
        armed = False
        for t in range(T):
            if up[e, t]:
                armed = True
            elif armed and below[e, t]:
                falls[e] += 1
                armed = False
    return falls


def legacy_ccil_v1(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Per-episode legacy return from (E, T, 3) pre-action states and (E, T, 1) applied actions."""
    theta = np.arctan2(states[..., 0], states[..., 1])
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    r = -0.5 * ((theta - np.pi) ** 2 + states[..., 2] ** 2) - 0.5 * actions[..., 0] ** 2
    return r.sum(axis=-1)


def gymnasium_pendulum_v1(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Per-episode return under Gymnasium Pendulum-v1's cost formula, applied to these trajectories (its dynamics
    differ): -(angle_normalize(theta_gym)^2 + 0.1 theta_dot^2 + 0.001 u^2), theta_gym = 0 upright, u clipped to +-2."""
    e = np.arctan2(states[..., 0], states[..., 1]) - np.pi
    e = (e + np.pi) % (2 * np.pi) - np.pi
    u = np.clip(actions[..., 0], -GYM_MAX_TORQUE, GYM_MAX_TORQUE)
    return -(e ** 2 + 0.1 * states[..., 2] ** 2 + 0.001 * u ** 2).sum(axis=-1)


def pendulum_swingup_v2(states: np.ndarray, actions: np.ndarray, commanded: np.ndarray | None = None) -> dict:
    """Per-episode arrays from (E, T, 3) pre-action states and (E, T, 1) applied actions; commanded actions, when the
    source records them separately (rollouts do, demonstrations do not), are checked against the action range."""
    if states.ndim != 3 or states.shape[-1] != 3 or actions.shape[:2] != states.shape[:2]:
        raise ValueError(f"expected (E, T, 3) states and (E, T, 1) actions, got {states.shape} and {actions.shape}")
    v = vertical(states)
    up = v >= UPRIGHT_COS
    reached = up.any(axis=1)
    first = np.where(reached, up.argmax(axis=1), -1)
    longest = _longest(up)
    a = actions[..., 0]
    out = {"upright_occupancy": up.mean(axis=1),
           "normalized_height": ((1 + v) / 2).mean(axis=1),
           "final_hold": up[:, -HOLD_STEPS:].all(axis=1),
           "reaches_upright": reached,
           "first_upright_s": np.where(reached, first * DT, np.nan),
           "upright_time_s": up.sum(axis=1) * DT,
           "longest_upright_s": longest * DT,
           "falls_after_first_upright": _falls(up, v < 0),
           "rms_angular_velocity": np.sqrt((states[..., 2] ** 2).mean(axis=1)),
           "normalized_effort": ((a / ACTION_BOUND) ** 2).mean(axis=1),
           "applied_out_of_range_fraction": (np.abs(a) > ACTION_BOUND).mean(axis=1),
           "legacy_ccil_v1_return": legacy_ccil_v1(states, actions),
           "gymnasium_pendulum_v1_return": gymnasium_pendulum_v1(states, actions)}
    if commanded is not None:
        out["commanded_out_of_range_fraction"] = (np.abs(commanded[..., 0]) > ACTION_BOUND).mean(axis=1)
    return out


def summarize(per_episode: dict, states: np.ndarray, actions: np.ndarray) -> dict:
    """Aggregate over episodes. Height, occupancy, RMS angular velocity and effort are means over every step of every
    episode (#143's definitions). With equal episode lengths they equal the means of the episode means, up to
    summation-order rounding."""
    reached = per_episode["reaches_upright"]
    first = per_episode["first_upright_s"][reached]
    out = {"profile": "pendulum_swingup_v2", "profile_version": VERSION, "episodes": int(len(reached)),
           "steps_per_episode": int(states.shape[1]),
           "mean_normalized_height": float(((1 + vertical(states)) / 2).mean()),
           "upright_occupancy": float(upright(states).mean()),
           "reaches_upright": float(reached.mean()),
           "final_two_second_hold": float(per_episode["final_hold"].mean()),
           "median_time_to_first_upright_s": float(np.median(first)) if first.size else None,
           "mean_upright_time_s": float(per_episode["upright_time_s"].mean()),
           "mean_longest_upright_s": float(per_episode["longest_upright_s"].mean()),
           "mean_falls_after_first_upright": float(per_episode["falls_after_first_upright"].mean()),
           "rms_angular_velocity_rad_s": float(np.sqrt((states[..., 2] ** 2).mean())),
           "mean_normalized_effort": float(((actions[..., 0] / ACTION_BOUND) ** 2).mean()),
           "applied_out_of_range_fraction": float(per_episode["applied_out_of_range_fraction"].mean()),
           "historical": {"legacy_ccil_v1_mean_return": float(per_episode["legacy_ccil_v1_return"].mean()),
                          "gymnasium_pendulum_v1_mean_return": float(per_episode["gymnasium_pendulum_v1_return"].mean())}}
    if "commanded_out_of_range_fraction" in per_episode:
        out["commanded_out_of_range_fraction"] = float(per_episode["commanded_out_of_range_fraction"].mean())
    return out


def score_rollout(rollout: dict) -> dict:
    """Both profiles for a stored rollout (tools/m3_pendulum or tools/vector_env arrays). The legacy return is the
    rollout's own stored returns, checked against a recomputation; v2 is computed from its states and actions."""
    T = rollout["actions"].shape[1]
    states, actions = rollout["states"][:, :T], rollout["actions"]
    per = pendulum_swingup_v2(states, actions, rollout.get("policy_actions"))
    drift = float(np.abs(per["legacy_ccil_v1_return"] - rollout["returns"]).max())
    if drift > 1e-9:
        raise ValueError(f"stored legacy returns differ from the legacy_ccil_v1 recomputation by {drift}")
    return {"legacy_ccil_v1": {"profile": "legacy_ccil_v1", "R": float(rollout["returns"].mean())},
            "pendulum_swingup_v2": summarize(per, states, actions)}
