"""The paper's Pendulum expert, reconstructed (#143, decision 1 of docs/experiments/pendulum-eval-v2-predeclaration.md).

CCIL (ICLR 2024), Appendix D, "Expert Formulation for Pendulum":

    pi_e(s) = -20.11 (theta - pi) - 7.08 theta_dot                          if |theta - pi| < 0.1   (LQR)
              -theta_dot (0.5 theta_dot^2 - 9.81 cos theta - 9.81)          otherwise               (energy shaping)

with the action clamped to [-3, 3]. The stored demonstrations hold exactly +-3 on about 24% of steps and never exceed it,
so the clamp is applied to the stored action.

The paper leaves two implementation details unstated. Each is a declared variant, fixed before any comparison
(docs/experiments/pendulum-expert-reconstruction.md):
- WRAPS: the angle theta = atan2(sin, cos) wrapped to [0, 2 pi) (the paper's normalized angle; primary), or left in
  (-pi, pi], where the LQR window covers only the positive side of upright;
- COSINES: cos theta taken as cos of that angle (primary), or as the state's cosine coordinate, which the integrator
  carries independently and can drift off the unit circle.

The printed gains are rounded to two decimals; the validation's LQR tolerance covers that rounding.
"""
from __future__ import annotations

import numpy as np

G, ACTION_BOUND, SWITCH = 9.81, 3.0, 0.1
K_ANGLE, K_RATE = 20.11, 7.08
GAIN_ROUNDING = 0.005  # half a unit in the printed gains' last place
WRAPS = ("zero_to_two_pi", "minus_pi_to_pi")
COSINES = ("of_angle", "state_coordinate")
PRIMARY = ("zero_to_two_pi", "of_angle")


def angle(states: np.ndarray, wrap: str) -> np.ndarray:
    theta = np.arctan2(states[..., 0], states[..., 1])
    if wrap == "zero_to_two_pi":
        return np.where(theta < 0, theta + 2 * np.pi, theta)
    if wrap == "minus_pi_to_pi":
        return theta
    raise ValueError(f"unknown wrap {wrap!r}")


def branches(states: np.ndarray, wrap: str = PRIMARY[0], cosine: str = PRIMARY[1],
             gains: tuple[float, float] = (K_ANGLE, K_RATE)) -> dict:
    """Unclamped LQR and energy-shaping actions, the LQR mask, and the switch margin | |theta - pi| - 0.1 |."""
    theta, theta_dot = angle(states, wrap), states[..., 2]
    if cosine == "of_angle":
        c = np.cos(theta)
    elif cosine == "state_coordinate":
        c = states[..., 1]
    else:
        raise ValueError(f"unknown cosine {cosine!r}")
    delta = theta - np.pi
    return {"lqr": -gains[0] * delta - gains[1] * theta_dot,
            "energy": -theta_dot * (0.5 * theta_dot ** 2 - G * c - G),
            "in_lqr": np.abs(delta) < SWITCH,
            "switch_margin": np.abs(np.abs(delta) - SWITCH),
            "delta": delta}


def act(states: np.ndarray, wrap: str = PRIMARY[0], cosine: str = PRIMARY[1],
        gains: tuple[float, float] = (K_ANGLE, K_RATE)) -> np.ndarray:
    """Clamped expert actions for states (..., 3)."""
    b = branches(states, wrap, cosine, gains)
    return np.clip(np.where(b["in_lqr"], b["lqr"], b["energy"]), -ACTION_BOUND, ACTION_BOUND)
