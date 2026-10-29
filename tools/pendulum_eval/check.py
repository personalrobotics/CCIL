#!/usr/bin/env python3
"""Behavior-level evidence for the Pendulum evaluation profiles (#143), on fixed synthetic trajectories.

    uv run --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/check.py

Each case is a 500-step trajectory with a known answer derived from its construction, not from the implementation:
always down, always upright, a fast crossing, swing-up then hold, swing-up then fall, and out-of-range actions. The
v2 ordering of the cases, scoring an episode alone against scoring it in a batch, refusal of malformed inputs, and a
stored-rollout round trip through the batched evaluator are checked too. Writes docs/fixtures/evidence/pendulum-eval-v2/profiles.json; exits
1 if any check fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum"):
    sys.path.insert(0, str(p))
import pendulum as vector  # noqa: E402  (tools/vector_env)
import profiles  # noqa: E402

OUT = REPO / "docs" / "fixtures" / "evidence" / "pendulum-eval-v2" / "profiles.json"
T, DT = 500, profiles.DT
UP = np.deg2rad(profiles.UPRIGHT_DEG)


def traj(theta: np.ndarray, action: float | np.ndarray = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(1, T, 3) states with theta_dot the finite-difference rate of theta, and (1, T, 1) actions."""
    theta_dot = np.gradient(theta, DT)
    s = np.stack([np.sin(theta), np.cos(theta), theta_dot], axis=-1)[None]
    return s, np.broadcast_to(np.asarray(action, float), (T,)).reshape(1, T, 1).copy()


def ramp(k0: int, k1: int, a: float, b: float) -> np.ndarray:
    return a + (b - a) * np.clip((np.arange(T) - k0) / (k1 - k0), 0, 1)


def first_up_step(theta: np.ndarray) -> int:
    """The first step within 8 degrees of upright, from the construction (angle distance to pi)."""
    return int(np.argmax(np.abs((theta - np.pi + np.pi) % (2 * np.pi) - np.pi) <= UP))


def cases() -> dict:
    t = np.arange(T) * DT
    up_hold = ramp(0, 200, 0.0, np.pi)
    up_fall = np.where(np.arange(T) < 300, ramp(0, 200, 0.0, np.pi), ramp(300, 400, np.pi, 0.0))
    omega = 10.0
    return {
        "always_down": (np.zeros(T), 0.0),
        "always_upright": (np.full(T, np.pi), 0.0),
        "fast_crossing": (omega * t, 0.0),
        "swing_up_then_hold": (up_hold, 0.0),
        "swing_up_then_fall": (up_fall, 0.0),
        "out_of_range_actions": (np.full(T, np.pi), 5.0),
    }


def expect(name: str, theta: np.ndarray, m: dict) -> list[tuple[str, bool, object]]:
    """The checks for one case: (description, pass, observed)."""
    g = {k: (v[0].item() if hasattr(v[0], "item") else v[0]) for k, v in m.items()}
    c = []

    def chk(what, ok, obs):
        c.append((what, bool(ok), obs))

    if name == "always_down":
        chk("height 0", g["normalized_height"] == 0.0, g["normalized_height"])
        chk("occupancy 0", g["upright_occupancy"] == 0.0, g["upright_occupancy"])
        chk("never reaches upright, no hold, no falls", not g["reaches_upright"] and not g["final_hold"]
            and g["falls_after_first_upright"] == 0, (g["reaches_upright"], g["final_hold"]))
        chk("legacy return -0.5 pi^2 per step", abs(g["legacy_ccil_v1_return"] + 0.5 * np.pi ** 2 * T) < 1e-9,
            g["legacy_ccil_v1_return"])
    elif name == "always_upright":
        chk("height 1, occupancy 1", g["normalized_height"] == 1.0 and g["upright_occupancy"] == 1.0,
            (g["normalized_height"], g["upright_occupancy"]))
        chk("final hold; first upright at 0 s; longest run 10 s", g["final_hold"] and g["first_upright_s"] == 0.0
            and abs(g["longest_upright_s"] - T * DT) < 1e-12, (g["first_upright_s"], g["longest_upright_s"]))
    elif name == "fast_crossing":
        turns = 10.0 * T * DT / (2 * np.pi)
        chk("occupancy near the upright arc's share 16/360 (+-0.01)",
            abs(g["upright_occupancy"] - 16 / 360) < 0.01, g["upright_occupancy"])
        chk("height near 0.5 (+-0.02)", abs(g["normalized_height"] - 0.5) < 0.02, g["normalized_height"])
        chk("reaches upright, no final hold", g["reaches_upright"] and not g["final_hold"],
            (g["reaches_upright"], g["final_hold"]))
        chk(f"falls about once per turn after the first upright (turns {turns:.1f})",
            abs(g["falls_after_first_upright"] - np.floor(turns - 0.5)) <= 1, g["falls_after_first_upright"])
        chk("RMS angular velocity 10 rad/s", abs(g["rms_angular_velocity"] - 10.0) < 1e-9, g["rms_angular_velocity"])
    elif name == "swing_up_then_hold":
        k = first_up_step(theta)
        chk("first upright at the constructed step", abs(g["first_upright_s"] - k * DT) < 1e-12,
            (g["first_upright_s"], k * DT))
        chk("occupancy (T - k) / T", abs(g["upright_occupancy"] - (T - k) / T) < 1e-12, g["upright_occupancy"])
        chk("final hold, no falls", g["final_hold"] and g["falls_after_first_upright"] == 0,
            (g["final_hold"], g["falls_after_first_upright"]))
    elif name == "swing_up_then_fall":
        k = first_up_step(theta)
        last_up = int(np.flatnonzero(np.abs((theta - np.pi + np.pi) % (2 * np.pi) - np.pi) <= UP)[-1])
        chk("reaches upright, one fall, no final hold", g["reaches_upright"] and g["falls_after_first_upright"] == 1
            and not g["final_hold"], (g["falls_after_first_upright"], g["final_hold"]))
        chk("longest run spans the constructed upright interval",
            abs(g["longest_upright_s"] - (last_up - k + 1) * DT) < 1e-12, (g["longest_upright_s"], (last_up - k + 1) * DT))
    elif name == "out_of_range_actions":
        chk("applied out-of-range fraction 1", g["applied_out_of_range_fraction"] == 1.0,
            g["applied_out_of_range_fraction"])
        chk("commanded out-of-range fraction 1", g["commanded_out_of_range_fraction"] == 1.0,
            g["commanded_out_of_range_fraction"])
        chk("effort (5/3)^2, unclipped", abs(g["normalized_effort"] - (5 / 3) ** 2) < 1e-12, g["normalized_effort"])
        chk("Gymnasium-v1 cost formula on the CCIL trajectory clips the torque to 2 inside the cost",
            abs(g["gymnasium_v1_cost_formula_on_ccil_trajectory"] + 0.001 * 4 * T) < 1e-9,
            g["gymnasium_v1_cost_formula_on_ccil_trajectory"])
    return c


def main() -> int:
    results, scored = {}, {}
    batch_s, batch_a = [], []
    for name, (theta, action) in cases().items():
        s, a = traj(theta, action)
        m = profiles.pendulum_swingup_v2(s, a, commanded=a)
        scored[name] = m
        batch_s.append(s[0])
        batch_a.append(a[0])
        results[name] = [{"check": w, "pass": ok, "observed": o if not isinstance(o, tuple) else list(o)}
                         for w, ok, o in expect(name, theta, m)]
    names = list(cases())
    bm = profiles.pendulum_swingup_v2(np.stack(batch_s), np.stack(batch_a), commanded=np.stack(batch_a))
    same = all(np.array_equal(bm[k][i], scored[n][k][0], equal_nan=True) for i, n in enumerate(names) for k in bm)
    results["batch_equals_single"] = [{"check": "scoring the six episodes as one batch equals scoring each alone",
                                       "pass": bool(same), "observed": same}]
    order = ["always_upright", "swing_up_then_hold", "swing_up_then_fall", "always_down"]
    occ = [scored[n]["upright_occupancy"][0] for n in order]
    hgt = [scored[n]["normalized_height"][0] for n in order]
    results["ordering"] = [{"check": "occupancy and height rank upright > swing-up-hold > swing-up-fall > down",
                            "pass": bool(all(x > y for x, y in zip(occ, occ[1:])) and
                                         all(x > y for x, y in zip(hgt, hgt[1:]))),
                            "observed": {"occupancy": [float(x) for x in occ], "height": [float(x) for x in hgt]}}]
    s, a = traj(np.full(T, np.pi))
    malformed = {"actions without the action axis (E, T)": (s, a[..., 0], None),
                 "two action dimensions (E, T, 2)": (s, np.concatenate([a, a], axis=-1), None),
                 "actions one step short": (s, a[:, :-1], None),
                 "commanded shape differs from applied": (s, a, a[..., 0]),
                 "episode shorter than the 100-step hold window": (s[:, :99], a[:, :99], None)}
    for what, bad in (("NaN", np.nan), ("+inf", np.inf)):
        bs, ba, bc = s.copy(), a.copy(), a.copy()
        bs[0, 250, 2], ba[0, 250, 0], bc[0, 250, 0] = bad, bad, bad
        malformed[f"{what} in states"] = (bs, a, None)
        malformed[f"{what} in applied actions"] = (s, ba, None)
        malformed[f"{what} in commanded actions"] = (s, a, bc)
    results["malformed_inputs"] = []
    for what, (ms, ma, mc) in malformed.items():
        try:
            profiles.pendulum_swingup_v2(ms, ma, commanded=mc)
            refused, msg = False, "accepted"
        except ValueError as e:
            refused, msg = True, str(e)
        results["malformed_inputs"].append({"check": f"refused: {what}", "pass": refused, "observed": msg})
    zero = vector.rollout(lambda obs: np.zeros((len(obs), 1)), "continuous", 1201)  # not a learned policy
    sc = profiles.score_rollout(zero)
    results["stored_rollout"] = [{"check": "a batched-evaluator zero-action rollout: the legacy profile reads its stored "
                                           "returns, and v2 scores it as never upright",
                                  "pass": sc["legacy_ccil_v1"]["R"] == float(zero["R"]) and
                                  sc["pendulum_swingup_v2"]["upright_occupancy"] == 0.0,
                                  "observed": {"legacy_R": sc["legacy_ccil_v1"]["R"],
                                               "v2_occupancy": sc["pendulum_swingup_v2"]["upright_occupancy"]}}]
    ok = all(r["pass"] for rs in results.values() for r in rs)
    report = {"issue": "#143", "profile_version": profiles.VERSION, "upright_threshold_deg": profiles.UPRIGHT_DEG,
              "cases": results, "all_pass": ok}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    for k, rs in results.items():
        for r in rs:
            print("PASS" if r["pass"] else "FAIL", k, "-", r["check"], "| observed", r["observed"])
    print("all pass" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
