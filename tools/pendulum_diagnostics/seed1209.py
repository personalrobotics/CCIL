#!/usr/bin/env python3
"""Diagnostic 1 of docs/experiments/pendulum-v6-diagnostics.md: the seed 1209 BC-10% unit, replayed and described.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python tools/pendulum_diagnostics/seed1209.py --root RUN_ROOT --out OUT_DIR

Reads the v6 run root (records, rollouts, stages) and writes OUT_DIR/seed1209.json and a figure. Nothing is trained
and nothing under RUN_ROOT is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "m3_pendulum", REPO / "tools" / "vector_env"):
    sys.path.insert(0, str(p))
import env as canonical  # noqa: E402  (tools/m3_pendulum)
import pendulum as vector  # noqa: E402  (tools/vector_env)

TASK, SEED, UNIT, CONTROLS = "continuous", 1209, "bc_frac10", ("bc", "ccil")
RECOMPUTE_TOL, OPEN_LOOP_TOL, RETURN_TOL = 1e-9, 1e-6, 1e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arm_dir(root: Path, arm: str) -> Path:
    base = root / "results" / "study" / TASK
    return base / arm if (base / arm).is_dir() else next(base.glob(f"{arm}@*"))


def rewards_of(states, actions):
    """env.reward for every (episode, step), vectorized: the pre-step state and the applied action."""
    theta = np.arctan2(states[:, :-1, 0], states[:, :-1, 1])
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    return -0.5 * ((theta - np.pi) ** 2 + states[:, :-1, 2] ** 2) - 0.5 * actions[..., 0] ** 2


def angle_error(states):
    theta = np.arctan2(states[..., 0], states[..., 1])
    return np.abs(np.where(theta < 0, theta + 2 * np.pi, theta) - np.pi)


def first_divergence(a, b):
    """Per episode, the first step where states differ, and the size of that first difference."""
    d = np.abs(a - b).max(axis=-1)
    out = []
    for e in range(d.shape[0]):
        nz = np.flatnonzero(d[e] > 0)
        out.append(None if nz.size == 0 else {"step": int(nz[0]), "size": float(d[e, nz[0]]),
                                              "final": float(d[e, -1])})
    return out


def check_unit(root: Path, arm: str, device: str) -> tuple[dict, dict]:
    from ccil.training.vector_bc import VectorBCPolicy
    rec_path = arm_dir(root, arm) / f"s{SEED}.json"
    rec = json.loads(rec_path.read_text())
    ev = rec["evaluations"]["0.0"]
    roll = root / ev["rollout"]
    ckpt = Path(rec["stages"]["policy.train"]) / "checkpoints" / "policy.npz"
    stored = dict(np.load(roll))
    out = {"record": str(rec_path.relative_to(root)), "stored_R": ev["R"],
           "rollout_sha256_matches": sha(roll) == ev["rollout_sha256"],
           "checkpoint_sha256_matches": sha(ckpt) == rec["checkpoint_sha256"]}

    r = rewards_of(stored["states"], stored["actions"])
    out["recompute"] = {"max_abs_reward_diff": float(np.abs(r - stored["rewards"]).max()),
                        "max_abs_return_diff": float(np.abs(r.sum(1) - stored["returns"]).max()),
                        "R_diff": float(abs(r.sum(1).mean() - ev["R"]))}
    out["recompute"]["ok"] = max(out["recompute"].values()) <= RECOMPUTE_TOL

    net, _, _ = VectorBCPolicy.load(ckpt, device=device)
    pa = np.stack([net.predict(stored["observations"][:, t]) for t in range(canonical.HORIZON)], axis=1)
    d = float(np.abs(pa - stored["policy_actions"]).max())
    out["open_loop"] = {"max_abs_action_diff": d, "ok": d <= OPEN_LOOP_TOL}

    v = vector.rollouts([(net.predict, SEED, 0.0)], TASK)[0]
    same = {k: bool(np.array_equal(v[k], stored[k])) for k in ("states", "policy_actions", "actions", "rewards")}
    rd = float(np.abs(v["returns"] - stored["returns"]).max())
    out["closed_loop_study_evaluator"] = {"bit_identical": same, "max_abs_return_diff": rd,
                                          "ok": all(same.values()) and rd <= RETURN_TOL}

    c = canonical.rollout(net.predict, TASK, SEED, 0.0)
    rd = np.abs(c["returns"] - stored["returns"])
    div = first_divergence(c["states"], stored["states"])
    over = [int(e) for e in np.flatnonzero(rd > RETURN_TOL)]
    out["closed_loop_canonical"] = {
        "max_abs_return_diff": float(rd.max()), "R_diff": float(abs(c["R"] - ev["R"])),
        "episodes_over_tolerance": over,
        "first_divergence_of_those": {e: div[e] for e in over},
        "largest_first_divergence_overall": max((x["size"] for x in div if x), default=0.0),
        # Explained by amplified rounding when every over-tolerance episode first diverges at the 1e-12 level.
        "ok": all(div[e] is not None and div[e]["size"] <= 1e-12 for e in over)}
    out["integrity_ok"] = all([out["rollout_sha256_matches"], out["checkpoint_sha256_matches"], out["recompute"]["ok"],
                               out["open_loop"]["ok"], out["closed_loop_study_evaluator"]["ok"],
                               out["closed_loop_canonical"]["ok"]])
    return out, stored


def describe(stored: dict, control_median: float) -> dict:
    ret = stored["returns"]
    order = np.argsort(ret)
    shortfall = np.clip(control_median - ret, 0, None)
    cum = np.cumsum(shortfall[order])
    half = int(np.searchsorted(cum, cum[-1] / 2) + 1) if cum[-1] > 0 else 0
    a, s = stored["actions"][..., 0], stored["states"]
    return {"returns_sorted": [round(float(x), 3) for x in ret[order]],
            "worst_episodes": [int(e) for e in order[:5]],
            "median_episode_return": float(np.median(ret)),
            "control_median_episode_return": control_median,
            "episodes_for_half_of_shortfall": half,
            "total_shortfall": float(cum[-1]),
            "max_abs_action": float(np.nanmax(np.abs(a))), "max_abs_theta_dot": float(np.nanmax(np.abs(s[..., 2]))),
            "steps_abs_action_over_3": int((np.abs(a) > 3).sum()),
            "episodes_with_abs_action_over_3": int((np.abs(a) > 3).any(axis=1).sum()),
            "non_finite_values": int(sum((~np.isfinite(stored[k])).sum()
                                         for k in ("states", "actions", "rewards")))}


def figure(stored: dict, episodes: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(canonical.HORIZON) * canonical.DT
    s = stored["states"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for e in episodes:
        lab = f"episode {e} (return {stored['returns'][e]:.0f})"
        axes[0].plot(t, angle_error(s[e, :-1]), lw=1.2, label=lab)
        axes[1].plot(t, s[e, :-1, 2], lw=1.2)
        axes[2].plot(t, stored["actions"][e, :, 0], lw=1.2)
        axes[3].plot(t, stored["rewards"][e], lw=1.2)
    for ax, y in zip(axes, ("angle error |θ − π| (rad)", "θ̇ (rad/s)", "applied action", "reward per step")):
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
    axes[2].axhline(3, color="gray", ls="--", lw=0.8)
    axes[2].axhline(-3, color="gray", ls="--", lw=0.8)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{TASK} {UNIT} seed {SEED}: worst episodes and the median episode", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report, rollouts = {"declared_in": "docs/experiments/pendulum-v6-diagnostics.md §1"}, {}
    for arm in (UNIT, *CONTROLS):
        report[arm], rollouts[arm] = check_unit(a.root, arm, a.device)
    control_median = float(np.median(np.concatenate([rollouts[c]["returns"] for c in CONTROLS])))
    for arm in (UNIT, *CONTROLS):
        report[arm]["description"] = describe(rollouts[arm], control_median)
    report["integrity_ok"] = all(report[arm]["integrity_ok"] for arm in (UNIT, *CONTROLS))
    d = report[UNIT]["description"]
    order = d["worst_episodes"][:3]
    med = int(np.argsort(rollouts[UNIT]["returns"])[len(rollouts[UNIT]["returns"]) // 2])
    figure(rollouts[UNIT], [*order, med], a.out / "seed1209-episodes.png")
    (a.out / "seed1209.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"integrity_ok": report["integrity_ok"],
                      **{arm: {"integrity_ok": report[arm]["integrity_ok"],
                               "episodes_for_half_of_shortfall": report[arm]["description"]["episodes_for_half_of_shortfall"]}
                         for arm in (UNIT, *CONTROLS)}}, indent=1))
    return 0 if report["integrity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
