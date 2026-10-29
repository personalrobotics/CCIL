#!/usr/bin/env python3
"""Diagnostic 2 of docs/experiments/pendulum-v6-diagnostics.md: zero action against the learned policies.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python tools/pendulum_diagnostics/behavior.py --root RUN_ROOT --out OUT_DIR

Reads the stored noise-free rollouts of the zero, bc, ccil, noise_bc and ccil_tuned arms (both tasks, all study seeds)
and writes OUT_DIR/behavior.json and figures. Nothing is re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
import env as canonical  # noqa: E402

TASKS = ("continuous", "discontinuous")
ARMS = ("zero", "bc", "noise_bc", "ccil", "ccil_tuned")
SEEDS = tuple(range(1201, 1221))
UPRIGHT, SENSITIVITY, HOLD_STEPS = 0.3, (0.1, 0.5), 100
CATEGORIES = ("holds", "swings_up_and_falls", "reaches_never_falls_not_held", "never_swings_up", "other")


def arm_dir(root: Path, task: str, arm: str) -> Path:
    base = root / "rollouts" / "study" / task
    return base / arm if (base / arm).is_dir() else next(base.glob(f"{arm}@*"))


def angle_error(states):
    theta = np.arctan2(states[..., 0], states[..., 1])
    return np.abs(np.where(theta < 0, theta + 2 * np.pi, theta) - np.pi)


def episode_metrics(z: dict, thr: float = UPRIGHT) -> dict:
    """Per-episode arrays for one rollout (100 episodes)."""
    s, a = z["states"][:, :-1], z["actions"][..., 0]
    e = angle_error(s)
    up = e < thr
    reached = up.any(axis=1)
    first = np.where(reached, up.argmax(axis=1), -1)
    longest = np.zeros(len(e), dtype=int)
    for i in range(len(e)):
        run = best = 0
        for u in up[i]:
            run = run + 1 if u else 0
            best = max(best, run)
        longest[i] = best
    holds = up[:, -HOLD_STEPS:].all(axis=1)
    after = np.array([(e[i, first[i]:] > np.pi / 2).any() if reached[i] else False for i in range(len(e))])
    never = (e > np.pi / 2).all(axis=1)
    cat = np.full(len(e), "other", dtype=object)
    cat[never] = "never_swings_up"
    cat[reached & ~after & ~holds] = "reaches_never_falls_not_held"
    cat[reached & after] = "swings_up_and_falls"
    cat[holds] = "holds"  # a hold overrides an earlier fall: it ends upright
    return {"reached": reached, "first": first, "steps_up": up.sum(axis=1), "longest": longest, "holds": holds,
            "category": cat,
            "angle_term": (-0.5 * e ** 2).sum(axis=1), "velocity_term": (-0.5 * s[..., 2] ** 2).sum(axis=1),
            "action_term": (-0.5 * a ** 2).sum(axis=1), "returns": z["returns"]}


def summarize(ms: list[dict]) -> dict:
    cat = np.concatenate([m["category"] for m in ms])
    reached = np.concatenate([m["reached"] for m in ms])
    first = np.concatenate([m["first"] for m in ms])
    out = {"episodes": int(cat.size),
           "categories": {c: float((cat == c).mean()) for c in CATEGORIES},
           "reaches_upright": float(reached.mean()),
           "first_upright_s_median_of_reached": float(np.median(first[reached]) * canonical.DT) if reached.any() else None,
           "steps_upright_mean": float(np.concatenate([m["steps_up"] for m in ms]).mean()),
           "longest_upright_run_s_mean": float(np.concatenate([m["longest"] for m in ms]).mean() * canonical.DT)}
    for k in ("angle_term", "velocity_term", "action_term", "returns"):
        out[f"{k}_mean"] = float(np.concatenate([m[k] for m in ms]).mean())
    # The three terms are the reward's own decomposition, so they must sum to the stored returns.
    out["decomposition_max_abs_error"] = float(max(
        np.abs(m["angle_term"] + m["velocity_term"] + m["action_term"] - m["returns"]).max() for m in ms))
    return out


def figure(rolls: dict, task: str, seed: int, episodes: list[int], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(canonical.HORIZON) * canonical.DT
    fig, axes = plt.subplots(4, len(episodes), figsize=(4 * len(episodes), 8), sharex=True, sharey="row",
                             layout="constrained", squeeze=False)
    colors = dict(zip(ARMS, ("#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
    for j, ep in enumerate(episodes):
        for arm in ARMS:
            z = rolls[(task, arm, seed)]
            s = z["states"][ep, :-1]
            kw = {"color": colors[arm], "lw": 1.1, "label": arm}
            axes[0, j].plot(t, angle_error(s), **kw)
            axes[1, j].plot(t, s[:, 2], **kw)
            axes[2, j].plot(t, z["actions"][ep, :, 0], **kw)
            axes[3, j].plot(t, z["rewards"][ep], **kw)
        axes[0, j].set_title(f"episode {ep}", fontsize=9)
        axes[0, j].axhline(UPRIGHT, color="gray", ls=":", lw=0.8)
        axes[-1, j].set_xlabel("time (s)")
    for ax, y in zip(axes[:, 0], ("|θ − π| (rad)", "θ̇ (rad/s)", "applied action", "reward")):
        ax.set_ylabel(y)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=5, loc="lower left")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rolls, metrics = {}, {}
    for task in TASKS:
        for arm in ARMS:
            d = arm_dir(a.root, task, arm)
            for seed in SEEDS:
                z = dict(np.load(d / f"s{seed}-noise0.0.npz"))
                rolls[(task, arm, seed)] = z
                metrics[(task, arm, seed)] = episode_metrics(z)
    report = {"declared_in": "docs/experiments/pendulum-v6-diagnostics.md §2", "upright_threshold_rad": UPRIGHT,
              "hold_steps": HOLD_STEPS, "summary": {}, "sensitivity": {}, "per_seed": {}, "representative": {}}
    for task in TASKS:
        for arm in ARMS:
            report["summary"][f"{task}/{arm}"] = summarize([metrics[(task, arm, s)] for s in SEEDS])
            for thr in SENSITIVITY:
                ms = [episode_metrics(rolls[(task, arm, s)], thr) for s in SEEDS]
                sm = summarize(ms)
                report["sensitivity"][f"{task}/{arm}/{thr}"] = {"categories": sm["categories"],
                                                                "reaches_upright": sm["reaches_upright"]}
            report["per_seed"][f"{task}/{arm}"] = {
                str(s): {k: float(metrics[(task, arm, s)][k].mean())
                         for k in ("angle_term", "velocity_term", "action_term", "returns", "steps_up")}
                for s in SEEDS}
        bcR = {s: rolls[(task, "bc", s)]["R"] for s in SEEDS}
        diff = {s: rolls[(task, "ccil", s)]["R"] - bcR[s] for s in SEEDS}
        ranked = sorted(SEEDS, key=lambda s: bcR[s])
        picks = {"median_bc": ranked[(len(ranked) - 1) // 2], "largest_ccil_minus_bc": max(SEEDS, key=diff.get),
                 "smallest_ccil_minus_bc": min(SEEDS, key=diff.get)}
        report["representative"][task] = {}
        for rule, seed in picks.items():
            ret = rolls[(task, "bc", seed)]["returns"]
            med_ep = int(np.argsort(ret)[(len(ret) - 1) // 2])
            eps = [med_ep, 0, 1, 2]
            report["representative"][task][rule] = {"seed": seed, "episodes": eps,
                                                    "ccil_minus_bc": float(diff[seed]), "bc_R": float(bcR[seed])}
            figure(rolls, task, seed, eps, f"{task}, seed {seed} ({rule}): episodes {eps}, noise-free",
                   a.out / f"behavior-{task}-{rule}.png")
    (a.out / "behavior.json").write_text(json.dumps(report, indent=1) + "\n")
    for k, v in report["summary"].items():
        print(k, {c: round(x, 3) for c, x in v["categories"].items()}, "R", round(v["returns_mean"]),
              "terms", round(v["angle_term_mean"]), round(v["velocity_term_mean"]), round(v["action_term_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
