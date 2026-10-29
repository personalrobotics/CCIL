#!/usr/bin/env python3
"""Post hoc comparison of the canonical corrected datasets with a second host's regeneration (descriptive only).

    uv run --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/corrected_cross_host.py --canonical DIR --other DIR --other-host NAME

The cross-host regeneration gate of docs/experiments/pendulum-corrected-demonstrator-predeclaration.md failed (the
second host's bytes differ) and remains failed. This report describes how the two generations differ. It has no pass
threshold and decides nothing; the second host's datasets are never used.

Per task it records raw differences and agreement counts:
- state and action differences: maximum, quantiles, episodes identical, first differing step;
- controller branch (LQR window or energy shaping) at every stored state;
- saturation decision (whether the unclamped float64 action exceeds the bound) at every stored state;
- discontinuity flips (discontinuous task only): whether each stored next state satisfies the flip condition (the
  condition depends on the angle only, which the flip leaves unchanged);
- per-episode upright occupancy and final hold under pendulum_swingup_v2.
Writes docs/fixtures/evidence/pendulum-corrected-demonstrator/cross-host-comparison.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum"):
    sys.path.insert(0, str(p))
import expert  # noqa: E402
import profiles  # noqa: E402
from corrected_demos import EVIDENCE, NAMES  # noqa: E402
from expert_alignment import DATASETS  # noqa: E402

from ccil.schema.io import read_trajectories  # noqa: E402

OUT = EVIDENCE / "cross-host-comparison.json"
FLIP_ANGLES, FLIP_WINDOW = (np.pi / 6, np.pi / 3), 1e-2  # tools/m3_pendulum/env.step


def load(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    ds = read_trajectories(path)
    s = np.stack([ep.observations["state"] for ep in ds.episodes])
    a = np.stack([ep.actions for ep in ds.episodes])[..., 0].astype(np.float64)
    return s, a, hashlib.sha256(path.read_bytes()).hexdigest()


def decisions(s: np.ndarray) -> dict:
    b = expert.branches(s)
    raw = np.where(b["in_lqr"], b["lqr"], b["energy"])
    theta_next = np.arctan2(s[:, 1:, 0], s[:, 1:, 1])
    flip = np.zeros(theta_next.shape, bool)
    for ang in FLIP_ANGLES:
        flip |= np.abs(theta_next - ang) < FLIP_WINDOW
    return {"branch_lqr": b["in_lqr"], "saturated": np.abs(raw) > expert.ACTION_BOUND, "flip_after_step": flip}


def agreement(x: np.ndarray, y: np.ndarray) -> dict:
    d = x != y
    return {"steps": int(d.size), "steps_disagreeing": int(d.sum()), "episodes_disagreeing": int(d.any(axis=1).sum()),
            "canonical_true": int(x.sum()), "other_true": int(y.sum())}


def diff_stats(d: np.ndarray) -> dict:
    per_ep = d.reshape(len(d), -1).max(axis=1)
    nz = d[d > 0]
    return {"max": float(d.max()), "quantiles_of_nonzero_q50_q99": [float(np.quantile(nz, q)) for q in (0.5, 0.99)]
            if nz.size else None, "elements_differing": int((d > 0).sum()), "elements": int(d.size),
            "episodes_identical": int((per_ep == 0).sum()), "episode_max_q50_q99": [float(np.quantile(per_ep, q))
                                                                                   for q in (0.5, 0.99)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--other", type=Path, required=True)
    ap.add_argument("--other-host", required=True)
    args = ap.parse_args()
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"label": "post hoc, descriptive only; no pass threshold; the regeneration gate failed and remains failed",
              "other_host": args.other_host, "provenance": {"commit": git("rev-parse", "HEAD"),
                                                            "dirty": git("status", "--porcelain").splitlines(),
                                                            "numpy": np.__version__},
              "tasks": {}}
    for task, name in NAMES.items():
        sc, ac, hc = load(args.canonical / name)
        so, ao, ho = load(args.other / name)
        ds, da = np.abs(sc - so).max(axis=2), np.abs(ac - ao)
        either = (ds > 0) | (da > 0)
        first = [int(np.argmax(e)) for e in either if e.any()]
        dc, do = decisions(sc), decisions(so)
        pc = profiles.pendulum_swingup_v2(sc, ac[..., None])
        po = profiles.pendulum_swingup_v2(so, ao[..., None])
        report["tasks"][task] = {
            "canonical_sha256": hc, "other_sha256": ho, "episodes": int(len(sc)), "frames": int(sc.shape[1]),
            "state_abs_difference": diff_stats(ds), "action_abs_difference": diff_stats(da),
            "first_differing_step": {"episodes_differing": len(first),
                                     "min": min(first) if first else None,
                                     "median": float(np.median(first)) if first else None,
                                     "max": max(first) if first else None},
            "controller_branch": agreement(dc["branch_lqr"], do["branch_lqr"]),
            "saturation_decision": agreement(dc["saturated"], do["saturated"]),
            "discontinuity_flip": agreement(dc["flip_after_step"], do["flip_after_step"]) if DATASETS[task][2]
            else "not applicable: the continuous task has no discontinuity",
            "upright_occupancy": {"episodes_differing": int((pc["upright_occupancy"] != po["upright_occupancy"]).sum()),
                                  "max_abs_difference": float(np.abs(pc["upright_occupancy"]
                                                                     - po["upright_occupancy"]).max()),
                                  "canonical_mean": float(pc["upright_occupancy"].mean()),
                                  "other_mean": float(po["upright_occupancy"].mean())},
            "final_hold": {"episodes_differing": int((pc["final_hold"] != po["final_hold"]).sum()),
                           "canonical_rate": float(pc["final_hold"].mean()),
                           "other_rate": float(po["final_hold"].mean())}}
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for task, t in report["tasks"].items():
        print(task, json.dumps({k: v for k, v in t.items() if k not in ("canonical_sha256", "other_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
