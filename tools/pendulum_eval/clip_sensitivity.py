#!/usr/bin/env python3
"""Clipped-action sensitivity of historical Part B (descriptive; the primary result is unchanged).

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/clip_sensitivity.py --root V6_ROOT [--device cuda]

The Part B audit found commanded actions above the declared range [-3, 3] by at most 5.4e-4. This re-evaluates every
reused Part B unit noise-free on the same fresh starts with each commanded action clipped to [-3, 3] before it is
applied, scores it under pendulum_swingup_v2, and reports, against the committed unclipped Part B report:
- per unit and seed, the change in occupancy, height and final hold;
- CCIL - BC and every secondary comparison, recomputed on the clipped rollouts (descriptive, the Part B levels);
- each checkpoint's action-scaler bounds (act_min, act_max), which set the range of the policy's raw output.
Writes docs/fixtures/evidence/pendulum-eval-v2/part-b-clip-sensitivity.json. Rollouts are not saved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "pendulum_eval"))
import historical as h  # noqa: E402
import profiles  # noqa: E402

from ccil.experiments import analysis  # noqa: E402

OUT = h.EVIDENCE / "part-b-clip-sensitivity.json"


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()}


def clipped(predict):
    return lambda obs: np.clip(predict(obs), -profiles.ACTION_BOUND, profiles.ACTION_BOUND)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    committed = json.loads((h.EVIDENCE / "part-b.json").read_text())
    pol = h.load_policies(args.root, args.device)
    scalers = {}
    for (task, arm, seed), (predict, _) in pol.items():
        net = predict.__self__
        scalers[f"{task}/{arm}/s{seed}"] = {k: None if getattr(net, k, None) is None
                                            else [float(x) for x in np.ravel(getattr(net, k))]
                                            for k in ("act_min", "act_max")}
    scores, status = {}, {}
    for task in h.TASKS:
        items, keys = [], []
        for arm in h.REUSED:
            for s in h.SEEDS:
                items.append((clipped(pol[(task, arm, s)][0]), s + h.FRESH_OFFSET, 0.0, "none"))
                keys.append((task, arm, "0.0", s))
        for res, (t, arm, key, s) in zip(h.channel_rollouts(items, task), keys):
            st = status.setdefault(t, {}).setdefault(arm, {})
            if len(res["failed_episodes"]):
                st[s] = "algorithmic_failure"
                continue
            st[s] = "ok"
            assert np.abs(res["actions"]).max() <= profiles.ACTION_BOUND
            scores[(t, arm, key, s)] = h.seed_scores(res)
        print(task, "evaluated", len(items), flush=True)
    tables = h.tables_from(scores, status)
    change = {}
    for (t, arm, key, s), v in scores.items():
        base = committed["per_seed"][f"{t}/{arm}/{key}/s{s}"]
        change[f"{t}/{arm}/s{s}"] = {m: float(v[m] - base[m]) for m in h.METRICS}
    by_arm = {}
    for k, v in change.items():
        t, arm, _ = k.split("/")
        by_arm.setdefault(f"{t}/{arm}", []).append(v)
    report = {
        "label": "descriptive sensitivity; the predeclared Part B result is unchanged",
        "provenance": provenance(), "profile_version": profiles.VERSION, "device": args.device,
        "action_scalers": scalers,
        "max_abs_change_by_arm": {k: {m: float(max(abs(x[m]) for x in v)) for m in h.METRICS} for k, v in by_arm.items()},
        "units_changed": {k: sum(any(x[m] != 0 for m in h.METRICS) for x in v) for k, v in by_arm.items()},
        "per_seed_change": change,
        "primary_rule_on_clipped": {t: analysis.decide(tables["upright_occupancy"], t, h.PRIMARY) for t in h.TASKS},
        "secondary_95_on_clipped": h.describe(tables, [(a, "bc") for a in h.REUSED if a != "bc"]
                                              + [("ccil_modern", "bc_modern"), *h.DATA_SIZE], ["0.0"]),
    }
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    for t in h.TASKS:
        c = report["primary_rule_on_clipped"][t]
        print(t, c["outcome"], c["comparison"]["mean"], c["comparison"]["interval"])
    print(json.dumps(report["max_abs_change_by_arm"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
