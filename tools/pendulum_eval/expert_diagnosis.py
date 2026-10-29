#!/usr/bin/env python3
"""Post hoc diagnosis of the failed expert reconstruction (docs/experiments/pendulum-expert-reconstruction.md).

    uv run --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/expert_diagnosis.py [--check]

Exploratory, run after revision 1 of the reconstruction failed as declared. It supports four claims and no others:
1. the printed Appendix D controller fails the declared per-state reconstruction (every variant, both tasks);
2. a wrapping inconsistency explains the in-window mismatch: the generator tested the LQR window on the angle wrapped
   to [0, 2 pi) but used the raw atan2 angle in (-pi, pi] in the LQR term, so on the theta > pi side the term is about
   -20.11 (-2 pi) and saturates at +3;
3. float32 evaluation plausibly explains the remaining small residuals: each is within a coarse float32 rounding scale
   of its branch's formula;
4. descriptive only: which demonstrations visit the affected side of the window, and their v2 occupancy and final
   hold. The wrapping defect may contribute to the expert's occupancy shortfall; this does not establish that it does.

Writes docs/fixtures/evidence/pendulum-eval-v2/expert-diagnosis.json; --check regenerates it and exits 1 if it differs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum",
          REPO / "tools" / "legacy_conversion"):
    sys.path.insert(0, str(p))
import expert  # noqa: E402
import profiles  # noqa: E402
from convert_pendulum import convert_pendulum  # noqa: E402
from expert_alignment import DATASETS, same_report  # noqa: E402
from expert_validation import AMBIGUOUS_SWITCH, STORAGE_TOL, VARIANTS, per_state  # noqa: E402

from ccil.schema.io import read_trajectories  # noqa: E402

OUT = REPO / "docs" / "fixtures" / "evidence" / "pendulum-eval-v2" / "expert-diagnosis.json"
EPS32 = float(np.finfo(np.float32).eps)
LARGE = 1e-3  # a disagreement larger than any rounding effect at |a| <= 3


def diagnose(states: np.ndarray, stored: np.ndarray) -> dict:
    theta, theta_dot = expert.angle(states, "zero_to_two_pi"), states[..., 2]
    raw = expert.angle(states, "minus_pi_to_pi")
    b = expert.branches(states)
    clamp = lambda x: np.clip(x, -expert.ACTION_BOUND, expert.ACTION_BOUND)  # noqa: E731
    window = b["in_lqr"] & (b["switch_margin"] >= AMBIGUOUS_SWITCH)
    tol_lqr = STORAGE_TOL + expert.GAIN_ROUNDING * (np.abs(b["delta"]) + np.abs(theta_dot))
    printed = clamp(b["lqr"])
    inconsistent = clamp(-expert.K_ANGLE * (raw - np.pi) - expert.K_RATE * theta_dot)
    err_printed, err_inc = np.abs(stored - printed), np.abs(stored - inconsistent)
    large = window & (err_printed > LARGE)
    affected = window & (raw < 0)  # the theta > pi side of the window, where the raw angle is near -pi
    claim2 = {"window_states": int(window.sum()),
              "printed_large_disagreements": int(large.sum()),
              "printed_large_disagreements_on_affected_side": int((large & affected).sum()),
              "affected_side_states": int(affected.sum()),
              "affected_side_stored_action_values": sorted({float(x) for x in stored[affected]}),
              "inconsistent_rule_failing_frozen_lqr_tolerance": int((window & (err_inc > tol_lqr)).sum()),
              "inconsistent_rule_large_disagreements": int((window & (err_inc > LARGE)).sum()),
              "inconsistent_rule_max_error_in_window": float(err_inc[window].max())}
    # Claim 3: residuals against the rule that matches, scaled by a coarse float32 rounding bound of each formula.
    energy = ~b["in_lqr"] & (b["switch_margin"] >= AMBIGUOUS_SWITCH)
    unsat_e = energy & (np.abs(b["energy"]) < expert.ACTION_BOUND)
    scale_e = EPS32 * (np.abs(theta_dot) * (0.5 * theta_dot ** 2 + expert.G * np.abs(np.cos(theta)) + expert.G)
                       * (1 + np.abs(theta_dot)) + np.abs(theta_dot) * expert.G * np.abs(np.sin(theta)) * np.pi)
    lqr_term = -expert.K_ANGLE * (raw - np.pi) - expert.K_RATE * theta_dot
    unsat_l = window & (np.abs(lqr_term) < expert.ACTION_BOUND)
    scale_l = EPS32 * (expert.K_ANGLE * (np.abs(raw) + np.pi) + expert.K_RATE * np.abs(theta_dot))
    claim3 = {}
    for name, m, err, scale in (("energy_unsaturated", unsat_e, np.abs(stored - b["energy"]), scale_e),
                                ("lqr_unsaturated_inconsistent_rule", unsat_l, err_inc, scale_l)):
        r = err[m] / scale[m]
        claim3[name] = {"states": int(m.sum()), "max_abs_residual": float(err[m].max()),
                        "median_abs_residual": float(np.median(err[m])),
                        "residual_over_float32_scale_q50_q99_q999_max": [float(x) for x in
                                                                         np.quantile(r, [0.5, 0.99, 0.999, 1.0])],
                        "storage_tolerance_exceeded": int((err[m] > STORAGE_TOL).sum())}
    # Claim 4 (descriptive, not causal).
    visits = affected.any(axis=1)
    per = profiles.pendulum_swingup_v2(states, stored[..., None])
    claim4 = {"episodes": int(len(visits)), "episodes_visiting_affected_side": int(visits.sum())}
    for name, m in (("visiting", visits), ("not_visiting", ~visits)):
        claim4[name] = {"episodes": int(m.sum()),
                        "mean_upright_occupancy": float(per["upright_occupancy"][m].mean()) if m.any() else None,
                        "final_hold_rate": float(per["final_hold"][m].mean()) if m.any() else None}
    claim4["note"] = ("descriptive association only; episodes that visit the affected side may differ in other ways, "
                      "so no causal effect on occupancy is established")
    return {"claim2_wrapping_inconsistency": claim2, "claim3_float32_residuals": claim3,
            "claim4_descriptive_occupancy": claim4}


def build() -> dict:
    report = {"issue": "#143", "status": "post hoc, exploratory; after revision 1 failed as declared",
              "reconstruction": "docs/experiments/pendulum-expert-reconstruction.md", "numpy": np.__version__,
              "tasks": {}}
    with tempfile.TemporaryDirectory() as tmp:
        convert_pendulum(Path(tmp))
        for task, (_, rel, _) in DATASETS.items():
            ds = read_trajectories(Path(tmp) / rel)
            states = np.stack([ep.observations["state"] for ep in ds.episodes]).astype(np.float64)
            stored = np.stack([ep.actions for ep in ds.episodes])[..., 0].astype(np.float64)
            claim1 = {f"{w}/{c}": per_state(states, stored, w, c)["pass"] for w, c in VARIANTS}
            report["tasks"][task] = {"claim1_printed_controller_passes_per_state": claim1, **diagnose(states, stored)}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    report = build()
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    for task, t in report["tasks"].items():
        print(task, json.dumps({k: v for k, v in t.items()}, indent=1))
    if ap.parse_args().check:
        same = OUT.exists() and same_report(json.loads(OUT.read_text()), json.loads(text))
        print("report matches" if same else f"{OUT} differs from a regeneration")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
