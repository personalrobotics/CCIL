#!/usr/bin/env python3
"""The expert-alignment gate of #143: pendulum_swingup_v2 on the retained demonstrations and same-start zero action.

    uv run --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/expert_alignment.py
    uv run --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/expert_alignment.py --check

The retained pickles cross the SHA-256-verified legacy boundary (tools/legacy_conversion/convert_pendulum.py) into a
temporary directory and are read with ccil's validated trajectory reader. Zero action starts each episode at its first
stored expert state and steps the batched integrator (tools/vector_env/pendulum.step, with the task's discontinuity)
until it has the same number of state frames. No learned policy is scored.

Writes docs/fixtures/evidence/pendulum-eval-v2/expert-alignment.json: every metric, the six gate checks declared in
#143 before outcomes were read, and a reconciliation of every value #143 states, at the precision it states. --check
regenerates the report and exits 1 if it differs from the committed one (floats to CROSS_HOST_RTOL) or if any stated
value does not reconcile.
The gate's outcome (including a failed check) is reported, not turned into an exit status.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum",
          REPO / "tools" / "legacy_conversion"):
    sys.path.insert(0, str(p))
import pendulum as vector  # noqa: E402  (tools/vector_env)
import profiles  # noqa: E402
from convert_pendulum import convert_pendulum  # noqa: E402

from ccil.schema.io import read_trajectories  # noqa: E402

OUT = REPO / "docs" / "fixtures" / "evidence" / "pendulum-eval-v2" / "expert-alignment.json"
# The batched integrator's zero-action states differ between macOS and Linux by at most 2 ulps (measured: 3.4e-16
# relative in the derived metrics; the expert values are identical). Floats are compared to this relative tolerance.
CROSS_HOST_RTOL = 1e-12
DATASETS = {"continuous": ("pendulum_cont", "data/pendulum_cont_100.npz", False),
            "discontinuous": ("pendulum_disc", "data/pendulum_disc_500.npz", True)}

# The six checks, as declared in #143 before outcomes were read: (name, source, metric, comparison, threshold).
CHECKS = (("zero_upright_occupancy", "zero", "upright_occupancy", "<=", 0.01),
          ("zero_final_hold", "zero", "final_two_second_hold", "==", 0.0),
          ("expert_reaches_upright", "expert", "reaches_upright", ">=", 0.95),
          ("expert_final_hold", "expert", "final_two_second_hold", ">=", 0.90),
          ("expert_minus_zero_height", "difference", "mean_normalized_height", ">=", 0.50),
          ("expert_upright_occupancy", "expert", "upright_occupancy", ">=", 0.50))

# Every value #143 states, as written there (strings keep the stated precision).
STATED = {
    "continuous": {"dataset_sha256": "a417269b62c3c5b444f9ac71aca5407fef18cf2c04f59a60df73f3234122f962",
                   "pickle_sha256": "ca9ee2f2f51b969d7d59ee26abd31fb6e537d379bd90dd645ba5e209a1f281fc",
                   "expert": {"episodes": "100", "steps_per_episode": "500",
                              "mean_normalized_height": "0.7382278119876428", "upright_occupancy": "0.497380",
                              "reaches_upright": "1.0", "final_two_second_hold": "0.95",
                              "median_time_to_first_upright_s": "3.95",
                              "rms_angular_velocity_rad_s": "2.134668817349325",
                              "mean_normalized_effort": "0.2501446008682251"},
                   "zero": {"episodes": "100", "steps_per_episode": "500",
                            "mean_normalized_height": "0.0444867223162392", "upright_occupancy": "0",
                            "reaches_upright": "0", "final_two_second_hold": "0",
                            "rms_angular_velocity_rad_s": "1.299136169418363", "mean_normalized_effort": "0"},
                   "expert_minus_zero_height": "0.6937410896714036"},
    "discontinuous": {"dataset_sha256": "b30cb074f77fdf094edc1fd91b6de9a1be28776740f630d33da1e24234c43b8e",
                      "pickle_sha256": "05c46cc52d6c8295e346709feca795f95956742bc1ab4f3924b4d838216549c8",
                      "expert": {"episodes": "500", "steps_per_episode": "500",
                                 "mean_normalized_height": "0.7336784857296410", "upright_occupancy": "0.490256",
                                 "reaches_upright": "1.0", "final_two_second_hold": "0.95",
                                 "median_time_to_first_upright_s": "3.80",
                                 "rms_angular_velocity_rad_s": "2.194756308477789",
                                 "mean_normalized_effort": "0.2499341666698456"},
                      "zero": {"episodes": "500", "steps_per_episode": "500",
                               "mean_normalized_height": "0.0458344462915480", "upright_occupancy": "0",
                               "reaches_upright": "0", "final_two_second_hold": "0",
                               "rms_angular_velocity_rad_s": "1.224300654218390", "mean_normalized_effort": "0"},
                      "expert_minus_zero_height": "0.6878440394380930"}}


def reconciles(value, stated: str) -> bool:
    """value equals the stated number. Values stated with at most two decimals (counts, rates over 100 or 500
    episodes, times in 0.02 s steps) are exact and must match up to float rounding of the decimal (1e-12); longer ones
    must match to half a unit in their last stated place."""
    d = Decimal(stated)
    exp = d.as_tuple().exponent
    if exp >= -2:
        return abs(float(value) - float(d)) <= 1e-12
    tol = max(Decimal(1).scaleb(exp) / 2, abs(d) * Decimal(repr(CROSS_HOST_RTOL)))
    return abs(Decimal(repr(float(value))) - d) <= tol


def same_report(a, b) -> bool:
    """Equal structure and non-float values; floats within CROSS_HOST_RTOL."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_report(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_report(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= CROSS_HOST_RTOL * max(abs(a), abs(b)) or a == b
    return a == b


def zero_rollout(first: np.ndarray, frames: int, discontinuous: bool) -> np.ndarray:
    states = np.empty((len(first), frames, 3))
    states[:, 0] = first
    for t in range(1, frames):
        states[:, t], _ = vector.step(states[:, t - 1], np.zeros(len(first)), discontinuous)
    return states


def gate(task: str, path: Path, discontinuous: bool) -> dict:
    ds = read_trajectories(path)
    expert_s = np.stack([ep.observations["state"] for ep in ds.episodes]).astype(np.float64)
    stored_a = np.stack([ep.actions for ep in ds.episodes])
    expert_a = stored_a.astype(np.float64)
    zero_s = zero_rollout(expert_s[:, 0], expert_s.shape[1], discontinuous)
    zero_a = np.zeros_like(expert_a)
    res = {}
    for name, s, a in (("expert", expert_s, expert_a), ("zero", zero_s, zero_a)):
        res[name] = profiles.summarize(profiles.pendulum_swingup_v2(s, a), s, a)
    res["expert_minus_zero_height"] = res["expert"]["mean_normalized_height"] - res["zero"]["mean_normalized_height"]
    checks = {}
    for cname, src, metric, op, thr in CHECKS:
        v = res["expert_minus_zero_height"] if src == "difference" else res[src][metric]
        ok = {"<=": v <= thr, ">=": v >= thr, "==": v == thr}[op]
        checks[cname] = {"value": v, "required": f"{op} {thr}", "pass": bool(ok)}
    res["checks"] = checks
    # The demonstrations store float32 actions. The canonical effort is computed in float64; the same formula in the
    # stored dtype is recorded too, because that is how #143's stated value was computed.
    res["expert_effort_in_stored_dtype"] = {"dtype": str(stored_a.dtype),
                                           "value": float(np.mean((stored_a[..., 0] / 3) ** 2))}
    res["all_checks_pass"] = all(c["pass"] for c in checks.values())
    return res


def reconcile(task: str, res: dict, hashes: dict) -> list[dict]:
    st, rows = STATED[task], []
    rows.append({"what": "converted dataset sha256", "stated": st["dataset_sha256"], "computed": hashes["dataset"],
                 "reconciles": st["dataset_sha256"] == hashes["dataset"]})
    rows.append({"what": "retained pickle sha256", "stated": st["pickle_sha256"], "computed": hashes["pickle"],
                 "reconciles": st["pickle_sha256"] == hashes["pickle"]})
    for src in ("expert", "zero"):
        for metric, stated in st[src].items():
            v = res[src][metric]
            rows.append({"what": f"{src} {metric}", "stated": stated, "computed": v, "reconciles": reconciles(v, stated)})
    stated = st["expert"]["mean_normalized_effort"]
    v = res["expert_effort_in_stored_dtype"]["value"]
    rows.append({"what": "expert mean_normalized_effort, recomputed in the stored float32 dtype", "stated": stated,
                 "computed": v, "reconciles": reconciles(v, stated),
                 "note": "the canonical float64 value differs from it by float32 rounding"})
    v = res["expert_minus_zero_height"]
    rows.append({"what": "expert minus zero mean height", "stated": st["expert_minus_zero_height"], "computed": v,
                 "reconciles": reconciles(v, st["expert_minus_zero_height"])})
    # The float64 effort row is explained, not reconciled, when the stored-dtype recomputation reconciles exactly.
    f32 = next(r for r in rows if r["what"].endswith("stored float32 dtype"))
    for r in rows:
        r["status"] = "reconciles" if r["reconciles"] else "differs"
        if not r["reconciles"] and r["what"] == "expert mean_normalized_effort" and f32["reconciles"]:
            r["status"] = "explained: #143 computed it in float32; the float32 recomputation reconciles"
            r["difference"] = float(r["computed"]) - float(r["stated"])
    return rows


def build() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        conv = convert_pendulum(Path(tmp))
        report = {"issue": "#143", "profile": "pendulum_swingup_v2", "profile_version": profiles.VERSION,
                  "upright_threshold_deg": profiles.UPRIGHT_DEG, "hold_steps": profiles.HOLD_STEPS,
                  "zero_action": "same-start, tools/vector_env/pendulum.step with the task's discontinuity",
                  "numpy": np.__version__, "tasks": {}}
        for task, (name, rel, disc) in DATASETS.items():
            info = conv["retained"][name]
            res = gate(task, Path(tmp) / rel, disc)
            hashes = {"dataset": info["dataset"]["sha256"], "pickle": info["legacy_pickle"]["sha256"]}
            rows = reconcile(task, res, hashes)
            report["tasks"][task] = {**res, "inputs": hashes, "reconciliation": rows,
                                     "all_stated_values_reconcile": all(r["status"] != "differs" for r in rows)}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    report = build()
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    ok = all(t["all_stated_values_reconcile"] for t in report["tasks"].values())
    for task, t in report["tasks"].items():
        print(task, "checks:", {k: c["pass"] for k, c in t["checks"].items()},
              "| stated values reconcile:", t["all_stated_values_reconcile"])
        for r in t["reconciliation"]:
            if r["status"] != "reconciles":
                print(" ", r["status"], "|", r["what"], r["stated"], r["computed"])
    if ap.parse_args().check:
        same = OUT.exists() and same_report(json.loads(OUT.read_text()), json.loads(text))
        print("report matches" if same else f"{OUT} differs from a regeneration")
        return 0 if same and ok else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
