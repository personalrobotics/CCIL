#!/usr/bin/env python3
"""The two frozen validations of the reconstructed Pendulum expert (docs/experiments/pendulum-expert-reconstruction.md).

    uv run --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/expert_validation.py [--check]

Check 1 compares the reconstructed clamped action with the stored action at every demonstration state, for each declared
variant. Check 2 replays the selected variant closed-loop from every demonstration's first state through v6's scalar
generator step and compares the complete trace, inside a conditioning envelope built from the gains' rounding corners,
float32-rounded applied actions and the batched integrator. Writes
docs/fixtures/evidence/pendulum-eval-v2/expert-reconstruction.json; --check regenerates it and exits 1 if it differs
from the committed one (floats to expert_alignment.CROSS_HOST_RTOL). Verdicts are outcomes, never exit statuses.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum",
          REPO / "tools" / "legacy_conversion"):
    sys.path.insert(0, str(p))
import env as scalar  # noqa: E402  (tools/m3_pendulum)
import expert  # noqa: E402
import pendulum as vector  # noqa: E402  (tools/vector_env)
from convert_pendulum import convert_pendulum  # noqa: E402
from expert_alignment import DATASETS, same_report  # noqa: E402

from ccil.schema.io import read_trajectories  # noqa: E402

OUT = REPO / "docs" / "fixtures" / "evidence" / "pendulum-eval-v2" / "expert-reconstruction.json"
# Frozen in docs/experiments/pendulum-expert-reconstruction.md before any comparison.
STORAGE_TOL, AMBIGUOUS_SWITCH = 1e-6, 1e-9
CONDITIONED, ENVELOPE_FACTOR, MIN_CONDITIONED = 0.01, 2.0, 0.90
VARIANTS = list(itertools.product(expert.WRAPS, expert.COSINES))  # the table's order; the primary is first
CORNERS = [(expert.K_ANGLE + i * expert.GAIN_ROUNDING, expert.K_RATE + j * expert.GAIN_ROUNDING)
           for i in (-1, 1) for j in (-1, 1)]


def per_state(states: np.ndarray, stored: np.ndarray, wrap: str, cosine: str) -> dict:
    b = expert.branches(states, wrap, cosine)
    clamp = lambda x: np.clip(x, -expert.ACTION_BOUND, expert.ACTION_BOUND)  # noqa: E731
    err_lqr, err_energy = np.abs(stored - clamp(b["lqr"])), np.abs(stored - clamp(b["energy"]))
    tol_lqr = STORAGE_TOL + expert.GAIN_ROUNDING * (np.abs(b["delta"]) + np.abs(states[..., 2]))
    ok_lqr, ok_energy = err_lqr <= tol_lqr, err_energy <= STORAGE_TOL
    ambiguous = b["switch_margin"] < AMBIGUOUS_SWITCH
    ok = np.where(ambiguous, ok_lqr | ok_energy, np.where(b["in_lqr"], ok_lqr, ok_energy))
    out = {"pass": bool(ok.all()), "states": int(ok.size), "failing_states": int((~ok).sum()),
           "switch_ambiguous_states": int(ambiguous.sum()), "branches": {}}
    for name, mask, err in (("lqr", b["in_lqr"] & ~ambiguous, err_lqr), ("energy", ~b["in_lqr"] & ~ambiguous, err_energy)):
        e = err[mask]
        out["branches"][name] = {"states": int(mask.sum()),
                                 "max_error": float(e.max()) if e.size else None,
                                 "p99_9_error": float(np.quantile(e, 0.999)) if e.size else None,
                                 "failing": int((~ok[mask]).sum())}
    bad = np.argwhere(~ok)
    out["first_failures"] = [{"episode": int(e), "step": int(t), "stored": float(stored[e, t]),
                              "lqr": float(b["lqr"][e, t]), "energy": float(b["energy"][e, t]),
                              "in_lqr": bool(b["in_lqr"][e, t])} for e, t in bad[:20]]
    return out


def _replay_one(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One episode through the scalar generator step: states (T, 3), applied actions (T,), flipped (T - 1,)."""
    s0, T, disc, wrap, cosine, gains, f32 = args
    states, actions, flips = np.empty((T, 3)), np.empty(T), np.zeros(T - 1, bool)
    states[0] = s0
    for t in range(T):
        a = float(expert.act(states[t], wrap, cosine, gains))
        actions[t] = float(np.float32(a)) if f32 else a
        if t + 1 < T:
            states[t + 1], flips[t] = scalar.step(states[t].copy(), actions[t], disc)
    return states, actions, flips


def replay_scalar(first: np.ndarray, T: int, disc: bool, variant, gains, f32: bool, pool) -> tuple:
    jobs = [(s0, T, disc, *variant, gains, f32) for s0 in first]
    res = list(pool.map(_replay_one, jobs, chunksize=4))
    return tuple(np.stack(x) for x in zip(*res))


def replay_batched(first: np.ndarray, T: int, disc: bool, variant) -> tuple:
    states, actions = np.empty((len(first), T, 3)), np.empty((len(first), T))
    states[:, 0] = first
    for t in range(T):
        actions[:, t] = expert.act(states[:, t], *variant)
        if t + 1 < T:
            states[:, t + 1], _ = vector.step(states[:, t].copy(), actions[:, t], disc)
    return states, actions


def deviation(sa, aa, sb, ab) -> np.ndarray:
    """(E, T) largest absolute difference over the state coordinates and the action."""
    return np.maximum(np.abs(sa - sb).max(axis=-1), np.abs(aa - ab))


def replay_check(states, stored, disc: bool, variant, pool) -> dict:
    E, T, _ = states.shape
    first = states[:, 0]
    main_s, main_a, main_f = replay_scalar(first, T, disc, variant, (expert.K_ANGLE, expert.K_RATE), False, pool)
    refs = [replay_scalar(first, T, disc, variant, g, False, pool)[:2] for g in CORNERS]
    refs.append(replay_scalar(first, T, disc, variant, (expert.K_ANGLE, expert.K_RATE), True, pool)[:2])
    refs.append(replay_batched(first, T, disc, variant))
    D = np.maximum.accumulate(np.max([deviation(main_s, main_a, s, a) for s, a in refs], axis=0), axis=1)
    main_a32 = main_a.astype(np.float32).astype(np.float64)
    Err = np.maximum(np.abs(states - main_s).max(axis=-1), np.abs(stored - main_a32))
    prefix = (D <= CONDITIONED).sum(axis=1)  # D is non-decreasing, so the conditioned steps are a prefix
    within = np.arange(T)[None] < prefix[:, None]
    ok_step = (Err <= STORAGE_TOL + ENVELOPE_FACTOR * D) | ~within
    ep_pass = ok_step.all(axis=1)
    full = prefix == T
    flips = {"episodes_with_flips": int(main_f.any(axis=1).sum()), "flips": int(main_f.sum()),
             "flips_in_prefix": 0, "stored_sign_reversal_disagreements": []}
    for e, t in np.argwhere(main_f):
        if t + 1 < prefix[e]:
            flips["flips_in_prefix"] += 1
            if np.sign(states[e, t + 1, 2]) == np.sign(states[e, t, 2]):
                flips["stored_sign_reversal_disagreements"].append({"episode": int(e), "step": int(t)})
    margin = np.where(within, Err - (STORAGE_TOL + ENVELOPE_FACTOR * D), -np.inf)
    verdict = "pass" if ep_pass.all() and full.mean() >= MIN_CONDITIONED else (
        "fail" if not ep_pass.all() else "inconclusive")
    return {"verdict": verdict, "episodes": E, "episodes_passing": int(ep_pass.sum()),
            "episodes_conditioned_full_horizon": int(full.sum()),
            "conditioned_fraction": float(full.mean()),
            "conditioning_limited": [{"episode": int(e), "prefix_steps": int(prefix[e])} for e in np.flatnonzero(~full)],
            "failing_episodes": [{"episode": int(e), "first_failing_step": int(np.argmax(~ok_step[e])),
                                  "max_error_in_prefix": float(Err[e][within[e]].max())}
                                 for e in np.flatnonzero(~ep_pass)][:50],
            "max_error_in_prefix": float(Err[within].max()),
            "max_error_full_horizon_conditioned_episodes": float(Err[full].max()) if full.any() else None,
            "max_envelope_in_prefix": float(D[within].max()),
            "worst_margin_in_prefix": float(margin.max()),
            "velocity_flips": flips}


def build(workers: int) -> dict:
    report = {"issue": "#143", "declaration": "docs/experiments/pendulum-expert-reconstruction.md",
              "controller": {"k_angle": expert.K_ANGLE, "k_rate": expert.K_RATE, "switch": expert.SWITCH,
                             "action_bound": expert.ACTION_BOUND, "gain_rounding": expert.GAIN_ROUNDING},
              "tolerances": {"storage": STORAGE_TOL, "ambiguous_switch": AMBIGUOUS_SWITCH,
                             "conditioned": CONDITIONED, "envelope_factor": ENVELOPE_FACTOR,
                             "min_conditioned_fraction": MIN_CONDITIONED},
              "numpy": np.__version__, "scipy": scipy.__version__, "tasks": {}}
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(workers) as pool:
        conv = convert_pendulum(Path(tmp))
        for task, (name, rel, disc) in DATASETS.items():
            ds = read_trajectories(Path(tmp) / rel)
            states = np.stack([ep.observations["state"] for ep in ds.episodes]).astype(np.float64)
            stored = np.stack([ep.actions for ep in ds.episodes])[..., 0].astype(np.float64)
            check1 = {f"{w}/{c}": per_state(states, stored, w, c) for w, c in VARIANTS}
            selected = next((v for v in VARIANTS if check1[f"{v[0]}/{v[1]}"]["pass"]), None)
            t = {"dataset_sha256": conv["retained"][name]["dataset"]["sha256"],
                 "check1_per_state": check1,
                 "selected_variant": None if selected is None else f"{selected[0]}/{selected[1]}",
                 "selected_is_primary": selected == expert.PRIMARY}
            if selected is None:
                t["check2_replay"] = {"verdict": "not run: no variant passed check 1"}
            else:
                t["check2_replay"] = replay_check(states, stored, disc, selected, pool)
            t["expert_validated"] = selected is not None and t["check2_replay"]["verdict"] == "pass"
            report["tasks"][task] = t
            print(task, "| check 1:", {k: v["pass"] for k, v in check1.items()}, "| selected:", t["selected_variant"],
                  "| check 2:", t["check2_replay"]["verdict"], flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    report = build(args.workers)
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.check:
        same = OUT.exists() and same_report(json.loads(OUT.read_text()), json.loads(text))
        print("report matches" if same else f"{OUT} differs from a regeneration")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
