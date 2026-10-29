#!/usr/bin/env python3
"""The corrected-demonstrator study's generation steps (docs/experiments/pendulum-corrected-demonstrator-predeclaration.md).

    R="uv run --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/corrected_demos.py"
    $R accept                      # the controller acceptance check (odd symmetry about upright)
    $R generate --out DIR          # corrected datasets, transition replay, demonstrator report
    $R replay --data DIR           # the transition replay of DIR's datasets on this host (cross-host check)

accept writes acceptance.json; generate refuses an existing DIR and, unless the acceptance check passes, does nothing.
It writes DIR/pendulum_{cont,disc}_corrected.npz (ccil.trajectories 1.0) and generation.json, both reports under
docs/fixtures/evidence/pendulum-corrected-demonstrator/. Datasets are recorded by sha256, not committed.

Action data contract: the controller computes and clamps in float64 (tools/pendulum_eval/expert.py, the primary
variant: one wrapped angle for the window and the action); that value is cast once to float32, and the float32 value
is both applied (tools/m3_pendulum/env.step) and stored. The pre-cast value is a diagnostic in the report only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum",
          REPO / "tools" / "legacy_conversion"):
    sys.path.insert(0, str(p))
import env as scalar  # noqa: E402  (tools/m3_pendulum)
import expert  # noqa: E402
import profiles  # noqa: E402
from convert_pendulum import convert_pendulum  # noqa: E402
from expert_alignment import CHECKS, DATASETS, zero_rollout  # noqa: E402

from ccil.schema.io import read_trajectories, write_trajectories  # noqa: E402
from ccil.schema.trajectories import Episode, TrajectoryDataset  # noqa: E402

EVIDENCE = REPO / "docs" / "fixtures" / "evidence" / "pendulum-corrected-demonstrator"
NAMES = {"continuous": "pendulum_cont_corrected.npz", "discontinuous": "pendulum_disc_corrected.npz"}
SYMMETRY_TOL, CROSS_HOST_TOL = 1e-12, 1e-12  # frozen in the predeclaration
SWITCH_GUARD = 1e-9  # grid points this close to the 0.1 switch are left out: either branch is legitimate there


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inconsistent_act(states: np.ndarray) -> np.ndarray:
    """The historical generator's rule (expert_diagnosis.py): window on the wrapped angle, LQR on the raw angle."""
    b = expert.branches(states)
    raw = expert.angle(states, "minus_pi_to_pi")
    lqr = -expert.K_ANGLE * (raw - np.pi) - expert.K_RATE * states[..., 2]
    return np.clip(np.where(b["in_lqr"], lqr, b["energy"]), -expert.ACTION_BOUND, expert.ACTION_BOUND)


def accept() -> dict:
    deltas = np.linspace(-0.2, 0.2, 401)
    deltas = deltas[np.abs(np.abs(deltas) - expert.SWITCH) > SWITCH_GUARD]
    rates = np.linspace(-3.0, 3.0, 121)
    d, w = np.meshgrid(deltas, rates, indexing="ij")
    plus = np.stack([np.sin(np.pi + d), np.cos(np.pi + d), -w], axis=-1)   # (pi + delta, -theta_dot)
    minus = np.stack([np.sin(np.pi - d), np.cos(np.pi - d), w], axis=-1)   # (pi - delta, theta_dot)
    out = {"grid": {"delta": [-0.2, 0.2, len(deltas)], "theta_dot": [-3.0, 3.0, len(rates)],
                    "switch_guard": SWITCH_GUARD}, "tolerance": SYMMETRY_TOL}
    for name, f in (("corrected", expert.act), ("historical_rule", inconsistent_act)):
        err = np.abs(f(plus) + f(minus))
        out[name] = {"max_asymmetry": float(err.max()), "violations": int((err > SYMMETRY_TOL).sum()),
                     "pass": bool((err <= SYMMETRY_TOL).all())}
    out["pass"] = out["corrected"]["pass"]
    return out


def generate_episode(s0: np.ndarray, T: int, disc: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T state frames and T actions: stored float32 actions, applied as stored; the pre-cast float64 actions."""
    states, stored, precast = np.empty((T, 3)), np.empty(T, np.float32), np.empty(T)
    states[0] = s0
    for t in range(T):
        precast[t] = float(expert.act(states[t]))
        stored[t] = np.float32(precast[t])
        if t + 1 < T:
            states[t + 1], _ = scalar.step(states[t].copy(), float(stored[t]), disc)
    return states, stored, precast


def replay(states: np.ndarray, actions: np.ndarray, disc: bool) -> float:
    """Largest absolute difference between each stored next state and the step from the stored (s_t, a_t)."""
    worst = 0.0
    for e in range(len(states)):
        for t in range(states.shape[1] - 1):
            nxt, _ = scalar.step(states[e, t].copy(), float(actions[e, t]), disc)
            worst = max(worst, float(np.abs(nxt - states[e, t + 1]).max()))
    return worst


def summary(states: np.ndarray, actions: np.ndarray) -> tuple[dict, dict]:
    per = profiles.pendulum_swingup_v2(states, actions[..., None].astype(np.float64))
    return per, profiles.summarize(per, states, actions[..., None].astype(np.float64))


def demonstrator_report(hist_s, hist_a, corr_s, corr_a, disc: bool) -> dict:
    zero_s = zero_rollout(hist_s[:, 0], hist_s.shape[1], disc)
    zero_a = np.zeros(hist_a.shape)
    res, per = {}, {}
    for name, s, a in (("historical", hist_s, hist_a), ("corrected", corr_s, corr_a), ("zero", zero_s, zero_a)):
        per[name], res[name] = summary(s, a)
    checks = {}
    for cname, src, metric, op, thr in CHECKS:
        if src == "zero":
            continue
        for demo in ("historical", "corrected"):
            v = res[demo]["mean_normalized_height"] - res["zero"]["mean_normalized_height"] \
                if src == "difference" else res[demo][metric]
            checks[f"{demo}/{cname}"] = {"value": v, "required": f"{op} {thr}",
                                         "pass": bool({"<=": v <= thr, ">=": v >= thr, "==": v == thr}[op])}
    paired = {}
    for m in ("upright_occupancy", "normalized_height"):
        d = per["corrected"][m] - per["historical"][m]
        paired[m] = {"mean_difference": float(d.mean()), "episodes_higher": int((d > 0).sum()),
                     "episodes_lower": int((d < 0).sum()), "episodes_equal": int((d == 0).sum())}
    hold = (per["corrected"]["final_hold"].astype(int) - per["historical"]["final_hold"].astype(int))
    paired["final_hold"] = {"gained": int((hold > 0).sum()), "lost": int((hold < 0).sum())}
    return {"label": "descriptive; same starts; not gating", "summaries": res, "gate_checks": checks,
            "paired_by_episode": paired}


def generate(out: Path) -> dict:
    acc = accept()
    if not acc["pass"]:
        raise SystemExit("generation refused: the controller acceptance check failed")
    out.mkdir(parents=True)
    report = {"acceptance_pass": True, "numpy": np.__version__, "scipy": scipy.__version__, "tasks": {}}
    with tempfile.TemporaryDirectory() as tmp:
        conv = convert_pendulum(Path(tmp))
        for task, (name, rel, disc) in DATASETS.items():
            hist = read_trajectories(Path(tmp) / rel)
            hist_s = np.stack([ep.observations["state"] for ep in hist.episodes]).astype(np.float64)
            hist_a = np.stack([ep.actions for ep in hist.episodes])[..., 0]
            E, T = hist_s.shape[:2]
            gen = [generate_episode(hist_s[e, 0], T, disc) for e in range(E)]
            corr_s = np.stack([g[0] for g in gen])
            corr_a = np.stack([g[1] for g in gen])
            precast = np.stack([g[2] for g in gen])
            ds = TrajectoryDataset(
                [Episode(episode_id=ep.episode_id, observations={"state": corr_s[e]}, actions=corr_a[e][:, None],
                         terminated=False) for e, ep in enumerate(hist.episodes)],
                {"corrected_demonstrator": {"predeclaration": "docs/experiments/pendulum-corrected-demonstrator-"
                                                              "predeclaration.md",
                                            "initial_states_from": conv["retained"][name]["dataset"]["sha256"],
                                            "controller": "tools/pendulum_eval/expert.py act() defaults",
                                            "action_contract": "float64 clamp, cast once to float32, applied and "
                                                               "stored"}}).validate()
            path = out / NAMES[task]
            write_trajectories(ds, path)
            back = read_trajectories(path)
            s_back = np.stack([ep.observations["state"] for ep in back.episodes])
            a_back = np.stack([ep.actions for ep in back.episodes])[..., 0]
            report["tasks"][task] = {
                "dataset": NAMES[task], "sha256": sha(path), "episodes": E, "frames": T,
                "historical_sha256": conv["retained"][name]["dataset"]["sha256"],
                "read_back_equal": bool(np.array_equal(s_back, corr_s) and np.array_equal(a_back, corr_a)),
                "initial_states_equal_historical": bool(np.array_equal(corr_s[:, 0], hist_s[:, 0])),
                "precast_minus_applied_max_abs": float(np.abs(precast - corr_a.astype(np.float64)).max()),
                "transition_replay_max_abs": replay(s_back, a_back, disc),
                "demonstrator": demonstrator_report(hist_s, hist_a, corr_s, corr_a, disc)}
            t = report["tasks"][task]
            t["transition_replay_pass_generating_host"] = t["transition_replay_max_abs"] == 0.0
            print(task, {k: v for k, v in t.items() if k != "demonstrator"}, flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("accept", "generate", "replay"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path)
    args = ap.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.step == "accept":
        rep, dest = accept(), EVIDENCE / "acceptance.json"
    elif args.step == "generate":
        if args.out is None:
            ap.error("generate needs --out")
        rep, dest = generate(args.out), EVIDENCE / "generation.json"
    else:
        if args.data is None:
            ap.error("replay needs --data")
        rep = {"tolerance": CROSS_HOST_TOL, "numpy": np.__version__, "scipy": scipy.__version__, "tasks": {}}
        for task, (_, _, disc) in DATASETS.items():
            ds = read_trajectories(args.data / NAMES[task])
            s = np.stack([ep.observations["state"] for ep in ds.episodes])
            a = np.stack([ep.actions for ep in ds.episodes])[..., 0]
            worst = replay(s, a, disc)
            rep["tasks"][task] = {"sha256": sha(args.data / NAMES[task]), "transition_replay_max_abs": worst,
                                  "pass": worst <= CROSS_HOST_TOL}
        dest = EVIDENCE / "replay-cross-host.json"
    dest.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep if args.step != "generate" else {"wrote": str(dest)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
