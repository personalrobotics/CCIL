#!/usr/bin/env python3
"""Evidence for the batched Pendulum environment (#111).

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/vector_env/check.py --out DIR

1. Transitions: every stored transition of both retained datasets (#23
   conversion) reproduced within 1e-6, with the canonical number of
   discontinuity crossings.
2. One-step equivalence: 3,000 random states and actions (including actions
   far outside the data), compared with the canonical scipy step: max |Δ| at
   most 1e-12.
3. Rollouts: zero, linear and a fixed tanh-MLP policy on both tasks, noise 0
   and 0.1, seed 9001, compared with the canonical environment's rollout:
   every state within 1e-12, R within 1e-9, identical discontinuity crossings.
4. Audit compatibility: the batched rollouts pass the #24 audit (rewards,
   returns, R, horizon, initial states, noise; replay off), and a batched
   replay reproduces every transition bit for bit.
5. A failing episode: a policy that returns a non-finite action for episode 7 at step 200 (otherwise the linear
   policy), on both tasks at noise 0 and 0.1. The batched rollout equals the canonical one; only episode 7 is a
   recorded failure (the same NaN positions, states within 1e-12, as in 3); its prefix equals the rollout without the failure and every later step, and its return,
   are NaN; every other episode is bit-identical to the rollout without the failure; and the saved rollout
   passes the independent audit as a recorded algorithmic failure, with the batched replay.
6. Gymnasium adapter: driving the VectorEnv with the linear policy gives the
   batched rollout's states and rewards exactly, truncates at 500 steps, and
   refuses a reset without a seed.
7. Timing, reported: one rollout and the 300k-transition replay, canonical
   against batched.
Writes DIR/report.json and DIR/rollouts/.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
sys.path.insert(0, str(REPO / "tools" / "legacy_conversion"))
sys.path.insert(0, str(HERE))
import audit as auditor  # noqa: E402
import env as canonical  # noqa: E402
import pendulum as vp  # noqa: E402
from convert_pendulum import convert_pendulum  # noqa: E402

from ccil.schema import read_trajectories  # noqa: E402

SEED = 9001
W1 = np.random.default_rng(1).normal(size=(3, 64))
W2 = np.random.default_rng(2).normal(size=(64, 1)) / 4
POLICIES = {"zero": lambda o: np.zeros((len(o), 1)), "linear": lambda o: o @ np.array([[-2.0], [1.0], [-0.5]]),
            "tanh_mlp": lambda o: 3 * np.tanh(np.tanh(o @ W1) @ W2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:700]}", flush=True)

    # 1. Stored transitions.
    convert_pendulum(out / "converted")
    for name, disc, flips in (("pendulum_cont_100", False, 0), ("pendulum_disc_500", True, 650)):
        ds = read_trajectories(out / "converted" / "data" / f"{name}.npz")
        x = np.concatenate([ep.observations["state"][:-1] for ep in ds.episodes]).astype(np.float64)
        u = np.concatenate([ep.actions[:-1, 0] for ep in ds.episodes]).astype(np.float64)
        y = np.concatenate([ep.observations["state"][1:] for ep in ds.episodes])
        t0 = time.time()
        nx, fl = vp.step(x, u, disc)
        sec = time.time() - t0
        err = float(np.abs(nx - y).max())
        record(f"transitions_{name}", err <= 1e-6 and int(fl.sum()) == flips, transitions=len(x), max_abs_error=err,
               flips=int(fl.sum()), canonical_flips=flips, batched_seconds=round(sec, 3))

    # 2. One-step equivalence on random states and actions.
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, 3000)
    y0 = np.stack([np.sin(th), np.cos(th), rng.uniform(-8, 8, 3000)], 1)
    a = np.concatenate([rng.uniform(-3, 3, 2000), rng.uniform(-50, 50, 1000)])
    ref = np.array([canonical.step(y0[i].copy(), float(a[i]), False)[0] for i in range(len(a))])
    d = np.abs(vp.integrate(y0, a) - ref)
    record("one_step_equivalence", d.max() <= 1e-12, samples=len(a), max_abs_difference=float(d.max()),
           exactly_equal_fraction=float((d == 0).all(axis=1).mean()))

    # 3-4. Rollouts against the canonical environment, then the audit and a batched replay.
    rows, audits = [], []
    for task in canonical.TASKS:
        for name, pol in POLICIES.items():
            for n in (0.0, 0.1):
                t0 = time.time()
                ref = canonical.rollout(pol, task, SEED, n)
                t_ref = time.time() - t0
                t0 = time.time()
                got = vp.rollout(pol, task, SEED, n)
                t_got = time.time() - t0
                path = out / "rollouts" / f"{task}-{name}-noise{n}.npz"
                canonical.save(got, path)
                au = auditor.audit(path, replay=False)
                rows.append({"task": task, "policy": name, "noise": n,
                             "max_state_difference": float(np.nanmax(np.abs(ref["states"] - got["states"]))),
                             "R_difference": abs(float(ref["R"]) - float(got["R"])),
                             "flips": [int(ref["flipped"].sum()), int(got["flipped"].sum())],
                             "canonical_seconds": round(t_ref, 2), "batched_seconds": round(t_got, 3)})
                audits.append({"rollout": path.name, "audit_ok": au["ok"], "issues": au["issues"],
                               "replay_mismatches": vp.replay_mismatches(got, task)})
    record("rollout_equivalence", all(r["max_state_difference"] <= 1e-12 and r["R_difference"] <= 1e-9
                                      and r["flips"][0] == r["flips"][1] for r in rows), rollouts=rows)
    record("audit_and_batched_replay", all(x["audit_ok"] and x["replay_mismatches"] == 0 for x in audits), audits=audits)

    # 5. A non-finite action for one episode partway through.
    FAIL_EP, FAIL_STEP = 7, 200

    def failing():
        calls = [0]

        def policy(o):
            act = POLICIES["linear"](o)
            if calls[0] == FAIL_STEP:  # every episode is alive until now, so row FAIL_EP is episode FAIL_EP
                act[FAIL_EP] = np.nan
            calls[0] += 1
            return act
        return policy
    rows = []
    for task in canonical.TASKS:
        for n in (0.0, 0.1):
            ref = canonical.rollout(failing(), task, SEED, n)
            got = vp.rollout(failing(), task, SEED, n)
            clean = vp.rollout(POLICIES["linear"], task, SEED, n)
            path = out / "rollouts" / f"{task}-failing-noise{n}.npz"
            canonical.save(got, path)
            au = auditor.audit(path, replay=False)
            others = np.setdiff1d(np.arange(canonical.EPISODES), [FAIL_EP])
            st, cl = got["states"], clean["states"]
            row = {"task": task, "noise": n,
                   "equals_canonical": bool(np.array_equal(ref["failed_episodes"], got["failed_episodes"])
                                            and all(np.array_equal(np.isnan(ref[k]), np.isnan(got[k]))
                                                    and np.nanmax(np.abs(ref[k] - got[k])) <= 1e-12
                                                    for k in ("states", "actions"))),
                   "failed_episodes": got["failed_episodes"].tolist(),
                   "prefix_equal": bool(np.array_equal(st[FAIL_EP, :FAIL_STEP + 1], cl[FAIL_EP, :FAIL_STEP + 1])),
                   "suffix_nan": bool(np.isnan(st[FAIL_EP, FAIL_STEP + 1:]).all()
                                      and np.isnan(got["actions"][FAIL_EP, FAIL_STEP:]).all()
                                      and np.isnan(got["rewards"][FAIL_EP, FAIL_STEP:]).all()),
                   "return_nan": bool(np.isnan(got["returns"][FAIL_EP]) and np.isnan(got["R"])),
                   "others_unchanged": bool(np.array_equal(st[others], cl[others])
                                            and np.array_equal(got["returns"][others], clean["returns"][others])),
                   "audit_ok": au["ok"], "audit_failed_episodes": au["failed_episodes"],
                   "audit_R_recomputed_nan": bool(np.isnan(au["R_recomputed"])),
                   "replay_mismatches": vp.replay_mismatches(got, task)}
            row["ok"] = (row["equals_canonical"] and row["failed_episodes"] == [FAIL_EP] and row["prefix_equal"]
                         and row["suffix_nan"] and row["return_nan"] and row["others_unchanged"] and row["audit_ok"]
                         and row["audit_failed_episodes"] == [FAIL_EP] and row["audit_R_recomputed_nan"]
                         and row["replay_mismatches"] == 0)
            rows.append(row)
    record("failing_episode", all(r["ok"] for r in rows), episode=FAIL_EP, step=FAIL_STEP, rollouts=rows)

    # 6. The Gymnasium adapter.
    venv = vp.vector_env("discontinuous")
    obs, _ = venv.reset(seed=SEED)
    states, rewards, trunc = [obs], [], None
    for _ in range(canonical.HORIZON):
        obs, r, term, trunc, _ = venv.step(POLICIES["linear"](obs)[:, 0])
        states.append(obs)
        rewards.append(r)
    ref = vp.rollout(POLICIES["linear"], "discontinuous", SEED)
    same = (np.array_equal(np.stack(states, 1), ref["states"]) and np.array_equal(np.stack(rewards, 1), ref["rewards"])
            and bool(trunc.all()) and not bool(term.any()))
    try:
        venv.reset()
        refuses = False
    except ValueError:
        refuses = True
    record("gymnasium_adapter", same and refuses, matches_batched_rollout=same, truncated_at_horizon=bool(trunc.all()),
           reset_without_seed_refused=refuses, autoreset_mode=str(venv.metadata["autoreset_mode"]))

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    import gymnasium
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": scipy.__version__, "gymnasium": gymnasium.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
