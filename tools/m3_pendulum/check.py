#!/usr/bin/env python3
"""The #24 acceptance checks (predeclaration §3), run before the tooling freeze.

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_pendulum/check.py --out DIR

They produce no tuning or study outcome: no policy is trained, and the
evaluator runs only fixed, non-learned policies on seed 9001, outside the
tuning (1-3) and study (101-120) seeds.
1. Transitions. Both retained datasets are converted (#23) and must match
   the predeclared SHA-256s; stepping the environment from every stored
   (s_t, a_t) must reproduce the stored s_{t+1} within 1e-6.
2. Evaluator audit. Zero, +1, -1 and a fixed linear controller are evaluated
   on both tasks, the linear controller also at evaluation noise 0.1, and the
   independent audit (audit.py) must reproduce every reward, the horizon,
   every return and R.
3. The audit is not vacuous: a reward changed by 1e-6, a moved state, a
   dropped step, a changed noise draw and a changed R are each refused.
4. Failures are results: a policy that emits NaN in one episode gives a
   non-finite R, recorded as a failed episode, and the audit agrees.
5. Repeatability: a second evaluation is identical array for array.
6. The ccil package is the frozen commit's (src, pyproject.toml, uv.lock).
Writes DIR/report.json, DIR/rollouts/*.npz, and DIR/converted/.
"""
import argparse
import hashlib
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "legacy_conversion"))
import audit as auditor  # noqa: E402
import env as pendulum  # noqa: E402
from convert_pendulum import convert_pendulum  # noqa: E402

from ccil.schema import read_trajectories  # noqa: E402

CCIL_FROZEN = "847b06057e8d36a16d13aa2f3b2e375ceb9147e5"  # predeclaration §2
DATASETS = {"pendulum_cont_100": ("continuous", "a417269b62c3c5b444f9ac71aca5407fef18cf2c04f59a60df73f3234122f962"),
            "pendulum_disc_500": ("discontinuous", None)}  # None: the SHA-256 in the committed conversion.json (§4)
SEED = 9001
POLICIES = {"zero": lambda o: np.zeros((len(o), 1)), "plus_one": lambda o: np.ones((len(o), 1)),
            "minus_one": lambda o: -np.ones((len(o), 1)),
            "linear": lambda o: o @ np.array([[-2.0], [1.0], [-0.5]])}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:900]}", flush=True)

    # 1. Transitions.
    convert_pendulum(out / "converted")
    committed = json.loads((REPO / "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json").read_text())
    for name, (task, expected) in DATASETS.items():
        expected = expected or committed["retained"][name.rsplit("_", 1)[0]]["dataset"]["sha256"]
        path = out / "converted" / "data" / f"{name}.npz"
        t0, errs, flips = time.time(), [], 0
        for ep in read_trajectories(path).episodes:
            x, u = ep.observations["state"], ep.actions
            for t in range(ep.length - 1):
                nxt, f = pendulum.step(x[t].astype(np.float64), float(u[t, 0]), pendulum.TASKS[task])
                errs.append(float(np.abs(nxt - x[t + 1]).max()))
                flips += int(f)
        errs = np.array(errs)
        record(f"transitions_{task}", sha(path) == expected and errs.max() <= 1e-6, dataset=name, sha256=sha(path),
               sha256_expected=expected, transitions=len(errs), max_abs_error=float(errs.max()),
               p99_abs_error=float(np.quantile(errs, 0.99)), over_tolerance=int((errs > 1e-6).sum()),
               discontinuity_flips=flips, seconds=round(time.time() - t0, 1))

    # 2. Evaluator audit with fixed policies.
    runs = [(p, task, 0.0) for p in POLICIES for task in pendulum.TASKS] + [("linear", t, 0.1) for t in pendulum.TASKS]
    audits = {}
    for p, task, n in runs:
        path = out / "rollouts" / f"{p}-{task}-noise{n}.npz"
        pendulum.save(pendulum.rollout(POLICIES[p], task, SEED, n), path)
        audits[path.stem] = auditor.audit(path)
    record("evaluator_audit", all(a["ok"] and not a["failed_episodes"] for a in audits.values()),
           seed=SEED, audits=audits, linear_controller="a = -2 sin θ + cos θ - 0.5 θ̇")

    # 3. The audit refuses tampering.
    base = out / "rollouts" / "linear-discontinuous-noise0.1.npz"
    with np.load(base, allow_pickle=False) as z:
        clean = {k: z[k] for k in z.files}

    def tampered(label, edit):
        r = {k: v.copy() for k, v in clean.items()}
        edit(r)
        p = out / "tampered" / f"{label}.npz"
        p.parent.mkdir(exist_ok=True)
        np.savez(p, **r)
        a = auditor.audit(p)
        return {"refused": not a["ok"], "issues": a["issues"]}

    def drop_step(r):
        meta = json.loads(str(r["meta"]))
        meta["horizon"] = 499
        r.update(states=r["states"][:, :-1], observations=r["observations"][:, :-1], actions=r["actions"][:, :-1],
                 policy_actions=r["policy_actions"][:, :-1], rewards=r["rewards"][:, :-1], flipped=r["flipped"][:, :-1],
                 obs_noise=r["obs_noise"][:, :-1], act_noise=r["act_noise"][:, :-1], meta=np.array(json.dumps(meta)))

    edits = {"reward_plus_1e-6": lambda r: r["rewards"].__setitem__((5, 100), r["rewards"][5, 100] + 1e-6),
             "state_moved": lambda r: r["states"].__setitem__((7, 250, 2), r["states"][7, 250, 2] + 1e-9),
             "step_dropped": drop_step,
             "noise_draw_changed": lambda r: r["act_noise"].__setitem__((0, 0, 0), 0.0),
             "R_changed": lambda r: r.__setitem__("R", r["R"] + 1e-6)}
    refusals = {k: tampered(k, f) for k, f in edits.items()}
    record("audit_refuses_tampering", all(v["refused"] for v in refusals.values()), cases=refusals)

    # 4. A non-finite action is a recorded failure.
    def nan_policy(o, calls=[0]):  # noqa: B006 -- counts steps
        calls[0] += 1
        a = np.zeros((len(o), 1))
        if calls[0] == 10:
            a[3] = np.nan
        return a
    path = out / "rollouts" / "nan-continuous-noise0.0.npz"
    res = pendulum.rollout(nan_policy, "continuous", SEED)
    pendulum.save(res, path)
    a = auditor.audit(path)
    record("nonfinite_action_is_a_failure", a["ok"] and a["failed_episodes"] == [3] and not np.isfinite(res["R"]),
           audit=a, R=str(res["R"]))

    # 5. Repeatability.
    again = pendulum.rollout(POLICIES["linear"], "discontinuous", SEED, 0.1)
    same = all(np.array_equal(again[k], clean[k], equal_nan=True) for k in clean if k != "meta")
    record("repeatable", same, compared=sorted(k for k in clean if k != "meta"))

    # 6. ccil frozen.
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True)  # noqa: E731
    diff = git("diff", "--name-only", CCIL_FROZEN, "HEAD", "--", "src", "pyproject.toml", "uv.lock").stdout.split()
    dirty = git("status", "--porcelain").stdout.strip()
    record("ccil_is_the_frozen_commit", not diff, frozen=CCIL_FROZEN, differing_paths=diff)

    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD").stdout.strip(), "code_dirty": bool(dirty),
                              "env_version": pendulum.VERSION, "python": platform.python_version(),
                              "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
