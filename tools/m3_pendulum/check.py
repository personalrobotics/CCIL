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
   dropped step, a changed noise draw and a changed R are each refused, and
   so are changes inside a failed episode's finite prefix or after its
   failure.
4. Failures are results: a policy that emits NaN in one episode, or in every
   episode at step 1, gives a non-finite R with the failed episodes recorded,
   and the audit agrees.
5. Repeatability: a second evaluation is identical array for array.
6. The code-state gate (provenance.py): the tree is clean and src,
   pyproject.toml and uv.lock equal the frozen ccil commit; on a scratch
   clone, an uncommitted src change and an untracked file are each refused.
   This is the version 4 freeze (tools/m3_pendulum/freeze.json). With
   --v6-freeze (tools/m3_experiments/freeze_evidence.sh) its two cases are
   recorded as skipped, not passed: the version 6 code-state gate is checked by
   tools/m3_experiments, and every other case still runs.
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
import provenance  # noqa: E402
from convert_pendulum import convert_pendulum  # noqa: E402

from ccil.schema import read_trajectories  # noqa: E402

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
    ap.add_argument("--v6-freeze", action="store_true",
                    help="skip the two version 4 freeze-gate cases (recorded as skipped); the version 6 gate is "
                         "checked by tools/m3_experiments")
    args = ap.parse_args()
    out = args.out.resolve()
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

    # 3. The audit refuses tampering, including inside a failed episode's finite prefix.
    def nan_policy(at, episodes):
        calls = [0]

        def policy(o):
            calls[0] += 1
            a = np.zeros((len(o), 1))
            if calls[0] == at:
                a[episodes] = np.nan
            return a
        return policy

    sources = {}
    for label, task, n, policy in (("linear-discontinuous-noise0.1", "discontinuous", 0.1, POLICIES["linear"]),
                                   ("nan-ep3-step9-continuous-noise0.0", "continuous", 0.0, nan_policy(10, [3])),
                                   ("nan-all-step1-continuous-noise0.0", "continuous", 0.0, nan_policy(2, slice(None)))):
        path = out / "rollouts" / f"{label}.npz"
        if not path.exists():
            pendulum.save(pendulum.rollout(policy, task, SEED, n), path)
        with np.load(path, allow_pickle=False) as z:
            sources[label] = {k: z[k] for k in z.files}

    def tampered(label, source, edit):
        r = {k: v.copy() for k, v in sources[source].items()}
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

    def add(key, index, delta):
        return lambda r: r[key].__setitem__(index, r[key][index] + delta)

    clean, ep3, all1 = ("linear-discontinuous-noise0.1", "nan-ep3-step9-continuous-noise0.0",
                        "nan-all-step1-continuous-noise0.0")
    edits = {"reward_plus_1e-6": (clean, add("rewards", (5, 100), 1e-6)),
             "state_moved": (clean, add("states", (7, 250, 2), 1e-9)),
             "step_dropped": (clean, drop_step),
             "noise_draw_changed": (clean, lambda r: r["act_noise"].__setitem__((0, 0, 0), 0.0)),
             "R_changed": (clean, lambda r: r.__setitem__("R", r["R"] + 1e-6)),
             "failed_episode_prefix_reward": (ep3, add("rewards", (3, 5), 1.0)),
             "failed_episode_prefix_state": (ep3, add("states", (3, 4, 0), 1e-9)),
             "failed_episode_value_after_failure": (ep3, lambda r: r["rewards"].__setitem__((3, 20), -1.0)),
             "all_failed_first_step_reward": (all1, add("rewards", (0, 0), 1.0))}
    refusals = {k: tampered(k, src, f) for k, (src, f) in edits.items()}
    record("audit_refuses_tampering", all(v["refused"] for v in refusals.values()), cases=refusals)

    # 4. A non-finite action is a recorded failure.
    failures = {}
    for label, expected in ((ep3, [3]), (all1, list(range(pendulum.EPISODES)))):
        a = auditor.audit(out / "rollouts" / f"{label}.npz")
        failures[label] = {"audit_ok": a["ok"], "failed_episodes": a["failed_episodes"], "R": str(a["R"]),
                           "pass": a["ok"] and a["failed_episodes"] == expected and not np.isfinite(a["R"])}
    record("nonfinite_action_is_a_failure", all(f["pass"] for f in failures.values()), rollouts=failures)

    # 5. Repeatability.
    again = pendulum.rollout(POLICIES["linear"], "discontinuous", SEED, 0.1)
    same = all(np.array_equal(again[k], sources[clean][k], equal_nan=True) for k in sources[clean] if k != "meta")
    record("repeatable", same, compared=sorted(k for k in sources[clean] if k != "meta"))

    # 6. The code-state gate: this tree, then a scratch clone made dirty, which the gate must refuse.
    state = provenance.tree_state(REPO)
    if args.v6_freeze:
        reason = "version 4 freeze gate (ccil 847b060), superseded by the version 6 gate (tools/m3_experiments)"
        for name in ("clean_tree_at_frozen_ccil", "gate_refuses_a_dirty_tree"):
            cases.append({"case": name, "skipped": reason})
            print(f"SKIP {name} {reason}", flush=True)
        return _report(out, cases, state)
    record("clean_tree_at_frozen_ccil", state["ok"], **{k: v for k, v in state.items() if k != "ok"})
    probe = out / "probe"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(REPO), str(probe)], check=True)
    subprocess.run(["git", "-C", str(probe), "checkout", "-q", state["head"]], check=True)
    probes = {"clean_clone": provenance.tree_state(probe)["ok"]}
    init = probe / "src" / "ccil" / "__init__.py"
    original = init.read_bytes()
    init.write_bytes(original + b"\n")
    probes["uncommitted_src_change"] = provenance.tree_state(probe)["ok"]
    init.write_bytes(original)
    (probe / "tools" / "untracked.txt").write_text("x")
    probes["untracked_file"] = provenance.tree_state(probe)["ok"]
    record("gate_refuses_a_dirty_tree", probes == {"clean_clone": True, "uncommitted_src_change": False,
                                                   "untracked_file": False}, gate_ok=probes)
    return _report(out, cases, state)


def _report(out, cases, state) -> int:
    """Write DIR/report.json. all_pass covers the cases that ran; skipped cases are listed, never counted as passed."""
    ran = [c for c in cases if "skipped" not in c]
    skipped = [c["case"] for c in cases if "skipped" in c]
    report = {"all_pass": all(c["pass"] for c in ran), "cases": cases,
              "environment": {"code_commit": state["head"], "code_dirty": bool(state["dirty"]),
                              "env_version": pendulum.VERSION, "python": platform.python_version(),
                              "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__}}
    if skipped:
        report["skipped"] = skipped
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    if skipped:
        print(f"{sum(c['pass'] for c in ran)}/{len(ran)} applicable cases pass; {len(skipped)} skipped "
              f"({', '.join(skipped)}); report: {out / 'report.json'}")
    else:
        print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
