#!/usr/bin/env python3
"""Evidence for the compiled engines (#134): no step runs uncompiled when compilation was requested.

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_compile/check.py --out DIR [--device cuda] [--preflight-root ROOT]

Every case runs in a fresh process, so each starts with torch.compile's caches empty. On a host without CUDA the
steps are compiled for the CPU (compile=True), which exercises the same guard.

1. policy_step: tools/m3_compile/workload.py trains seven groups of one bucket whose tables, learning rates,
   NoiseBC and recipes differ, plus a streaming group, with chunks of 5 steps. The last chunk of every epoch is
   shorter, which is the stride that recompiled before #134. It runs against this tree and against the source
   before #134 (the parent of the commit that introduced the step-major layout):
   - every step's row indices and NoiseBC draws are identical in all runs;
   - eager: this tree's parameters and losses equal the pre-#134 engine's, bit for bit;
   - compiled: this tree's parameters and losses do not depend on the chunking (5 or 1000 steps per chunk), and
     equal the pre-#134 engine's whenever that engine ran one specialization per group (1000 steps per chunk).
     The pre-#134 engine's own dependence on the chunking is reported;
   - this tree compiles one specialization per data mode, NoiseBC flag and recipe. Groups that differ only in
     table size or learning rate reuse it, and every record states the requested and effective engine and the
     specialization counts.
2. refusal: a deliberately excessive sequence of shapes, nine policy groups with distinct recipes and then nine
   fit groups with distinct learning rates. The first eight of each compile, and their records say so with
   specializations 1 to 8. The ninth raises CompilationRefused on its first step, before the step runs, and
   returns nothing.
   Control (the same sequence with the guard switched off after the engine is built): the ninth group returns
   records claiming a compiled engine with no new specialization, because it ran uncompiled. That is the
   failure the guard prevents.
3. runner_refusal: ccil.experiments.runner on a small smoke plan whose two fit groups differ in their settings,
   with torch.compile's recompile limit set to 1, so the second fit group is refused. The phase stops with
   CompilationRefused. No unit record exists, no labels or policy stage completed, and no completed
   transition.fit stage holds a model of the refused group (by fit identity).
4. v6_schedule (with --preflight-root, a root of `run_study.py ... --preflight` for every step and shard): every
   fit and policy group of the exact v6 schedule ran with its effective engine equal to the requested one,
   compiled; the largest specialization count per step function is reported against the limit. A compilation
   refusal there would have stopped the preflight, so every record existing and verified is the claim.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKLOAD = HERE / "workload.py"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def _base() -> str:
    """The commit before #134: the parent of the commit that introduced the step-major chunk layout."""
    first = _git("log", "--format=%H", "-S", "_step_major", "--", "src/ccil/training/vector_bc_v2.py").splitlines()
    return _git("rev-parse", f"{first[-1]}^")


def _sub(*args, env=None, timeout=3600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def _json(p: subprocess.CompletedProcess) -> dict:
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {p.stderr[-3000:]}")
    return json.loads(p.stdout)


def policy_step(device: str, out: Path) -> dict:
    base = _base()
    src = Path(tempfile.mkdtemp(prefix="ccil-base-", dir=out))
    subprocess.run(f"git -C {REPO} archive {base} src | tar -x -C {src}", shell=True, check=True)
    runs = {}
    for tree, env in (("this", {}), ("base", {"PYTHONPATH": str(src / "src")})):
        for mode, extra in (("eager", ["--compile", "0"]), ("compiled_5", ["--compile", "1", "--chunk-steps", "5"]),
                            ("compiled_1000", ["--compile", "1", "--chunk-steps", "1000"])):
            runs[f"{tree}_{mode}"] = _json(_sub(str(WORKLOAD), "--device", device, *extra, env=env))
            (out / f"workload-{tree}-{mode}.json").write_text(json.dumps(runs[f"{tree}_{mode}"], indent=1))
    imported = {k: r["ccil"] for k, r in runs.items()}
    groups = [g["name"] for g in runs["this_eager"]["groups"]]

    def col(run, field):
        return [g[field] for g in runs[run]["groups"]]

    def same(a, b, field="parameters_sha256"):
        return col(a, field) == col(b, field) and col(a, "epoch_losses") == col(b, "epoch_losses")
    draws = all(col(k, "rows_and_draws_sha256") == col("this_eager", "rows_and_draws_sha256") for k in runs)
    eager = same("this_eager", "base_eager")
    chunking = same("this_compiled_5", "this_compiled_1000")
    vs_base = same("this_compiled_5", "base_compiled_1000")
    base_chunking = [a == b for a, b in zip(col("base_compiled_5", "parameters_sha256"),
                                            col("base_compiled_1000", "parameters_sha256"))]
    comp = col("this_compiled_5", "compilation")
    # One specialization per (data mode, NoiseBC, recipe): bc_a, noise_bc, modern_a and the streaming step are new.
    expected_new = [1, 0, 0, 1, 1, 0, 1]
    records = (all(c["requested"] == c["effective"] == "compiled (default)" for c in comp)
               and [c["new_specializations"] for c in comp] == expected_new
               and all(isinstance(c["specializations"], int) for c in comp)
               and all(c["requested"] == c["effective"] == "eager" for c in col("this_eager", "compilation")))
    imports_ok = (all(v.startswith(str(src)) for k, v in imported.items() if k.startswith("base"))
                  and all(v.startswith(str(REPO)) for k, v in imported.items() if k.startswith("this")))
    shutil.rmtree(src)  # the extracted pre-#134 source: its commit is recorded
    return {"pass": draws and eager and chunking and vs_base and records and imports_ok,
            "base_commit": base, "groups": groups, "imported_from": imported, "imports_ok": imports_ok,
            "rows_and_draws_identical": draws, "eager_identical_to_base": eager,
            "compiled_independent_of_chunking": chunking, "compiled_identical_to_base_one_specialization": vs_base,
            "base_compiled_independent_of_chunking_per_group": dict(zip(groups, base_chunking)),
            "compilation": dict(zip(groups, comp)), "expected_new_specializations": dict(zip(groups, expected_new)),
            "records_ok": records, "steps": dict(zip(groups, col("this_eager", "steps")))}


REFUSAL = r'''
import json, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from workload import supervision
from ccil.training import vector_bc_v2 as v2
from ccil.models.mlp_batched import FitJob, fit_group
device, control = sys.argv[2], sys.argv[3] == "control"
if control:  # the guard switched off after each engine is built: torch.compile's own behaviour past the limit
    real_init = v2._Engine.__init__
    def init(self, *a, **k):
        real_init(self, *a, **k)
        import torch._dynamo
        torch._dynamo.config.fail_on_recompile_limit_hit = False
    v2._Engine.__init__ = init
calls = []
real_call = v2._Engine.__call__
def counted(self, *a, **k):
    calls.append(self.eager.__qualname__)
    return real_call(self, *a, **k)
v2._Engine.__call__ = counted
BASE = {"hidden_layers": [16], "learning_rate": 1e-3, "batch_size": 16, "epochs": 1,
        "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None}
FIT = {"hidden_layers": [16], "activation": "relu", "lipschitz_type": "slack", "lipschitz_constraint": 1.0,
       "soft_penalty_weight": 1.0, "soft_sampling_eps": 0.1, "soft_samples": 2, "slack_weight": 1.0,
       "learning_rate": 1e-3, "weight_decay": 1e-5, "epochs": 1, "batch_size": 16}
r = np.random.default_rng(0)
z, a = r.normal(size=(64, 2)), r.normal(size=(64, 1))
zn = z + 0.1 * np.tanh(a)
out = {"policy": [], "fit": []}
for kind in ("policy", "fit"):
    for i in range(9):
        before = len(calls)
        try:
            if kind == "policy":  # grad_clip_norm is compiled in: each value is a specialization
                res = v2.train_group([v2.Job(supervision(64, i, "bc"), 1)], {**BASE, "grad_clip_norm": 1.0 + i},
                                     device=device, compile=True)
            else:  # so is the fit's learning rate (the fit step is unchanged by #134)
                res = fit_group([FitJob(z, a, zn, 1)], {**FIT, "learning_rate": 1e-3 * (i + 1)}, device=device,
                                compile=True)
            out[kind].append({"group": i, "returned": True, "compilation": res[0][1]["execution"]["compilation"],
                              "engine_calls": len(calls) - before})
        except v2.CompilationRefused as e:
            out[kind].append({"group": i, "returned": False, "refused": str(e), "engine_calls": len(calls) - before})
print(json.dumps(out))
'''


def refusal(device: str) -> dict:
    res = {}
    for mode in ("guard", "control"):
        p = _sub("-c", REFUSAL, str(HERE), device, mode)
        res[mode] = _json(p)
    ok = True
    for kind in ("policy", "fit"):
        g, c = res["guard"][kind], res["control"][kind]
        first8 = all(x["returned"] and x["compilation"]["requested"] == x["compilation"]["effective"]
                     == "compiled (default)" and x["compilation"]["specializations"] == x["group"] + 1
                     and x["compilation"]["new_specializations"] == 1 for x in g[:8])
        ninth = (not g[8]["returned"] and g[8]["engine_calls"] == 1 and "refused before it ran" in g[8]["refused"])
        silent = (c[8]["returned"] and c[8]["compilation"]["effective"] == "compiled (default)"
                  and c[8]["compilation"]["new_specializations"] == 0 and c[8]["compilation"]["specializations"] == 8)
        res[f"{kind}_first_eight_compiled"], res[f"{kind}_ninth_refused_on_first_step"] = first8, ninth
        res[f"{kind}_control_ninth_ran_uncompiled_claiming_compiled"] = silent
        ok = ok and first8 and ninth and silent
    res["pass"] = ok
    return res


RUNNER = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch, torch._dynamo
torch._dynamo.config.recompile_limit = 1
from pendulum_suite import PendulumSuite
from ccil.experiments import plan, runner
from ccil.models import mlp_batched
from ccil.training.vector_bc_v2 import CompilationRefused
root, device = Path(sys.argv[2]), sys.argv[3]
study = plan.from_dict({
    "study": "compile-check", "version": 1, "tasks": ["continuous"],
    "settings": ["transition_model.trainer_version=2", "policy.trainer_version=2",
                 "policy.exclude_validation_episodes=false"],
    "phases": {"study": {"seeds": [9002], "arms": ["ccil", "ccil_nolip"]}},
    "arms": {"ccil": {"settings": ["policy.method=ccil"]},
             "ccil_nolip": {"settings": ["policy.method=ccil", "transition_model.lipschitz_type=none"]}}})
groups = []
real_fit = mlp_batched.fit_group
def fit(jobs, settings, **k):  # compiled on any device; each group's fit identities, to find their stages
    ids = [mlp_batched.fit_identity(j.z, j.a, j.z_next, settings, j.seed) for j in jobs]
    groups.append({"fits": len(jobs), "fit_identities": ids, "outcome": "running"})
    try:
        out = real_fit(jobs, settings, **{**k, "compile": True})
    except CompilationRefused as e:
        groups[-1]["outcome"] = f"refused: {e}"
        raise
    groups[-1]["outcome"] = "returned"
    return out
mlp_batched.fit_group = fit
try:
    runner.run_phase(root, study, "study", PendulumSuite(root, study, smoke=True),
                     gate={"ok": True, "real": False, "head": "check", "tooling": {}}, smoke=True, device=device,
                     log=lambda m: None)
    stopped = None
except CompilationRefused as e:
    stopped = f"CompilationRefused: {e}"
print(json.dumps({"stopped": stopped, "groups": groups}))
'''


def _find(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find(v, key)


def _complete_stages(root: Path) -> dict:
    from ccil.runs.validate import validate_run
    out: dict = {}
    for st in root.rglob("stage.json"):
        if validate_run(st.parent)[0] == "complete":
            out.setdefault(json.loads(st.read_text())["stage"], []).append(st.parent)
    return out


def runner_refusal(device: str, out: Path) -> dict:
    root = out / "runner-refusal"
    tools = [str(REPO / "tools" / "m3_experiments"), str(REPO / "tools" / "m3_pendulum"),
             str(REPO / "tools" / "vector_env")]
    p = _sub("-c", RUNNER, tools[0], str(root), device,
             env={"PYTHONPATH": os.pathsep.join(tools + [os.environ.get("PYTHONPATH", "")])})
    res = _json(p)
    stages = _complete_stages(root)
    records = sorted(str(q.relative_to(root)) for q in (root / "results").rglob("*.json")) if (root / "results").exists() else []
    refused = [g for g in res["groups"] if g["outcome"].startswith("refused")]
    returned = [g for g in res["groups"] if g["outcome"] == "returned"]
    fit_ids = set()
    for d in stages.get("transition.fit", []):
        fit_ids |= set(_find(json.loads((d / "metrics.json").read_text()), "fit_identity_sha256"))
    leaked = sorted(set(i for g in refused for i in g["fit_identities"]) & fit_ids)
    later = {k: len(v) for k, v in stages.items() if k in ("labels.generate", "labels.certify", "policy.train",
                                                           "policy.evaluate")}
    res.update({"complete_stages": {k: len(v) for k, v in stages.items()}, "records": records,
                "refused_fits_with_a_complete_stage": leaked})
    res["pass"] = (bool(res["stopped"]) and res["stopped"].startswith("CompilationRefused") and len(refused) == 1
                   and len(returned) >= 1 and not records and not leaked and not any(later.values())
                   and len(stages.get("transition.fit", [])) >= 1)
    return res


def v6_schedule(root: Path) -> dict:
    comps = {"fit": [], "policy": []}
    for st in root.rglob("stage.json"):
        stage = json.loads(st.read_text())["stage"]
        if stage not in ("transition.fit", "policy.train"):
            continue
        m = json.loads((st.parent / "metrics.json").read_text())
        for g in _find(m, "group"):
            if isinstance(g, dict) and "compilation" in g:
                comps["fit" if stage == "transition.fit" else "policy"].append(g["compilation"])
    import torch._dynamo
    limit = torch._dynamo.config.recompile_limit
    summary = {}
    for kind, cs in comps.items():
        summary[kind] = {"group_records": len(cs),
                         "effective_equals_requested_compiled": all(c["requested"] == c["effective"]
                                                                    == "compiled (default)" for c in cs),
                         "max_specializations": max((c["specializations"] for c in cs), default=None),
                         "recompile_limit": limit}
    results = list((root / "results").rglob("*.json")) if (root / "results").exists() else []
    ok = all(s["group_records"] > 0 and s["effective_equals_requested_compiled"]
             and s["max_specializations"] <= limit for s in summary.values()) and bool(results)
    return {"pass": ok, "root": str(root), "unit_records": len(results), **summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight-root", type=Path)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    report = {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
              "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
              "device": a.device, "cases": {}}
    if a.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    cases = {"policy_step": lambda: policy_step(a.device, a.out), "refusal": lambda: refusal(a.device),
             "runner_refusal": lambda: runner_refusal(a.device, a.out)}
    if a.preflight_root:
        cases["v6_schedule"] = lambda: v6_schedule(a.preflight_root.resolve())
    for name, fn in cases.items():
        try:
            report["cases"][name] = fn()
        except Exception as e:  # noqa: BLE001 - a case that cannot run is a failed case, reported
            report["cases"][name] = {"pass": False, "error": f"{type(e).__name__}: {e}"[-4000:]}
        print(("PASS" if report["cases"][name]["pass"] else "FAIL"), name, flush=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    n = sum(c["pass"] for c in report["cases"].values())
    print(f"{n}/{len(report['cases'])} cases pass; report: {a.out / 'report.json'}")
    return 0 if n == len(report["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
