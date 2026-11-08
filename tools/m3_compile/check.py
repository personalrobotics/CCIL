#!/usr/bin/env python3
"""Evidence for the compiled fit engine (#134): no fit step runs uncompiled unrecorded when compilation was requested.

    uv run --group evidence python tools/m3_compile/check.py --out DIR [--device cuda]

Every case runs in a fresh process, so each starts with torch.compile's caches empty. On a host without CUDA the
steps are compiled for the CPU (compile=True), which exercises the same guard. The policy trainer's cases moved
with it to il-supervision-benchmarks (#168).

1. fit_isolation: nine fit groups with distinct learning rates in one process, more specializations than
   torch.compile's limit of 8. Every group runs compiled. The first eight report specializations 1 to 8, and the
   ninth starts a fresh compiler context (specializations 1).
2. fit_refusal: the fit step's compiler fails on its first step (injected). With the eager fallback off, as a
   study process sets it, fit_group raises CompilationRefused before that step runs and returns nothing. With
   the default fallback, the group returns and its record states the eager engine and the compiler failure.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

FIT = {"hidden_layers": [16], "activation": "relu", "lipschitz_type": "slack", "lipschitz_constraint": 1.0,
       "soft_penalty_weight": 1.0, "soft_sampling_eps": 0.1, "soft_samples": 2, "slack_weight": 1.0,
       "learning_rate": 1e-3, "weight_decay": 1e-5, "epochs": 1, "batch_size": 16}
DATA = f'''
import json, sys
import numpy as np
from ccil.models import engine, mlp_batched
from ccil.models.mlp_batched import FitJob, fit_group
device = sys.argv[1]
FIT = {FIT!r}
r = np.random.default_rng(0)
z, a = r.normal(size=(64, 2)), r.normal(size=(64, 1))
zn = z + 0.1 * np.tanh(a)
'''
ISOLATION = DATA + r'''
out = []
for i in range(9):  # the fit's learning rate is compiled into its step: each value is a specialization
    res = fit_group([FitJob(z, a, zn, 1)], {**FIT, "learning_rate": 1e-3 * (i + 1)}, device=device, compile=True)
    out.append(res[0][1]["execution"]["compilation"])
print(json.dumps(out))
'''
REFUSAL = DATA + r'''
engine.EAGER_FALLBACK = sys.argv[2] == "fallback"
real_compiled = engine._compiled
def compiled(fn):  # the fit step's compiler fails on its first step (injected)
    if fn is not mlp_batched._step:
        return real_compiled(fn)
    def broken(*a, **k):
        raise RuntimeError("injected compiler failure")
    return broken
engine._compiled = compiled
try:
    res = fit_group([FitJob(z, a, zn, 1)], FIT, device=device, compile=True)
    print(json.dumps({"returned": True, "compilation": res[0][1]["execution"]["compilation"]}))
except engine.CompilationRefused as e:
    print(json.dumps({"returned": False, "refused": str(e), "refusal_class": type(e).__name__}))
'''


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _json(*args) -> dict:
    p = subprocess.run([sys.executable, "-c", *args], capture_output=True, text=True, timeout=3600)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def fit_isolation(device: str) -> dict:
    cs = _json(ISOLATION, device)
    compiled = all(c["requested"] == c["effective"] == "compiled (default)" and c["new_specializations"] == 1
                   for c in cs)
    counts = [c["specializations"] for c in cs] == [1, 2, 3, 4, 5, 6, 7, 8, 1]
    fresh = [c["fresh_context"] for c in cs] == [False] * 8 + [True]
    return {"pass": compiled and counts and fresh, "all_compiled": compiled, "ninth_in_a_fresh_context": counts and fresh,
            "compilation": cs}


def fit_refusal(device: str) -> dict:
    res = {mode: _json(REFUSAL, device, mode) for mode in ("refuse", "fallback")}
    r, f = res["refuse"], res["fallback"]
    res["refused_before_the_first_step"] = (not r["returned"] and r["refusal_class"] == "CompilationRefused"
                                            and "compilation failed on the first step" in r["refused"])
    res["fallback_recorded_as_eager"] = (f["returned"] and f["compilation"]["effective"].startswith("eager")
                                         and "injected compiler failure" in f["compilation"]["effective"])
    res["pass"] = res["refused_before_the_first_step"] and res["fallback_recorded_as_eager"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    report = {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
              "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
              "device": a.device, "cases": {}}
    if a.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    cases = {"fit_isolation": lambda: fit_isolation(a.device), "fit_refusal": lambda: fit_refusal(a.device)}
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
