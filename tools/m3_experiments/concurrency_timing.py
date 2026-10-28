#!/usr/bin/env python3
"""Study-phase timing per shard layout (#114): wall time of one shard process, to compare one process per GPU
with several.

Derived from tools/m3_provenance/shard_timing.py (#127), with the shard as an argument. Two processes, shards
0/4 and 2/4, run together on one GPU cover exactly the units of shard 0/2, so their wall time (the later of
the two) compares directly with one process running 0/2.


Run from a checkout that has the study runner (ccil.experiments, tools/m3_experiments), on a quiet CUDA host:

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python <this file> --repo CHECKOUT --out OUT.json [--strict]

It runs the study phase of the version 6 plan, shard 0/2: 10 seeds x 15 arms x 2 tasks = 300 units, the size of
one real shard, in a fresh root. The seeds are development seeds 9401-9420, never the predeclaration's, and tuned
arms use the middle value of their grid, so nothing here is a tuning or study outcome. No return is read or
printed: the output is timing only.

The same file runs against each code state (the runner before #127, and after it, cached and strict), one process
at a time on one GPU. Provenance time is measured in this process by wrapping ccil.runs.provenance's functions
(sha256 and its full hash, code, implementation, and verify_code_state where it exists); label workers are
spawned processes, so their provenance time is in the wall time only.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEV_SEEDS = {"from": 9401, "to": 9420}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="the checkout whose code state is measured")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="a new directory for the shard's stages and records")
    ap.add_argument("--strict", action="store_true", help="CCIL_PROVENANCE_STRICT=1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/2", help="i/n: the units of every n-th seed, from the i-th")
    a = ap.parse_args()
    if a.out.exists() or a.root.exists():
        raise SystemExit("--out and --root must be new")
    if a.strict:
        os.environ["CCIL_PROVENANCE_STRICT"] = "1"
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo / "tools" / "m3_experiments"))

    import torch
    import yaml

    import ccil
    from ccil.experiments import plan, runner
    from ccil.runs import provenance
    from pendulum_suite import PendulumSuite
    if not ccil.__file__.startswith(str(repo)):
        raise SystemExit(f"ccil is imported from {ccil.__file__}, not {repo}")

    spent: dict = {}

    def timed(name, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                s = spent.setdefault(name, {"calls": 0, "seconds": 0.0})
                s["calls"] += 1
                s["seconds"] += time.perf_counter() - t0
        return wrapper
    import importlib
    for m in ("ccil.pipelines.offline", "ccil.pipelines.policy_harness", "ccil.pipelines.stages", "ccil.supervision",
              "ccil.core.certificates", "ccil.core.labels", "ccil.data.prepare", "ccil.runs.validate",
              "ccil.runs.writer", "ccil.models.mlp_batched", "ccil.training.vector_bc_v2"):
        importlib.import_module(m)  # loaded now, so every reference below is wrapped
    originals = {name: getattr(provenance, name) for name in
                 ("sha256", "_full_sha256", "code", "implementation", "verify_code_state")
                 if hasattr(provenance, name)}
    wrapped = {name: timed(name, fn) for name, fn in originals.items()}
    for mod in [m for k, m in sys.modules.items() if k == "ccil" or k.startswith("ccil.")]:
        for attr, value in list(vars(mod).items()):  # module globals, and names imported from provenance
            for name, fn in originals.items():
                if value is fn:
                    setattr(mod, attr, wrapped[name])
    spec = yaml.safe_load((repo / "tools/m3_experiments/pendulum_v6.yaml").read_text())
    spec["phases"]["tuning"]["seeds"] = [9400]
    spec["phases"]["study"]["seeds"] = DEV_SEEDS
    study = plan.from_dict(spec)
    choice = {t: {arm: {"value": (g.grid[t] if isinstance(g.grid, dict) else g.grid)[2]}
                  for arm, g in study.tuning.items()} for t in study.tasks}
    git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    gate = {"ok": True, "head": git("rev-parse", "HEAD"), "note": "timing only (#127); development seeds"}
    root = a.root.resolve()
    shard = tuple(int(x) for x in a.shard.split("/"))
    suite = PendulumSuite(root, study, smoke=False)
    t_start = time.time()
    t0 = time.perf_counter()
    summary = runner.run_phase(root, study, "study", suite, gate=gate, smoke=False, device=a.device,
                               choice=choice, choice_sha256="development-choice", shard=shard, log=lambda m: None)
    wall = time.perf_counter() - t0
    out = {"code_commit": gate["head"], "code_dirty": bool(git("status", "--porcelain")),
           "strict": a.strict, "units": summary["units"], "wall_seconds": wall,
           "phase_timing": summary["timing"], "provenance_in_process": spent,
           "hash_stats": getattr(provenance, "HASH_STATS", None), "seeds": DEV_SEEDS, "shard": list(shard), "started": t_start, "finished": time.time(),
           "device": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
           "torch": torch.__version__, "load_at_start": os.getloadavg()}
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("code_commit", "strict", "units", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
