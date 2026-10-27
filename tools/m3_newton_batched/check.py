#!/usr/bin/env python3
"""Evidence for stacked Newton solves (#110).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_newton_batched/check.py --out DIR [--device cpu]

The reference is the per-row solve this change replaced (PER_ROW below, as it was in
src/ccil/core/generators.py at 7a7a5e8).
1. Synthetic systems (D = 3, 17, 64), with singular and non-finite rows mixed in: steps and
   success flags equal the reference exactly; timing is reported.
2. Real labels: noisy_action label generation on discontinuous Pendulum (the ordinary pipeline at
   seed 501, paper settings, with the dynamics fit shortened to 5 epochs for speed) with the
   reference solve and with the stacked solve. Both runs have the same stage identity, so the whole candidate
   file must be byte-identical (its SHA-256): every array, including source identity, weights, measurements and
   any action chunks, and the metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))

from ccil.core import generators  # noqa: E402


def PER_ROW(J, r):  # the replaced implementation, kept verbatim as the reference
    ok = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(r).all(axis=1)
    step = np.zeros_like(r)
    for i in np.flatnonzero(ok):
        try:
            step[i] = np.linalg.solve(J[i], r[i])
        except np.linalg.LinAlgError:
            ok[i] = False
    return step, ok & np.isfinite(step).all(axis=1)


def synthetic():
    rng, rows = np.random.default_rng(0), []
    for D, N in ((3, 200000), (17, 20000), (64, 2000)):
        for singular in (False, True):
            J = np.eye(D) + 0.3 * rng.normal(size=(N, D, D))
            r = rng.normal(size=(N, D))
            if singular:
                J[5] = 0
                J[7, 0] = np.nan
                r[9, 1] = np.inf
                J[11] = np.outer(np.ones(D), np.arange(D))
                J[N // 2] = 0
            t0 = time.time()
            a = PER_ROW(J, r)
            t1 = time.time()
            b = generators._solve_rows(J, r)
            t2 = time.time()
            rows.append({"D": D, "N": N, "singular_or_nonfinite_rows": bool(singular),
                         "identical": bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])),
                         "failed_rows": int((~b[1]).sum()), "per_row_seconds": t1 - t0, "stacked_seconds": t2 - t1})
    return rows


def real_labels(out: Path, device: str):
    import study

    from ccil.config import resolve
    from ccil.pipelines import offline
    from ccil.schema.io import read_candidates
    study.prepare(out)
    data = out / "inputs" / "data" / f"{study.DATASETS['discontinuous']}.npz"
    os.chdir(out / "inputs")
    results = {}
    for name, fn in (("per_row", PER_ROW), ("stacked", generators._solve_rows)):
        original = generators._solve_rows
        generators._solve_rows = fn
        try:
            cfg = resolve(config=str(out / "inputs" / "config" / study.CONFIGS["discontinuous"]),
                          sets=["run.seed=501", f"compute.device={device}", f"data.path={data}",
                                f"run.output_dir={out / ('stages-' + name)}", "transition_model.epochs=5"])
            t0 = time.time()
            labeled = offline.generate_offline(cfg)[2]
            path = labeled.out("labels/candidates.npz")
            results[name] = (read_candidates(path), time.time() - t0,
                             __import__("hashlib").sha256(path.read_bytes()).hexdigest(), labeled.run_dir.name)
        finally:
            generators._solve_rows = original
    a = results["per_row"][0]
    sha = {k: v[2] for k, v in results.items()}
    run_ids = {k: v[3] for k, v in results.items()}
    same = sha["per_row"] == sha["stacked"] and run_ids["per_row"] == run_ids["stacked"]
    return {"identical_candidates": bool(same), "candidates_sha256": sha, "stage_run_ids": run_ids,
            "candidates": int(len(a.accepted)), "accepted": int(np.sum(a.accepted)),
            "pipeline_seconds": {k: v[1] for k, v in results.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    syn = synthetic()
    real = real_labels(out, a.device)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(r["identical"] for r in syn) and real["identical_candidates"],
              "synthetic": syn, "real_labels": real,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "device": a.device}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({"all_pass": report["all_pass"], "real": real,
                      "synthetic": [(r["D"], r["singular_or_nonfinite_rows"], r["identical"],
                                     round(r["per_row_seconds"], 3), round(r["stacked_seconds"], 3)) for r in syn]}, indent=1))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
