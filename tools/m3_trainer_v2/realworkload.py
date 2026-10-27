"""Real-workload timing of trainer v2 (#108): 20 distinct Pendulum label sets per task, trained as 20-seed groups.

Labels come from the ordinary pipeline (the #23 corrected configuration, the #24 drivers' settings) for seeds
301-320 per task, run in parallel worker processes across the visible GPUs. Each group is then timed in this
process: wall, host setup, one-time (first-epoch excess), steady ms per step (median of later epochs), peak
device memory, and the shared table's rows and blocks.
"""
from __future__ import annotations

import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY = REPO / "tools" / "m3_pendulum"
SEEDS = tuple(range(301, 321))


def _labels(args):
    """One labels pipeline (a worker process): returns the dataset and candidates paths."""
    task, seed, root, gpu = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(STUDY))
    import study

    from ccil.config import resolve
    from ccil.pipelines.offline import generate_offline
    data = Path(root) / "inputs" / "data" / f"{study.DATASETS[task]}.npz"
    os.chdir(Path(root) / "inputs")
    cfg = resolve(config=str(Path(root) / "inputs" / "config" / study.CONFIGS[task]),
                  sets=[f"run.seed={seed}", "compute.device=cuda", f"data.path={data}",
                        f"run.output_dir={Path(root) / 'stages' / task / f's{seed}'}",
                        "policy.exclude_validation_episodes=false"])
    prepared, _, labeled = generate_offline(cfg)[:3]
    return task, seed, str(prepared.out("data/dataset.npz")), str(labeled.out("labels/candidates.npz"))


def run(out: Path, device: str) -> dict:
    import torch
    sys.path.insert(0, str(STUDY))
    import study

    from ccil.config import resolve
    from ccil.core import IdentityRepresentation
    from ccil.pipelines.policy_harness import settings_of
    from ccil.supervision import build_supervision
    from ccil.training import vector_bc_v2 as v2
    from ccil.training.noise_bc import NoiseBC
    root = out / "realworkload"
    study.prepare(root)
    gpus = list(range(torch.cuda.device_count())) or [0]
    jobs = [(task, s, str(root), gpus[i % len(gpus)]) for i, (task, s) in enumerate((t, s) for t in study.DATASETS
                                                                                     for s in SEEDS)]
    t0 = time.time()
    with ProcessPoolExecutor(4 * len(gpus), mp_context=multiprocessing.get_context("spawn")) as pool:
        paths = {(t, s): (d, c) for t, s, d, c in pool.map(_labels, jobs)}
    label_seconds = time.time() - t0
    rep, rows = IdentityRepresentation("state"), []
    for task in study.DATASETS:
        cfg = resolve(config=str(root / "inputs" / "config" / study.CONFIGS[task]), sets=["compute.device=cuda"])
        base = settings_of(cfg.values)
        for arm, n in (("bc", 20), ("noise_bc", 20), ("ccil", 20), ("ccil", 1)):
            settings = dict(base) if arm == "ccil" else {**base, "balance_labels": None}
            group = []
            for s in SEEDS[:n]:
                d, c = paths[(task, s)]
                sup = build_supervision(d, arm, representation=rep, candidates=c if arm == "ccil" else None)
                group.append(v2.Job(sup, s, NoiseBC("symmetric_uniform_half_batch", 1e-4, seed=s) if arm == "noise_bc"
                                    else None))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            res = v2.train_group(group, settings, device=device)
            wall = time.time() - t
            g, r0 = res[0][1]["group"], res[0][1]
            es = g["epoch_seconds"]
            steady = statistics.median(es[1:]) if len(es) > 1 else es[0]
            rows.append({"task": task, "arm": arm, "models": n, "balanced": bool(settings.get("balance_labels")),
                         "wall_seconds": wall, "setup_seconds": g["setup_seconds"],
                         "one_time_seconds": max(0.0, es[0] - steady), "epochs": len(es),
                         "steady_ms_per_step": 1000 * steady / r0["steps_per_epoch"],
                         "steps_per_epoch": r0["steps_per_epoch"], "rows_per_model": r0["rows"],
                         "table_rows": g["table_rows"], "table_blocks": len(g["table_blocks"]),
                         "data_mode": g["data_mode"], "engine": g["engine"],
                         "peak_device_gb": torch.cuda.max_memory_allocated() / 2**30})
            print(rows[-1], flush=True)
    return {"pass": all(r["engine"].startswith("compiled") for r in rows), "label_pipelines": len(jobs),
            "label_seconds_parallel": label_seconds, "seeds": list(SEEDS), "measurements": rows,
            "criterion": "every group trains with the compiled engine (the timings themselves are reported)"}
