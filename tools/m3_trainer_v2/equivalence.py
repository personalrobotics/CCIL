"""Statistical equivalence of trainer v1 and v2 on Pendulum (#108), declared in check.py before running.

One supervision set per task comes from the ordinary pipeline at seed 501, as the #24 drivers configure it
(the #23 corrected configuration, paper settings). v1 and v2 each train BC and CCIL on it with training
seeds 201-210. Each policy is evaluated on the canonical environment at its seed's initial states, and the
95% union interval of mean(R_v2 - R_v1) must contain 0 for every (task, arm).
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY = REPO / "tools" / "m3_pendulum"
ARMS = ("bc", "ccil")


def _supervision(paths: dict, arm: str):
    from ccil.core import IdentityRepresentation
    from ccil.supervision import build_supervision
    return build_supervision(paths["dataset"], arm, representation=IdentityRepresentation("state"),
                             candidates=paths["candidates"] if arm == "ccil" else None)


def _v1_unit(args):
    """Train one v1 policy and evaluate it (a worker process)."""
    task, arm, seed, paths, settings, out = args
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.path.insert(0, str(STUDY))
    import env as pendulum
    from ccil.training import vector_bc
    policy, _ = vector_bc.train_vector_bc(_supervision(paths, arm), settings, seed=seed, device="cuda")
    policy.save(Path(out) / f"v1-{task}-{arm}-s{seed}.npz")
    return task, arm, seed, float(pendulum.rollout(policy.predict, task, seed, 0.0)["R"])


def _evaluate(args):
    task, seed, path = args
    sys.path.insert(0, str(STUDY))
    import env as pendulum
    from ccil.training.vector_bc import VectorBCPolicy
    policy, _, _ = VectorBCPolicy.load(path)
    return float(pendulum.rollout(policy.predict, task, seed, 0.0)["R"])


def run(out: Path, device: str, seeds, label_seed: int) -> dict:
    sys.path.insert(0, str(STUDY))
    import analyze
    import study

    from ccil.config import resolve
    from ccil.pipelines.offline import generate_offline
    from ccil.pipelines.policy_harness import settings_of
    from ccil.training import vector_bc_v2 as v2
    root = out / "equivalence"
    study.prepare(root)
    results, per_unit = {}, []
    jobs_v1, v2_ckpts = [], []
    for task in study.DATASETS:
        data = root / "inputs" / "data" / f"{study.DATASETS[task]}.npz"
        cwd = os.getcwd()
        os.chdir(root / "inputs")
        try:
            cfg = resolve(config=str(root / "inputs" / "config" / study.CONFIGS[task]),
                          sets=[f"run.seed={label_seed}", "compute.device=cuda", f"data.path={data}",
                                f"run.output_dir={root / 'stages'}", "policy.exclude_validation_episodes=false"])
            prepared, _, labeled = generate_offline(cfg)[:3]
        finally:
            os.chdir(cwd)
        paths = {"dataset": str(prepared.out("data/dataset.npz")), "candidates": str(labeled.out("labels/candidates.npz"))}
        for arm in ARMS:
            settings = settings_of(cfg.values)
            if arm == "bc":
                settings["balance_labels"] = None
            sup = _supervision(paths, arm)
            trained = v2.train_group([v2.Job(sup, s) for s in seeds], settings, device=device)
            for s, (policy, record) in zip(seeds, trained):
                path = out / f"v2-{task}-{arm}-s{s}.npz"
                policy.save(path)
                v2_ckpts.append((task, arm, s, str(path)))
            jobs_v1 += [(task, arm, s, paths, settings, str(out)) for s in seeds]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(6, mp_context=ctx) as pool:
        r1 = {(t, a, s): R for t, a, s, R in pool.map(_v1_unit, jobs_v1)}
        r2 = dict(zip([(t, a, s) for t, a, s, _ in v2_ckpts],
                      pool.map(_evaluate, [(t, s, p) for t, a, s, p in v2_ckpts])))
    for task in study.DATASETS:
        for arm in ARMS:
            d = np.array([r2[(task, arm, s)] - r1[(task, arm, s)] for s in seeds])
            iv = analyze.union_interval(d, 0.95)
            results[f"{task}/{arm}"] = {"contains_zero": iv["interval"][0] <= 0 <= iv["interval"][1],
                                        "interval": iv["interval"], "width": iv["interval"][1] - iv["interval"][0],
                                        "mean_difference": iv["mean"],
                                        "mean_R_v1": float(np.mean([r1[(task, arm, s)] for s in seeds])),
                                        "mean_R_v2": float(np.mean([r2[(task, arm, s)] for s in seeds]))}
            per_unit += [{"task": task, "arm": arm, "seed": s, "R_v1": r1[(task, arm, s)], "R_v2": r2[(task, arm, s)]}
                         for s in seeds]
    (out / "equivalence.json").write_text(json.dumps({"results": results, "per_seed": per_unit}, indent=1) + "\n")
    return {"pass": all(v["contains_zero"] for v in results.values()), "results": results, "seeds": list(seeds),
            "label_seed": label_seed, "criterion": "95% union interval (BCa with t) of mean(R_v2 - R_v1) contains 0",
            "per_seed": "<out>/equivalence.json"}
