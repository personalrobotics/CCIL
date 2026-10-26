#!/usr/bin/env python3
"""Phase-0 baseline timing of the current CCIL stack on Pendulum (#114).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/fast_stack/baseline.py --out DIR [--reps 3] [--device cuda:0]

The contract is fixed in #114 before any run:
- seed 501 for every repetition: outside tuning (1-3), study (101-120) and smoke (9001-9003) seeds;
- per task, one representative seed group run in-process exactly as the #24
  drivers configure it (the #23 corrected configuration, the study's arm
  settings): data.prepare, transition.fit, labels.generate, labels.certify, and
  policy.train + policy.evaluate for BC, NoiseBC (1e-4), CCIL and CCIL-filtered,
  then evaluation rollouts at every noise level for BC and CCIL;
- each repetition is a fresh process with a fresh stage store (cold); the
  process wall clock includes interpreter start, imports and CUDA initialization;
- the GPU is synchronized at every timing boundary; per stage: wall time, peak
  torch memory (allocated, reserved), peak process RSS, and device utilization
  and memory from a streaming nvidia-smi trace (250 ms);
- steady state: per-epoch policy-training times after the first epoch;
- 3 cold repetitions per task; the reported number is the median, with min and max.

Writes DIR/<task>-rep<k>/ (timing.json, gpu.csv, rss.csv, stage logs, stdout)
unmodified, and DIR/report.json with the medians, the workload sizes and the
projections, each projection labeled as such.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY_TOOLS = REPO / "tools" / "m3_pendulum"
SEED = 501
TASKS = ("continuous", "discontinuous")
ARMS = ("bc", "noise_bc", "ccil", "ccil_filtered")
NOISE_BC_SCALE = 1e-4
ROLLOUT_ARMS = ("bc", "ccil")


# --- one cold repetition (child process) ---------------------------------------------------------------------------
def _rss_sampler(path: Path, stop: threading.Event):
    with open(path, "w") as f:
        f.write("t,rss_kb\n")
        while not stop.is_set():
            try:
                rss = next(int(ln.split()[1]) for ln in open("/proc/self/status") if ln.startswith("VmRSS:"))
            except (OSError, StopIteration):
                rss = -1
            f.write(f"{time.time():.3f},{rss}\n")
            f.flush()
            stop.wait(0.25)


def child(task: str, out: Path, device: str) -> None:
    t_process = float(os.environ["BASELINE_T0"])
    t_imports = time.time()
    import numpy as np
    import torch

    sys.path.insert(0, str(STUDY_TOOLS))
    import env as pendulum
    import study

    from ccil.config import resolve
    from ccil.schema import read_trajectories
    from ccil.pipelines import offline, policy_harness, stages
    from ccil.training import vector_bc
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    t_ready = time.time()

    stop = threading.Event()
    rss_thread = threading.Thread(target=_rss_sampler, args=(out / "rss.csv", stop), daemon=True)
    rss_thread.start()
    events: list[dict] = []
    current = {"arm": None}

    original_run_stage = stages.run_stage

    def timed_run_stage(stage, cfg, inputs, body, components=None, **kw):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        result = original_run_stage(stage, cfg, inputs, body, components=components, **kw)
        torch.cuda.synchronize(device)
        t1 = time.time()
        events.append({"stage": stage, "arm": current["arm"], "start": t0, "end": t1, "seconds": t1 - t0,
                       "reused": bool(getattr(result, "reused", False)), "run_dir": str(result.run_dir),
                       "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                       "torch_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20})
        return result
    for mod in (stages, offline, policy_harness):
        mod.run_stage = timed_run_stage
    original_prepare = offline.prepare_stage

    def timed_prepare(cfg):  # dataset loading and validation happen before run_stage
        torch.cuda.synchronize(device)
        t0 = time.time()
        result = original_prepare(cfg)
        t1 = time.time()
        events.append({"stage": "data.prepare.total", "arm": current["arm"], "start": t0, "end": t1, "seconds": t1 - t0,
                       "reused": result.reused})
        return result
    offline.prepare_stage = timed_prepare

    epoch_times: dict = {}
    original_train = vector_bc.train_vector_bc

    def timed_train(*a, **kw):
        marks = [None]

        def on_epoch(policy, opt, epoch, state):
            torch.cuda.synchronize(device)
            marks.append(time.time())
        torch.cuda.synchronize(device)
        marks[0] = time.time()
        out_ = original_train(*a, on_epoch=on_epoch, **kw)
        epoch_times[current["arm"]] = [b - a_ for a_, b in zip(marks, marks[1:])]
        return out_
    vector_bc.train_vector_bc = timed_train

    root = out / "root"
    inputs = study.prepare(root)[task]
    config = root / "inputs" / "config" / study.CONFIGS[task]
    data = root / "inputs" / "data" / f"{study.DATASETS[task]}.npz"
    resolved, workloads = {}, {}
    for arm in ARMS:
        sets = [f"run.seed={SEED}", f"compute.device={device}", f"data.path={data}", f"run.output_dir={root / 'stages'}",
                "policy.exclude_validation_episodes=false", *study.ARMS[arm][0]]
        if arm == "noise_bc":
            sets.append(f"policy.noise_scale={NOISE_BC_SCALE!r}")
        if os.environ.get("BASELINE_SMOKE"):  # harness check only: never reported as a baseline
            balanced = arm.startswith("ccil") and task == "discontinuous"
            sets += ["transition_model.epochs=1", f"policy.epochs={20 if balanced else 1}"]
        cwd = os.getcwd()
        os.chdir(root / "inputs")
        try:
            cfg = resolve(config=str(config), sets=sets)
            resolved[arm] = {"sets": sets, "values": json.loads(json.dumps(cfg.values, default=str))}
            current["arm"] = arm
            results = policy_harness.run_policy_harness(cfg)
        finally:
            os.chdir(cwd)
        by = {r.stage: r for r in results}
        tr = json.loads((by["policy.train"].run_dir / "metrics.json").read_text())["values"]["training"]
        lab = json.loads((by["labels.generate"].run_dir / "metrics.json").read_text())["values"]
        workloads[arm] = {"policy_rows": tr["rows"], "steps_per_epoch": tr["steps_per_epoch"], "epochs": tr["epochs_run"],
                          "row_sources": tr["row_sources"], "candidates": lab.get("candidates"),
                          "accepted": lab.get("accepted"),
                          "checkpoint": str(by["policy.train"].out(policy_harness.POLICY_CHECKPOINT))}
    workloads["transitions"] = int(sum(ep.length - 1 for ep in read_trajectories(data).episodes))

    for arm in ROLLOUT_ARMS:
        net, _, _ = vector_bc.VectorBCPolicy.load(workloads[arm]["checkpoint"], device=device)
        for n in (0.0, *pendulum.NOISE_LEVELS):
            current["arm"] = arm
            torch.cuda.synchronize(device)
            t0 = time.time()
            res = pendulum.rollout(net.predict, task, SEED, n)
            torch.cuda.synchronize(device)
            t1 = time.time()
            events.append({"stage": "rollout", "arm": arm, "noise": n, "start": t0, "end": t1, "seconds": t1 - t0,
                           "episodes": pendulum.EPISODES, "horizon": pendulum.HORIZON, "R": None})
            del res  # outcomes are not part of the baseline
    stop.set()
    rss_thread.join()
    t_end = time.time()
    (out / "timing.json").write_text(json.dumps({
        "task": task, "seed": SEED, "device": device, "process_start": t_process, "imports_start": t_imports,
        "ready": t_ready, "end": t_end, "cold_process_seconds": t_end - t_process,
        "import_and_cuda_init_seconds": t_ready - t_process, "events": events, "policy_epoch_seconds": epoch_times,
        "workloads": workloads, "inputs": inputs, "resolved": resolved,
        "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__}, indent=1, default=str) + "\n")


# --- the outer driver ----------------------------------------------------------------------------------------------
def _sh(*cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _host() -> dict:
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    mem = next((ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemTotal")), "?")
    return {"platform": platform.platform(), "python": platform.python_version(), "cpu": cpu, "cores": os.cpu_count(),
            "ram_kb": mem, "gpus": _sh("nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                         "--format=csv,noheader").splitlines()}


def _window(rows, t0, t1):
    sel = [r for r in rows if t0 <= r[0] <= t1]
    return sel


def summarize(out: Path, reps: int) -> dict:
    """Medians (with min and max) per task and stage; projections labeled as such."""
    import csv
    per_task = {}
    for task in TASKS:
        runs = []
        for k in range(1, reps + 1):
            d = out / f"{task}-rep{k}"
            t = json.loads((d / "timing.json").read_text())
            gpu = []
            for row in csv.reader(open(d / "gpu.csv")):
                try:
                    ts = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()  # local time
                    gpu.append((ts, float(row[1]), float(row[2])))
                except (ValueError, IndexError):
                    continue
            rss = [(float(a), int(b)) for a, b in (ln.strip().split(",") for ln in open(d / "rss.csv").readlines()[1:]) if b != "-1"]
            stages_ = {}
            for e in t["events"]:
                key = f"{e['arm']}:{e['stage']}" + (f"@{e['noise']}" if "noise" in e else "")
                g = _window(gpu, e["start"], e["end"])
                r = _window(rss, e["start"], e["end"])
                stages_[key] = {"seconds": e["seconds"], "reused": e.get("reused", False),
                                "gpu_util_mean": statistics.mean(x[1] for x in g) if g else None,
                                "gpu_util_max": max(x[1] for x in g) if g else None,
                                "device_mem_max_mb": max(x[2] for x in g) if g else None,
                                "rss_max_mb": max(x[1] for x in r) / 1024 if r else None,
                                "torch_peak_allocated_mb": e.get("torch_peak_allocated_mb"),
                                "torch_peak_reserved_mb": e.get("torch_peak_reserved_mb")}
            steady = {arm: statistics.median(ts[1:]) if len(ts) > 1 else None for arm, ts in t["policy_epoch_seconds"].items()}
            runs.append({"cold_process_seconds": t["cold_process_seconds"],
                         "import_and_cuda_init_seconds": t["import_and_cuda_init_seconds"],
                         "stages": stages_, "steady_epoch_seconds": steady, "workloads": t["workloads"]})

        def med(values):
            v = [x for x in values if x is not None]
            return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} if v else None
        keys = sorted({k for r in runs for k in r["stages"]})
        per_task[task] = {
            "cold_process_seconds": med([r["cold_process_seconds"] for r in runs]),
            "import_and_cuda_init_seconds": med([r["import_and_cuda_init_seconds"] for r in runs]),
            "stages": {k: {m: med([r["stages"].get(k, {}).get(m) for r in runs])
                           for m in ("seconds", "gpu_util_mean", "gpu_util_max", "device_mem_max_mb", "rss_max_mb",
                                     "torch_peak_allocated_mb", "torch_peak_reserved_mb")}
                       | {"reused_in_any_rep": any(r["stages"].get(k, {}).get("reused") for r in runs)} for k in keys},
            "steady_epoch_seconds": {arm: med([r["steady_epoch_seconds"].get(arm) for r in runs]) for arm in ARMS},
            "workloads": runs[0]["workloads"]}
    return {"statistic": "median of 3 cold repetitions (min, max shown)", "seed": SEED, "tasks": per_task,
            "projection": project(per_task)}


def project(per_task: dict) -> dict:
    """Serial single-process projections of the #24 tuning and study from the measured medians. PROJECTED, not measured.

    Stage counts follow the #24 design with stage reuse within a (task, seed) group. Data-size subsets are projected
    by scaling the full-data stage time with the fraction of episodes (an assumption, stated); certify, BC and NoiseBC
    use their measured stages; CCIL-tuned and CCIL-noLip reuse the CCIL stages' times.
    """
    out = {}
    for task, t in per_task.items():
        s = {k: v["seconds"]["median"] for k, v in t["stages"].items() if v["seconds"]}

        def get(arm, stage):
            return s.get(f"{arm}:{stage}", 0.0)
        prep, fit, gen = get("bc", "data.prepare"), get("bc", "transition.fit"), get("bc", "labels.generate")
        cert = get("ccil_filtered", "labels.certify")
        train = {a: get(a, "policy.train") + get(a, "policy.evaluate") for a in ARMS}
        roll = statistics.median([v for k, v in s.items() if k.endswith(("@0.0", "@0.01", "@0.05", "@0.1"))] or [0.0])
        frac = sum((0.1, 0.25, 0.5))
        per_seed_study = (prep * (1 + frac) + fit * (2 + frac) + gen * (3 + frac) + cert
                          + train["bc"] + train["noise_bc"] + train["ccil"] + train["ccil_filtered"]
                          + 2 * train["ccil"] + frac * (train["bc"] + train["ccil"]) + 28 * roll)
        per_seed_tuning = prep + fit + gen * 5 + 5 * train["noise_bc"] + 5 * train["ccil"] + 40 * roll
        out[task] = {"study_seconds_per_seed": per_seed_study, "study_seconds_20_seeds": 20 * per_seed_study,
                     "tuning_seconds_per_seed": per_seed_tuning, "tuning_seconds_3_seeds": 3 * per_seed_tuning}
    total = sum(v["study_seconds_20_seeds"] + v["tuning_seconds_3_seeds"] for v in out.values())
    return {"label": "PROJECTED (serial, one process, one GPU) from measured medians; not measured",
            "per_task": out, "total_hours": total / 3600,
            "formula": "per seed: prep*(1+Σf) + fit*(2+Σf) + generate*(3+Σf) + certify + train(bc,noise_bc,ccil,ccil_filtered) "
                       "+ 2*train(ccil) [noLip, tuned] + Σf*(train(bc)+train(ccil)) + 28 rollouts; tuning per seed: "
                       "prep + fit + 5*generate + 5*train(noise_bc) + 5*train(ccil) + 40 rollouts; Σf = 0.85"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--child", nargs=2, metavar=("TASK", "DIR"))
    ap.add_argument("--smoke", action="store_true", help="1-epoch harness check; its numbers are not a baseline")
    a = ap.parse_args()
    if a.child:
        child(a.child[0], Path(a.child[1]), a.device)
        return 0
    sys.path.insert(0, str(STUDY_TOOLS))
    import provenance
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = provenance.tree_state(REPO)
    if a.smoke:
        os.environ["BASELINE_SMOKE"] = "1"
    meta = {"smoke": a.smoke, "commit": state["head"], "tree": state, "host": _host(), "contract": "#114 phase 0", "reps": a.reps,
            "device": a.device, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not state["ok"]:
        raise SystemExit(f"code-state gate refused: {state}")
    gpu_index = a.device.split(":")[1] if ":" in a.device else "0"
    for k in range(1, a.reps + 1):
        for task in TASKS:
            d = out / f"{task}-rep{k}"
            d.mkdir()
            load = _sh("cat", "/proc/loadavg")
            trace = subprocess.Popen(["nvidia-smi", "-i", gpu_index, "--query-gpu=timestamp,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits", "-lms", "250"], stdout=open(d / "gpu.csv", "w"))
            t0 = time.time()
            p = subprocess.run([sys.executable, __file__, "--out", str(out), "--device", a.device, "--child", task, str(d)],
                               env={**os.environ, "BASELINE_T0": repr(t0)}, capture_output=True, text=True)
            wall = time.time() - t0
            trace.terminate()
            trace.wait()
            (d / "stdout.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
            meta.setdefault("runs", []).append({"task": task, "rep": k, "exit": p.returncode, "outer_wall_seconds": wall,
                                                "loadavg_before": load})
            print(f"{task} rep{k}: exit {p.returncode}, {wall:.0f} s, load before {load}", flush=True)
            if p.returncode != 0:
                (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
                raise SystemExit(f"{task} rep{k} failed; see {d / 'stdout.log'}")
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    report = {**summarize(out, a.reps), "meta": meta}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({t: {"cold": v["cold_process_seconds"]} for t, v in report["tasks"].items()}, indent=1))
    print(json.dumps(report["projection"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
