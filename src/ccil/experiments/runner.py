"""Batched execution of a study phase (#112): every unit's stages, with the expensive ones run in groups.

A unit (task, arm, seed, value) runs the pipeline of `ccil policy train` (data.prepare -> transition.fit ->
labels.generate [-> labels.certify] -> policy.train -> policy.evaluate), then the suite's simulator evaluation,
and gets one write-once record (records.result_path). The stage artifacts are the ones the CLI would write, with
the same identities; the runner only changes how the work is scheduled:

- transition.fit: units whose fit is missing are deduplicated by stage identity (arms share a seed's fit) and
  fitted together by models.mlp_batched.fit_group, grouped by model settings (transition_model.trainer_version 2);
- labels.generate and labels.certify: per unit (each is batched internally);
- policy.train: units are grouped by training settings and trained together by training.vector_bc_v2.train_group
  (policy.trainer_version 2), in chunks of at most max_group models, then evaluated and recorded chunk by chunk,
  so a long phase writes records as it goes and resumes where it stopped;
- stages that are already complete are reused, never recomputed (stages.completed_stage).

A group that runs out of device memory is split in half and retried, down to single units; a single unit out of
memory is retried once and then recorded as an infrastructure fault. A stage failure of a unit's labels is an
algorithmic failure of that unit, as is a non-finite training loss or return (the #24 rules). Configuration
errors and any other exception are tooling defects: they stop the phase without a record.

The suite supplies what is task-specific (see Suite).
"""
from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .records import result_path, sha, unit_name, write_once

VERSION = "ccil.experiments.runner/1"


class Suite(Protocol):
    """The task-specific half of a study."""

    name: str
    version: str

    def config(self, phase: str, task: str, seed: int, settings: list[str], data_fraction, device: str,
               output_dir: Path):
        """(resolved configuration, {config_sha256, data_sha256}) of a trained unit; settings already include the
        study's, the arm's and the tuned value; the suite adds its data path and anything task-specific."""

    def inputs(self, task: str) -> dict:
        """{config_sha256, data_sha256} of an arm with no training (it uses no data)."""

    def evaluate(self, phase: str, task: str, items: list) -> dict:
        """Simulator evaluation of many units: items are (key, predict or None for the zero action, seed,
        noise levels); returns {key: {str(noise): {R, rollout, rollout_sha256, failed_episodes, audit_ok,
        audit_issues}}}, rollout paths relative to the study root."""

    def unit_report(self, task: str, arm: str, seed: int, cfg, stages: dict) -> dict:
        """Extra record fields from a trained unit's stage directories (for example label accuracy)."""

    def versions(self) -> dict:
        """Versions of the suite's own dependencies (simulator, integrator)."""


@dataclass
class _Unit:
    task: str
    arm: str
    seed: int
    value: object
    record: dict
    cfg: object = None
    prepared: object = None
    status: str | None = None          # set once the unit has failed
    stages: dict = field(default_factory=dict)
    recorded: bool = False

    @property
    def key(self):
        return (self.task, self.arm, self.seed, self.value)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _versions(suite) -> dict:
    import scipy
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "gpu": gpu, "suite": suite.versions()}


def _oom(e: BaseException) -> bool:
    import torch
    while e is not None:
        if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in str(e).lower():
            return True
        e = getattr(e, "error", None) or e.__cause__
    return False


def _free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_on_oom(items: list, run, log) -> list:
    """run(items) -> results in order; a group out of memory is split in half and retried, down to one item.

    Returns per item a result, or the exception of a single item that ran out of memory.
    """
    try:
        return run(items)
    except Exception as e:  # noqa: BLE001 - only memory exhaustion is handled; anything else propagates
        if not _oom(e):
            raise
        _free()
        if len(items) == 1:
            return [e]
        log(f"  out of memory with {len(items)} in a group; splitting")
        h = len(items) // 2
        return _split_on_oom(items[:h], run, log) + _split_on_oom(items[h:], run, log)


def _chunks(xs: list, cap: int, per: int = 1, overhead: float = 0.0) -> list:
    """xs in chunks sized to minimize the group work, at most cap models per chunk (per models an item).

    The trainers pad a group to the next bucket with masked models that still compute, and chunks run one after
    another, each paying a per-step overhead. A step of a group of bucket b costs about b + overhead model units
    (measured on siddh: overhead about 4 for transition fits and 14 for policies), and the chunking minimizes
    the total: 84 policies run as 52 (padded to 64) + 32 rather than as 128, and 15 as one padded group of 16.
    """
    from ..training.vector_bc_v2 import BUCKETS
    buckets = [b for b in BUCKETS if per <= b <= cap] or [per]
    n = len(xs)
    best = [(0.0, ())] + [None] * n  # best[m]: (cost, chunk sizes in items) for m items; ties to fewer chunks
    for m in range(1, n + 1):
        best[m] = min(((best[max(0, m - b // per)][0] + b + overhead, best[max(0, m - b // per)][1] + (min(b // per, m),))
                       for b in buckets), key=lambda c: (c[0], len(c[1])))
    out, i = [], 0
    for size in sorted(best[n][1], reverse=True):
        out.append(xs[i:i + size])
        i += size
    return out


def _stage_key(cfg, stages: tuple) -> str:
    """The configuration the given stages consume (their inputs follow from it): equal keys share those stages."""
    from ..pipelines.stages import STAGE_KEYS
    prefixes = {p for st in stages for p in STAGE_KEYS[st]}
    return json.dumps({k: v for k, v in cfg.values.items()
                       if any(k == p or (p.endswith(".") and k.startswith(p)) for p in prefixes)},
                      sort_keys=True, default=str)


def _labels_one(cfg, attempt: int):
    """generate_offline for one unit (a worker process): None, or (failure record, out of memory)."""
    from ..config import ConfigError
    from ..pipelines import offline
    from ..pipelines.stages import StageError
    try:
        offline.generate_offline(cfg)
        return None
    except StageError as e:
        if isinstance(e.error, ConfigError):
            raise
        return {"stage": e.stage, "error": str(e)[-2000:], "attempt": attempt}, _oom(e)


def _labels(units: list, workers: int, log) -> None:
    """The label stages of every unit, in worker processes, in waves so that no two processes write one stage.

    Each wave runs at most one unit per distinct stage at its level (transition.fit, labels.generate,
    labels.certify, on the stages earlier waves completed); the last wave only reuses. A stage failure is the unit's algorithmic failure; a unit out of memory (possibly from
    sharing the device with other workers) is retried alone, and a second exhaustion is an infrastructure fault.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    levels = [("data.prepare", "transition.fit"), ("data.prepare", "transition.fit", "labels.generate"),
              ("data.prepare", "transition.fit", "labels.generate", "labels.certify")]
    waves, seen = [[] for _ in range(len(levels) + 1)], [set() for _ in levels]
    for u in units:  # a unit goes to the first level whose stages it does not share with an earlier unit
        keys = [_stage_key(u.cfg, lv) for lv in levels]
        w = next((i for i, k in enumerate(keys) if k not in seen[i]), len(levels))
        for i in range(w, len(levels)):
            seen[i].add(keys[i])
        waves[w].append(u)
    retry = []

    def settle(u, res):
        if res is None:
            return
        failure, oom = res
        u.record.setdefault("failures", []).append(failure)
        if oom:
            retry.append(u)
        else:
            u.status = "algorithmic_failure"
    ctx = multiprocessing.get_context("spawn")  # the parent has run torch: never fork it
    with ProcessPoolExecutor(max(1, workers), mp_context=ctx) as pool:
        for wave in waves:
            for u, res in zip(wave, pool.map(_labels_one, [u.cfg for u in wave], [1] * len(wave))):
                settle(u, res)
    for u in retry:  # alone, in this process
        _free()
        u.record["attempt"] = 2
        res = _labels_one(u.cfg, 2)
        if res is not None:
            u.record["failures"].append(res[0])
            u.status = "infrastructure_fault" if res[1] else "algorithmic_failure"
    log(f"  labels: waves of {[len(w) for w in waves]} units, {workers} workers, {len(retry)} retried alone")


def run_phase(root, study, phase: str, suite: Suite, *, gate: dict, smoke: bool, device: str = "cuda",
              choice: dict | None = None, choice_sha256: str | None = None, max_group: int = 128,
              shard: tuple[int, int] = (0, 1), workers: int = 8, log=print) -> dict:
    """Run every unit of phase that has no record yet; return a summary with stage timings.

    shard (i, n) runs only the units of every n-th seed of the phase, starting at the i-th, so n processes (one
    per device) run a phase together: a seed's units share their stage directories and stay in one shard, and
    records are per unit.

    gate is the code-state gate the caller required (gate.require), recorded in every record; choice and its
    SHA-256 are required for a phase that runs a tuned arm at its chosen value.
    workers is the number of processes that generate labels at once. They are spawned, so a driver script that
    calls run_phase must guard its entry point with `if __name__ == "__main__":`.
    """
    from ..core import IdentityRepresentation
    from ..data.prepare import prepare_stage
    from ..models.mlp_batched import FitJob, fit_group
    from ..pipelines import offline
    from ..pipelines.policy_harness import (POLICY_CHECKPOINT, VectorMLP, run_policy_harness, settings_of,
                                            unit_supervision)
    from ..pipelines.stages import completed_stage, stage_dest
    from ..runs import layout
    from ..training import vector_bc_v2
    from ..training.vector_bc import VectorBCPolicy

    root = Path(root).resolve()
    timing = {}
    t_phase = time.time()
    units = []
    i_shard, n_shards = shard
    seeds = list(study.phases[phase]["seeds"])
    for task, arm, seed, value in study.units(phase, choice):
        if seeds.index(seed) % n_shards != i_shard or result_path(root, phase, task, arm, seed, value).exists():
            continue
        a = study.arms[arm]
        record = {"version": VERSION, "suite": f"{suite.name}/{suite.version}", "phase": phase, "task": task,
                  "arm": arm, "seed": seed, "value": value, "value_key": a.tuned, "smoke": smoke, "attempt": 1,
                  "gate": gate, "versions": _versions(suite), "started": _now()}
        if choice_sha256 is not None:
            record["tuning_choice_sha256"] = choice_sha256
        u = _Unit(task, arm, seed, value, record)
        if a.train:
            store = root / "stages" / phase / task / f"s{seed}"
            u.cfg, record["inputs"] = suite.config(phase, task, seed, study.unit_settings(arm, value), a.data_fraction,
                                                   device, store)
            record["overrides"] = study.unit_settings(arm, value)
        else:
            record["inputs"] = suite.inputs(task)
        units.append(u)
    trained = [u for u in units if u.cfg is not None]
    log(f"{phase}: {len(units)} units without a record ({len(trained)} trained)")

    # 1. data.prepare, then transition.fit in groups --------------------------------------------------------------
    t0 = time.time()
    for u in trained:
        offline.preflight(u.cfg)
        u.prepared = prepare_stage(u.cfg)
    fits = {}  # stage destination -> the units sharing that fit
    for u in trained:
        v = u.cfg.values
        if v["transition_model.trainer_version"] != "2" or v["transition_model.kind"] != "mlp":
            continue  # fitted per unit by generate_offline
        rep = IdentityRepresentation(v["representation.modality"])
        inputs = offline.transition_inputs(u.prepared)
        comps = {"representation": rep, "transition_model": offline.build_model(v)}
        if completed_stage("transition.fit", u.cfg, inputs, comps) is None:
            fits.setdefault(stage_dest("transition.fit", u.cfg, inputs, comps), []).append(u)
    by_settings = {}
    for dest, us in fits.items():
        v = us[0].cfg.values
        key = json.dumps([dict(offline.build_model(v).settings), v["compute.device"]], sort_keys=True, default=str)
        by_settings.setdefault(key, []).append(us)
    fit_groups = []
    for key, members in by_settings.items():
        settings, dev = json.loads(key)
        per = 2 if any(us[0].prepared.out(layout.SPLIT).exists()  # a label and a separate evaluation model
                       and us[0].cfg.values["transition_model.fit_episodes"] == "all" for us in members) else 1
        for chunk in _chunks(members, max_group, per, overhead=4.0):
            jobs, owners = [], []
            for us in chunk:
                u = us[0]
                v = u.cfg.values
                z, a, zn, tr, _ = offline.transition_data(v, offline.transition_inputs(u.prepared),
                                                          IdentityRepresentation(v["representation.modality"]))
                if v["transition_model.fit_episodes"] == "train":  # the label model is the train-split fit
                    jobs.append(FitJob(z[tr], a[tr], zn[tr], v["run.seed"]))
                    owners.append((us, "label"))
                    continue
                jobs.append(FitJob(z, a, zn, v["run.seed"]))
                owners.append((us, "label"))
                if tr is not None:
                    jobs.append(FitJob(z[tr], a[tr], zn[tr], v["run.seed"]))
                    owners.append((us, "evaluation"))
            tg = time.time()
            out = _split_on_oom(list(range(len(jobs))),
                                lambda idx: fit_group([jobs[i] for i in idx], settings, device=dev), log)
            fit_groups.append({"fits": len(jobs), "seconds": round(time.time() - tg, 1)})
            prefit = {}
            for (us, role), res in zip(owners, out):
                prefit.setdefault(id(us), ({"label": None, "evaluation": None}, us))[0][role] = res
            for pf, us in prefit.values():
                if any(isinstance(x, BaseException) for x in pf.values()):
                    continue  # fitted again per unit below, where a second failure is recorded
                offline.fit_transition_stage(us[0].cfg, us[0].prepared,
                                             IdentityRepresentation(us[0].cfg.values["representation.modality"]),
                                             prefit=pf)
            jobs = out = prefit = None
            _free()
    timing["prepare_and_fit_seconds"] = round(time.time() - t0, 1)
    timing["fit_groups"] = fit_groups
    log(f"  fits: {len(fits)} distinct for {sum(len(us) for us in fits.values())} units, {len(fit_groups)} groups, "
        f"{timing['prepare_and_fit_seconds']}s")

    # 2. labels, several units at once in worker processes ---------------------------------------------------
    t0 = time.time()
    _labels(trained, workers, log)
    timing["labels_seconds"] = round(time.time() - t0, 1)
    log(f"  labels: {timing['labels_seconds']}s")

    # 3. policies in groups, each chunk evaluated and recorded as it finishes ---------------------------------------
    t0 = time.time()
    groups, single = {}, []
    for u in trained:
        if u.status:
            continue
        v = u.cfg.values
        if v["policy.trainer_version"] == "2":
            key = json.dumps([settings_of(v), v["compute.device"], v["compute.deterministic"]], sort_keys=True)
            groups.setdefault(key, []).append(u)
        else:
            single.append(u)
    train_groups = []
    for key, members in groups.items():
        settings, dev, det = json.loads(key)
        for chunk in _chunks(members, max_group, overhead=14.0):
            sups, todo = {}, []
            for u in chunk:
                sups[id(u)] = unit_supervision(u.cfg)
                _, inputs, sup, noise, held = sups[id(u)]
                if completed_stage("policy.train", u.cfg, inputs, {"trainer": VectorMLP()}) is None:
                    todo.append((u, vector_bc_v2.Job(sup, u.seed, noise, held)))
            tg = time.time()
            out = _split_on_oom(todo, lambda items: vector_bc_v2.train_group(
                [j for _, j in items], settings, device=dev, deterministic=det), log) if todo else []
            if todo:
                train_groups.append({"models": len(todo), "seconds": round(time.time() - tg, 1)})
            trained_by = {id(u): res for (u, _), res in zip(todo, out)}
            for u in chunk:
                res = trained_by.get(id(u))
                if isinstance(res, BaseException):
                    u.record.setdefault("failures", []).append({"stage": "policy.train", "error": str(res)[-2000:]})
                    try:  # one more attempt alone
                        _, _, sup, noise, held = sups[id(u)]
                        res = vector_bc_v2.train_group([vector_bc_v2.Job(sup, u.seed, noise, held)], settings,
                                                       device=dev, deterministic=det)[0]
                    except Exception as e:  # noqa: BLE001
                        if not _oom(e):
                            raise
                        u.status, u.record["attempt"] = "infrastructure_fault", 2
                        _free()
                        continue
                u.stages = {r.stage: r.run_dir for r in run_policy_harness(u.cfg, trained=res, unit=sups[id(u)])}
            sups = todo = out = trained_by = None
            _free()
            _finish(root, phase, chunk, suite, study, POLICY_CHECKPOINT, VectorBCPolicy, device, log)
    for u in single:
        u.stages = {r.stage: r.run_dir for r in run_policy_harness(u.cfg)}
    _finish(root, phase, [u for u in units if not u.recorded], suite, study, POLICY_CHECKPOINT,
            VectorBCPolicy, device, log)
    timing["policies_and_evaluation_seconds"] = round(time.time() - t0, 1)
    timing["train_groups"] = train_groups
    timing["phase_seconds"] = round(time.time() - t_phase, 1)
    log(f"  policies and evaluation: {timing['policies_and_evaluation_seconds']}s; phase {timing['phase_seconds']}s")
    return {"units": len(units), "timing": timing}


def _finish(root, phase, units, suite, study, policy_checkpoint, policy_cls, device, log):
    """Evaluate units in the simulator and write their records (failed units are recorded without evaluation)."""
    by_task = {}
    for u in units:
        if u.status is None and u.cfg is not None:
            _describe(u, suite)
        if u.status is None:
            by_task.setdefault(u.task, []).append(u)
    for task, us in by_task.items():
        items = []
        for u in us:
            predict = None
            if u.cfg is not None:
                net, _, _ = policy_cls.load(u.stages["policy.train"] / policy_checkpoint, device=device)
                predict = net.predict
            items.append((u.key, predict, u.seed, study.arms[u.arm].noise_levels))
        t0 = time.time()
        evaluations = suite.evaluate(phase, task, items)
        log(f"  evaluated {len(items)} {task} units in {time.time() - t0:.1f}s")
        for u in us:
            ev = evaluations[u.key]
            if not all(e["audit_ok"] for e in ev.values()):
                raise SystemExit(f"the audit refused a rollout of {u.key}: {ev}")
            u.record["evaluations"] = ev
            if not all(np.isfinite(e["R"]) for e in ev.values()):
                u.status, u.record["failure"] = "algorithmic_failure", "non-finite return"
            else:
                u.status = "ok"
    for u in units:
        u.record["status"] = u.status
        u.record["finished"] = _now()
        write_once(u.record, result_path(root, phase, u.task, u.arm, u.seed, u.value))
        u.recorded = True


def _describe(u, suite):
    """Fill a trained unit's record from its stage directories; a non-finite training loss fails the unit."""
    st = u.stages
    r = u.record
    r["stages"] = {k: str(p) for k, p in st.items()}
    r["resolved_config"] = json.loads((st["policy.train"] / "config.json").read_text())
    r["checkpoint_sha256"] = sha(st["policy.train"] / "checkpoints" / "policy.npz")
    training = json.loads((st["policy.train"] / "metrics.json").read_text())["values"]["training"]
    losses = [e["mean_loss"] for e in training["epochs"]]
    r["training"] = {k: training.get(k) for k in ("trainer", "trainer_version", "epochs_run", "rows", "row_sources",
                                                  "execution", "recipe", "validation", "seconds")}
    r["training"]["mean_loss"] = losses
    r["dynamics_held_out"] = json.loads((st["transition.fit"] / "metrics.json").read_text())["values"].get("held_out")
    r["labels_report"] = json.loads((st["labels.generate"] / "metrics.json").read_text())["values"]
    if "labels.certify" in st:
        r["certify_report"] = json.loads((st["labels.certify"] / "metrics.json").read_text())["values"]
    r["supervision"] = json.loads((st["policy.train"] / "supervision.json").read_text())
    r.update(suite.unit_report(u.task, u.arm, u.seed, u.cfg, st))
    if not all(x is not None and np.isfinite(x) for x in losses):
        u.status, r["failure"] = "algorithmic_failure", "non-finite training loss"


__all__ = ["Suite", "run_phase", "VERSION", "unit_name"]
