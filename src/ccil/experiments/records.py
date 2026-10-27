"""Write-once unit records and their verification (#112), generalized from the #24 drivers (tools/m3_pendulum/study.py).

A record is one JSON file per unit, at results/<phase>/<task>/<arm>[@value]/s<seed>.json, written atomically
and never overwritten. verify_phase checks a phase's records against the plan before anything reads them; the
suite supplies what is suite-specific:
- audit(path) -> {ok, issues, task, seed, noise_level, failed_episodes, R_recomputed}: re-audits a saved
  evaluation from its own arrays;
- expected_inputs(record) -> {config_sha256, data_sha256}: the input hashes a record must carry.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TOL = 1e-9


class RecordsRefused(SystemExit):
    """A phase's records do not verify; the message lists the problems."""


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def unit_name(arm, value=None) -> str:
    return arm if value is None else f"{arm}@{value!r}"


def result_path(root, phase, task, arm, seed, value=None) -> Path:
    return Path(root) / "results" / phase / task / unit_name(arm, value) / f"s{seed}.json"


def write_once(record: dict, path) -> None:
    """Write a unit's record; refuse to overwrite one."""
    if Path(path).exists():
        raise FileExistsError(f"{path} exists; records are written once")
    atomic_json(record, path)


def _verify_one(args):
    root, rel, expected, audit = args
    path = Path(root) / rel
    same = sha(path) == expected
    return rel, same, audit(path) if same else None


def verify_phase(root, study, phase, *, smoke: bool, audit, expected_inputs, choice=None, jobs: int = 8) -> list:
    """Every record of a phase, verified; each returned record carries "verified": {noise: R recomputed by the audit}.

    Refuses (RecordsRefused, listing every problem) unless:
    - the records are exactly the plan's units for the phase (no missing, duplicate or extra unit), each at its
      own path;
    - every record is of this kind (smoke or real), passed its gate, and all share one code state and one tooling
      commit; real records passed the frozen-tooling gate;
    - every record carries the input hashes expected_inputs gives;
    - every evaluation file has its recorded SHA-256 and passes the suite's audit, and the record's R, failed
      episodes, task, seed and noise level agree with the audit;
    - a record is ok exactly when it has every planned evaluation and every recomputed R is finite.
    """
    root = Path(root).resolve()
    files = sorted(p for p in (root / "results" / phase).rglob("s*.json") if ".fault" not in p.name)
    records = [json.loads(p.read_text()) for p in files]
    problems = []
    expected = Counter(study.units(phase, choice))
    have = Counter((r["task"], r["arm"], r["seed"], r["value"]) for r in records)
    if have != expected:
        problems.append(f"records are not exactly the {phase} units: missing {sorted((expected - have).elements())[:3]}, "
                        f"duplicate or extra {sorted((have - expected).elements())[:3]}")
    for p, r in zip(files, records):
        if p != result_path(root, phase, r["task"], r["arm"], r["seed"], r["value"]) or r["phase"] != phase:
            problems.append(f"{p.relative_to(root)} does not hold the unit its path names")
    if {r["smoke"] for r in records} - {smoke}:
        problems.append("smoke and real records are mixed")
    if not all(r["gate"]["ok"] and r["gate"]["real"] == (not smoke) for r in records):
        problems.append("a record was made without passing its code-state gate")
    if len({(r["gate"]["head"], r["gate"]["tooling"].get("tooling_commit")) for r in records}) > 1:
        problems.append("records come from more than one code state")
    for r in records:
        if r.get("arm") in study.arms and r.get("inputs") != expected_inputs(r):
            problems.append(f"{r['task']}/{unit_name(r['arm'], r['value'])}/s{r['seed']}: input hashes differ")
    jobs_list = [(str(root), e["rollout"], e["rollout_sha256"], audit) for r in records
                 for e in r.get("evaluations", {}).values()]
    if jobs > 1:  # fork, so the suite's audit function needs no import path in the workers
        ctx = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else None)
        with ProcessPoolExecutor(jobs, mp_context=ctx) as pool:
            audits = {rel: (same, a) for rel, same, a in pool.map(_verify_one, jobs_list, chunksize=4)}
    else:
        audits = {rel: (same, a) for rel, same, a in map(_verify_one, jobs_list)}
    for r in records:
        name = f"{r['task']}/{unit_name(r['arm'], r['value'])}/s{r['seed']}"
        r["verified"] = {}
        for n, e in r.get("evaluations", {}).items():
            same, a = audits[e["rollout"]]
            if not same or not a["ok"]:
                problems.append(f"{name} noise {n}: the rollout fails the re-audit "
                                f"({'changed bytes' if not same else a['issues']})")
                continue
            agree = (a["task"], a["seed"], str(a["noise_level"])) == (r["task"], r["seed"], n) \
                and a["failed_episodes"] == e["failed_episodes"] \
                and (e["R"] == a["R_recomputed"] or abs(e["R"] - a["R_recomputed"]) <= TOL
                     or (np.isnan(e["R"]) and np.isnan(a["R_recomputed"])))
            if not agree:
                problems.append(f"{name} noise {n}: the record disagrees with its rollout's audit")
                continue
            r["verified"][n] = a["R_recomputed"]
        planned = ({str(x) for x in study.arms[r["arm"]].noise_levels}
                   if r["arm"] in study.arms and r["arm"] not in study.deferred(phase) else set())
        if r["status"] == "ok" and (set(r["verified"]) != planned
                                    or not all(np.isfinite(v) for v in r["verified"].values())):
            problems.append(f"{name}: status ok without every finite, verified evaluation")
    if problems:
        raise RecordsRefused(f"{phase} records refused ({len(problems)} problems): " + "; ".join(problems[:10]))
    return records


CHOICE = "tuning-choice.json"


def record_choice(root, study, tuning: list, phase: str = "tuning", acceptance_arms=("ccil_tuned",)) -> str:
    """compute_choice from verified tuning records, written once to root/tuning-choice.json; returns its SHA-256.

    Arms selected by validation loss have no returns in the tuning records, so their choice is made, and fixed
    here, before any of their tuning-seed returns exists."""
    path = Path(root) / CHOICE
    write_once({"choice": compute_choice(study, tuning, phase, acceptance_arms), "phase": phase}, path)
    return sha(path)


def verify_deferred(root, study, phase, *, smoke: bool, audit, jobs: int = 8) -> list:
    """The deferred returns of a tuning phase (runner.evaluate_deferred), verified like a phase's records.

    Refuses unless they are exactly the phase's deferred units that are ok, each names the recorded choice by its
    SHA-256 and its tuning record by its SHA-256 (so it was computed after the choice, from that unit's policy),
    and every rollout passes the audit and agrees with its record."""
    root = Path(root).resolve()
    choice_sha = sha(root / CHOICE)
    tuning = {(r["task"], r["arm"], r["seed"], r["value"]): (p, r) for p in (root / "results" / phase).rglob("s*.json")
              if ".fault" not in p.name for r in [json.loads(p.read_text())]}
    want = {k for k, (_, r) in tuning.items() if k[1] in study.deferred(phase) and r["status"] == "ok"}
    files = sorted((root / "results" / f"{phase}_returns").rglob("s*.json"))
    recs = [json.loads(p.read_text()) for p in files]
    problems = []
    have = Counter((r["task"], r["arm"], r["seed"], r["value"]) for r in recs)
    if have != Counter(want):
        problems.append(f"deferred returns are not exactly the ok deferred units of {phase}")
    for p, r in zip(files, recs):
        k = (r["task"], r["arm"], r["seed"], r["value"])
        if r.get("choice_sha256") != choice_sha:
            problems.append(f"{p.name}: does not name the recorded choice")
        if k in tuning and r.get("tuning_record_sha256") != sha(tuning[k][0]):
            problems.append(f"{p.name}: does not name its tuning record")
        if r.get("smoke") != smoke:
            problems.append(f"{p.name}: smoke and real records are mixed")
    jobs_list = [(str(root), e["rollout"], e["rollout_sha256"], audit) for r in recs for e in r["evaluations"].values()]
    audits = {rel: (same, a) for rel, same, a in map(_verify_one, jobs_list)}
    for r in recs:
        r["verified"] = {}
        for n, e in r["evaluations"].items():
            same, a = audits[e["rollout"]]
            if not same or not a["ok"] or a["failed_episodes"] != e["failed_episodes"] or (
                    abs(e["R"] - a["R_recomputed"]) > TOL and not (np.isnan(e["R"]) and np.isnan(a["R_recomputed"]))):
                problems.append(f"{r['task']}/{unit_name(r['arm'], r['value'])}/s{r['seed']} noise {n}: refused")
                continue
            r["verified"][n] = a["R_recomputed"]
    if problems:
        raise RecordsRefused(f"{phase} deferred returns refused ({len(problems)} problems): " + "; ".join(problems[:10]))
    return recs


def _finite(x):
    return x if x is not None and np.isfinite(x) else None  # JSON-stable: NaN never equals itself


def validation_loss(record):
    """A unit's selected-checkpoint validation loss (checkpoint_selection validation), or None."""
    v = (record.get("training") or {}).get("validation")
    return None if not v else v.get("selected_loss")


def compute_choice(study, tuning: list, phase: str = "tuning", acceptance_arms=("ccil_tuned",)) -> dict:
    """The plan's tuning rule from verified tuning records: per task and tuned arm, the eligible value with the
    best mean over the phase's seeds, ties to the smaller. The rule is the arm's select: highest_mean_noise_free_R
    (the audited noise-free R) or lowest_mean_validation_loss (the selected checkpoint's held-out validation loss,
    from the training record; returns are never used). Every value-by-seed result is kept, with the label
    acceptance rate for acceptance_arms."""
    seeds = study.phases[phase]["seeds"]
    by = {(r["task"], r["arm"], r["value"], r["seed"]): r for r in tuning}
    out = {}
    for task in study.tasks:
        out[task] = {}
        for arm in study.phases[phase]["arms"]:
            grid = study.tuning[arm].grid[task]
            by_loss = study.tuning[arm].select == "lowest_mean_validation_loss"
            table, eligible = {}, {}
            for v in grid:
                recs = [by[(task, arm, v, s)] for s in seeds]
                ok = all(r["status"] == "ok" for r in recs)
                row = {"R": {str(r["seed"]): _finite(r["verified"].get("0.0")) for r in recs},
                       "status": {str(r["seed"]): r["status"] for r in recs}}
                if by_loss:
                    row["validation_loss"] = {str(r["seed"]): _finite(validation_loss(r)) for r in recs}
                    ok = ok and all(x is not None for x in row["validation_loss"].values())
                row["eligible"] = ok
                if arm in acceptance_arms:
                    row["label_acceptance_rate"] = {str(r["seed"]): r.get("label_accuracy", {}).get("acceptance_rate")
                                                    for r in recs}
                if ok and by_loss:  # lower is better: rank by the negated mean loss
                    row["mean_validation_loss"] = float(np.mean(list(row["validation_loss"].values())))
                    eligible[v] = -row["mean_validation_loss"]
                elif ok:
                    eligible[v] = row["mean_R"] = float(np.mean([r["verified"]["0.0"] for r in recs]))
                table[repr(v)] = row
            if eligible:
                best = max(eligible.values())
                value = min(v for v, m in eligible.items() if m == best)
                rule = ("lowest mean validation loss" if by_loss else "highest mean R") + "; ties to the smaller value"
            else:
                value, rule = grid[0], "every value ineligible: the paper's value"
            out[task][arm] = {"value": value, "rule": rule, "table": table}
    return out
