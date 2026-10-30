#!/usr/bin/env python3
"""The corrected-demonstrator study (docs/experiments/pendulum-corrected-demonstrator-predeclaration.md).

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/corrected_study.py"
    $R preflight --root NEW_ROOT --data CANONICAL_DIR --v6 V6_ROOT [--smoke]
    $R train     --root ROOT --data CANONICAL_DIR
    $R audit     --root ROOT --v6 V6_ROOT
    $R evaluate  --root ROOT --out NEW_DIR
    $R smoke     --work NEW_DIR --data CANONICAL_DIR --v6 V6_ROOT        # the whole sequence, reduced, plus refusals

Training is v6's: the same ccil source (v6's ccil commit), runner, stages, shared settings and BC and CCIL arm
settings (corrected_study.yaml). Only the dataset changes: CorrectedSuite places the canonical bytes
(canonical-datasets.json, verified by sha256) where v6's suite reads each task's dataset. Identity is by hash.

Order and state are enforced, and every check fails closed (exit 1):
- Every step runs the code-state gate (corrected_freeze.json: ccil paths equal to v6's ccil commit; tooling paths
  equal to the commit it names; a clean tree). A smoke root records the tooling check without requiring it.
- preflight creates the root and ROOT/root.json (plan, canonical datasets, seeds, smoke); every later step requires
  the same identity, and each requires its predecessor's passing report in ROOT/reports, written under the same
  frozen tooling commit. For a real root, train also requires the preflight report committed in the evidence folder,
  and evaluate the audit report, byte for byte.
- audit validates the v6 inventory, the exact planned units, every completed stage (manifest hashes), each stage's
  recorded inputs traced back to the canonical dataset within this root, and that no run ID or .npz artifact hash
  appears in v6.
- evaluate applies v6's section 9 rules: an algorithmic failure in a primary arm makes the task inconclusive; at most
  two infrastructure exclusions per primary arm; at least 16 paired seeds. It evaluates only units whose records are
  ok and records every exclusion and failed rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dataclasses import dataclass

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (HERE, REPO / "tools" / "m3_experiments", REPO / "tools" / "m3_pendulum", REPO / "tools" / "vector_env",
          REPO / "tools" / "legacy_conversion"):
    sys.path.insert(0, str(p))

EVIDENCE = REPO / "docs" / "fixtures" / "evidence" / "pendulum-corrected-demonstrator"
CANONICAL = EVIDENCE / "canonical-datasets.json"
PLAN = HERE / "corrected_study.yaml"
FREEZE = "tools/pendulum_eval/corrected_freeze.json"
V6_CCIL_COMMIT = "6fbcce543e8a1ebdf243063c179dd3af82d5db8a"
TOOLING_PATHS = ("tools/m3_experiments", "tools/m3_pendulum", "tools/vector_env", "tools/legacy_conversion",
                 "tools/pendulum_eval", "correct_il", "data", "config", "docs/artifacts/manifest.json",
                 "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json", str(CANONICAL.relative_to(REPO)))
PRIMARY_ARMS, TASKS = ("bc", "ccil"), ("continuous", "discontinuous")
STUDY_SEEDS, SMOKE_SEEDS = tuple(range(1201, 1221)), (9201, 9202, 9203)
C_PRIMARY = 1 - 0.05 / 4      # two tasks x two primary analyses
MIN_VALID, MAX_EXCLUDED = 16, 2  # v6 section 9
NEXT = {"train": "preflight", "audit": "train", "evaluate": "audit"}
COMMITTED_PREREQ = {"train": "preflight", "evaluate": "audit"}


class Refused(SystemExit):
    pass


@dataclass(frozen=True)
class Spec:
    """What distinguishes one study run by this driver: its plan, freeze manifest and evidence folder."""
    plan: Path
    freeze: str
    evidence: Path


CORRECTED = Spec(PLAN, FREEZE, EVIDENCE)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_commit() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()}


def canonical() -> dict:
    return json.loads(CANONICAL.read_text())["datasets"]


def study_plan(smoke: bool, spec: Spec = CORRECTED):
    import yaml

    from ccil.experiments import plan
    d = yaml.safe_load(spec.plan.read_text())
    if smoke:
        d["phases"]["study"]["seeds"] = list(SMOKE_SEEDS)
    return plan.from_dict(d)


def gate_state(smoke: bool, spec: Spec = CORRECTED) -> dict:
    from ccil.experiments import gate
    return gate.require(REPO, gate.Freeze(ccil_commit=V6_CCIL_COMMIT, tooling_paths=TOOLING_PATHS,
                                          manifest=spec.freeze), real=not smoke)


# --- root identity and step order -------------------------------------------------------------------------------------
def identity(smoke: bool, spec: Spec = CORRECTED) -> dict:
    return {"plan_sha256": sha(spec.plan), "canonical_manifest_sha256": sha(CANONICAL), "smoke": smoke,
            "seeds": list(SMOKE_SEEDS if smoke else STUDY_SEEDS),
            "datasets": {t: c["sha256"] for t, c in canonical().items()}}


def require_order(root: Path, step: str, spec: Spec = CORRECTED) -> tuple[dict, dict]:
    """The root's identity, the gate, and the predecessor's passing report under the same frozen tooling."""
    ident_path = root / "root.json"
    if not ident_path.exists():
        raise Refused(f"refused: {root} has no root.json; run preflight first")
    ident = json.loads(ident_path.read_text())
    smoke = ident.get("smoke") is True
    if {k: v for k, v in ident.items() if k != "root"} != identity(smoke, spec) or ident.get("root") != str(root):
        raise Refused(f"refused: {root} was created for a different plan, dataset, seeds or path")
    state = gate_state(smoke, spec)
    prev = root / "reports" / f"{NEXT[step]}.json"
    if not prev.exists():
        raise Refused(f"refused: {step} needs a passing {NEXT[step]} report in {root / 'reports'}")
    rep = json.loads(prev.read_text())
    if rep.get("all_pass") is not True:
        raise Refused(f"refused: the {NEXT[step]} report did not pass")
    if rep["gate"]["tooling"].get("tooling_commit") != state["tooling"].get("tooling_commit"):
        raise Refused(f"refused: the {NEXT[step]} report was written under different frozen tooling")
    if not smoke and step in COMMITTED_PREREQ:
        committed = spec.evidence / f"study-{COMMITTED_PREREQ[step]}.json"
        if not committed.exists() or committed.read_bytes() != prev.read_bytes():
            raise Refused(f"refused: {step} needs the {NEXT[step]} report committed as {committed.name}, unchanged")
    return ident, state


def write_report(root: Path, step: str, rep: dict) -> Path:
    rep["process"] = process_commit()
    dest = root / "reports" / f"{step}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    return dest


# --- the suite --------------------------------------------------------------------------------------------------------
def make_suite(root: Path, study, data: Path, smoke: bool):
    import study as s24
    from pendulum_suite import PendulumSuite

    class CorrectedSuite(PendulumSuite):
        """v6's suite with each task's dataset replaced by the canonical corrected bytes."""

        def __init__(self):
            self.root, self.study, self.smoke, self.preflight = root, study, smoke, False
            self._inputs = prepare_inputs(root, data, s24)

    return CorrectedSuite()


def prepare_inputs(root: Path, data: Path, s24) -> dict:
    """root/inputs as v6's suite reads it, each task's dataset path holding the canonical bytes. Refuses other bytes."""
    from convert_pendulum import convert_pendulum
    inputs = root / "inputs"
    if not (inputs / "conversion.json").exists():
        convert_pendulum(inputs)
    committed = json.loads((REPO / "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json").read_text())
    out = {}
    for task, c in canonical().items():
        dest = inputs / "data" / f"{s24.DATASETS[task]}.npz"
        if sha(dest) != c["sha256"]:
            src = data / c["local_name"]
            if not src.exists() or sha(src) != c["sha256"]:
                raise Refused(f"refused: {src} is not the canonical {task} dataset {c['sha256']}")
            shutil.copyfile(src, dest)
        want_config = committed["retained"][s24.CONVERTED[task]]["configs"]["corrected"]["sha256"]
        got = {"data": sha(dest), "config": sha(inputs / "config" / s24.CONFIGS[task])}
        if got != {"data": c["sha256"], "config": want_config}:
            raise Refused(f"refused: {task} inputs {got} are not the canonical dataset and committed configuration")
        out[task] = got
    return out


# --- v6 inventory and stage checks --------------------------------------------------------------------------------------
def v6_inventory(v6: Path) -> dict:
    """Every v6 study stage's run ID and .npz artifact hashes, from its manifest; the v6 BC and CCIL records must all
    be present with complete stages, and each policy checkpoint must equal its record."""
    from ccil.runs import validate_run
    missing = []
    for t in TASKS:
        for a in PRIMARY_ARMS:
            for s in STUDY_SEEDS:
                f = v6 / "results" / "study" / t / a / f"s{s}.json"
                if not f.exists():
                    missing.append(f"{t}/{a}/s{s}")
                    continue
                r = json.loads(f.read_text())
                for stage, d in r["stages"].items():
                    if validate_run(Path(d))[0] != "complete":
                        missing.append(f"{t}/{a}/s{s} {stage}")
                m = json.loads((Path(r["stages"]["policy.train"]) / "manifest.json").read_text())
                if m["files"]["checkpoints/policy.npz"]["sha256"] != r["checkpoint_sha256"]:
                    missing.append(f"{t}/{a}/s{s} checkpoint")
    if missing:
        raise Refused(f"refused: {v6} is not the v6 study root: {missing[:5]} ({len(missing)} problems)")
    ids, npz = set(), set()
    for d in (v6 / "stages" / "study").glob("*/s*/*"):
        mf = d / "manifest.json"
        if mf.exists():
            ids.add(d.name)
            npz |= {f["sha256"] for k, f in json.loads(mf.read_text())["files"].items() if k.endswith(".npz")}
    return {"run_ids": ids, "npz": npz}


def check_stages(root: Path, v6: dict, datasets: set | None = None) -> dict:
    """Every stage directory under root: complete; its recorded inputs are the study's datasets (by default the
    canonical datasets) or artifacts of other stages in this root; no run ID or .npz artifact hash appears in v6."""
    from ccil.runs import validate_run
    stages = sorted(d for d in (root / "stages").glob("*/*/s*/*") if d.is_dir())
    produced, info = {}, {}
    for d in stages:
        status = validate_run(d)[0]
        files = json.loads((d / "manifest.json").read_text())["files"] if status == "complete" else {}
        produced.update({f["sha256"]: d.name for f in files.values()})
        info[d] = (status, files)
    can = datasets if datasets is not None else {c["sha256"] for c in canonical().values()}
    out, ok = {}, bool(stages)
    for d in stages:
        status, files = info[d]
        inputs = json.loads((d / "stage.json").read_text())["inputs"] if (d / "stage.json").exists() else None
        untraced = None if inputs is None else {k: v for k, v in inputs.items() if v not in can and v not in produced}
        npz_in_v6 = [k for k, f in files.items() if k.endswith(".npz") and f["sha256"] in v6["npz"]]
        row = {"status": status, "inputs_untraced": untraced, "run_id_in_v6": d.name in v6["run_ids"],
               "npz_in_v6": npz_in_v6}
        row["pass"] = status == "complete" and untraced == {} and not row["run_id_in_v6"] and not npz_in_v6
        if d.name.startswith("data-prepare-"):
            row["pass"] &= inputs.get("dataset") in can
        ok &= row["pass"]
        out[str(d.relative_to(root))] = row
    return {"stages": out, "all_pass": ok}


# --- steps ------------------------------------------------------------------------------------------------------------
def preflight(root: Path, data: Path, v6_root: Path, smoke: bool, device: str, spec: Spec = CORRECTED,
              prior: tuple = (), datasets=None) -> dict:
    """datasets(root) gives the study's dataset hashes for check_stages once the root's inputs exist (default: the
    canonical datasets); prior lists earlier study roots whose run IDs and .npz artifacts must not reappear."""
    from ccil.data.prepare import prepare_stage
    if root.exists():
        raise Refused(f"refused: {root} exists; the corrected run root must be new")
    state = gate_state(smoke, spec)
    v6 = inventory(v6_root, prior)
    study = study_plan(smoke, spec)
    root.mkdir(parents=True)
    (root / "root.json").write_text(json.dumps({**identity(smoke, spec), "root": str(root)}, indent=1,
                                               sort_keys=True))
    suite = make_suite(root, study, data, smoke)
    for task, arm, seed, value in study.units("study"):
        cfg, _ = suite.config("study", task, seed, study.unit_settings(arm, value), study.arms[arm].data_fraction,
                              device, root / "stages" / "study" / task / f"s{seed}")
        prepare_stage(cfg)
    checks = check_stages(root, v6, datasets(root) if datasets else None)
    return {"step": "preflight", "gate": state, "smoke": smoke, **checks}


def inventory(v6_root: Path, prior: tuple = ()) -> dict:
    """v6's inventory, joined with each prior study root's (each must pass the same completeness checks)."""
    out = v6_inventory(v6_root)
    for p in prior:
        more = v6_inventory(p)
        out = {k: out[k] | more[k] for k in out}
    return out


def train(root: Path, data: Path, device: str, workers: int, spec: Spec = CORRECTED) -> dict:
    import time

    from ccil.experiments import gate, runner
    ident, state = require_order(root, "train", spec)
    smoke = ident["smoke"]
    study = study_plan(smoke, spec)
    suite = make_suite(root, study, data, smoke)

    def recheck():
        now = gate_state(smoke, spec)
        if now["head"] != state["head"]:
            raise gate.GateRefused(f"code-state gate refused: HEAD moved from {state['head']} to {now['head']}")

    out = runner.run_phase(root, study, "study", suite, gate=state, smoke=smoke, device=device, workers=workers,
                           recheck=recheck, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    return {"step": "train", "gate": state, "summary": out, "all_pass": True}


def records(root: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted((root / "results" / "study").glob("*/*/s*.json"))]


def audit(root: Path, v6_root: Path) -> dict:
    ident, state = require_order(root, "audit")
    v6 = v6_inventory(v6_root)
    planned = {(t, a, s) for t, a, s, _ in study_plan(ident["smoke"]).units("study")}
    recs = records(root)
    got = {(r["task"], r["arm"], r["seed"]) for r in recs}
    not_canonical = [f"{r['task']}/{r['arm']}/s{r['seed']}" for r in recs
                     if r["inputs"]["data_sha256"] != ident["datasets"][r["task"]]]
    stages = check_stages(root, v6)
    ok = got == planned and len(recs) == len(planned) and not not_canonical and stages["all_pass"]
    return {"step": "audit", "gate": state, "planned_units": len(planned), "records": len(recs),
            "missing_units": sorted(map(str, planned - got)), "unplanned_units": sorted(map(str, got - planned)),
            "statuses": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["status"] for r in recs},
            "records_not_canonical": not_canonical, **stages, "all_pass": ok}


def primary_decision(d: dict, status: dict, c: float) -> dict:
    """v6 section 9 on paired per-seed differences d {seed: value} with status {arm: {seed: status}} of the two
    primary arms: any algorithmic failure is inconclusive; more than MAX_EXCLUDED infrastructure exclusions in an arm
    is inconclusive; fewer than MIN_VALID paired seeds is inconclusive; otherwise the union interval decides."""
    from ccil.experiments import analysis
    reasons = []
    for arm, st in status.items():
        if any(v == "algorithmic_failure" for v in st.values()):
            reasons.append(f"algorithmic failure in {arm}")
        if sum(v == "infrastructure_fault" for v in st.values()) > MAX_EXCLUDED:
            reasons.append(f"more than {MAX_EXCLUDED} infrastructure exclusions in {arm}")
    seeds = sorted(s for s in d if all(st.get(s) == "ok" for st in status.values()))
    if len(seeds) < MIN_VALID:
        reasons.append(f"{len(seeds)} paired seeds (< {MIN_VALID})")
    res = analysis.union_interval(np.array([d[s] for s in seeds]), c) if len(seeds) >= 2 else {"interval": None}
    iv = res.get("interval")
    if reasons or iv is None:
        outcome = "inconclusive"
    else:
        outcome = "increases" if iv[0] > 0 else "decreases" if iv[1] < 0 else "no_detectable_difference"
    return {**res, "seeds": seeds, "per_seed": {str(s): float(d[s]) for s in seeds}, "outcome": outcome,
            "inconclusive_reasons": reasons}


def known_answer_cases() -> dict:
    """The section 9 rules on constructed inputs whose outcome follows from the rule, not from the code."""
    seeds = list(range(20))
    d = {s: 0.1 + 0.01 * (s % 3) for s in seeds}  # clearly positive
    ok = {s: "ok" for s in seeds}

    def with_(changes):
        st = dict(ok)
        st.update(changes)
        return st

    cases = {
        "all_ok_positive": ({"bc": ok, "ccil": ok}, "increases"),
        "algorithmic_failure_in_ccil": ({"bc": ok, "ccil": with_({3: "algorithmic_failure"})}, "inconclusive"),
        "two_infrastructure_exclusions": ({"bc": with_({1: "infrastructure_fault", 2: "infrastructure_fault"}),
                                           "ccil": ok}, "increases"),
        "three_infrastructure_exclusions": ({"bc": with_({1: "infrastructure_fault", 2: "infrastructure_fault",
                                                          3: "infrastructure_fault"}), "ccil": ok}, "inconclusive"),
        "fifteen_paired_seeds": ({"bc": with_({1: "infrastructure_fault", 2: "infrastructure_fault"}),
                                  "ccil": with_({3: "infrastructure_fault", 4: "infrastructure_fault",
                                                 5: "infrastructure_fault"})}, "inconclusive"),
    }
    out = {}
    for name, (status, want) in cases.items():
        got = primary_decision(d, status, C_PRIMARY)
        out[name] = {"expected": want, "got": got["outcome"], "n": len(got["seeds"]),
                     "reasons": got["inconclusive_reasons"], "pass": got["outcome"] == want}
    out["two_infrastructure_exclusions"]["pass"] &= out["two_infrastructure_exclusions"]["n"] == 18
    return out


def evaluate(root: Path, out: Path, device: str) -> dict:
    import env as canonical_env
    import expert
    import historical as h

    from ccil.experiments import analysis
    from ccil.training.vector_bc import VectorBCPolicy
    ident, state = require_order(root, "evaluate")
    smoke = ident["smoke"]
    seeds = ident["seeds"]
    hist_seed = dict(zip(seeds, STUDY_SEEDS))  # smoke pairs its seeds with study seeds to exercise the code path only
    out.mkdir(parents=True)
    part_b = json.loads((h.EVIDENCE / "part-b.json").read_text())
    hist = part_b["per_seed"]
    recs = {(r["task"], r["arm"], r["seed"]): r for r in records(root)}
    status = {t: {a: {s: recs[(t, a, s)]["status"] for s in seeds} for a in PRIMARY_ARMS} for t in TASKS}

    def demonstrator(obs):  # the corrected controller under its action contract: float64 clamp, cast to float32
        return expert.act(obs).astype(np.float32).astype(np.float64).reshape(-1, 1)

    scores, rollouts, failed_rollouts, excluded = {}, {}, [], []
    conds = [("0.0", 0.0, "none")] + [(f"{ch}/{n}", n, ch) for ch in h.CHANNELS for n in h.LEVELS]
    for t in TASKS:
        items, keys = [], []
        for a in PRIMARY_ARMS:
            for s in seeds:
                r = recs[(t, a, s)]
                if r["status"] != "ok":
                    excluded.append({"unit": f"{t}/{a}/s{s}", "status": r["status"]})
                    continue
                ckpt = Path(r["stages"]["policy.train"]) / "checkpoints" / "policy.npz"
                if sha(ckpt) != r["checkpoint_sha256"]:
                    raise Refused(f"refused: {t}/{a}/s{s} checkpoint does not match its record")
                pol = VectorBCPolicy.load(ckpt, device=device)[0].predict
                for key, n, ch in conds:
                    items.append((pol, s + h.FRESH_OFFSET, n, ch))
                    keys.append((t, a, key, s))
        for s in seeds:
            items.append((demonstrator, s + h.FRESH_OFFSET, 0.0, "none"))
            keys.append((t, "corrected_demonstrator", "0.0", s))
        for res, (tt, a, key, s) in zip(h.channel_rollouts(items, t), keys):
            path = out / tt / a / f"s{s}-{key.replace('/', '-')}.npz"
            canonical_env.save(res, path)
            rollouts[f"{tt}/{a}/s{s}/{key}"] = sha(path)
            if len(res["failed_episodes"]):
                failed_rollouts.append(f"{tt}/{a}/s{s}/{key}")
                if key == "0.0" and a in PRIMARY_ARMS:
                    status[tt][a][s] = "algorithmic_failure"  # non-finite actions on the fresh starts
                continue
            scores[(tt, a, key, s)] = h.seed_scores(res)

    def diff(t, m, key="0.0"):
        return {s: scores[(t, "ccil", key, s)][m] - scores[(t, "bc", key, s)][m]
                for s in seeds if (t, "ccil", key, s) in scores and (t, "bc", key, s) in scores}

    def hist_diff(t, m):
        return {s: hist[f"{t}/ccil/0.0/s{hist_seed[s]}"][m] - hist[f"{t}/bc/0.0/s{hist_seed[s]}"][m] for s in seeds}

    rep = {"step": "evaluate", "gate": state, "smoke": smoke, "device": device, "primary": {}, "secondary_95": {},
           "noise_95": {}, "failures": {"excluded_units": excluded, "failed_rollouts": failed_rollouts,
                                        "status": status}}
    for t in TASKS:
        d1, hd = diff(t, "upright_occupancy"), hist_diff(t, "upright_occupancy")
        rep["primary"][t] = {
            "corrected_ccil_minus_bc": primary_decision(d1, status[t], C_PRIMARY),
            "change_in_ccil_minus_bc": {**primary_decision({s: d1[s] - hd[s] for s in d1}, status[t], C_PRIMARY),
                                        "reading": "the effect of replacing the demonstrations on the whole CCIL-"
                                                   "versus-BC pipeline; not the isolated effect of the wrapping defect"}}
        for m in h.METRICS:
            dm, hm = diff(t, m), hist_diff(t, m)
            valid = [s for s in dm if all(status[t][a][s] == "ok" for a in PRIMARY_ARMS)]
            rep["secondary_95"][f"{t}/corrected_ccil_minus_bc/{m}"] = analysis.union_interval(
                np.array([dm[s] for s in valid]), 0.95)
            rep["secondary_95"][f"{t}/change_in_ccil_minus_bc/{m}"] = analysis.union_interval(
                np.array([dm[s] - hm[s] for s in valid]), 0.95)
            for a in PRIMARY_ARMS:
                ch = [scores[(t, a, "0.0", s)][m] - hist[f"{t}/{a}/0.0/s{hist_seed[s]}"][m]
                      for s in seeds if (t, a, "0.0", s) in scores]
                rep["secondary_95"][f"{t}/{a}_corrected_minus_historical/{m}"] = analysis.union_interval(
                    np.array(ch), 0.95)
        for ch in h.CHANNELS:
            for n in h.LEVELS:
                dn = diff(t, "upright_occupancy", f"{ch}/{n}")
                rep["noise_95"][f"{t}/ccil-bc/{ch}/{n}"] = analysis.union_interval(np.array(list(dn.values())), 0.95)
    rep["zero_action_reference"] = {"source": "historical Part B (part-b.json), unchanged",
                                    **{t: part_b["arm_summaries"][f"{t}/zero/0.0"] for t in TASKS}}
    rep["arm_summaries"] = h.arm_summaries(scores)
    rep["per_seed"] = {f"{t}/{a}/{n}/s{s}": {m: v[m] for m in h.METRICS} for (t, a, n, s), v in scores.items()}
    rep["rollout_sha256"] = rollouts
    rep["all_pass"] = True
    return rep


# --- smoke: the whole sequence, reduced, with refusals ------------------------------------------------------------------
def smoke(work: Path, data: Path, v6_root: Path, device: str, workers: int) -> dict:
    if work.exists():
        raise Refused(f"refused: {work} exists")
    work.mkdir(parents=True)
    root = (work / "root").resolve()
    out = {"known_answer_cases": known_answer_cases(), "refusals": {}, "steps": {}}

    def refused(name, fn):
        try:
            fn()
            out["refusals"][name] = {"refused": False}
        except SystemExit as e:
            out["refusals"][name] = {"refused": True, "message": str(e)[:300]}

    wrong = work / "wrong-data"
    wrong.mkdir()
    for c in canonical().values():  # the historical bytes under the canonical names: a wrong dataset
        (wrong / c["local_name"]).write_bytes(b"not the canonical dataset")
    empty_v6 = work / "empty-v6"
    empty_v6.mkdir()
    refused("preflight_wrong_dataset", lambda: preflight((work / "r-wrong").resolve(), wrong, v6_root, True, device))
    refused("preflight_empty_v6_root", lambda: preflight((work / "r-v6").resolve(), data, empty_v6, True, device))
    no_pre = (work / "r-nopre").resolve()
    no_pre.mkdir()
    refused("train_without_preflight", lambda: train(no_pre, data, device, workers))

    for step, fn in (("preflight", lambda: preflight(root, data, v6_root, True, device)),
                     ("train", lambda: train(root, data, device, workers))):
        rep = fn()
        write_report(root, step, rep)
        out["steps"][step] = rep["all_pass"]
        if step == "train":
            refused("evaluate_before_audit", lambda: evaluate(root, work / "eval-early", device))
    refused("audit_empty_v6_root", lambda: audit(root, empty_v6))
    rep = audit(root, v6_root)
    write_report(root, "audit", rep)
    out["steps"]["audit"] = rep["all_pass"]
    corrupt = (work / "root-corrupt").resolve()
    shutil.copytree(root, corrupt)
    ident = json.loads((corrupt / "root.json").read_text())
    (corrupt / "root.json").write_text(json.dumps({**ident, "root": str(corrupt)}, indent=1, sort_keys=True))
    victim = next(corrupt.glob("stages/study/*/s*/policy-train-*/checkpoints/policy.npz"))
    before = victim.read_bytes()
    mid = len(before) // 2
    victim.write_bytes(before[:mid] + bytes([before[mid] ^ 0xFF]) + before[mid + 1:])  # invert one byte
    if victim.read_bytes() == before:
        raise SystemExit("smoke harness error: the corruption did not change the file")
    rc = audit(corrupt, v6_root)
    out["refusals"]["audit_corrupted_stage"] = {"refused": not rc["all_pass"], "corrupted": str(victim.relative_to(work))}
    rep = evaluate(root, work / "eval", device)
    write_report(root, "evaluate", rep)
    out["steps"]["evaluate"] = rep["all_pass"]
    out["smoke_primary_outcomes"] = {t: {k: v["outcome"] for k, v in p.items()} for t, p in rep["primary"].items()}
    out["all_pass"] = (all(v["refused"] for v in out["refusals"].values()) and all(out["steps"].values())
                       and all(c["pass"] for c in out["known_answer_cases"].values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("preflight", "train", "audit", "evaluate", "smoke"))
    ap.add_argument("--root", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--v6", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--smoke", action="store_true", help="preflight only: a smoke root on non-study seeds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.step == "smoke":
        rep = smoke(a.work.resolve(), a.data, a.v6, a.device, a.workers)
        rep["process"] = process_commit()
        dest = EVIDENCE / "study-smoke.json"
        dest.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    else:
        root = a.root.resolve()
        rep = {"preflight": lambda: preflight(root, a.data, a.v6, a.smoke, a.device),
               "train": lambda: train(root, a.data, a.device, a.workers),
               "audit": lambda: audit(root, a.v6),
               "evaluate": lambda: evaluate(root, a.out, a.device)}[a.step]()
        dest = write_report(root, a.step, rep)
    print(json.dumps({"step": a.step, "pass": rep["all_pass"], "report": str(dest)}, indent=1))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
