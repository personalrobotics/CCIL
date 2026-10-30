#!/usr/bin/env python3
"""The demonstration-count study (docs/experiments/pendulum-demonstration-count-predeclaration.md).

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_eval/count_study.py"
    $R preflight --root NEW_ROOT --data CANONICAL_DIR --v6 V6_ROOT --corrected CORRECTED_ROOT [--smoke]
    $R train     --root ROOT --data CANONICAL_DIR
    $R audit     --root ROOT --v6 V6_ROOT --corrected CORRECTED_ROOT
    $R evaluate  --root ROOT --corrected CORRECTED_ROOT --out NEW_DIR
    $R smoke     --work NEW_DIR --data CANONICAL_DIR --v6 V6_ROOT --corrected CORRECTED_ROOT

The corrected study's driver (corrected_study.py) runs this plan (count_study.yaml): the same frozen ccil source,
suite, settings, gate and step order, with its own freeze manifest and evidence folder. What differs:
- Each unit trains on its seed's 100-of-500 subset of the canonical discontinuous dataset (v6's subset rule at
  fraction 0.2; BC and CCIL at a seed share it). Every record and every data.prepare stage must carry its seed's
  subset hash.
- No run ID or .npz artifact may appear in v6 or in the corrected study's root. The corrected root is identified by
  its audit report, which must equal the committed study-audit.json byte for byte.
- evaluate runs the two units per seed noise-free on the fresh starts, and takes the 500-demonstration values from the
  committed corrected study-evaluate.json (pinned by sha256). Primary at c = 0.975: (CCIL - BC)_100, and
  (CCIL - BC)_100 - (CCIL - BC)_500, by seed, under v6's section 9 rules. Secondary, descriptive at 95%: each arm's
  100 - 500 change per metric, CCIL label accuracy at 100 against 500, and continuous CCIL - BC beside discontinuous
  at 100.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corrected_study as cs  # noqa: E402

SPEC = cs.Spec(HERE / "count_study.yaml", "tools/pendulum_eval/count_freeze.json",
               cs.REPO / "docs" / "fixtures" / "evidence" / "pendulum-demonstration-count")
TASK, ARMS, BASE = "discontinuous", ("bc_100", "ccil_100"), {"bc_100": "bc", "ccil_100": "ccil"}
C_PRIMARY = 1 - 0.05 / 2  # two primary analyses
CORRECTED_EVAL = cs.EVIDENCE / "study-evaluate.json"
CORRECTED_EVAL_SHA256 = "6a1f154b55a79524f8db2ad4b9cb542b8a8519d197d8f778c7a726845ed3e19d"
Refused = cs.Refused


def require_corrected(corrected: Path) -> None:
    rep = corrected / "reports" / "audit.json"
    if not rep.exists() or rep.read_bytes() != (cs.EVIDENCE / "study-audit.json").read_bytes():
        raise Refused(f"refused: {corrected} is not the corrected study root (its audit report differs from the "
                      "committed study-audit.json)")


def subsets(root: Path) -> dict:
    """{seed: sha256} of each seed's subset, rebuilt from the root's canonical dataset (an existing file must equal it)."""
    import study as s24
    seeds = json.loads((root / "root.json").read_text())["seeds"]
    fracs = {a.data_fraction for a in cs.study_plan(True, SPEC).arms.values()}
    if len(fracs) != 1:
        raise Refused(f"refused: the arms do not share one data fraction: {fracs}")
    frac = fracs.pop()
    return {s: s24.subset(root, TASK, s, frac)[1] for s in seeds}


def preflight(root, data, v6, corrected, smoke, device):
    require_corrected(corrected)
    return cs.preflight(root, data, v6, smoke, device, SPEC, prior=(corrected,),
                        datasets=lambda r: set(subsets(r).values()))


def train(root, data, device, workers):
    return cs.train(root, data, device, workers, SPEC)


def audit(root: Path, v6_root: Path, corrected: Path) -> dict:
    require_corrected(corrected)
    ident, state = cs.require_order(root, "audit", SPEC)
    inv = cs.inventory(v6_root, (corrected,))
    planned = {(t, a, s) for t, a, s, _ in cs.study_plan(ident["smoke"], SPEC).units("study")}
    recs = cs.records(root)
    got = {(r["task"], r["arm"], r["seed"]) for r in recs}
    sub = subsets(root)
    not_subset = [f"{r['task']}/{r['arm']}/s{r['seed']}" for r in recs if r["inputs"]["data_sha256"] != sub[r["seed"]]]
    stages = cs.check_stages(root, inv, set(sub.values()))
    ok = got == planned and len(recs) == len(planned) and not not_subset and stages["all_pass"]
    return {"step": "audit", "gate": state, "planned_units": len(planned), "records": len(recs),
            "missing_units": sorted(map(str, planned - got)), "unplanned_units": sorted(map(str, got - planned)),
            "statuses": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["status"] for r in recs},
            "subset_sha256": {str(s): d for s, d in sub.items()}, "records_not_their_seed_subset": not_subset,
            **stages, "all_pass": ok}


def label_summary(rec: dict) -> dict:
    la = rec["label_accuracy"]
    e = la["true_label_error"]
    return {"acceptance_rate": la["acceptance_rate"], "true_label_error_mean": e["mean"],
            "true_label_error_q50": e["quantiles"]["0.5"], "true_label_error_q95": e["quantiles"]["0.95"]}


def evaluate(root: Path, out: Path, corrected: Path, device: str) -> dict:
    import env as canonical_env
    import historical as h

    from ccil.experiments import analysis
    from ccil.training.vector_bc import VectorBCPolicy
    require_corrected(corrected)
    ident, state = cs.require_order(root, "evaluate", SPEC)
    if cs.sha(CORRECTED_EVAL) != CORRECTED_EVAL_SHA256:
        raise Refused(f"refused: {CORRECTED_EVAL.name} is not the audited corrected evaluation report")
    smoke, seeds = ident["smoke"], ident["seeds"]
    base_seed = dict(zip(seeds, cs.STUDY_SEEDS))  # smoke pairs its seeds with study seeds to exercise the code path
    out.mkdir(parents=True)
    r500 = json.loads(CORRECTED_EVAL.read_text())
    v500 = r500["per_seed"]
    recs = {(r["arm"], r["seed"]): r for r in cs.records(root) if r["task"] == TASK}
    status = {a: {s: recs[(a, s)]["status"] for s in seeds} for a in ARMS}

    scores, rollouts, failed, excluded, items, keys = {}, {}, [], [], [], []
    for a in ARMS:
        for s in seeds:
            r = recs[(a, s)]
            if r["status"] != "ok":
                excluded.append({"unit": f"{TASK}/{a}/s{s}", "status": r["status"]})
                continue
            ckpt = Path(r["stages"]["policy.train"]) / "checkpoints" / "policy.npz"
            if cs.sha(ckpt) != r["checkpoint_sha256"]:
                raise Refused(f"refused: {TASK}/{a}/s{s} checkpoint does not match its record")
            items.append((VectorBCPolicy.load(ckpt, device=device)[0].predict, s + h.FRESH_OFFSET, 0.0, "none"))
            keys.append((a, s))
    for res, (a, s) in zip(h.channel_rollouts(items, TASK), keys):
        path = out / TASK / a / f"s{s}-0.0.npz"
        canonical_env.save(res, path)
        rollouts[f"{TASK}/{a}/s{s}/0.0"] = cs.sha(path)
        if len(res["failed_episodes"]):
            failed.append(f"{TASK}/{a}/s{s}/0.0")
            status[a][s] = "algorithmic_failure"  # non-finite actions on the fresh starts
            continue
        scores[(a, s)] = h.seed_scores(res)

    def d100(m):
        return {s: scores[("ccil_100", s)][m] - scores[("bc_100", s)][m]
                for s in seeds if ("ccil_100", s) in scores and ("bc_100", s) in scores}

    def d500(m):
        return {s: v500[f"{TASK}/ccil/0.0/s{base_seed[s]}"][m] - v500[f"{TASK}/bc/0.0/s{base_seed[s]}"][m]
                for s in seeds}

    occ100, occ500 = d100("upright_occupancy"), d500("upright_occupancy")
    rep = {"step": "evaluate", "gate": state, "smoke": smoke, "device": device,
           "corrected_evaluation": {"report": CORRECTED_EVAL.name, "sha256": CORRECTED_EVAL_SHA256},
           "primary": {
               "ccil_minus_bc_100": cs.primary_decision(occ100, status, C_PRIMARY),
               "change_100_minus_500": {
                   **cs.primary_decision({s: occ100[s] - occ500[s] for s in occ100}, status, C_PRIMARY),
                   "reading": "a positive value supports the hypothesis that more demonstrations shrink CCIL's "
                              "incremental benefit; averaged jointly over subset selection and training randomness"}},
           "secondary_95": {}, "failures": {"excluded_units": excluded, "failed_rollouts": failed, "status": status}}
    valid = [s for s in seeds if all(status[a][s] == "ok" for a in ARMS)]
    for m in h.METRICS:
        dm, dm5 = d100(m), d500(m)
        rep["secondary_95"][f"ccil_minus_bc_100/{m}"] = analysis.union_interval([dm[s] for s in valid], 0.95)
        rep["secondary_95"][f"change_100_minus_500/{m}"] = analysis.union_interval(
            [dm[s] - dm5[s] for s in valid], 0.95)
        for a in ARMS:
            rep["secondary_95"][f"{a}_minus_{BASE[a]}_500/{m}"] = analysis.union_interval(
                [scores[(a, s)][m] - v500[f"{TASK}/{BASE[a]}/0.0/s{base_seed[s]}"][m] for s in seeds
                 if (a, s) in scores], 0.95)

    # CCIL's labels at 100 against 500, by seed (the label_accuracy each CCIL record carries).
    c500 = {r["seed"]: r for r in cs.records(corrected) if r["task"] == TASK and r["arm"] == "ccil"}
    per = {}
    for s in seeds:
        if status["ccil_100"][s] == "ok" and base_seed[s] in c500:
            per[str(s)] = {"100": label_summary(recs[("ccil_100", s)]), "500": label_summary(c500[base_seed[s]])}
    rep["label_accuracy"] = {"per_seed": per, "difference_100_minus_500_95": {
        k: analysis.union_interval([v["100"][k] - v["500"][k] for v in per.values()], 0.95)
        for k in ("acceptance_rate", "true_label_error_mean")}}
    rep["continuous_beside"] = {
        "label": "descriptive; the tasks also differ in demonstration starts and subsets, so this does not isolate "
                 "the discontinuity",
        "continuous_ccil_minus_bc_100_demonstrations": r500["secondary_95"]["continuous/corrected_ccil_minus_bc/"
                                                                            "upright_occupancy"],
        "discontinuous_ccil_minus_bc_100_demonstrations": rep["secondary_95"]["ccil_minus_bc_100/upright_occupancy"]}
    rep["arm_summaries"] = h.arm_summaries({(TASK, a, "0.0", s): v for (a, s), v in scores.items()})
    rep["per_seed"] = {f"{TASK}/{a}/0.0/s{s}": {m: v[m] for m in h.METRICS} for (a, s), v in scores.items()}
    rep["rollout_sha256"] = rollouts
    rep["all_pass"] = True
    return rep


# --- smoke: the whole sequence, reduced, with refusals ------------------------------------------------------------------
def smoke(work: Path, data: Path, v6_root: Path, corrected: Path, device: str, workers: int) -> dict:
    if work.exists():
        raise Refused(f"refused: {work} exists")
    work.mkdir(parents=True)
    root = (work / "root").resolve()
    out = {"known_answer_cases": cs.known_answer_cases(), "refusals": {}, "steps": {}}

    def refused(name, fn):
        try:
            fn()
            out["refusals"][name] = {"refused": False}
        except SystemExit as e:
            out["refusals"][name] = {"refused": True, "message": str(e)[:300]}

    wrong = work / "wrong-data"
    wrong.mkdir()
    for c in cs.canonical().values():
        (wrong / c["local_name"]).write_bytes(b"not the canonical dataset")
    empty = work / "empty-root"
    empty.mkdir()
    refused("preflight_wrong_dataset",
            lambda: preflight((work / "r-wrong").resolve(), wrong, v6_root, corrected, True, device))
    refused("preflight_empty_v6_root",
            lambda: preflight((work / "r-v6").resolve(), data, empty, corrected, True, device))
    refused("preflight_not_corrected_root",
            lambda: preflight((work / "r-corr").resolve(), data, v6_root, empty, True, device))
    no_pre = (work / "r-nopre").resolve()
    no_pre.mkdir()
    refused("train_without_preflight", lambda: train(no_pre, data, device, workers))

    for step, fn in (("preflight", lambda: preflight(root, data, v6_root, corrected, True, device)),
                     ("train", lambda: train(root, data, device, workers))):
        rep = fn()
        cs.write_report(root, step, rep)
        out["steps"][step] = rep["all_pass"]
    refused("evaluate_before_audit", lambda: evaluate(root, work / "eval-early", corrected, device))
    refused("audit_empty_v6_root", lambda: audit(root, empty, corrected))
    rep = audit(root, v6_root, corrected)
    cs.write_report(root, "audit", rep)
    out["steps"]["audit"] = rep["all_pass"]
    out["smoke_subsets"] = rep["subset_sha256"]

    def copy_root(name):
        c = (work / name).resolve()
        shutil.copytree(root, c)
        ident = json.loads((c / "root.json").read_text())
        (c / "root.json").write_text(json.dumps({**ident, "root": str(c)}, indent=1, sort_keys=True))
        return c

    def corrupt(path: Path):
        before = path.read_bytes()
        mid = len(before) // 2
        path.write_bytes(before[:mid] + bytes([before[mid] ^ 0xFF]) + before[mid + 1:])  # invert one byte
        if path.read_bytes() == before:
            raise SystemExit("smoke harness error: the corruption did not change the file")
        return str(path.relative_to(work))

    c = copy_root("root-corrupt-stage")
    victim = corrupt(next(c.glob("stages/study/*/s*/policy-train-*/checkpoints/policy.npz")))
    rc = audit(c, v6_root, corrected)
    out["refusals"]["audit_corrupted_stage"] = {"refused": not rc["all_pass"], "corrupted": victim}
    c = copy_root("root-corrupt-subset")
    victim = corrupt(next(c.glob("inputs/subsets/*.npz")))
    refused("audit_corrupted_subset", lambda: audit(c, v6_root, corrected))
    out["refusals"]["audit_corrupted_subset"]["corrupted"] = victim
    rep = evaluate(root, work / "eval", corrected, device)
    cs.write_report(root, "evaluate", rep)
    out["steps"]["evaluate"] = rep["all_pass"]
    out["smoke_primary_outcomes"] = {k: v["outcome"] for k, v in rep["primary"].items()}
    out["all_pass"] = (all(v["refused"] for v in out["refusals"].values()) and all(out["steps"].values())
                       and all(k["pass"] for k in out["known_answer_cases"].values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("preflight", "train", "audit", "evaluate", "smoke"))
    ap.add_argument("--root", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--v6", type=Path)
    ap.add_argument("--corrected", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--smoke", action="store_true", help="preflight only: a smoke root on non-study seeds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.step == "smoke":
        rep = smoke(a.work.resolve(), a.data, a.v6, a.corrected, a.device, a.workers)
        rep["process"] = cs.process_commit()
        SPEC.evidence.mkdir(parents=True, exist_ok=True)
        dest = SPEC.evidence / "study-smoke.json"
        dest.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    else:
        root = a.root.resolve()
        rep = {"preflight": lambda: preflight(root, a.data, a.v6, a.corrected, a.smoke, a.device),
               "train": lambda: train(root, a.data, a.device, a.workers),
               "audit": lambda: audit(root, a.v6, a.corrected),
               "evaluate": lambda: evaluate(root, a.out, a.corrected, a.device)}[a.step]()
        dest = cs.write_report(root, a.step, rep)
    print(json.dumps({"step": a.step, "pass": rep["all_pass"], "report": str(dest)}, indent=1))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
