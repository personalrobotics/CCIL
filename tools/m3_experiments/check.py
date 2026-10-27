#!/usr/bin/env python3
"""Evidence for ccil.experiments (#112).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_experiments/check.py --out DIR

1. analysis: on synthetic R tables covering every outcome (improvement, deterioration, nulls with a passing
   and a failing positive control, a primary-arm algorithmic failure, too many exclusions, too few valid
   seeds, a degenerate BCa interval), ccil.experiments.analysis.decide with the #24 predeclaration gives
   exactly the #24 analysis's result: outcome, intervals, reasons, control and relative effect.
2. gate: on scratch clones of this commit (clean, with main's #24 v4 freeze manifest; an uncommitted src change; an untracked file;
   a manifest naming the current commit; a later committed tooling change; a manifest naming no
   commit), ccil.experiments.gate with the #24 freeze decides exactly as the #24 provenance module, for real
   and smoke runs.
3. plan: tools/m3_experiments/pendulum_v4.yaml (the #24 v4 design as a plan) expands to exactly the #24
   drivers' units for tuning and for the study (with a tuning choice), in the same order, including every
   arm's settings, noise levels and data fractions; malformed specs are refused with every problem listed.
4. records (with --smoke-root, a #24 check_study smoke root): ccil.experiments.records.verify_phase with the
   Pendulum hooks accepts the clean root with the same recomputed R as the #24 verify_phase, recomputes the
   same tuning choice, and refuses exactly when the #24 verifier refuses: an edited study-record R and an extra
   study unit (each restored afterwards).
5. runner (with --runner): ccil.experiments.runner on a small smoke plan (both tasks; bc, ccil, noise_bc, bc_frac10
   and zero; two seeds) with the Pendulum suite:
   - every record verifies (verify_phase, every rollout replayed through the integrator that produced it);
   - a tuning phase of a modern-recipe BC arm (AdamW, cosine, step budget, clipping, EMA, checkpoint selection on
     held-out episodes) chooses, per task, the learning rate with the lowest selected-checkpoint validation loss;
   - `ccil policy train` with each trained record's overrides reuses every stage the runner wrote, in the same
     directories (the runner's stage identities are the CLI's);
   - a second run writes nothing; with two records removed, a run writes exactly those, trains nothing and
     reuses every stage, and the phase verifies again;
   - batched evaluation: each trained unit's policy rolled out alone (tools/vector_env rollout) gives the same
     saved rollout, bit for bit (every array), as its batched evaluation;
   - memory exhaustion (injected: any fit or training group of more than two models raises
     torch.cuda.OutOfMemoryError, and every training of one bc_frac10 unit does) splits groups down to the size
     that fits, records the always-failing unit as an infrastructure fault after its retry, and the phase
     verifies.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
import analyze as m24  # noqa: E402  (the #24 analysis, the reference)
import check_study  # noqa: E402  (its synthetic tables)

import provenance as p24  # noqa: E402  (the #24 gate, the reference)

from ccil.experiments.analysis import Decision, decide  # noqa: E402
from ccil.experiments import plan  # noqa: E402
from ccil.experiments.gate import Freeze, gate  # noqa: E402

FREEZE_24 = Freeze(ccil_commit=p24.CCIL_FROZEN, tooling_paths=p24.TOOLING_PATHS, manifest=p24.FREEZE)

PREDECLARED_24 = Decision(treatment="ccil", baseline="bc", control=("bc", "bc_frac10"), confidence=0.975,
                          min_valid=16, max_excluded=2,
                          outcomes=("ccil_improves_on_bc", "ccil_is_worse_than_bc", "no_detectable_difference"))


def analysis_matches_24() -> dict:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 20, 20)
    sym = np.tile([15.0, -15.0], 10)
    syn = check_study.synthetic
    scenarios = {
        "improves": syn(100 + noise), "worse": syn(-100 + noise), "null_control_passes": syn(sym),
        "null_control_fails": syn(sym, control_gap=0.0),
        "algorithmic_failure": syn(100 + noise, status={("ccil", 105): "algorithmic_failure"}),
        "three_exclusions": syn(100 + noise, status={("bc", s): "infrastructure_fault" for s in (101, 102, 103)}),
        "fifteen_valid": syn(100 + noise, status={**{("bc", s): "infrastructure_fault" for s in (101, 102)},
                                                  **{("ccil", s): "infrastructure_fault" for s in (103, 104, 105)}}),
        "degenerate_bca": syn(np.full(20, 50.0))}
    rows = {}
    for name, tables in scenarios.items():
        ref = m24.primary(tables, "t")
        got = decide(tables, "t", PREDECLARED_24)
        keys = ("outcome", "inconclusive_reasons")
        same = (all(ref[k] == got[k] for k in keys) and ref["comparison"]["interval"] == got["comparison"]["interval"]
                and ref["positive_control"]["passes"] == got["positive_control"]["passes"]
                and ref["relative_effect"] == got["relative_effect"])
        rows[name] = {"identical": same, "outcome": got["outcome"]}
    return {"pass": all(r["identical"] for r in rows.values()), "scenarios": rows}


def gate_matches_24(out: Path) -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    clone = out / "clone"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(REPO), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", head], check=True)
    git = ["git", "-C", str(clone), "-c", "user.name=check", "-c", "user.email=check@invalid"]
    rows = {}
    # Freeze this commit's ccil for both implementations, so the scenarios produce passes as well as refusals.
    saved, p24.CCIL_FROZEN = p24.CCIL_FROZEN, head
    freeze = Freeze(ccil_commit=head, tooling_paths=p24.TOOLING_PATHS, manifest=p24.FREEZE)

    def both(name):
        for real in (True, False):
            a, b = p24.gate(clone, real)["ok"], gate(clone, freeze, real)["ok"]
            rows[f"{name}/{'real' if real else 'smoke'}"] = {"p24": a, "experiments": b, "same": a == b}
    both("clean_with_main_v4_manifest")  # main carries the #24 v4 freeze manifest
    init = clone / "src" / "ccil" / "__init__.py"
    data = init.read_bytes()
    init.write_bytes(data + b"\n")
    both("uncommitted_src_change")
    init.write_bytes(data)
    (clone / "tools" / "untracked.txt").write_text("x")
    both("untracked_file")
    (clone / "tools" / "untracked.txt").unlink()
    (clone / p24.FREEZE).parent.mkdir(parents=True, exist_ok=True)
    (clone / p24.FREEZE).write_text(json.dumps({"tooling_commit": head}) + "\n")
    subprocess.run([*git, "add", "-f", p24.FREEZE], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "freeze"], check=True)
    both("manifest_at_this_commit")
    readme = clone / "tools" / "m3_pendulum" / "README.md"
    readme.write_text(readme.read_text() + "\nchanged\n")
    subprocess.run([*git, "commit", "-q", "-am", "change the tooling"], check=True)
    both("committed_tooling_change")
    (clone / p24.FREEZE).write_text(json.dumps({"tooling_commit": "0" * 40}) + "\n")
    subprocess.run([*git, "commit", "-q", "-am", "name no commit"], check=True)
    both("manifest_names_no_commit")
    p24.CCIL_FROZEN = saved
    shutil.rmtree(clone)
    outcomes = {r["experiments"] for r in rows.values()}
    return {"pass": all(r["same"] for r in rows.values()) and outcomes == {True, False}, "scenarios": rows,
            "both_outcomes_exercised": outcomes == {True, False}}


def plan_matches_24(out: Path) -> dict:
    import study as s24  # the #24 drivers, the reference
    spec = plan.load(HERE / "pendulum_v4.yaml")
    root = out / "plan-root"
    root.mkdir()
    choice = {t: {"noise_bc": {"value": 1e-4}, "ccil_tuned": {"value": 0.3}} for t in s24.DATASETS}
    (root / "tuning-choice.json").write_text(json.dumps(choice))
    tuning_same = spec.units("tuning") == s24.units(root, "tuning", smoke=False)
    study_same = spec.units("study", choice) == s24.units(root, "study", smoke=False)
    arms_same = all(spec.arms[n].settings == tuple(a[0] or ()) and spec.arms[n].noise_levels == tuple(a[1])
                    and spec.arms[n].data_fraction == a[2] and (spec.arms[n].tuned or None) == a[3]
                    and spec.arms[n].train == (a[0] is not None) for n, a in s24.ARMS.items())
    bad = [{"study": "x"}, {**plan.yaml.safe_load((HERE / "pendulum_v4.yaml").read_text()), "phases": {
        "tuning": {"seeds": [1, 2], "arms": ["bc"], "tuning": True}, "study": {"seeds": [2, 3], "arms": ["nope"]}}}]
    refusals = []
    for b in bad:
        try:
            plan.from_dict(b)
            refusals.append(None)
        except plan.PlanError as e:
            refusals.append(e.args[0])
    refused = all(r for r in refusals) and len(refusals[1]) >= 3  # tuned-only, unknown arm, shared seed
    return {"pass": tuning_same and study_same and arms_same and refused, "tuning_units_same": tuning_same,
            "study_units_same": study_same, "arms_same": arms_same, "units": {"tuning": len(spec.units("tuning")),
                                                                           "study": len(spec.units("study", choice))},
            "refusals": refusals}


def records_match_24(root: Path) -> dict:
    import audit as a24
    import study as s24

    from ccil.experiments import records
    spec = plan.load(HERE / "pendulum_v4.yaml")
    smoke = plan.Study(spec.name, spec.version, spec.tasks,
                       {"tuning": {**spec.phases["tuning"], "seeds": s24.SMOKE_TUNING_SEEDS},
                        "study": {**spec.phases["study"], "seeds": s24.SMOKE_STUDY_SEEDS}}, spec.arms, spec.tuning)
    inputs = s24.prepare(root)

    def expected_inputs(r):  # the Pendulum suite's hook: the converted inputs, or the rebuilt subset
        want = dict(inputs[r["task"]])
        if spec.arms[r["arm"]].data_fraction:
            want["data"] = s24.subset(root, r["task"], r["seed"], spec.arms[r["arm"]].data_fraction)[1]
        return {"config_sha256": want["config"], "data_sha256": want["data"]}
    choice = json.loads((root / "tuning-choice.json").read_text())

    def both(phase):
        out = {}
        for name, fn in (("p24", lambda: s24.verify_phase(root, phase, True)),
                         ("experiments", lambda: records.verify_phase(root, smoke, phase, smoke=True, audit=a24.audit,
                                                                      expected_inputs=expected_inputs, choice=choice))):
            try:
                out[name] = {(r["task"], r["arm"], r["seed"], r["value"]): r["verified"] for r in fn()}
            except SystemExit:
                out[name] = "refused"
        return out
    clean = {ph: both(ph) for ph in ("tuning", "study")}
    same_clean = all(v["p24"] == v["experiments"] and v["p24"] != "refused" for v in clean.values())
    tuning = s24.verify_phase(root, "tuning", True)
    same_choice = all(json.loads(json.dumps(records.compute_choice(smoke, tuning)[t])) ==
                      json.loads(json.dumps(s24.compute_choice(tuning, True)[t])) for t in spec.tasks)
    rec = next((root / "results" / "study" / "continuous" / "bc").glob("s*.json"))
    tampered = {}
    data = rec.read_bytes()
    obj = json.loads(data)
    obj["evaluations"]["0.0"]["R"] += 1.0
    rec.write_text(json.dumps(obj))
    tampered["edited_R"] = both("study")
    rec.write_bytes(data)
    extra = rec.with_name("s9999.json")
    obj = json.loads(data)
    obj["seed"] = 9999
    extra.write_text(json.dumps(obj))
    tampered["extra_unit"] = both("study")
    extra.unlink()
    refusals = {k: (v["p24"] == "refused", v["experiments"] == "refused") for k, v in tampered.items()}
    return {"pass": same_clean and same_choice and all(a and b for a, b in refusals.values()),
            "clean_root_same": same_clean, "choice_same": same_choice, "refusals": refusals,
            "verified_records": {ph: len(v["experiments"]) if v["experiments"] != "refused" else 0
                                 for ph, v in clean.items()}}


RUNNER_PLAN = {
    "study": "runner-check", "version": 1, "tasks": ["continuous", "discontinuous"],
    "settings": ["transition_model.trainer_version=2", "policy.trainer_version=2",
                 "policy.exclude_validation_episodes=false"],
    "phases": {"tuning": {"seeds": [9001], "arms": ["bc_modern"], "tuning": True},
               "study": {"seeds": [9002, 9003], "arms": ["bc", "ccil", "noise_bc", "bc_frac10", "zero"]}},
    "arms": {"bc": {"settings": ["policy.method=bc", "policy.balance_noisy_labels=false"], "noise_levels": [0.0, 0.01]},
             "ccil": {"settings": ["policy.method=ccil"], "noise_levels": [0.0, 0.01]},
             "noise_bc": {"settings": ["policy.method=noise_bc", "policy.noise_model=symmetric_uniform_half_batch",
                                       "policy.balance_noisy_labels=false", "policy.noise_scale=0.01"]},
             "bc_frac10": {"settings": ["policy.method=bc", "policy.balance_noisy_labels=false"], "data_fraction": 0.1},
             "zero": {"train": False, "noise_levels": [0.0, 0.01]},
             "bc_modern": {"settings": ["policy.method=bc", "policy.balance_noisy_labels=false",
                                        "policy.exclude_validation_episodes=true", "policy.optimizer=adamw",
                                        "policy.weight_decay=0.0001", "policy.lr_schedule=cosine",
                                        "policy.warmup_steps=20", "policy.step_budget=200", "policy.grad_clip_norm=1.0",
                                        "policy.ema_decay=0.99", "policy.checkpoint_selection=validation"],
                           "tuned": "policy.learning_rate"}},
    "tuning": {"bc_modern": {"grid": [1.0e-4, 3.0e-3], "select": "lowest_mean_validation_loss"}},
}
GATE_DEV = {"ok": True, "real": False, "head": "check", "tooling": {}}


def _cli_reuses(root: Path, records: list, device: str) -> list:
    out = []
    for r in records:
        if "stages" not in r:
            continue
        cfg = {k: v["value"] for k, v in r["resolved_config"]["values"].items()}
        sets = [f"run.seed={r['seed']}", f"compute.device={device}", f"data.path={cfg['data.path']}",
                f"run.output_dir={cfg['run.output_dir']}", *r["overrides"], "transition_model.epochs=1",
                f"policy.epochs={cfg['policy.epochs']}"]
        conf = "pendulum_cont.corrected.yaml" if r["task"] == "continuous" else "pendulum_disc.corrected.yaml"
        cmd = [sys.executable, "-m", "ccil", "policy", "train", "--config", str(root / "inputs" / "config" / conf)]
        for x in sets:
            cmd += ["--set", x]
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln.split() for ln in p.stdout.splitlines() if len(ln.split()) == 3 and ln.split()[1] in ("wrote", "reused")]
        out.append({"unit": f"{r['task']}/{r['arm']}/s{r['seed']}", "exit": p.returncode,
                    "stages": {x[0]: x[1] for x in lines},
                    "same_directories": bool(lines) and all(Path(x[2]).resolve() == Path(r["stages"][x[0]]).resolve()
                                                            for x in lines)})
    return out


def runner_check(out: Path, device: str) -> dict:
    import torch

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "tools" / "vector_env"))
    import pendulum as vector
    from pendulum_suite import PendulumSuite

    from ccil.experiments import records, runner
    from ccil.models import mlp_batched
    from ccil.training import vector_bc_v2
    from ccil.training.vector_bc import VectorBCPolicy

    study = plan.from_dict(RUNNER_PLAN)
    root = out / "runner"
    suite = PendulumSuite(root, study, smoke=True)
    res = {}

    def verify(r, s):
        return records.verify_phase(r, study, "study", smoke=True, audit=PendulumSuite.audit,
                                    expected_inputs=s.expected_inputs, jobs=4)
    first = runner.run_phase(root, study, "study", suite, gate=GATE_DEV, smoke=True, device=device, log=lambda m: None)
    recs = verify(root, suite)
    res["first_run"] = {"units": first["units"], "timing": first["timing"],
                        "status": {f"{r['task']}/{r['arm']}/s{r['seed']}": r["status"] for r in recs}}
    ok_first = first["units"] == 20 and all(r["status"] == "ok" for r in recs)

    tun = runner.run_phase(root, study, "tuning", suite, gate=GATE_DEV, smoke=True, device=device, log=lambda m: None)
    trecs = records.verify_phase(root, study, "tuning", smoke=True, audit=PendulumSuite.audit,
                                 expected_inputs=suite.expected_inputs, jobs=4)
    choice = records.compute_choice(study, trecs, acceptance_arms=())
    losses = {(r["task"], r["value"]): records.validation_loss(r) for r in trecs}
    want = {t: min(v for (tt, v) in losses if tt == t and losses[(tt, v)] == min(x for (u, _), x in losses.items()
                                                                                    if u == t))
            for t in study.tasks}
    res["validation_loss_tuning"] = {"units": tun["units"], "losses": {f"{t}@{v}": x for (t, v), x in losses.items()},
                                     "choice": {t: choice[t]["bc_modern"]["value"] for t in study.tasks},
                                     "selected_epochs": [r["training"]["validation"]["selected_epoch"] for r in trecs]}
    ok_tuning = (tun["units"] == 4 and all(x is not None for x in losses.values())
                 and all(choice[t]["bc_modern"]["value"] == want[t] for t in study.tasks)
                 and all(r["training"]["recipe"]["step_budget"] == 200 for r in trecs))
    cli = _cli_reuses(root, recs, device)
    res["cli"] = cli
    ok_cli = len(cli) == 16 and all(c["exit"] == 0 and c["same_directories"]
                                    and set(c["stages"].values()) == {"reused"} for c in cli)

    again = runner.run_phase(root, study, "study", suite, gate=GATE_DEV, smoke=True, device=device, log=lambda m: None)
    removed = [records.result_path(root, "study", "discontinuous", "ccil", 9003),
               records.result_path(root, "study", "continuous", "zero", 9002)]
    for p in removed:
        p.unlink()
    resumed = runner.run_phase(root, study, "study", suite, gate=GATE_DEV, smoke=True, device=device, log=lambda m: None)
    recs2 = verify(root, suite)
    res["resume"] = {"second_run_units": again["units"], "after_removal_units": resumed["units"],
                     "train_groups": resumed["timing"]["train_groups"], "fit_groups": resumed["timing"]["fit_groups"]}
    ok_resume = (again["units"] == 0 and resumed["units"] == 2 and not resumed["timing"]["train_groups"]
                 and not resumed["timing"]["fit_groups"] and all(p.exists() for p in removed) and len(recs2) == 20)

    diffs = []
    for r in recs2:
        if "stages" not in r:
            continue
        net, _, _ = VectorBCPolicy.load(Path(r["stages"]["policy.train"]) / "checkpoints" / "policy.npz", device=device)
        for n, e in r["evaluations"].items():
            alone = vector.rollout(net.predict, r["task"], r["seed"], float(n))
            with np.load(root / e["rollout"], allow_pickle=False) as z:
                differ = [k for k in z.files if k != "meta" and not np.array_equal(z[k], alone[k], equal_nan=True)]
            diffs.append({"unit": f"{r['task']}/{r['arm']}/s{r['seed']}", "noise": n, "arrays_differing": differ,
                          "R_alone": float(alone["R"]), "R_recorded": e["R"]})
    res["batched_evaluation"] = {"rollouts": len(diffs), "per_rollout": diffs}
    ok_eval = len(diffs) == 24 and not any(d["arrays_differing"] for d in diffs)

    # Injected memory exhaustion.
    real_fit, real_train = mlp_batched.fit_group, vector_bc_v2.train_group
    calls = {"fit": [], "train": []}
    doomed = None

    def fit(jobs, *a, **k):
        calls["fit"].append(len(jobs))
        if len(jobs) > 2:
            raise torch.cuda.OutOfMemoryError("injected: CUDA out of memory")
        return real_fit(jobs, *a, **k)

    def train(jobs, *a, **k):
        calls["train"].append(len(jobs))
        if len(jobs) > 2 or any(j.supervision is doomed[0] for j in jobs if doomed):
            raise torch.cuda.OutOfMemoryError("injected: CUDA out of memory")
        return real_train(jobs, *a, **k)
    from ccil.pipelines import policy_harness
    real_sup = policy_harness.unit_supervision

    def sup(cfg):
        nonlocal doomed
        got = real_sup(cfg)
        if cfg["run.seed"] == 9002 and cfg["data.path"].endswith("-f0.1.npz") and "cont" in cfg["data.path"]:
            doomed = (got[2],)
        return got
    root2 = out / "runner-oom"
    suite2 = PendulumSuite(root2, study, smoke=True)
    mlp_batched.fit_group, vector_bc_v2.train_group, policy_harness.unit_supervision = fit, train, sup
    try:
        oom = runner.run_phase(root2, study, "study", suite2, gate=GATE_DEV, smoke=True, device=device,
                               log=lambda m: None)
    finally:
        mlp_batched.fit_group, vector_bc_v2.train_group, policy_harness.unit_supervision = real_fit, real_train, real_sup
    recs3 = verify(root2, suite2)
    status = {f"{r['task']}/{r['arm']}/s{r['seed']}": r["status"] for r in recs3}
    res["memory_exhaustion"] = {"fit_group_sizes": calls["fit"], "train_group_sizes": calls["train"], "status": status,
                                "timing": oom["timing"]}
    faults = {k for k, v in status.items() if v != "ok"}
    ok_oom = (faults == {"continuous/bc_frac10/s9002"} and status["continuous/bc_frac10/s9002"] == "infrastructure_fault"
              and max(calls["fit"]) > 2 and max(calls["train"]) > 2)
    res["checks"] = {"validation_loss_tuning": ok_tuning, "first_run_verifies": ok_first, "cli_reuses_every_stage": ok_cli, "resume": ok_resume,
                     "batched_evaluation": ok_eval, "memory_exhaustion": ok_oom}
    return {"pass": all(res["checks"].values()), **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke-root", type=Path, help="a #24 check_study smoke root, for the records case")
    ap.add_argument("--runner", action="store_true", help="run the runner case")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)
    r = analysis_matches_24()
    record("analysis_matches_24", r.pop("pass"), **r)
    r = gate_matches_24(out)
    record("gate_matches_24", r.pop("pass"), **r)
    r = plan_matches_24(out)
    record("plan_matches_24", r.pop("pass"), **r)
    if args.smoke_root:
        r = records_match_24(args.smoke_root.resolve())
        record("records_match_24", r.pop("pass"), **r)
    if args.runner:
        r = runner_check(out, args.device)
        record("runner", r.pop("pass"), **r)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "python": platform.python_version(), "platform": platform.platform()}}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
