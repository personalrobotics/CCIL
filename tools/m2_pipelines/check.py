#!/usr/bin/env python3
"""Evidence for the offline and integrated pipelines (#19).

    uv run --group evidence python tools/m2_pipelines/check.py --out DIR

1. Offline analytic run: three complete stage runs whose labels pass the #10
   oracle (I1-I5), byte-identical replay, and full reuse on rerun.
   The labels carry the run seed, the uniform weighting, explicit weights (0 on
   rejected rows), and input hashes equal to the labels stage's manifest inputs.
   Each stage's stage.json records the identity its run ID hashes (consumed
   configuration, components with versions, implementation source hash and
   dependency versions, input hashes). Changed source, even a comment, or a
   bumped component version writes distinct runs and leaves the old ones intact;
   an unreadable implementation fails instead of reusing. With --result, a
   ccil.labels-result 1.0 file names the same stages, reuse and run
   directories the printed lines do, the absolute path of each artifact, and
   the labels stage's configuration record (#168).
2. Offline generation never imports the integrated pipeline or a policy trainer, and with torch installed
   the core, supervision, offline pipeline and dynamics fitters load no downstream trainer, experiment
   code, policy harness or SciPy, and raise no deprecation warning (#168). The moved code is refused,
   naming its il-supervision-benchmarks replacement: importing ccil.training or ccil.experiments raises
   ImportError, and ccil policy train exits 2 without a traceback.
3. Integrated stub flow on a synthetic state+image dataset: seven complete stages,
   latent labels, shared manifest vocabulary, checkpoint handoff by hash.
4. Replacing the downstream trainer reuses the four CCIL stages. Changing only
   supervision.exclude_validation_episodes reuses them too, and the new
   supervision holds no episode out where the default held the split's
   validation episodes out.
5. A failure in policy.train leaves earlier stages valid; the rerun resumes there.
   A failure in policy.evaluate leaves policy.train complete; the rerun reuses
   it and completes.
   Unsupported configurations fail before any stage writes.
6. Help output states inputs, outputs, optional dependencies, and exit codes.
Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "m2_interfaces"))
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
import oracle  # noqa: E402
from check import fixture, to_oracle  # noqa: E402  (tools/m2_interfaces/check.py)

from ccil.config import resolve  # noqa: E402
from ccil.pipelines.integrated import run_integrated  # noqa: E402
from ccil.pipelines.policy import LinearPolicyTrainer  # noqa: E402
from ccil.pipelines.stages import StageError  # noqa: E402
from ccil.runs import layout, validate_run  # noqa: E402
from ccil.schema import Episode, TrajectoryDataset, read_candidates, read_trajectories, write_trajectories  # noqa: E402


def ccil(*args, cwd):
    return subprocess.run([sys.executable, "-m", "ccil", *map(str, args)], cwd=cwd, capture_output=True, text=True)


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def runs(d: Path) -> dict:
    return {p.name: p for p in sorted(d.iterdir()) if p.is_dir() and not p.name.startswith(".")} if d.exists() else {}


def image_dataset(path: Path) -> None:
    rng = np.random.default_rng(7)
    render = rng.normal(size=(3, 64))
    eps = []
    for e in range(4):
        s = np.cumsum(rng.normal(scale=0.1, size=(15, 3)), axis=0)
        pix = np.clip(128 + 40 * (s @ render), 0, 255).astype(np.uint8).reshape(15, 8, 8, 1)
        eps.append(Episode(f"img{e}", {"state": s.astype(np.float32), "image.top": pix}, rng.normal(size=(15, 2)).astype(np.float32)))
    write_trajectories(TrajectoryDataset(eps), path)


INTEGRATED = ["--set", "representation.kind=pca", "--set", "representation.modality=image.top", "--set", "representation.dim=4",
              "--set", "transition_model.kind=linear", "--set", "generation.method=backward_euler",
              "--set", "generation.support_radius=0.5", "--set", "data.validation_fraction=0.25",
              "--set", "data.path=images.npz", "--set", "run.output_dir=runs",
              # The stub tests the latent-space handoff, not admission: declare the recovery length.
              "--set", "generation.recovery_steps=1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    config["episodes"] = [e for e in config["episodes"] if not e["corrupt_actions"]]
    offline_args = ["labels", "generate", "--recipe", "analytic_double_integrator", "--set", "data.path=fixture.npz",
                    "--set", "run.output_dir=runs"]

    # 1. Offline analytic run.
    wa, wb = out / "offline_a", out / "offline_b"
    for w in (wa, wb):
        w.mkdir()
        write_trajectories(fixture(config), w / "fixture.npz")
    r1 = ccil(*offline_args, cwd=wa)
    ra = runs(wa / "runs")
    statuses = {n: validate_run(p)[0] for n, p in ra.items()}
    labels_dir = next(p for n, p in ra.items() if n.startswith("labels-generate"))
    dataset = read_trajectories(next(p for n, p in ra.items() if n.startswith("data-prepare")) / layout.DATASET)
    cands = read_candidates(labels_dir / layout.CANDIDATES, dataset)
    _, verdicts = oracle.evaluate({**config, "methods": {"backward_euler_fast": config["methods"]["backward_euler_fast"]}},
                                  oracle.episodes_to_transitions(config), {"backward_euler_fast": to_oracle(cands, dataset)})
    oracle_ok = {k: v["backward_euler_fast"]["pass"] for k, v in verdicts.items() if k != "I6_episode_boundaries"}
    r2 = ccil(*offline_args, cwd=wb)
    rb = runs(wb / "runs")
    replay = sorted(ra) == sorted(rb) and all(tree(ra[n]) == tree(rb[n]) for n in ra)
    r3 = ccil(*offline_args, cwd=wa)
    reused = r3.returncode == 0 and r3.stdout.count("reused") == 5 and r3.stdout.count("wrote") == 0
    record("offline_analytic_run", r1.returncode == 0 and len(ra) == 5 and set(statuses.values()) == {"complete"}
           and all(oracle_ok.values()) and replay and reused,
           stages=statuses, oracle=oracle_ok, replay_identical=replay, rerun_output=r3.stdout.strip().splitlines(),
           report=json.loads((labels_dir / layout.METRICS).read_text())["values"], stderr=r1.stderr[-400:])

    # 1b. Pipeline-produced candidates carry their own provenance and explicit weights.
    lm = json.loads((labels_dir / layout.MANIFEST).read_text())
    cfg_seed = json.loads((labels_dir / "config.json").read_text())["values"]["run.seed"]["value"]
    manifest_inputs = {k: v["sha256"] for k, v in lm["inputs"].items()}
    record("offline_candidates_carry_provenance",
           cands.inputs == manifest_inputs and "transition_model" in cands.inputs and cands.seed == cfg_seed
           and cands.run_id == lm["run_id"] and cands.weighting == {"name": "uniform", "version": "1", "params": {}}
           and np.array_equal(cands.weight, cands.accepted.astype(np.float64)),
           inputs=cands.inputs, manifest_inputs=manifest_inputs, seed=cands.seed, weighting=cands.weighting,
           weight_on_rejected=sorted(set(cands.weight[~cands.accepted].tolist())),
           weight_on_accepted=sorted(set(cands.weight[cands.accepted].tolist())))

    # 1c. Reuse is decided by implementation and component identity, recorded in stage.json.
    from ccil.pipelines.stages import stage_run_id
    from ccil.runs.provenance import implementation
    impl = implementation()
    recorded = {n: json.loads((p / layout.STAGE).read_text()) for n, p in ra.items()}
    consistent = all(st["run_id"] == n and stage_run_id({k: v for k, v in st.items() if k != "run_id"}) == n
                     and st["implementation"] == impl and layout.STAGE in json.loads((ra[n] / layout.MANIFEST).read_text())["files"]
                     for n, st in recorded.items())
    record("stage_identity_recorded_and_rederives_run_id", consistent and reused,
           implementation=impl, components={n: st["components"] for n, st in recorded.items()})

    # 1d. The machine-readable result names every stage and artifact the printed lines name.
    r4 = ccil(*offline_args, "--result", "result.json", cwd=wa)
    res = json.loads((wa / "result.json").read_text())
    printed = {ln.split()[0]: (ln.split()[1], (wa / ln.split()[2]).resolve()) for ln in r4.stdout.splitlines()}
    named = {s["stage"]: ("reused" if s["reused"] else "wrote", Path(s["run_dir"])) for s in res["stages"]}
    gen_dir = named["labels.generate"][1]
    want = {"dataset": ra[next(n for n in ra if n.startswith("data-prepare"))] / layout.DATASET,
            "split": ra[next(n for n in ra if n.startswith("data-prepare"))] / layout.SPLIT,
            "candidates": gen_dir / layout.CANDIDATES, "selection": named["labels.curate"][1] / layout.SELECTION}
    arts = {k: Path(v) for k, v in res["artifacts"].items()}
    record("labels_result_names_stages_and_artifacts",
           r4.returncode == 0 and (res["schema"], res["version"]) == ("ccil.labels-result", "1.0")
           and named == printed and all(arts[k] == v.resolve() for k, v in want.items())
           and all(p.is_absolute() and p.exists() for p in arts.values())
           and res["config"] == json.loads((gen_dir / layout.CONFIG).read_text()),
           artifacts=sorted(arts), stages=sorted(named))

    def variant_run(name, prelude):
        """Rerun the offline pipeline in work_a from a child process that first runs prelude."""
        code = prelude + "\nimport sys\nfrom ccil.__main__ import main\nsys.exit(main(sys.argv[1:]))\n"
        before = set(runs(wa / "runs"))
        r = subprocess.run([sys.executable, "-c", code, *offline_args], cwd=wa, capture_output=True, text=True)
        status = {l.split()[0]: l.split()[1] for l in r.stdout.strip().splitlines() if len(l.split()) > 1}
        new = sorted(set(runs(wa / "runs")) - before)
        return r, status, new

    originals = {n: tree(p) for n, p in ra.items()}
    pkg = out / "changed_source"
    shutil.copytree(REPO / "src" / "ccil", pkg / "ccil", ignore=shutil.ignore_patterns("__pycache__"))
    with open(pkg / "ccil" / "core" / "generators.py", "a") as f:
        f.write("# an edit that changes no behavior still changes the implementation\n")
    r, status, new = variant_run("changed_source", f"import sys; sys.path.insert(0, {str(pkg)!r})\n"
                                                   f"import ccil; assert ccil.__file__.startswith({str(pkg)!r})")
    new_impl = {json.loads((wa / "runs" / n / layout.STAGE).read_text())["implementation"]["source_sha256"] for n in new}
    record("changed_implementation_writes_distinct_runs", r.returncode == 0 and set(status.values()) == {"wrote"}
           and len(new) == 5 and new_impl and impl["source_sha256"] not in new_impl
           and all(tree(ra[n]) == h for n, h in originals.items()),
           stage_status=status, new_runs=new, stderr=r.stderr[-300:])

    r, status, new = variant_run("component_version", "from ccil.core import LinearTransitionModel\n"
                                                      "LinearTransitionModel.version = '2'")
    record("component_version_change_writes_distinct_runs", r.returncode == 0 and status.get("data.prepare") == "reused"
           and status.get("transition.fit") == "wrote" and status.get("labels.generate") == "wrote" and len(new) == 4,
           stage_status=status, new_runs=new, stderr=r.stderr[-300:])

    (out / "empty_package").mkdir()
    r, status, new = variant_run("unidentifiable", f"import ccil.runs.provenance as p\n"
                                                   f"from pathlib import Path\np.PACKAGE_ROOT = Path({str(out / 'empty_package')!r})")
    record("unidentifiable_implementation_never_reuses", r.returncode == 2 and not new and "reused" not in r.stdout
           and "no ccil source files" in r.stderr, stage_status=status, error=r.stderr.strip()[-300:])

    # 2. Offline generation imports no trainer or integrated pipeline.
    probe = ("import sys, json; from ccil.config import resolve; from ccil.pipelines.offline import generate_offline; "
             "generate_offline(resolve(recipe='analytic_double_integrator', sets=['data.path=fixture.npz', 'run.output_dir=probe_runs'])); "
             "print(json.dumps(sorted(m for m in sys.modules if (m.startswith('ccil.') and ('integrated' in m or 'policy' in m)) "
             "or m.split('.')[0] == 'torch')))")
    p = subprocess.run([sys.executable, "-c", probe], cwd=wa, capture_output=True, text=True)
    mods = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
    record("offline_imports_no_trainer", mods == [], loaded=mods, stderr=p.stderr[-300:])

    # The ownership boundary (#168): CCIL's core, supervision, offline pipeline and dynamics fitters load no
    # downstream trainer, experiment code, policy harness or SciPy, also with torch installed.
    probe = ("import sys, json; import ccil, ccil.core, ccil.supervision, ccil.pipelines.offline, "
             "ccil.models.mlp, ccil.models.mlp_batched; "
             "bad = ('ccil.training', 'ccil.experiments', 'ccil.pipelines.policy_harness', 'scipy'); "
             "print(json.dumps({'forbidden': sorted(m for m in sys.modules if m.startswith(bad)), "
             "'torch_loaded': 'torch' in sys.modules}))")
    p = subprocess.run([sys.executable, "-W", "error::FutureWarning", "-c", probe], cwd=wa, capture_output=True,
                       text=True)  # and with no deprecation warning: the core never touches moved code
    got = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
    record("core_loads_no_downstream_learner", got is not None and got["forbidden"] == [] and got["torch_loaded"],
           invariant="ccil, ccil.core, ccil.supervision, ccil.pipelines.offline and ccil.models.mlp(_batched) load "
                     "nothing under ccil.training, ccil.experiments, ccil.pipelines.policy_harness, or scipy",
           result=got, stderr=p.stderr[-300:])

    # Moved code (#168) is refused and says where it went: an ImportError, and ccil policy train exits 2.
    moved = {}
    for mod, replacement in (("ccil.training", "ilsb.learners.vector"), ("ccil.experiments", "ilsb run")):
        p = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=wa, capture_output=True, text=True)
        last = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
        moved[mod] = {"exit": p.returncode, "refused": last.startswith("ImportError:") and replacement in last}
    p = ccil("policy", "train", "--recipe", "analytic_double_integrator", "--set", "data.path=missing.npz", cwd=wa)
    moved["ccil policy train"] = {"exit": p.returncode, "refused": p.returncode == 2 and "Traceback" not in p.stderr
                                  and "ilsb policy train CONFIG" in p.stderr}
    record("moved_code_refused_naming_its_replacement", all(v["refused"] for v in moved.values()), moved=moved)

    # 3. Integrated stub flow.
    wi = out / "integrated"
    wi.mkdir()
    image_dataset(wi / "images.npz")
    r = ccil("pipeline", "integrated", *INTEGRATED, cwd=wi)
    ri = runs(wi / "runs")
    st = {n: validate_run(p)[0] for n, p in ri.items()}
    by_stage = {json.loads((p / layout.MANIFEST).read_text())["workflow"]: p for p in ri.values()}
    ok = r.returncode == 0 and len(ri) == 7 and set(st.values()) == {"complete"}
    detail = {"stages": st, "stderr": r.stderr[-400:]}
    if ok:
        lab = by_stage["labels.generate"]
        lc = read_candidates(lab / layout.CANDIDATES, read_trajectories(by_stage["data.prepare"] / layout.DATASET))
        manifests = [json.loads((p / layout.MANIFEST).read_text()) for p in [*ri.values(), *ra.values()]]
        same_vocab = len({tuple(sorted(m)) for m in manifests}) == 1
        lab_inputs = json.loads((lab / layout.MANIFEST).read_text())["inputs"]
        handoff = (lab_inputs["representation"]["sha256"] == hashlib.sha256(
            (by_stage["representation.fit"] / "checkpoints/representation.npz").read_bytes()).hexdigest()
            and lab_inputs["transition_model"]["sha256"] == hashlib.sha256(
                (by_stage["transition.fit"] / "checkpoints/transition_model.npz").read_bytes()).hexdigest())
        ev = json.loads((by_stage["policy.evaluate"] / layout.METRICS).read_text())["values"]
        ok = set(lc.observations) == {"latent.pca"} and lc.accepted.any() and same_vocab and handoff
        detail = {"stages": st, "label_modality": sorted(lc.observations), "accepted": int(lc.accepted.sum()),
                  "candidates": int(len(lc.accepted)), "same_manifest_vocabulary": same_vocab,
                  "checkpoint_handoff_by_hash": handoff, "evaluation": ev,
                  "labels_inputs": sorted(lab_inputs)}
    record("integrated_stub_flow", ok, **detail)

    # 4. Replacing the downstream trainer reuses the four CCIL stages.
    r = ccil("pipeline", "integrated", *INTEGRATED, "--set", "integrated.trainer=mean_action", cwd=wi)
    lines = [l.split() for l in r.stdout.strip().splitlines()]
    reuse = {l[0]: l[1] for l in lines}
    record("downstream_trainer_replaceable", r.returncode == 0
           and all(reuse.get(s) == "reused" for s in ("data.prepare", "representation.fit", "transition.fit", "labels.generate"))
           and reuse.get("policy.train") == "wrote" and reuse.get("policy.evaluate") == "wrote" and len(runs(wi / "runs")) == 9,
           stage_status=reuse)

    # 4b. Changing only supervision.exclude_validation_episodes reuses the CCIL stages; the supervision then
    # holds no episode out.
    def held_out(train_dir):
        return json.loads((train_dir / layout.METRICS).read_text())["values"]["supervision"]["counts"]
    first_train = {n for n in runs(wi / "runs") if n.startswith("policy-train")}
    r = ccil("pipeline", "integrated", *INTEGRATED, "--set", "supervision.exclude_validation_episodes=false", cwd=wi)
    status = {l.split()[0]: l.split()[1] for l in r.stdout.strip().splitlines() if len(l.split()) > 1}
    new_train = [p for n, p in runs(wi / "runs").items() if n.startswith("policy-train") and n not in first_train]
    counts = {"default": held_out(by_stage["policy.train"]) if ok else None,
              "false": held_out(new_train[0]) if r.returncode == 0 and len(new_train) == 1 else None}
    record("supervision_holdout_follows_the_flag", r.returncode == 0 and len(new_train) == 1
           and all(status.get(s) == "reused" for s in ("data.prepare", "representation.fit", "transition.fit",
                                                        "labels.generate"))
           and status.get("policy.train") == "wrote" and counts["default"] is not None and counts["false"] is not None
           and counts["default"]["demonstration_rows_held_out"] > 0 and counts["false"]["demonstration_rows_held_out"] == 0
           and counts["false"]["demonstration_rows"] == counts["default"]["demonstration_rows"]
           + counts["default"]["demonstration_rows_held_out"],
           stage_status=status, counts=counts, stderr=r.stderr.strip()[-300:])

    # 5. Failure in policy.train, then resume; unsupported configs fail before writing.
    wf = out / "failure"
    wf.mkdir()
    image_dataset(wf / "images.npz")

    class FlakyTrainer(LinearPolicyTrainer):
        name = "flaky"
        fail = True

        def fit(self, z, a, w):
            if FlakyTrainer.fail:
                raise RuntimeError("simulated trainer failure")
            return super().fit(z, a, w)

    sets = [s for s in INTEGRATED if s != "--set"]
    cwd = os.getcwd()
    os.chdir(wf)
    try:
        cfg = resolve(sets=sets)
        try:
            run_integrated(cfg, trainer=FlakyTrainer())
            failed_stage = None
        except StageError as e:
            failed_stage = e.stage
        after_fail = {n: validate_run(p)[0] for n, p in runs(wf / "runs").items()}
        staging = [p.name for p in (wf / "runs").iterdir() if p.name.startswith(".")]
        FlakyTrainer.fail = False
        resumed = run_integrated(cfg, trainer=FlakyTrainer())
    finally:
        os.chdir(cwd)
    resume_ok = [r.reused for r in resumed] == [True, True, True, True, False, False, True]  # labels.diagnose is listed last
    record("failure_leaves_earlier_stages_and_resumes", failed_stage == "policy.train" and len(after_fail) == 5
           and set(after_fail.values()) == {"complete"} and not staging and resume_ok,
           failed_stage=failed_stage, stages_after_failure=after_fail, staging_left=staging,
           resume=[[r.stage, r.reused] for r in resumed])

    class FlakyEvalTrainer(LinearPolicyTrainer):
        name = "flaky_eval"
        fail = True

        def predict(self, z):
            if FlakyEvalTrainer.fail:
                raise RuntimeError("simulated evaluation failure")
            return super().predict(z)

    before_eval_fail = set(runs(wf / "runs"))
    os.chdir(wf)
    try:
        try:
            run_integrated(cfg, trainer=FlakyEvalTrainer())
            failed_stage = None
        except StageError as e:
            failed_stage = e.stage
        new_runs = {n: validate_run(p)[0] for n, p in runs(wf / "runs").items() if n not in before_eval_fail}
        FlakyEvalTrainer.fail = False
        try:
            resumed = run_integrated(cfg, trainer=FlakyEvalTrainer())
            resume_error = None
        except StageError as e:
            resumed, resume_error = [], str(e)
    finally:
        os.chdir(cwd)
    record("evaluate_failure_resumes_with_reused_policy", failed_stage == "policy.evaluate"
           and [n.split("-")[1] for n in new_runs] == ["train"] and set(new_runs.values()) == {"complete"}
           and [r.reused for r in resumed] == [True, True, True, True, True, False, True],
           failed_stage=failed_stage, runs_after_failure=new_runs, resume=[[r.stage, r.reused] for r in resumed],
           resume_error=resume_error)
    for name, extra in (("unsupported_method", ["--set", "generation.method=backward_model"]),
                        ("mlp_model", ["--set", "transition_model.kind=mlp"])):
        w = out / f"preflight_{name}"
        w.mkdir()
        write_trajectories(fixture(config), w / "fixture.npz")
        r = ccil(*offline_args, *extra, cwd=w)
        record(f"fails_before_any_stage[{name}]", r.returncode == 2 and not (w / "runs").exists(), error=r.stderr.strip())

    # 6. Help output.
    helps = {cmd: ccil(*cmd.split(), "--help", cwd=out).stdout for cmd in ("labels generate", "pipeline integrated")}
    record("help_states_inputs_outputs_dependencies_exit_codes",
           all(all(k in h for k in ("required:", "outputs:", "optional:", "exit:")) for h in helps.values()),
           help=helps)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
