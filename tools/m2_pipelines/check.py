#!/usr/bin/env python3
"""Evidence for the offline and integrated pipelines (#19).

    uv run --group evidence python tools/m2_pipelines/check.py --out DIR

1. Offline analytic run: three complete stage runs whose labels pass the #10
   oracle (I1-I5), byte-identical replay, and full reuse on rerun.
   The labels carry the run seed, the uniform weighting, explicit weights (0 on
   rejected rows), and input hashes equal to the labels stage's manifest inputs.
   Changed source, even a comment, or a
   bumped component version writes distinct runs and leaves the old ones intact;
   an unreadable implementation fails instead of reusing. With --result, a
   ccil.labels-result 1.0 file names the same stages, reuse and run
   directories the printed lines do, the absolute path of each artifact, and
   the labels stage's configuration record (#168).
2. Offline generation never imports the integrated pipeline, a policy trainer or torch.
3. Integrated stub flow on a synthetic state+image dataset: seven complete stages,
   latent labels, shared manifest vocabulary, checkpoint handoff by hash.
4. Replacing the downstream trainer reuses the four CCIL stages. Changing only
   supervision.exclude_validation_episodes reuses them too, and the new
   supervision holds no episode out where the default held the split's
   validation episodes out.
5. A failure in policy.train leaves earlier stages valid; the rerun resumes there.
   A failure in policy.evaluate leaves policy.train complete; the rerun reuses
   it and completes.
   Unsupported or incomplete configurations fail before any stage writes (exit 2). An input a stage refuses
   exits 2 and completes nothing; a defect inside a stage exits 1 with its traceback, never as an input error.
6. The golden path (#170), through the CLI on 20 double-integrator episodes: diagnose then generate reuses the
   three diagnose stages; generate writes candidates and quality evidence, a generate result and a completion
   summary, and no selection; two curations of one generate result write two nested selections and regenerate
   nothing; curate refuses a Generate key and a changed candidates file (exit 2, nothing written), and so does
   inspect; inspect reports every section with the build_supervision call; labels generate, deprecated, still
   writes its selection and ccil.labels-result 1.0, reusing the same candidates.
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


GOLDEN = ["--set", "data.path=demos.npz", "--set", "run.output_dir=runs", "--set", "compute.device=cpu",
          "--set", "transition_model.kind=linear", "--set", "generation.method=noisy_action",
          "--set", "generation.labels_per_transition=4", "--set", "generation.action_noise_std=3.0",
          "--set", "certificate.profile=disturbed_action", "--set", "certificate.true_state_lipschitz=1.0",
          "--set", "certificate.true_action_lipschitz=0.2", "--set", "certificate.demonstration_residual_bound=0.0"]


def golden(w: Path, record) -> None:
    from system import DoubleIntegrator
    w.mkdir()
    sys_, r, eps = DoubleIntegrator(0.1), np.random.default_rng(0), []
    for e in range(20):
        s = [r.uniform(-1, 1, size=2)]
        a = r.uniform(-1, 1, size=(30, 1))
        for t in range(29):
            s.append(sys_.step_np(s[-1][None], a[t][None])[0])
        eps.append(Episode(f"e{e}", {"state": np.array(s, np.float32)}, a.astype(np.float32)))
    write_trajectories(TrajectoryDataset(eps), w / "demos.npz")

    def status(out):
        return {ln.split()[0]: ln.split()[1] for ln in out.splitlines() if ln.split()[:1] and ln.split()[0].count(".") == 1
                and len(ln.split()) > 2 and ln.split()[1] in ("wrote", "reused")}
    d = ccil("diagnose", *GOLDEN, cwd=w)
    g = ccil("generate", *GOLDEN, cwd=w)
    result = w / "runs" / "generate-result.json"
    gs = status(g.stdout)
    record("diagnose_then_generate_reuses_three_stages", d.returncode == 0 and g.returncode == 0
           and [gs.get(k) for k in ("data.prepare", "transition.fit", "labels.diagnose")] == ["reused"] * 3
           and gs.get("labels.generate") == "wrote", diagnose=status(d.stdout), generate=gs)
    doc = json.loads(result.read_text()) if result.exists() else {}
    selections = [n for n in runs(w / "runs") if n.startswith("labels-curate")]
    record("generate_writes_candidates_and_evidence_not_a_selection", g.returncode == 0 and not selections
           and set(doc.get("artifacts", {})) == {"dataset", "split", "transition_model", "diagnosis", "candidates",
                                                  "certificates"}
           and all(k in g.stdout for k in ("candidates", "rejected", "evidence", "result", "next           ccil curate")),
           artifacts=sorted(doc.get("artifacts", {})), summary=g.stdout.splitlines()[-6:])
    before = set(runs(w / "runs"))
    c1 = ccil("curate", "runs/generate-result.json", "--set", "curate.tolerance=0.5", cwd=w)
    c2 = ccil("curate", "runs/generate-result.json", "--set", "curate.tolerance=1.0", cwd=w)
    new = sorted(set(runs(w / "runs")) - before)
    masks = []
    for c in (c1, c2):
        line = next((ln for ln in c.stdout.splitlines() if ln.startswith("result")), "")
        res = json.loads((w / line.split()[-1]).read_text()) if line else None
        if res:
            with np.load(w / "runs" / res["artifacts"]["selection"]["path"]) as z:
                masks.append(z["mask"])
    nested = len(masks) == 2 and masks[0].sum() < masks[1].sum() and not (masks[0] & ~masks[1]).any()
    record("two_curations_of_one_generate_regenerate_nothing", c1.returncode == 0 and c2.returncode == 0
           and len(new) == 2 and all(n.startswith("labels-curate") for n in new) and nested,
           new_runs=new, selected=[int(m.sum()) for m in masks])
    refused = ccil("curate", "runs/generate-result.json", "--set", "data.path=other.npz", cwd=w)
    cands = w / "runs" / doc["artifacts"]["candidates"]["path"]
    original = cands.read_bytes()
    flipped = bytearray(original)
    flipped[len(flipped) // 2] ^= 1
    cands.write_bytes(bytes(flipped))
    before = set(runs(w / "runs"))
    tampered = ccil("curate", "runs/generate-result.json", "--set", "curate.tolerance=0.7", cwd=w)
    inspected_tampered = ccil("inspect", "runs/generate-result.json", cwd=w)
    cands.write_bytes(original)
    record("curate_refuses_a_changed_artifact_and_generate_keys", refused.returncode == 2 and "belong to Generate"
           in refused.stderr and tampered.returncode == 2 and "changed since Generate wrote it" in tampered.stderr
           and inspected_tampered.returncode == 2 and set(runs(w / "runs")) == before and "Traceback" not in
           tampered.stderr + refused.stderr, refused=refused.stderr.strip()[-200:], tampered=tampered.stderr.strip()[-200:])
    line = next((ln for ln in c1.stdout.splitlines() if ln.startswith("result")), "")
    i = ccil("inspect", line.split()[-1] if line else "missing", cwd=w)
    sections = [ln for ln in i.stdout.splitlines() if ln and not ln.startswith(" ")]
    record("inspect_reports_every_section", i.returncode == 0 and sections == [
        "configuration", "provenance", "quality", "rejections", "selection", "usable rows", "next step"]
        and "build_supervision(" in i.stdout and "curate.tolerance = 0.5 (cli)" in i.stdout, sections=sections)
    alias = ccil("labels", "generate", *GOLDEN, "--set", "curate.tolerance=0.5", "--result", "alias.json", cwd=w)
    adoc = json.loads((w / "alias.json").read_text()) if (w / "alias.json").exists() else {}
    record("labels_generate_alias_keeps_its_outputs", alias.returncode == 0 and "deprecated" in alias.stderr
           and adoc.get("schema") == "ccil.labels-result" and "selection" in adoc.get("artifacts", {})
           and status(alias.stdout).get("labels.curate") in ("reused", "wrote")
           and status(alias.stdout).get("labels.generate") == "reused", stages=status(alias.stdout))


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
    ccil(*offline_args, cwd=wb)
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
    from ccil.runs.provenance import implementation
    impl = implementation()
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
    record("unidentifiable_implementation_never_reuses", r.returncode == 1 and not new and "reused" not in r.stdout
           and "no ccil source files" in r.stderr and "Traceback" in r.stderr, stage_status=status, error=r.stderr.strip()[-300:])

    # 2. Offline generation imports no trainer or integrated pipeline.
    probe = ("import sys, json; from ccil.config import resolve; from ccil.pipelines.offline import generate_offline; "
             "generate_offline(resolve(recipe='analytic_double_integrator', sets=['data.path=fixture.npz', 'run.output_dir=probe_runs'])); "
             "print(json.dumps(sorted(m for m in sys.modules if (m.startswith('ccil.') and ('integrated' in m or 'policy' in m)) "
             "or m.split('.')[0] == 'torch')))")
    p = subprocess.run([sys.executable, "-c", probe], cwd=wa, capture_output=True, text=True)
    mods = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
    record("offline_imports_no_trainer", mods == [], loaded=mods, stderr=p.stderr[-300:])

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
                        ("mlp_model", ["--set", "transition_model.kind=mlp"]),
                        ("noisy_action_without_its_noise", ["--set", "generation.method=noisy_action"])):
        w = out / f"preflight_{name}"
        w.mkdir()
        write_trajectories(fixture(config), w / "fixture.npz")
        r = ccil(*offline_args, *extra, cwd=w)
        record(f"fails_before_any_stage[{name}]", r.returncode == 2 and not (w / "runs").exists()
               and "Traceback" not in r.stderr, error=r.stderr.strip())

    # An input a stage refuses exits 2 and completes nothing; a defect inside a stage exits 1 with its traceback.
    def completed(w):
        return sorted(n for n, p in runs(w / "runs").items() if validate_run(p)[0] == "complete") if (w / "runs").exists() else []
    w = out / "stage_input_error"
    w.mkdir()
    write_trajectories(fixture(config), w / "fixture.npz")
    r = ccil(*offline_args, "--set", "data.validation_fraction=0.9", cwd=w)  # two episodes: no train split remains
    record("stage_input_error_exits_2", r.returncode == 2 and "Traceback" not in r.stderr and not completed(w),
           error=r.stderr.strip()[-300:])
    w = out / "stage_defect"
    w.mkdir()
    write_trajectories(fixture(config), w / "fixture.npz")
    prelude = ("import ccil.pipelines.offline as o\n"
               "def broken(*a, **k):\n    raise TypeError('simulated defect')\n"
               "o.generate_labels = broken\n")
    r = subprocess.run([sys.executable, "-c", prelude + "import sys\nfrom ccil.__main__ import main\nsys.exit(main(sys.argv[1:]))\n",
                        *offline_args], cwd=w, capture_output=True, text=True)
    done = completed(w)
    record("stage_defect_exits_1_with_its_traceback", r.returncode == 1 and "Traceback" in r.stderr
           and "simulated defect" in r.stderr and not any(n.startswith("labels-generate") for n in done)
           and any(n.startswith("data-prepare") for n in done), completed_before_the_defect=done,
           error=r.stderr.strip()[-300:])
    # A generator that returns a malformed CandidateBatch is a defect in that implementation, not a refused input.
    w = out / "malformed_candidates"
    w.mkdir()
    write_trajectories(fixture(config), w / "fixture.npz")
    prelude = ("import dataclasses, ccil.core.generators as g\n"
               "real = g.BackwardEuler.generate\n"
               "def broken(self, batch, model, seed):\n"
               "    c = real(self, batch, model, seed)\n    return dataclasses.replace(c, a=c.a[:, :0])\n"
               "g.BackwardEuler.generate = broken\n")
    r = subprocess.run([sys.executable, "-c", prelude + "import sys\nfrom ccil.__main__ import main\nsys.exit(main(sys.argv[1:]))\n",
                        *offline_args], cwd=w, capture_output=True, text=True)
    done = completed(w)
    record("malformed_candidate_batch_exits_1_with_its_traceback", r.returncode == 1 and "Traceback" in r.stderr
           and "malformed CandidateBatch" in r.stderr and not any(n.startswith("labels-generate") for n in done),
           completed_before_the_defect=done, error=r.stderr.strip()[-300:])

    # 6. The golden path (#170): diagnose, generate, curate, inspect, on 20 double-integrator episodes.
    golden(out / "golden", record)

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
