#!/usr/bin/env python3
"""Evidence for the configuration contract and `data prepare` (#4).

    PYTHONPATH=src python tools/m2_config/check.py --out DIR

Runs `python -m ccil data prepare` on the Pendulum data with defaults and with an
override, checks per-value provenance, byte-identical replay, failures before
any output, legacy migration, and generated-docs drift. Writes DIR/report.json.
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "correct_il"))
from ccil.config import ConfigError, resolve  # noqa: E402
from ccil.schema import read_trajectories, write_trajectories  # noqa: E402
from ccil.schema.legacy import from_m1_npz  # noqa: E402
from trajectory_io import convert_legacy_pickle  # noqa: E402

ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}


def ccil(*args, cwd):
    return subprocess.run([sys.executable, "-m", "ccil", *map(str, args)], cwd=cwd, env=ENV, capture_output=True, text=True)


def tree_hashes(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    # Two working directories with identical relative inputs, for replay checks.
    sha = {a["path"]: a["sha256"] for a in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
           if a.get("path")}
    work = [out / "work_a", out / "work_b"]
    for w in work:
        (w / "data").mkdir(parents=True)
        convert_legacy_pickle(REPO / "data/pendulum_cont_100.pkl", w / "data/pendulum_cont_100.npz",
                              expect_sha256=sha["data/pendulum_cont_100.pkl"])
    base = ["--recipe", "pendulum_cont", "--set", "run.output_dir=runs"]

    # --- default and override runs ------------------------------------------
    r_def = ccil("data", "prepare", *base, cwd=work[0])
    r_ovr = ccil("data", "prepare", *base, "--set", "data.validation_fraction=0.3", cwd=work[0])
    runs = sorted((work[0] / "runs").iterdir())
    ok = r_def.returncode == 0 and r_ovr.returncode == 0 and len(runs) == 2
    detail = {"stderr": (r_def.stderr + r_ovr.stderr)[-800:]}
    if ok:
        def load(run):
            return (json.loads((run / "config.json").read_text()), json.loads((run / "manifest.json").read_text()),
                    json.loads((run / "split.json").read_text()))
        by_frac = {load(r)[0]["values"]["data.validation_fraction"]["value"]: (r, *load(r)) for r in runs}
        d_run, d_cfg, d_man, d_split = by_frac[0.2]
        o_run, o_cfg, o_man, o_split = by_frac[0.3]
        changed = sorted(k for k in d_cfg["values"] if d_cfg["values"][k] != o_cfg["values"][k])
        sources_ok = (d_cfg["values"]["data.validation_fraction"]["source"] == "recipe:pendulum_cont"
                      and o_cfg["values"]["data.validation_fraction"]["source"] == "cli"
                      and d_cfg["values"]["run.output_dir"]["source"] == "cli"
                      and d_cfg["values"]["transition_model.lipschitz_type"]["source"] == "recipe:pendulum_cont"
                      and d_cfg["values"]["generation.root_tolerance"]["source"] == "default")
        hashes_ok = all(d_man["files"][n] == hashlib.sha256((d_run / n).read_bytes()).hexdigest() for n in d_man["files"])
        ok = sources_ok and hashes_ok and changed == ["data.validation_fraction"] \
            and len(d_split["validation"]) == 20 and len(o_split["validation"]) == 30 and d_man["run_id"] != o_man["run_id"]
        detail = {"default_run": d_man["run_id"], "override_run": o_man["run_id"], "changed_values": changed,
                  "override_source": o_cfg["values"]["data.validation_fraction"]["source"],
                  "validation_episodes": [len(d_split["validation"]), len(o_split["validation"])],
                  "seed": d_man["seed"], "default_manifest": d_man, "override_manifest": o_man,
                  "default_value_sources": {k: v["source"] for k, v in d_cfg["values"].items()}}
    record("default_and_override_runs_record_provenance", ok, **detail)

    # --- byte-identical replay from another working directory ---------------
    r_b = ccil("data", "prepare", *base, cwd=work[1])
    a_dirs, b_dirs = sorted((work[0] / "runs").iterdir()), sorted((work[1] / "runs").iterdir())
    default_a = [d for d in a_dirs if json.loads((d / "split.json").read_text())["validation_fraction"] == 0.2][0]
    same = r_b.returncode == 0 and len(b_dirs) == 1 and b_dirs[0].name == default_a.name \
        and tree_hashes(b_dirs[0]) == tree_hashes(default_a)
    record("replay_is_byte_identical", same, run_id=b_dirs[0].name if b_dirs else None, files=tree_hashes(default_a))

    # --- existing run directory: refused, then atomically replaced ---------
    before = tree_hashes(default_a)
    r_again = ccil("data", "prepare", *base, cwd=work[0])
    refused = r_again.returncode == 2 and "exists" in r_again.stderr and tree_hashes(default_a) == before
    r_over = ccil("data", "prepare", "--recipe", "pendulum_cont", "--set", "run.output_dir=runs", "--set", "run.overwrite=true",
                  cwd=work[0])
    leftovers = [p.name for p in (work[0] / "runs").iterdir() if p.name.startswith(".")]
    record("existing_run_refused_then_replaced_atomically", refused and r_over.returncode == 0 and not leftovers,
           refused_stderr=r_again.stderr.strip()[-200:], leftovers=leftovers)

    # --- failures before any output -----------------------------------------
    def fails(name, sets, needle, extra=()):
        d = out / f"fail_{name}"
        d.mkdir()
        shutil.copytree(work[0] / "data", d / "data")
        r = ccil("data", "prepare", *extra, "--recipe", "pendulum_cont", "--set", "run.output_dir=runs",
                 *sum((["--set", s] for s in sets), []), cwd=d)
        record(f"fails_before_output[{name}]", r.returncode == 2 and needle in r.stderr and not (d / "runs").exists(),
               error=r.stderr.strip()[-400:])

    fails("misspelled_key", ["data.validaton_fraction=0.3"], "did you mean 'data.validation_fraction'")
    fails("wrong_type", ["run.seed=abc"], "is not a valid integer")
    fails("ambiguous_boolean", ["run.overwrite=yes"], "expected true or false")
    fails("wrong_unit", ["evaluation.max_episode_steps=10 s"], "does not match the declared unit 'steps'")
    fails("out_of_range", ["data.validation_fraction=1.5"], "outside [0.0, 1.0)")
    fails("incompatible_combination", ["data.action_low=-1"], "must be declared together")
    fails("noise_bc_without_scale", ["policy.method=noise_bc"], "requires a positive policy.noise_scale")
    fails("unknown_recipe_value", ["generation.method=magic"], "is not one of")
    bad = out / "fail_invalid_data"
    bad.mkdir()
    ds = from_m1_npz(work[0] / "data/pendulum_cont_100.npz")
    ds.episodes[7].actions = ds.episodes[7].actions.copy()
    ds.episodes[7].actions[123, 0] = np.nan
    (bad / "data").mkdir()
    write_trajectories(ds, bad / "data/corrupt.npz", validate=False)
    r = ccil("data", "prepare", "--recipe", "pendulum_cont", "--set", "run.output_dir=runs", "--set", "data.path=data/corrupt.npz",
             cwd=bad)
    record("fails_before_output[invalid_dataset]", r.returncode == 2
           and "non_finite at episode 7 (7), frame 123: actions has non-finite values" in r.stderr
           and not (bad / "runs").exists(), error=r.stderr.strip()[-300:])

    # --- typed parsing and units --------------------------------------------
    cfg = resolve(recipe="pendulum_cont", sets=["run.overwrite=false", "evaluation.max_episode_steps=500 steps",
                                                "data.action_low=[-3, -3]", "data.action_high=3,3"])
    record("typed_parsing", cfg["run.overwrite"] is False and cfg["evaluation.max_episode_steps"] == 500
           and cfg["data.action_low"] == [-3.0, -3.0] and cfg["data.action_high"] == [3.0, 3.0],
           values={k: cfg[k] for k in ("run.overwrite", "evaluation.max_episode_steps", "data.action_low", "data.action_high")})
    py = resolve(recipe="pendulum_cont", overrides={"data.validation_fraction": 0.3})
    record("python_interface_precedence", py["data.validation_fraction"] == 0.3
           and py.sources["data.validation_fraction"] == "python")

    # --- migration -----------------------------------------------------------
    legacy = resolve(config=REPO / "config/pendulum_cont.yml")
    recipe = resolve(recipe="pendulum_cont")
    record("legacy_config_replays_to_recipe_values", legacy.values == recipe.values,
           migration_notes=len(legacy.notes), differing=sorted(k for k in recipe.values if legacy.values[k] != recipe.values[k]))
    lf = out / "label_filtering.yml"
    lf.write_text((REPO / "config/pendulum_cont.yml").read_text().replace("  epsilon:", "  label_err_quantile: 0.3\n  epsilon:"))
    try:
        resolve(config=lf)
        got = None
    except ConfigError as e:
        got = e.errors
    record("unsupported_legacy_key_fails_with_guidance", got is not None and any("ported in #22" in g for g in got), errors=got)
    v2 = out / "future.yml"
    v2.write_text("config_version: 2\nrun: {seed: 1}\n")
    try:
        resolve(config=v2)
        got = None
    except ConfigError as e:
        got = e.errors
    record("unknown_config_version_rejected", got is not None and any("config_version 2" in g for g in got), errors=got)

    # --- one canonical source for docs ---------------------------------------
    drift = {name: ccil(*cmd, cwd=REPO) for name, cmd in (
        ("parameters", ("config", "docs", "--check")), ("audit", ("config", "audit", "--check")),
        ("schemas", ("schema", "docs", "--check")))}
    record("generated_docs_match_sources", all(r.returncode == 0 for r in drift.values()),
           output={k: (r.stdout + r.stderr).strip() for k, r in drift.items()})

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "docs/config")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
