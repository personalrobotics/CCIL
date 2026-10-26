#!/usr/bin/env python3
"""Evidence for #23: explicit legacy dataset and configuration conversion.

    uv run --group evidence python tools/legacy_conversion/check.py --out DIR

- Both retained Pendulum datasets convert through the trusted boundary and
  ccil data convert, and equal the unpickled legacy arrays exactly (values and
  dtypes); manifests carry the pickle, M1, and output hashes and converter
  versions; a second conversion is byte-identical.
- Both Pendulum configurations convert in both fidelities, with a disposition
  for every legacy key; every supplied value is preserved; the fidelities
  differ exactly in the implicit legacy behaviors; the outputs resolve; and
  the shipped recipes resolve identically to the corrected conversions.
- Unknown, duplicate, ambiguous, incompatible, and mistyped settings, a
  non-.pkl dataset path without a map, non-M1 and truncated datasets, and
  unacknowledged or tampered pickles all fail with a diagnostic naming the
  problem, write nothing, and (for pickles) execute nothing.
- The converted dataset and configuration prepare a complete run, so they can
  be inspected before any model is trained.
- policy.exclude_validation_episodes (written false by the historical
  fidelity) trains the experimental policy on every episode, as legacy BC did;
  true holds the split's validation episodes out.
Writes DIR/report.json and DIR/artifacts/ (manifests and converted configurations).
"""
import argparse
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "correct_il"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_pendulum import RETAINED, convert_pendulum  # noqa: E402
from trajectory_io import LegacyInputError, convert_legacy_pickle  # noqa: E402

from ccil.config import resolve  # noqa: E402
from ccil.config.convert import LEGACY_IMPLICIT  # noqa: E402
from ccil.config import REGISTRY  # noqa: E402
from ccil.config.migrate import RENAMED, flatten  # noqa: E402
from ccil.config.spec import coerce  # noqa: E402
from ccil.runs import validate_run  # noqa: E402
from ccil.schema import read_trajectories  # noqa: E402


class Exploit:
    """Unpickling this creates a marker file: proof that a pickle was executed."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


def ccil(*args, cwd=REPO):
    return subprocess.run([sys.executable, "-m", "ccil", *map(str, args)], cwd=cwd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:900]}", flush=True)

    a, b = out / "convert_a", out / "convert_b"
    summary = convert_pendulum(a)
    convert_pendulum(b)
    artifacts = {x["path"]: x["sha256"] for x in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
                 if x.get("path")}

    # 1. Datasets.
    rows = {}
    for name, (pkl, _) in RETAINED.items():
        stem = Path(pkl).stem
        ds = read_trajectories(a / "data" / f"{stem}.npz")
        assert sha(REPO / pkl) == artifacts[pkl]
        with open(REPO / pkl, "rb") as f:  # trusted: its sha256 was verified against the artifact manifest above
            legacy = pickle.load(f)
        exact = len(ds.episodes) == len(legacy) and all(
            np.array_equal(np.asarray(l["observations"]), e.observations["state"])
            and np.asarray(l["observations"]).dtype == e.observations["state"].dtype
            and np.array_equal(np.asarray(l["actions"]).reshape(len(l["actions"]), -1), e.actions)
            and np.asarray(l["actions"]).dtype == e.actions.dtype
            for e, l in zip(ds.episodes, legacy))
        m = json.loads((a / "manifests" / f"{stem}.dataset.json").read_text())
        chain = ds.meta["converted_from"]
        rows[name] = {"episodes": len(ds.episodes), "exact": exact, "manifest": m,
                      "chain_ok": m["legacy_pickle"]["sha256"] == artifacts[pkl] == chain["meta"]["source"]["sha256"]
                      and m["output"]["sha256"] == sha(a / "data" / f"{stem}.npz") and m["trust"] == "sha256"}
    reproducible = tree(a) == tree(b)
    record("datasets_convert_exactly_with_provenance",
           all(r["exact"] and r["chain_ok"] for r in rows.values()) and reproducible,
           datasets={k: {"episodes": v["episodes"], "exact": v["exact"], "chain_ok": v["chain_ok"],
                         "legacy_pickle_sha256": v["manifest"]["legacy_pickle"]["sha256"],
                         "output_sha256": v["manifest"]["output"]["sha256"], "dtypes": v["manifest"]["dtypes"]}
                     for k, v in rows.items()},
           second_conversion_byte_identical=reproducible)

    # 2. Configurations.
    import yaml
    cfg_rows = {}
    for name, (_, yml) in RETAINED.items():
        raw = yaml.safe_load((REPO / yml).read_text())
        legacy_keys = set(flatten(raw))
        man = {f: json.loads((a / "manifests" / f"{name}.{f}.config.json").read_text()) for f in ("corrected", "historical")}
        res = {f: resolve(config=a / "config" / f"{name}.{f}.yaml") for f in man}
        covered = all({r["legacy"] for r in m["dispositions"]} == legacy_keys for m in man.values())
        # Independently: every renamed legacy value, coerced from the legacy file, equals the resolved canonical value.
        raw_flat = flatten(raw)
        preserved = all(coerce(REGISTRY[RENAMED[k]], v) == res[f][RENAMED[k]]
                        for f in res for k, v in raw_flat.items() if k in RENAMED)
        renamed_equal = all(r["canonical"] == RENAMED[r["legacy"]] for m in man.values() for r in m["dispositions"]
                            if r["disposition"] == "renamed")
        differ = sorted(k for k in res["corrected"].values if res["corrected"][k] != res["historical"][k])
        recipe = resolve(recipe=name)
        recipe_diff = {k: (recipe[k], res["corrected"][k]) for k in recipe.values if recipe[k] != res["corrected"][k]}
        cfg_rows[name] = {"legacy_keys": len(legacy_keys), "dispositions_cover_every_key": covered,
                          "counts": man["corrected"]["counts"], "supplied_values_preserved": preserved and renamed_equal,
                          "fidelities_differ_in": differ, "recipe_differences": recipe_diff,
                          "implicit": man["historical"]["implicit_legacy_behavior"]}
    record("configurations_convert_with_dispositions",
           all(r["dispositions_cover_every_key"] and r["supplied_values_preserved"]
               and r["fidelities_differ_in"] == sorted(LEGACY_IMPLICIT) and not r["recipe_differences"]
               for r in cfg_rows.values()),
           configurations=cfg_rows)

    # 3. Malformed inputs fail, write nothing, and execute nothing.
    bad = out / "malformed"
    bad.mkdir()
    base = (REPO / "config/pendulum_cont.yml").read_text()
    cases_yaml = {
        "unknown_key": (base.replace("  epsilon:", "  bogus_knob: 3\n  epsilon:"), "aug.bogus_knob"),
        "duplicate_key": (base.replace("  epsilon:", "  num_labels: 7\n  epsilon:"), "duplicate key"),
        "ambiguous_naive_with_noise": (base.replace("noise_bc: 0.0", "noise_bc: 0.5"), "policy.naive"),
        "incompatible_data_frac": (base.replace("data:\n", "data:\n  data_frac: 0.5\n"), "data.data_frac"),
        "incompatible_state_encoder": (base + "state_encoder: resnet\n", "state_encoder"),
        "mistyped_value": (base.replace("num_labels: 20", "num_labels: twenty"), "aug.num_labels"),
        "dataset_path_not_pkl_without_map": (base.replace("  path: 'data/pendulum_cont_100.npz'", "  pkl: 'data/pendulum_cont_100.h5'"),
                                             "data.pkl"),
    }
    malformed = {}
    for label, (text, needle) in cases_yaml.items():
        assert text != base or label == "incompatible_state_encoder", f"{label}: the edit did not apply"
        src = bad / f"{label}.yml"
        src.write_text(text)
        o, m = bad / f"{label}.out.yaml", bad / f"{label}.manifest.json"
        r = ccil("config", "convert", src, "--out", o, "--manifest", m)
        malformed[label] = {"exit": r.returncode, "names_problem": needle in r.stderr, "nothing_written": not o.exists() and not m.exists(),
                            "diagnostic": r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else ""}
    np.savez(bad / "not_m1.npz", x=np.zeros(3))
    m1 = a / "m1" / "pendulum_cont_100.npz"
    (bad / "truncated.npz").write_bytes(m1.read_bytes()[: len(m1.read_bytes()) // 2])
    for label, src in (("dataset_not_m1", bad / "not_m1.npz"), ("dataset_truncated", bad / "truncated.npz"),
                       ("dataset_already_canonical", a / "data" / "pendulum_cont_100.npz")):
        o = bad / f"{label}.out.npz"
        r = ccil("data", "convert", src, o, "--manifest", bad / f"{label}.json")
        malformed[label] = {"exit": r.returncode, "names_problem": any(t in r.stderr for t in ("ccil-trajectories/1", "readable", "format")),
                            "nothing_written": not o.exists() and not (bad / f"{label}.json").exists(),
                            "diagnostic": r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else ""}
    marker = bad / "pickle-executed"
    exploit = bad / "exploit.pkl"
    with open(exploit, "wb") as f:
        pickle.dump(Exploit(str(marker)), f)
    for label, kwargs in (("pickle_unacknowledged", {}), ("pickle_tampered", {"expect_sha256": "0" * 64})):
        try:
            convert_legacy_pickle(exploit, bad / f"{label}.npz", **kwargs)
            outcome = "accepted"
        except LegacyInputError as e:
            outcome = f"LegacyInputError: {str(e)[:160]}"
        malformed[label] = {"exit": 2 if outcome != "accepted" else 0, "names_problem": "refusing to unpickle" in outcome or "sha256" in outcome,
                            "nothing_written": not (bad / f"{label}.npz").exists(), "diagnostic": outcome,
                            "pickle_executed": marker.exists()}
    record("malformed_inputs_fail_closed",
           all(v["exit"] == 2 and v["names_problem"] and v["nothing_written"] and not v.get("pickle_executed")
               for v in malformed.values()),
           cases=malformed)

    # 4. Consumable before any model is trained.
    prep = {}
    for name in RETAINED:
        for f in ("corrected", "historical"):
            r = ccil("data", "prepare", "--config", f"config/{name}.{f}.yaml", "--set", f"run.output_dir=prep/{name}-{f}",
                     "--set", "compute.device=cpu", cwd=a)
            run = next((a / "prep" / f"{name}-{f}").glob("data-prepare-*"), None)
            prep[f"{name}.{f}"] = {"exit": r.returncode, "status": validate_run(run)[0] if run else None,
                                   "stderr": r.stderr[-200:]}
    hist = resolve(config=a / "config" / "pendulum_disc.historical.yaml")
    record("converted_artifacts_prepare_a_complete_run",
           all(v["exit"] == 0 and v["status"] == "complete" for v in prep.values())
           and hist["policy.exclude_validation_episodes"] is False and hist["policy.noise_model"] == "legacy_uniform_half_batch",
           runs=prep, historical_policy={"exclude_validation_episodes": hist["policy.exclude_validation_episodes"],
                                         "noise_model": hist["policy.noise_model"]})

    # 5. policy.exclude_validation_episodes: historical (false) trains on every episode, corrected holds validation out.
    sys.path.insert(0, str(REPO / "tools" / "m3_supervision"))
    sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
    from check import rollouts  # noqa: E402  (tools/m3_supervision/check.py)
    from system import DoubleIntegrator  # noqa: E402
    from ccil.schema import write_trajectories  # noqa: E402
    h = out / "harness"
    h.mkdir()
    write_trajectories(rollouts(DoubleIntegrator(0.1), 20261026), h / "rollouts.npz")
    held = {}
    for flag in ("true", "false"):
        r = ccil("policy", "train", "--recipe", "analytic_double_integrator", "--set", "data.path=rollouts.npz",
                 "--set", "run.output_dir=runs", "--set", "data.validation_fraction=0.25", "--set", "compute.device=cpu",
                 "--set", "policy.method=bc", "--set", "policy.batch_size=16", "--set", "policy.epochs=1",
                 "--set", f"policy.exclude_validation_episodes={flag}", cwd=h)
        stages = {ln.split()[0]: ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip()}
        counts = json.loads((h / stages["policy.train"] / "supervision.json").read_text())["counts"] if r.returncode == 0 else {}
        ev = json.loads((h / stages["policy.evaluate"] / "metrics.json").read_text())["values"] if r.returncode == 0 else {}
        held[flag] = {"exit": r.returncode, "demonstration_rows": counts.get("demonstration_rows"),
                      "held_out": counts.get("demonstration_rows_held_out"),
                      "validation_episodes_in_training": ev.get("validation_episodes_in_training")}
    record("policy_validation_holdout_follows_the_flag",
           held["true"]["exit"] == 0 and held["false"]["exit"] == 0 and held["true"]["held_out"] > 0
           and held["false"]["held_out"] == 0 and held["false"]["demonstration_rows"] == held["true"]["demonstration_rows"] + held["true"]["held_out"]
           and held["false"]["validation_episodes_in_training"] is True and held["true"]["validation_episodes_in_training"] is False,
           runs=held)

    art = out / "artifacts"
    shutil.copytree(a / "manifests", art / "manifests")
    shutil.copytree(a / "config", art / "config")
    shutil.copy(a / "conversion.json", art / "conversion.json")
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases, "summary": summary,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "correct_il", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    for p in art.rglob("*.json"):
        p.write_text(p.read_text().replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
