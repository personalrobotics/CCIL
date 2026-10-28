#!/usr/bin/env python3
"""The CCIL supervision overlay on a LeRobot dataset (#28): export, verification, and use by LeRobot releases.

    uv run --group evidence --extra lerobot python tools/m4_overlay/check.py --out DIR [--releases 0.4.0,...]

The analytic double-integrator fixture is written as a LeRobot v3.0 dataset, converted (#27), labelled by the
ordinary pipeline (`ccil labels generate`, once with single actions and once with 4-step action chunks), resolved
by build_supervision, and exported as an overlay bundle. On the CCIL side:
1. the source is unchanged, and the bundle's LeRobot portion is a byte-for-byte copy of it;
2. the overlay holds exactly the supervision's synthetic rows (values, weights, candidate rows, source frames,
   classifications), only accepted candidates, the run ID once in ccil/meta.json, and every input hash;
3. refusals: an existing destination, a source that differs from the conversion manifest, a supervision dataset
   with no link to the conversion, a non-CCIL supervision; a failed export leaves nothing; a tampered shard or LeRobot
   portion is refused on reading.
For each LeRobot release (each in its own environment, with ccil's core wheel installed beside it):
4. stock LeRobot loads the bundle and sees exactly the source's episodes and frames: the overlay is invisible;
5. CCILTrainingDataset sees every usable label with its weight, the demonstration frames t <= T - 2 of the training
   episodes, and nothing else; its weights, balanced or not, are as recorded;
6. temporal windows: an observation history is refused; a future-action window is refused without stored chunks,
   and with them every label's window is exactly its own chunk, never a neighbouring label.
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tools" / "m2_interfaces"))
from check import fixture  # noqa: E402  (tools/m2_interfaces/check.py)

from ccil.core import IdentityRepresentation  # noqa: E402
from ccil.lerobot import LeRobotFormatError, convert, write_dataset  # noqa: E402
from ccil.lerobot.overlay import export_overlay, read_overlay  # noqa: E402
from ccil.schema import read_candidates, read_trajectories  # noqa: E402
from ccil.supervision import build_supervision  # noqa: E402

SUPPORTED = ("0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.5.0", "0.5.1", "0.6.0", "0.6.1")
CHUNK = 4


def tree(d: Path, skip=None) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file() and not (skip and p.relative_to(d).parts[0] == skip)}


def labels(work: Path, data: Path, chunk: int | None, extra=()):
    args = [sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
            "--set", f"data.path={data}", "--set", f"run.output_dir={work / 'runs'}"]
    if chunk:
        args += ["--set", f"generation.action_chunk_size={chunk}"]
    for kv in extra:
        args += ["--set", kv]
    r = subprocess.run(args, capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    runs = {line.split()[0]: Path(line.split()[2]) for line in r.stdout.splitlines() if len(line.split()) == 3}
    return {k: (v if v.is_absolute() else work / v) for k, v in runs.items()}


def bundle(work: Path, src: Path, conv: Path, chunk: int | None, extra=()):
    work.mkdir(parents=True)
    runs = labels(work, conv / "trajectories.npz", chunk, extra)
    prepared, labelled = runs["data.prepare"], runs["labels.generate"]
    cands_path = labelled / "labels" / "candidates.npz"
    sup = build_supervision(prepared / "data" / "dataset.npz", "ccil", representation=IdentityRepresentation("state"),
                            candidates=cands_path, split=prepared / "data" / "split.json",
                            action_chunks="consume" if chunk else "refuse")  # the chunk size is the candidates'
    cands = read_candidates(cands_path, read_trajectories(prepared / "data" / "dataset.npz"))
    dest = work / "bundle"
    meta = export_overlay(src, dest, sup, conversion_manifest=conv / "manifest.json", run_id=cands.run_id,
                          generator=cands.generator, measurements=cands.measurements, prepared_run=prepared)
    return dest, meta, sup, cands, prepared


def ccil_side(src_tree: dict, src: Path, dest: Path, meta: dict, sup, cands) -> dict:
    m, rows = read_overlay(dest)
    syn = np.asarray(sup.source) == "synthetic"
    order = np.argsort(rows["candidate_row"])
    ref = np.argsort(np.asarray(sup.candidate_row)[syn])
    feat = meta["feature_map"]["observation"]["lerobot_feature"]
    chunked = "action_chunk" in rows
    act_ok = (np.array_equal(rows["action_chunk"][order].reshape(np.asarray(sup.action_chunks)[syn].shape),
                             np.asarray(sup.action_chunks)[syn][ref].astype(rows["action_chunk"].dtype))
              if chunked else np.array_equal(rows["action"][order], np.asarray(sup.actions)[syn][ref].astype(rows["action"].dtype)))
    ok = {"source_unchanged": tree(src) == src_tree,
          "lerobot_portion_is_the_source": tree(dest, skip="ccil") == src_tree,
          "rows_are_the_synthetic_supervision": len(rows["candidate_row"]) == int(syn.sum())
          == sup.manifest()["counts"]["synthetic_rows"],
          "values_weights_and_identity": act_ok
          and np.array_equal(rows[feat][order], np.asarray(sup.observations)[syn][ref].astype(rows[feat].dtype))
          and np.array_equal(rows["weight"][order], np.asarray(sup.weights)[syn][ref])
          and np.array_equal(rows["candidate_row"][order], np.asarray(sup.candidate_row)[syn][ref])
          and np.array_equal(rows["source_frame_index"][order], np.asarray(sup.frame)[syn][ref]),
          "only_accepted_candidates": bool(cands.accepted[rows["candidate_row"]].all()),
          "rejected_candidates_absent": int((~cands.accepted).sum()) > 0
          and not np.isin(np.flatnonzero(~cands.accepted), rows["candidate_row"]).any(),
          "run_id_once": m["run_id"] == cands.run_id and "run_id" not in rows,
          "input_hashes_recorded": m["inputs"] == sup.manifest()["inputs"],
          "measurements_per_label": all(np.array_equal(rows[f"measurement.{k}"], np.asarray(v, np.float64)[rows["candidate_row"]])
                                        for k, v in cands.measurements.items())}
    return {"pass": all(ok.values()), **ok, "rows": int(len(rows["candidate_row"])), "rejected": int((~cands.accepted).sum()),
            "classifications": m["classifications"], "columns": m["columns"]}


def refusals(work: Path, src: Path, conv: Path, sup, cands, prepared, dest: Path) -> dict:
    res = {}

    def attempt(name, fn, expect):
        before = sorted(p.name for p in work.iterdir())
        try:
            fn()
            res[name] = {"refused": False}
        except (LeRobotFormatError, FileExistsError, ValueError, IndexError) as e:
            res[name] = {"refused": expect in str(e) and sorted(p.name for p in work.iterdir()) == before,
                         "message": str(e)[-250:]}
    kw = dict(conversion_manifest=conv / "manifest.json", run_id=cands.run_id, generator=cands.generator)
    attempt("existing_destination", lambda: export_overlay(src, dest, sup, prepared_run=prepared, **kw), "exists")
    tampered = work / "tampered-source"
    shutil.copytree(src, tampered)
    (tampered / "meta" / "stats.json").write_text((tampered / "meta" / "stats.json").read_text() + " ")
    attempt("source_differs_from_the_conversion", lambda: export_overlay(tampered, work / "o1", sup,
                                                                         prepared_run=prepared, **kw), "differs")
    other = json.loads((conv / "manifest.json").read_text())
    other["output"]["sha256"] = "0" * 64  # a conversion whose output is not the supervision's dataset
    (work / "other-conversion.json").write_text(json.dumps(other))
    kw_other = {**kw, "conversion_manifest": work / "other-conversion.json"}
    attempt("no_link_to_the_conversion", lambda: export_overlay(src, work / "o2", sup, prepared_run=prepared, **kw_other),
            "is not the conversion's output")
    bc = build_supervision(prepared / "data" / "dataset.npz", "bc", representation=IdentityRepresentation("state"),
                           split=prepared / "data" / "split.json")
    attempt("non_ccil_supervision", lambda: export_overlay(src, work / "o3", bc, prepared_run=prepared, **kw), "method")
    bad = {"distance": np.zeros(1)}  # too short: fails after the copy has started, inside the staging directory
    attempt("failed_export_leaves_nothing", lambda: export_overlay(src, work / "o4", sup, prepared_run=prepared,
                                                                   measurements=bad, **kw), "index")
    for name, edit in (("tampered_label_shard", lambda d: _flip(next((d / "ccil" / "labels").rglob("*.parquet")))),
                       ("modified_lerobot_portion", lambda d: _flip(next((d / "data").rglob("*.parquet"))))):
        copy = work / f"read-{name}"
        shutil.copytree(dest, copy)
        edit(copy)
        try:
            read_overlay(copy)
            res[name] = {"refused": False}
        except LeRobotFormatError as e:
            res[name] = {"refused": True, "message": str(e)[-200:]}
    return {"pass": all(v["refused"] for v in res.values()), **res}


def _flip(p: Path):
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 1
    p.write_bytes(bytes(b))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m4_lerobot_check", REPO / "tools" / "m4_lerobot" / "check.py")
    m4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m4)  # its per-release environments (tools/m4_lerobot/check.py)
    env_for = m4.env_for
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--releases", default=",".join(SUPPORTED))
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)

    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    config["episodes"] = [e for e in config["episodes"] if not e["corrupt_actions"]]
    src = out / "source"
    write_dataset(fixture(config), src, fps=10)
    src_tree = tree(src)
    conv = out / "conversion"
    conv.mkdir()
    convert(src, conv / "trajectories.npz", conv / "manifest.json")
    single = bundle(out / "single", src, conv, None)
    chunked = bundle(out / "chunked", src, conv, CHUNK)
    for name, (dest, meta, sup, cands, _) in (("single", single), ("chunked", chunked)):
        r = ccil_side(src_tree, src, dest, meta, sup, cands)
        record(f"overlay_{name}", r.pop("pass"), **r)
    # Zero accepted labels is an outcome, not an error: a valid, empty overlay.
    for name, chunk in (("none", None), ("none-chunked", CHUNK)):  # with chunks too: an empty (0, K, A) reshapes
        none = bundle(out / name, src, conv, chunk, extra=("generation.support_radius=1e-12",))
        m, rows = read_overlay(none[0])
        record(f"zero_accepted_labels_give_an_empty_overlay[{'chunked' if chunk else 'single'}]", m["rows"] == 0
               and len(rows["candidate_row"]) == 0 and int((~none[3].accepted).sum()) == len(none[3].accepted) > 0
               and tree(none[0], skip="ccil") == src_tree and m["classifications"] == {},
               candidates=int(len(none[3].accepted)), rows=m["rows"], columns=m["columns"])
    r = refusals(out / "single", src, conv, single[2], single[3], single[4], single[0])
    record("refusals", r.pop("pass"), **r)

    dist = out / "dist"
    subprocess.run(["uv", "build", "-q", "--wheel", "--out-dir", str(dist), str(REPO)], check=True)
    wheel = next(dist.glob("ccil-*.whl"))
    matrix = {}
    for release in a.releases.split(","):
        python, freeze = env_for(out, release)
        row = {}
        if python is None:
            matrix[release] = {"pass": False, "install_error": freeze}
            continue
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), str(wheel)], check=True)
        work = out / "work" / release
        work.mkdir(parents=True)

        def side(*args):
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_LEROBOT_HOME": str(work / "hf-home")}
            return subprocess.run([str(python), str(HERE / "lerobot_side.py"), *map(str, args)], capture_output=True,
                                  text=True, cwd=work, env=env)
        dest, meta, sup, cands, _ = single
        r = side("stock", dest, src, work / "stock.npz")
        if r.returncode == 0:
            z = dict(np.load(work / "stock.npz", allow_pickle=False))
            keys = sorted(k[len("source/"):] for k in z if k.startswith("source/"))
            row["stock_sees_exactly_the_source"] = bool(z["episodes"][0] == z["episodes"][1]) and all(
                np.array_equal(z[f"bundle/{k}"], z[f"source/{k}"]) for k in keys) and keys == sorted(
                k[len("bundle/"):] for k in z if k.startswith("bundle/"))
        else:
            row["stock_sees_exactly_the_source"], row["stock_error"] = False, r.stderr[-600:]
        r = side("aware", dest, work / "aware.npz")
        if r.returncode == 0:
            z = dict(np.load(work / "aware.npz", allow_pickle=False))
            _, rows = read_overlay(dest)
            syn = np.asarray(sup.source) == "synthetic"
            n_demo = sup.manifest()["counts"]["demonstration_rows"]
            w = z["weights"]
            row["aware_sees_every_label_and_weight"] = (
                np.array_equal(z["synthetic/ccil.candidate_row"], rows["candidate_row"])
                and np.array_equal(z["synthetic/ccil.weight"], rows["weight"])
                and np.array_equal(z["synthetic/action"], rows["action"])
                and int(z["demo_rows"]) == n_demo == int((~syn).sum())
                and float(z["demo_first_weight"]) == sup.manifest()["demonstration_weight"]
                and not bool(z["demo_first_synthetic"])
                and abs(z["weights_balanced"][:n_demo].sum() - 0.5) < 1e-9 and abs(z["weights_balanced"][n_demo:].sum() - 0.5) < 1e-9
                and np.array_equal(w[n_demo:], rows["weight"]))
            row["labels"], row["demonstration_rows"] = int(len(rows["candidate_row"])), n_demo
        else:
            row["aware_sees_every_label_and_weight"], row["aware_error"] = False, r.stderr[-600:]
        t = {}
        for name, (dest_, k) in (("single", (single[0], 0)), ("chunked", (chunked[0], CHUNK))):
            r = side("temporal", dest_, work / f"temporal-{name}.json", k)
            t[name] = json.loads((work / f"temporal-{name}.json").read_text()) if r.returncode == 0 else {"error": r.stderr[-600:]}
        row["temporal"] = t
        row["temporal_fails_closed"] = (t["single"].get("observation_history", {}).get("refused") is True
                                        and t["single"].get("future_actions", {}).get("refused") is True
                                        and t["chunked"].get("observation_history", {}).get("refused") is True
                                        and t["chunked"].get("future_actions", {}).get("windows_are_each_labels_own_chunk") is True)
        row["pass"] = row["stock_sees_exactly_the_source"] and row["aware_sees_every_label_and_weight"] and row["temporal_fails_closed"]
        matrix[release] = row
        print(f"  {release}: stock {row['stock_sees_exactly_the_source']} aware {row['aware_sees_every_label_and_weight']} "
              f"temporal {row['temporal_fails_closed']}", flush=True)
    record("lerobot_releases", all(r["pass"] for r in matrix.values()), matrix=matrix)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform()}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
