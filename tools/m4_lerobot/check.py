#!/usr/bin/env python3
"""LeRobot Dataset v3.0 compatibility workflow (#26): the normative gate for which LeRobot releases are supported.

    uv run --extra lerobot python tools/m4_lerobot/check.py --out DIR [--releases 0.4.0,...,0.6.1]

Needs network access (PyPI, and the Hugging Face Hub for the --hub case). Each LeRobot release runs in its own
Python 3.12 environment under DIR/envs; ccil never imports lerobot. For each release:
1. CCIL writes a state-only v3.0 dataset (several data shards, two state modalities, two tasks);
2. LeRobot loads it and iterates it through its public dataset and DataLoader;
3. LeRobot writes a fixture of known values and CCIL reads it;
4. both sides must agree on episode boundaries, ordering, feature values, dtypes, shapes, timestamps, task
   mappings and statistics: the stored statistics exactly; count, min, max, mean and std against LeRobot's own
   computation on the same frames to float tolerance; quantiles are reported (LeRobot's are approximate);
5. a small policy trains on CCIL's dataset through LeRobot's dataset, with no rewriting.
A release is supported only when all of these pass. Once, on the CCIL side:
6. malformed metadata, missing shards, unsupported modalities, inconsistent episode offsets, unknown versions,
   unmapped features, a path that leaves the dataset (lexically, or through a symlink to a data shard or a metadata
   file), a malformed path format and multi-task episodes are refused; a streamed write with a non-finite episode is
   refused; nothing partial is written or returned;
7. the boundary: the built wheel's ccil[lerobot] installs pyarrow (and huggingface_hub on Python 3.11+) and no
   LeRobot runtime; without the extra, using the bridge fails with install guidance; at the declared minimum
   versions, on Python 3.10 (pyarrow only: fetch says it needs 3.11) and 3.12, CCIL's write and read round-trip.
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
RELEASES = ("0.3.3", "0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.5.0", "0.5.1", "0.6.0", "0.6.1")
MINIMUM = {"pyarrow": "21.0.0", "huggingface_hub": "1.0.0"}
FPS = 20
FIXTURE_LENGTHS, FIXTURE_FPS = (5, 7, 4), 10
HUB = ("lerobot/pusht_keypoints", None)  # repository; the revision is resolved and recorded at run time
FORBIDDEN = ("lerobot", "torch", "datasets", "av", "torchcodec", "pandas", "opencv-python-headless")

from ccil.lerobot import (LeRobotFormatError, fetch, read_dataset, read_info, read_stats,  # noqa: E402
                          write_dataset)
from ccil.schema import Episode, TrajectoryDataset  # noqa: E402


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def source_dataset() -> TrajectoryDataset:
    """60 episodes of 150-249 frames, a 32-d state, a 16-d second state and an 8-d action: about 2.7 MB, so a 1 MB
    shard cap gives several shards. Actions are a fixed linear function of both states plus small noise, so the
    training smoke test has something to learn. Seeded, so the written dataset is byte-identical on a rerun."""
    r = np.random.default_rng(26)
    W = r.normal(size=(48, 8)) / np.sqrt(48)
    eps = []
    for i in range(60):
        T = int(r.integers(150, 250))
        state, env = r.normal(size=(T, 32)), r.normal(size=(T, 16))
        action = np.hstack([state, env]) @ W + 0.05 * r.normal(size=(T, 8))
        eps.append(Episode(str(i), {"state": state.astype(np.float32), "state.env": env.astype(np.float32)},
                           action.astype(np.float32), timestamps=None, task=["reach", "push"][i % 2]))
    return TrajectoryDataset(eps)


def fixture_expected():
    """What lerobot_side.py write records: episode e, frame t has v = 100 e + t."""
    out = []
    for e, length in enumerate(FIXTURE_LENGTHS):
        v = (e * 100 + np.arange(length)).astype(np.float32)
        out.append({"state": np.stack([v, v + 0.25, -v], 1), "state.environment_state": np.repeat((v / 7)[:, None], 4, 1),
                    "action": np.stack([v / 3, 1 - v], 1), "timestamps": (np.arange(length) / FIXTURE_FPS).astype(np.float32),
                    "task": f"task {e % 2}"})
    return out


def env_for(out: Path, release: str) -> tuple[Path | None, str]:
    env = out / "envs" / release
    if not env.exists():
        subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(env)], check=True)
    spec = f"lerobot[dataset]=={release}" if tuple(map(int, release.split("."))) >= (0, 6, 0) else f"lerobot=={release}"
    r = subprocess.run(["uv", "pip", "install", "-q", "--python", str(env / "bin" / "python"), spec],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-1500:]
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(env / "bin" / "python")], capture_output=True,
                            text=True).stdout
    return env / "bin" / "python", freeze


def lerobot(python: Path, *args, cwd: Path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_LEROBOT_HOME": str(cwd / "hf-home")}
    return subprocess.run([str(python), str(HERE / "lerobot_side.py"), *map(str, args)], capture_output=True,
                          text=True, env=env, cwd=cwd)


def close(a, b, rtol=1e-5, atol=1e-6) -> bool:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol))


def compare_written(src: TrajectoryDataset, ccil_root: Path, dump: dict) -> dict:
    """LeRobot's view of CCIL's dataset against the source episodes and CCIL's statistics."""
    lens = np.array([ep.length for ep in src.episodes])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    ok = {"frames": int(lens.sum()) == len(dump["frame/index"]),
          "episode_boundaries": np.array_equal(dump["episodes/from"], starts)
          and np.array_equal(dump["episodes/to"], starts + lens)}
    cat = lambda f: np.concatenate([f(ep) for ep in src.episodes])  # noqa: E731
    expect = {"observation.state": cat(lambda e: e.observations["state"]),
              "observation.env": cat(lambda e: e.observations["state.env"]), "action": cat(lambda e: e.actions),
              "timestamp": cat(lambda e: (np.arange(e.length) / FPS).astype(np.float32)),
              "index": np.arange(lens.sum()), "episode_index": np.repeat(np.arange(len(lens)), lens),
              "frame_index": cat(lambda e: np.arange(e.length))}
    ok["values_and_order"] = all(np.array_equal(dump[f"frame/{k}"].reshape(len(v), -1), v.reshape(len(v), -1))
                                 for k, v in expect.items())
    ok["dtypes_and_shapes"] = all(dump[f"frame/{k}"].dtype == v.dtype or k in ("index", "episode_index", "frame_index")
                                  for k, v in expect.items()) and dump["frame/observation.state"].shape[1:] == (32,)
    tasks = cat(lambda e: np.array([e.task] * e.length))
    ok["tasks"] = np.array_equal(dump["frame/task"], tasks)
    stats = read_stats(ccil_root)
    ok["stored_stats_read_as_written"] = all(close(dump[f"stored_stats/{f}/{s}"], v, rtol=0, atol=0)
                                             for f, sv in stats.items() for s, v in sv.items())
    exact = ("count", "min", "max", "mean", "std")
    ok["stats_match_lerobot"] = all(close(dump[f"lerobot_stats/{f}/{s}"], stats[f][s]) for f in stats for s in exact
                                    if f"lerobot_stats/{f}/{s}" in dump)
    quantile_gap = {f"{f}/{s}": float(np.abs(np.asarray(dump[f"lerobot_stats/{f}/{s}"]) - np.asarray(stats[f][s])).max())
                    for f in stats for s in stats[f] if s.startswith("q") and f"lerobot_stats/{f}/{s}" in dump}
    return {"pass": all(ok.values()), **ok,
            "quantiles_max_abs_difference_reported": dict(sorted(quantile_gap.items()))}


def compare_read(ds: TrajectoryDataset, root: Path, dump: dict) -> dict:
    """CCIL's reading of LeRobot's fixture against the known values and LeRobot's own view of it."""
    exp = fixture_expected()
    ok = {"episodes_and_order": [ep.episode_id for ep in ds.episodes] == [str(i) for i in range(len(exp))]
          and [ep.length for ep in ds.episodes] == list(FIXTURE_LENGTHS)}
    ok["values"] = all(np.array_equal(ep.observations["state"], e["state"])
                       and np.array_equal(ep.observations["state.environment_state"], e["state.environment_state"])
                       and np.array_equal(ep.actions, e["action"]) for ep, e in zip(ds.episodes, exp))
    ok["dtypes_and_shapes"] = all(ep.actions.dtype == np.float32 and ep.observations["state"].shape[1:] == (3,)
                                  for ep in ds.episodes)
    ok["timestamps"] = all(np.array_equal(ep.timestamps, e["timestamps"]) for ep, e in zip(ds.episodes, exp))
    ok["tasks"] = [ep.task for ep in ds.episodes] == [e["task"] for e in exp]
    lens = np.array(FIXTURE_LENGTHS)
    ok["episode_boundaries_agree_with_lerobot"] = (np.array_equal(dump["episodes/from"], np.cumsum(lens) - lens)
                                                   and np.array_equal(dump["episodes/to"], np.cumsum(lens)))
    ok["values_agree_with_lerobot"] = np.array_equal(
        np.concatenate([ep.observations["state"] for ep in ds.episodes]), dump["frame/observation.state"])
    stats = read_stats(root)
    ok["stats_agree_with_lerobot"] = all(close(dump[f"stored_stats/{f}/{s}"], v, rtol=0, atol=0)
                                         for f, sv in stats.items() for s, v in sv.items())
    return {"pass": all(ok.values()), **ok}


def refusals(work: Path, ccil_root: Path, fixture: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    res = {}

    def case(name, edit, expect, ignore=(), base=fixture):
        d = work / f"refuse-{name}"
        shutil.copytree(base, d)
        edit(d)
        try:
            read_dataset(d, ignore=ignore)
            res[name] = {"refused": False}
        except LeRobotFormatError as e:
            res[name] = {"refused": expect in str(e), "message": str(e)[-300:]}

    def info(d, f):
        p = d / "meta" / "info.json"
        i = json.loads(p.read_text())
        f(i)
        p.write_text(json.dumps(i))
    case("unreadable_info", lambda d: (d / "meta" / "info.json").write_text("{not json"), "unreadable JSON")
    case("missing_fields", lambda d: info(d, lambda i: i.pop("data_path")), "data_path")
    case("unknown_version", lambda d: info(d, lambda i: i.update(codebase_version="v2.1")), "not supported")
    case("future_version", lambda d: info(d, lambda i: i.update(codebase_version="v3.1")), "not supported")
    case("video_modality", lambda d: info(d, lambda i: i["features"].update(
        {"observation.images.top": {"dtype": "video", "shape": [96, 96, 3], "names": None}})), "M5")
    case("depth_modality", lambda d: info(d, lambda i: i["features"].update(
        {"observation.depth": {"dtype": "float32", "shape": [64, 64], "names": None}})), "M5")
    case("unmapped_feature", lambda d: info(d, lambda i: i["features"].update(
        {"next.reward": {"dtype": "float32", "shape": [1], "names": None}})), "no CCIL slot")
    case("path_leaves_dataset", lambda d: info(d, lambda i: i.update(
        data_path="../escape/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")), "inside the dataset")
    case("missing_shard", lambda d: next((d / "data").rglob("*.parquet")).unlink(), "is missing")
    case("malformed_path_format", lambda d: info(d, lambda i: i.update(
        data_path="data/chunk-{chunk_index:03q}/file-{file_index:03d}.parquet")), "cannot be formatted")

    def escape(d, rel):  # the file moves outside the dataset and a symlink takes its place
        inside = d / rel
        outside = d.parent / f"outside-{d.name}-{inside.name}"
        shutil.move(inside, outside)
        inside.symlink_to(outside)
    case("data_shard_symlink_escape",
         lambda d: escape(d, next((d / "data").rglob("*.parquet")).relative_to(d).as_posix()), "outside the dataset")
    case("metadata_symlink_escape", lambda d: escape(d, "meta/tasks.parquet"), "outside the dataset")

    def shift(d):
        p = next((d / "meta" / "episodes").rglob("*.parquet"))
        t = pq.read_table(p)
        i = t.schema.get_field_index("dataset_from_index")
        t = t.set_column(i, "dataset_from_index", pa.array([v + 1 for v in t.column(i).to_pylist()], pa.int64()))
        pq.write_table(t, p)
    case("inconsistent_offsets", shift, "contiguous")

    def two_tasks(d):
        p = next((d / "data").rglob("*.parquet"))
        t = pq.read_table(p)
        i = t.schema.get_field_index("task_index")
        v = t.column(i).to_pylist()
        v[1] = 1 - v[1]
        pq.write_table(t.set_column(i, "task_index", pa.array(v, pa.int64())), p)
    case("multi_task_episode", two_tasks, "tasks within one episode")
    # An unmapped feature named in ignore is read without it, and recorded.
    d = work / "ignore-accepted"
    shutil.copytree(fixture, d)
    info(d, lambda i: i["features"].update({"next.reward": {"dtype": "float32", "shape": [1], "names": None}}))
    try:
        ds = read_dataset(d, ignore=["next.reward"])
        res["ignored_feature_read_and_recorded"] = {"refused": ds.meta["dataset"]["ignored_features"] == ["next.reward"]}
    except LeRobotFormatError as e:
        res["ignored_feature_read_and_recorded"] = {"refused": False, "message": str(e)}
    # Writing: images, off-rate timestamps and an existing destination are refused, and nothing is left behind.
    src = source_dataset().episodes[:2]
    img = [Episode("0", {"image.top": np.zeros((5, 4, 4, 3), np.uint8)}, np.zeros((5, 1), np.float32))]
    off = [Episode("0", {"state": np.zeros((5, 2), np.float32)}, np.zeros((5, 1), np.float32),
                   timestamps=np.arange(5) * 0.07)]
    nan = Episode("1", {"state": np.zeros((5, 2), np.float32)}, np.full((5, 1), np.nan, np.float32))
    good = Episode("0", {"state": np.zeros((5, 2), np.float32)}, np.zeros((5, 1), np.float32))
    for name, eps, expect, dest in (("write_streamed_non_finite", (e for e in (good, nan)), "episode 1",
                                     work / "w-nan"),
                                    ("write_image", img, "M5", work / "w-img"),
                                    ("write_off_rate_timestamps", off, "1 / fps", work / "w-off"),
                                    ("write_existing_destination", src, "exists", ccil_root)):
        before = sorted(p.name for p in dest.parent.iterdir())
        try:
            write_dataset(eps, dest, fps=FPS)
            res[name] = {"refused": False}
        except (LeRobotFormatError, FileExistsError) as e:
            res[name] = {"refused": expect in str(e) and sorted(p.name for p in dest.parent.iterdir()) == before,
                         "message": str(e)[-200:]}
    return {"pass": all(v["refused"] for v in res.values()), **res}


BOUNDARY = """
import json, sys
import ccil, ccil.lerobot
try:
    ccil.lerobot.read_dataset(sys.argv[1])
    print(json.dumps({"guidance": None}))
except ImportError as e:
    print(json.dumps({"guidance": str(e)}))
"""

ROUND_TRIP = """
import json, sys
import numpy as np, pyarrow
try:
    import huggingface_hub
    hub = huggingface_hub.__version__
except ImportError:
    hub = None
from ccil.schema import Episode, TrajectoryDataset
from ccil.lerobot import fetch, read_dataset, write_dataset
eps = [Episode(str(i), {"state": np.arange(12, dtype=np.float32).reshape(6, 2) + i}, np.ones((6, 1), np.float32),
               task="t") for i in range(3)]
write_dataset(TrajectoryDataset(eps), sys.argv[1], fps=10)
back = read_dataset(sys.argv[1])
fetch_guidance = None
if hub is None:
    try:
        fetch("lerobot/pusht_keypoints", "main")
    except ImportError as e:
        fetch_guidance = str(e)
print(json.dumps({"pyarrow": pyarrow.__version__, "huggingface_hub": hub, "fetch_guidance": fetch_guidance,
                  "equal": all(np.array_equal(a.observations["state"], b.observations["state"])
                               for a, b in zip(eps, back.episodes))}))
"""


def boundary(work: Path, fixture: Path) -> dict:
    dist = work / "dist"
    subprocess.run(["uv", "build", "-q", "--wheel", "--out-dir", str(dist), str(REPO)], check=True)
    wheel = next(dist.glob("ccil-*.whl"))
    res = {}
    for name, extra, py, pins in (("core", "", "3.12", []), ("extra", "[lerobot]", "3.12", []),
                                  ("minimum_py310", "[lerobot]", "3.10", [f"pyarrow=={MINIMUM['pyarrow']}"]),
                                  ("minimum_py312", "[lerobot]", "3.12", [f"{k}=={v}" for k, v in MINIMUM.items()])):
        env = work / f"env-{name}"
        subprocess.run(["uv", "venv", "-q", "--python", py, str(env)], check=True)
        pyx = str(env / "bin" / "python")
        r = subprocess.run(["uv", "pip", "install", "-q", "--python", pyx, f"{wheel}{extra}", *pins],
                           capture_output=True, text=True)
        installed = {l.split("==")[0].lower() for l in subprocess.run(["uv", "pip", "freeze", "--python", pyx],
                                                                       capture_output=True, text=True).stdout.split()}
        entry = {"install_exit": r.returncode, "forbidden_installed": sorted(set(FORBIDDEN) & installed)}
        if name == "core":
            g = subprocess.run([pyx, "-c", BOUNDARY, str(fixture)], capture_output=True, text=True, cwd=work)
            guidance = json.loads(g.stdout.strip().splitlines()[-1])["guidance"] if g.returncode == 0 else None
            entry.update(pyarrow_absent="pyarrow" not in installed, guidance=guidance,
                         ok=bool(guidance) and "ccil[lerobot]" in guidance and "does not install the LeRobot runtime"
                         in guidance and "pyarrow" not in installed)
        else:
            t = subprocess.run([pyx, "-c", ROUND_TRIP, str(work / f"rt-{name}")], capture_output=True, text=True,
                               cwd=work)
            rt = json.loads(t.stdout.strip().splitlines()[-1]) if t.returncode == 0 else {"error": t.stderr[-500:]}
            py310 = py == "3.10"  # huggingface_hub is Python 3.11+ only (filelock's fixed releases need 3.11)
            entry.update(round_trip=rt, ok=r.returncode == 0 and rt.get("equal") is True
                         and not entry["forbidden_installed"]
                         and (not pins or all(rt.get(k) == v for k, v in MINIMUM.items() if not (py310 and k != "pyarrow")))
                         and ((rt.get("huggingface_hub") is None and "3.11" in (rt.get("fetch_guidance") or ""))
                              if py310 else rt.get("huggingface_hub") is not None))
        res[name] = entry
    return {"pass": all(v["ok"] for v in res.values()), **res}


def hub(work: Path) -> dict:
    """A real tabular dataset from the Hub at its current revision, recorded: read, or refused with a reason."""
    try:
        path, sha = fetch(HUB[0], revision="main", local_dir=work / "hub")
        info = read_info(path)
        out = {"repo_id": HUB[0], "revision": sha, "features": sorted(info["features"])}
        try:
            read_dataset(path)
            out["read_as_is"] = True
        except LeRobotFormatError as e:
            out["read_as_is"], out["refusal"] = False, str(e)[-400:]
        ignore = sorted(f for f in info["features"] if f.startswith("next."))
        ds = read_dataset(path, ignore=ignore)  # what a caller gets after naming the features it drops
        return {**out, "ignored": ignore, "episodes": len(ds.episodes), "frames": sum(ep.length for ep in ds.episodes),
                "feature_map": ds.meta["dataset"]["feature_map"]}
    except Exception as e:  # noqa: BLE001 - reported: network, or a dataset not in v3.0
        return {"repo_id": HUB[0], "read": False, "error": f"{type(e).__name__}: {e}"[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--releases", default=",".join(RELEASES))
    ap.add_argument("--hub", action="store_true", help="also fetch and read a real dataset from the Hub")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:500]}", flush=True)

    src = source_dataset()
    ccil_root = out / "ccil-written"
    write_dataset(src, ccil_root, fps=FPS, data_files_size_in_mb=1)
    again = out / "ccil-written-again"
    write_dataset(source_dataset(), again, fps=FPS, data_files_size_in_mb=1)
    shards = sorted(p.relative_to(ccil_root).as_posix() for p in (ccil_root / "data").rglob("*.parquet"))
    back = read_dataset(ccil_root)
    record("ccil_writes_deterministically_and_reads_back", tree(ccil_root) == tree(again) and len(shards) > 1
           and all(np.array_equal(a.observations[k], b.observations[k]) for a, b in zip(src.episodes, back.episodes)
                   for k in a.observations),
           shards=shards, files=tree(ccil_root))

    matrix = {}
    fixtures = {}
    for release in a.releases.split(","):
        python, freeze = env_for(out, release)
        row = {"installed": None if python is None else {l.split("==")[0]: l.split("==")[1] for l in freeze.split()
                                                          if l.split("==")[0].lower() in
                                                          ("lerobot", "torch", "pyarrow", "datasets", "numpy")}}
        if python is None:
            row.update(install_error=freeze, write_direction=False, read_direction=False, training=False)
            matrix[release] = row
            continue
        work = out / "work" / release
        work.mkdir(parents=True)
        r = lerobot(python, "load", ccil_root, work / "ccil.npz", cwd=work)
        if r.returncode == 0:
            with np.load(work / "ccil.npz", allow_pickle=False) as z:
                cmp = compare_written(src, ccil_root, dict(z))
            row["write_direction"], row["write_detail"] = cmp["pass"], cmp
        else:
            row["write_direction"], row["write_error"] = False, r.stderr[-800:]
        fx = work / "fixture"
        r = lerobot(python, "write", fx, cwd=work)
        if r.returncode == 0:
            fixtures[release] = fx
            r2 = lerobot(python, "load", fx, work / "fixture.npz", cwd=work)
            try:
                ds = read_dataset(fx)
                with np.load(work / "fixture.npz", allow_pickle=False) as z:
                    cmp = compare_read(ds, fx, dict(z)) if r2.returncode == 0 else {"pass": False, "load": r2.stderr[-400:]}
                row["read_direction"], row["read_detail"] = cmp["pass"], cmp
            except LeRobotFormatError as e:
                row["read_direction"], row["read_refused"] = False, str(e)[-400:]
        else:
            row["read_direction"], row["read_error"] = False, r.stderr[-800:]
        r = lerobot(python, "train", ccil_root, work / "train.json", cwd=work)
        if r.returncode == 0:
            t = json.loads((work / "train.json").read_text())
            row["training"] = t["finite"] and t["loss_last"] < 0.5 * t["loss_first"]
            row["training_detail"] = t
        else:
            row["training"], row["training_error"] = False, r.stderr[-800:]
        row["supported"] = bool(row["write_direction"] and row["read_direction"] and row["training"])
        matrix[release] = row
        print(f"  {release}: write {row['write_direction']} read {row['read_direction']} training {row['training']}",
              flush=True)
    supported = [v for v, r in matrix.items() if r["supported"]]
    record("compatibility_matrix", bool(supported), supported=supported,
           unsupported={v: [k for k in ("write_direction", "read_direction", "training") if not r[k]]
                        for v, r in matrix.items() if not r["supported"]}, matrix=matrix)

    base = fixtures.get(supported[-1]) if supported else None
    if base is not None:
        r = refusals(out / "refusals", ccil_root, base)
        record("refusals", r.pop("pass"), **r)
    r = boundary(out / "boundary", base or ccil_root)
    record("install_boundary", r.pop("pass"), **r)
    if a.hub:
        h = hub(out / "hub-work")
        record("hub_dataset_read_with_named_ignores", h.get("episodes", 0) > 0 and h.get("read_as_is") is False, **h)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "supported_releases": supported,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; supported: {supported}; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
