#!/usr/bin/env python3
"""Validated LeRobot ingestion (#27): endings, feature dispositions, source-local diagnostics, conversion manifests.

    uv run --extra lerobot python tools/m4_ingest/check.py --out DIR [--hub]

Small v3.0 fixtures are written by CCIL's bridge and then given next.* columns with PyArrow, so each case controls
exactly what the source records. Cases:
1. endings: unknown by default (termination_known False); an exhaustive terminal_feature gives terminated and
   truncated; a terminal flag before the final frame, a non-boolean feature and a missing feature are refused;
   the Push-T pattern (next.done true on the last two frames) reads with unknown endings by default and is refused
   when next.done is declared the terminal feature;
2. dispositions: every source feature has one (mapped, checked index, terminal indicator, ignored, preserved);
   an undisposed feature is refused; a preserved feature is summarized and hashed, not copied;
3. identity: episode and frame identity survive exactly; duplicate (episode_index, frame_index) identities and
   non-monotonic indices are refused with the shard, row, episode and frame; identical consecutive frames are not;
4. conversion: convert writes a ccil.trajectories file and a manifest atomically, leaves the source byte-for-byte
   unchanged, writes nothing when it refuses, and records dataset metadata apart from any algorithm parameter;
5. schema: trajectories 1.1 adds the optional termination_known; ccil.trajectories 1.0's reader (main before #27)
   reads a 1.1 file, seeing every unknown ending as non-terminal; a file without the field stays 1.0;
6. with --hub: lerobot/pusht_keypoints at a recorded revision, the same rules on real data.
Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
BASE_COMMIT = "e2e7e67af5ee8571efe79ad52912b44c8e3a264b"  # main before #26 and #27: ccil.trajectories 1.0's reader
LENGTHS = (6, 5, 7)

from ccil.lerobot import LeRobotFormatError, convert, fetch, read_dataset, write_dataset  # noqa: E402
from ccil.schema import Episode, TrajectoryDataset, read_trajectories, write_trajectories  # noqa: E402


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def source() -> TrajectoryDataset:
    eps = []
    for e, T in enumerate(LENGTHS):
        v = (10 * e + np.arange(T)).astype(np.float32)
        state, act = np.stack([v, -v], 1), (v / 2)[:, None]
        if e == 1:  # two identical consecutive frames: legitimate data, never a duplicate identity
            state[3], act[3] = state[2], act[2]
        eps.append(Episode(str(e), {"state": state}, act, task=f"task {e % 2}"))
    return TrajectoryDataset(eps)


def fixture(work: Path, name: str, columns: dict) -> Path:
    """The source dataset, plus extra per-frame columns: {feature: (dtype, function(episode, length) -> values)}."""
    root = work / name
    write_dataset(source(), root, fps=10)
    info = json.loads((root / "meta" / "info.json").read_text())
    for path in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(path)
        ep = t.column("episode_index").to_numpy()
        for feat, (dtype, fn) in columns.items():
            vals = np.concatenate([fn(e, int((ep == e).sum())) for e in np.unique(ep)])
            t = t.append_column(feat, pa.array(vals, pa.bool_() if dtype == "bool" else pa.from_numpy_dtype(np.dtype(dtype))))
        pq.write_table(t, path)
    for feat, (dtype, _) in columns.items():
        info["features"][feat] = {"dtype": dtype, "shape": [1], "names": None}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return root


def last_two(e, T):
    return np.arange(T) >= T - 2


def refused(call, needle):
    try:
        call()
        return {"refused": False}
    except LeRobotFormatError as e:
        return {"refused": needle in str(e), "message": str(e).split(": ", 1)[-1][-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hub", action="store_true")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    work = out / "fixtures"
    work.mkdir()
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)

    nexts = {"next.done": ("bool", last_two), "next.reward": ("float32", lambda e, T: np.linspace(0, 1, T)),
             "next.success": ("bool", lambda e, T: (np.arange(T) == T - 1) & (e != 1))}
    rich = fixture(work, "rich", nexts)
    all_next = ["next.done", "next.reward", "next.success"]

    # 1. Endings.
    ds = read_dataset(rich, preserve=all_next)
    unknown = all(ep.termination_known is False and ep.terminated is False for ep in ds.episodes)
    record("endings_unknown_by_default", unknown and ds.meta["dataset"]["endings"]["unknown"] == len(LENGTHS),
           endings=ds.meta["dataset"]["endings"])
    ds = read_dataset(rich, preserve=["next.done", "next.reward"], terminal_feature="next.success")
    got = [(ep.termination_known, ep.terminated) for ep in ds.episodes]
    record("explicit_terminal_and_truncation", got == [(True, True), (True, False), (True, True)],
           episodes=got, endings=ds.meta["dataset"]["endings"])
    early = fixture(work, "early", {"next.success": ("bool", lambda e, T: np.arange(T) == 2)})
    r = refused(lambda: read_dataset(early, terminal_feature="next.success"), "before the final frame")
    record("early_terminal_flag_refused", r["refused"] and "row 2 (episode 0, frame 2" in r.get("message", ""), **r)
    r = refused(lambda: read_dataset(rich, preserve=["next.done", "next.success"], terminal_feature="next.reward"),
                "must be bool")
    record("non_boolean_terminal_feature_refused", r["refused"], **r)
    r = refused(lambda: read_dataset(rich, preserve=all_next, terminal_feature="next.terminal"), "is not in meta/info.json")
    record("missing_terminal_feature_refused", r["refused"], **r)
    pusht = fixture(work, "pusht_pattern", {"next.done": ("bool", last_two)})
    ds = read_dataset(pusht, preserve=["next.done"])
    r = refused(lambda: read_dataset(pusht, terminal_feature="next.done"), "before the final frame")
    record("pusht_pattern_done_true_twice", ds.meta["dataset"]["endings"]["unknown"] == len(LENGTHS) and r["refused"]
           and ds.meta["dataset"]["features"]["next.done"]["summary"]["true_count"] == 2 * len(LENGTHS),
           default_endings=ds.meta["dataset"]["endings"], selected=r)

    # 2. Dispositions.
    ds = read_dataset(rich, ignore=["next.reward"], preserve=["next.done"], terminal_feature="next.success")
    info = json.loads((rich / "meta" / "info.json").read_text())
    disp = ds.meta["dataset"]["features"]
    kinds = {k: v["disposition"] for k, v in disp.items()}
    record("every_feature_has_a_disposition", set(disp) == set(info["features"]) and kinds["next.reward"] == "ignored"
           and kinds["next.done"] == "preserved_as_source_metadata" and kinds["next.success"] == "terminal_indicator"
           and kinds["observation.state"] == "mapped" and kinds["index"] == "checked_index"
           and all(isinstance(v, (int, float, str)) for v in disp["next.done"]["summary"].values()),
           dispositions=disp)
    r = refused(lambda: read_dataset(rich, preserve=["next.done", "next.success"]), "'next.reward'")
    record("undisposed_feature_refused", r["refused"], **r)
    r = refused(lambda: read_dataset(rich, ignore=["next.done"], preserve=["next.done", "next.reward", "next.success"]),
                "both ignored and preserved")
    record("conflicting_dispositions_refused", r["refused"], **r)

    # 3. Identity.
    src = source()
    ds = read_dataset(rich, preserve=all_next)
    record("episode_and_frame_identity_exact",
           [ep.episode_id for ep in ds.episodes] == [ep.episode_id for ep in src.episodes]
           and all(np.array_equal(a_.observations["state"], b.observations["state"]) and np.array_equal(a_.actions, b.actions)
                   for a_, b in zip(ds.episodes, src.episodes))
           and np.array_equal(ds.episodes[1].observations["state"][2], ds.episodes[1].observations["state"][3]),
           lengths=[ep.length for ep in ds.episodes], identical_consecutive_frames_accepted=True)

    def edit_column(name, fn):
        d = work / f"edit-{name}"
        shutil.copytree(pusht, d)
        p = next((d / "data").rglob("*.parquet"))
        t = pq.read_table(p)
        i = t.schema.get_field_index(name)
        t = t.set_column(i, name, pa.array(fn(t.column(i).to_numpy().copy()), pa.int64()))
        pq.write_table(t, p)
        return d

    def dup_frame(x):
        x[3] = x[2]  # episode 0 frames 2 and 3 both claim frame 2
        return x

    def swap_index(x):
        x[[4, 5]] = x[[5, 4]]
        return x
    r = refused(lambda: read_dataset(edit_column("frame_index", dup_frame), preserve=["next.done"]), "each once")
    record("duplicate_frame_identity_refused", r["refused"] and "row 3 (episode 0, frame 3)" in r.get("message", ""), **r)
    r = refused(lambda: read_dataset(edit_column("index", swap_index), preserve=["next.done"]), "row 4: index 5 after 3")
    record("non_monotonic_index_refused", r["refused"], **r)

    # 4. Conversion.
    before = tree(rich)
    conv = out / "conversion"
    conv.mkdir()
    doc = convert(rich, conv / "trajectories.npz", conv / "manifest.json", ignore=["next.reward"],
                  preserve=["next.done"], terminal_feature="next.success")
    back = read_trajectories(conv / "trajectories.npz")
    meta = json.loads(np.load(conv / "trajectories.npz", allow_pickle=False)["meta"].item())
    record("conversion_manifest_and_source_unchanged",
           tree(rich) == before and doc["source_unchanged"] and set(doc["features"]) == set(info["features"])
           and doc["output"]["sha256"] == hashlib.sha256((conv / "trajectories.npz").read_bytes()).hexdigest()
           and [ep.termination_known for ep in back.episodes] == [True] * len(LENGTHS) and meta["version"] == "1.1"
           and "termination_known" in meta["optional_fields"] and "config" not in doc,
           manifest=doc, file_meta_version=meta["version"])
    failed = out / "conversion-refused"
    failed.mkdir()
    r = refused(lambda: convert(early, failed / "t.npz", failed / "m.json", terminal_feature="next.success"),
                "before the final frame")
    record("refused_conversion_writes_nothing", r["refused"] and not any(failed.iterdir()), **r)

    # 5. Schema compatibility with the 1.0 reader.
    old = out / "reader-1.0"
    old.mkdir()
    subprocess.run(f"git -C {REPO} archive {BASE_COMMIT} src/ccil | tar -x -C {old}", shell=True, check=True)
    probe = (f"import sys; sys.path.insert(0, {str(old / 'src')!r}); import ccil, json; "
             f"assert ccil.__file__.startswith({str(old)!r}); from ccil.schema import read_trajectories; "
             "d = read_trajectories(sys.argv[1]); "
             "print(json.dumps([ep.terminated for ep in d.episodes]))")
    plain = out / "plain.npz"
    write_trajectories(TrajectoryDataset([Episode(ep.episode_id, ep.observations, ep.actions) for ep in source().episodes]),
                       plain)
    unk = out / "unknown-endings.npz"
    write_trajectories(read_dataset(rich, preserve=all_next), unk)
    res = {}
    for name, f in (("1.1 with termination_known", unk), ("1.1 converted", conv / "trajectories.npz"), ("1.0", plain)):
        p = subprocess.run([sys.executable, "-c", probe, str(f)], capture_output=True, text=True)
        res[name] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else p.stderr[-300:]
    plain_version = json.loads(np.load(plain, allow_pickle=False)["meta"].item())["version"]
    record("schema_1_1_read_by_the_1_0_reader", res["1.1 with termination_known"] == [False] * len(LENGTHS)
           and res["1.1 converted"] == [True, False, True] and plain_version == "1.0" and isinstance(res["1.0"], list),
           old_reader=res, file_without_the_field_version=plain_version, base_commit=BASE_COMMIT)
    try:
        write_trajectories(TrajectoryDataset([Episode("0", {"state": np.zeros((3, 1))}, np.zeros((3, 1)),
                                                      terminated=True, termination_known=False)]), out / "bad.npz")
        bad = False
    except Exception as e:  # noqa: BLE001 - the schema error names the rule
        bad = "unknown ending cannot be terminal" in str(e)
    record("unknown_and_terminated_is_invalid", bad and not (out / "bad.npz").exists())

    # 6. The Hub.
    if a.hub:
        try:
            path, sha = fetch("lerobot/pusht_keypoints", revision="main", local_dir=out / "hub")
            ds = read_dataset(path, preserve=["next.done", "next.reward", "next.success"])
            r = refused(lambda: read_dataset(path, preserve=["next.reward", "next.success"], terminal_feature="next.done"),
                        "before the final frame")
            record("pusht_keypoints_from_the_hub", ds.meta["dataset"]["endings"]["unknown"] == len(ds.episodes) and r["refused"],
                   revision=sha, episodes=len(ds.episodes), endings=ds.meta["dataset"]["endings"],
                   next_done=ds.meta["dataset"]["features"]["next.done"], next_done_as_terminal=r)
        except OSError as e:
            record("pusht_keypoints_from_the_hub", False, error=str(e)[-300:])

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "pyarrow": pa.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
