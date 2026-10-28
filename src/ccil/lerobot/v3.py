"""The tabular subset of LeRobotDataset v3.0, read and written with PyArrow (#26).

Layout: meta/info.json declares the format version, the features and the data path template; data shards are
Parquet files at data_path.format(chunk_index=..., file_index=...); meta/episodes/chunk-XXX/file-YYY.parquet holds
one row per episode (its data shard, its global index range, its tasks and per-episode statistics);
meta/tasks.parquet maps task_index to the task string; meta/stats.json holds the dataset statistics.

Mapping to ccil.trajectories: observation.state -> observations["state"], any other observation.<name> ->
observations["state.<name>"]; action -> actions; timestamp -> timestamps; the episode's single task -> task.
Index columns (index, frame_index, episode_index, task_index) are checked, not copied. Every other feature needs
an explicit disposition from the caller: ignore, or preserve (recorded as source metadata: its schema, a summary
and a content hash, never its values).

Endings (#27): an episode boundary says that recording stopped, not why. By default no next.* feature is
interpreted and every ending is unknown (termination_known False, terminated False). A caller may declare
terminal_feature, an exhaustive boolean terminal-state indicator: true only on the final frame (terminated), false
there (truncated), and true earlier refuses the episode. next.done gets no special treatment. Writing: LeRobot v3.0
has no terminal flag, so write_dataset refuses an episode with a known ending unless drop_endings=True, the
documented lossy mode; unknown non-terminal endings are written.

Fails closed: an unknown codebase_version, a feature of an unsupported dtype or shape (images, video, depth, audio,
strings, multi-dimensional arrays), a feature with no CCIL slot unless the caller names it in ignore, a missing
shard, or episode offsets that disagree with the data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import string
from pathlib import Path, PurePosixPath

import numpy as np

from ..schema import Episode, SchemaError, TrajectoryDataset

CODEBASE_VERSION = "v3.0"
INSTALL = ("the LeRobot format bridge needs the ccil[lerobot] extra: pip install 'ccil[lerobot]' "
           "(it installs CCIL's LeRobot Dataset v3.0 format bridge: pyarrow, and huggingface_hub on Python 3.11+; "
           "it does not install the LeRobot runtime)")
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
INDEX_COLUMNS = ("index", "frame_index", "episode_index", "task_index")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
FLOATS = ("float32", "float64")
DEFERRED = ("image", "video", "audio")  # dtypes of modalities that arrive with image CCIL (M5)
TIMESTAMP_TOLERANCE_S = 1e-4  # LeRobot's own tolerance when it checks timestamps against 1 / fps


class LeRobotFormatError(ValueError):
    """The directory is not a LeRobotDataset v3.0 dataset this bridge can read, or a dataset cannot be written."""

    def __init__(self, where, problems):
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(INSTALL) from e
    return pa, pq


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(root: Path, rel: str, where: str, problems: list) -> Path | None:
    """root / rel, refusing a path that is absolute, has empty, '.' or '..' parts, or resolves (through any
    symlink) outside the resolved dataset root."""
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        problems.append(f"{where} {rel!r} is not a relative path inside the dataset")
        return None
    path = root.joinpath(*p.parts)
    if not _contained(root, path):
        problems.append(f"{where} {rel!r} resolves outside the dataset (through a symlink)")
        return None
    return path


def _contained(root: Path, path: Path) -> bool:
    """Whether path, resolved through every symlink, is inside root, resolved."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (ValueError, OSError):
        return False


def _meta_file(root: Path, rel: str) -> Path:
    """A metadata file of the dataset, refused if it resolves outside it."""
    path = root.joinpath(*PurePosixPath(rel).parts)
    if not _contained(root, path):
        raise LeRobotFormatError(root, [f"{rel} resolves outside the dataset (through a symlink)"])
    return path


# --- reading -------------------------------------------------------------------------------------------------------

def read_info(root) -> dict:
    """meta/info.json, validated: the format version, the fields the bridge relies on and the path template."""
    root = Path(root)
    if not root.is_dir():
        raise LeRobotFormatError(root, ["not a directory"])
    path = _meta_file(root, "meta/info.json")
    if not path.is_file():
        raise LeRobotFormatError(root, ["meta/info.json is missing"])
    try:
        info = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LeRobotFormatError(path, [f"unreadable JSON: {e}"]) from e
    problems = []
    if not isinstance(info, dict):
        raise LeRobotFormatError(path, ["not a JSON object"])
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise LeRobotFormatError(path, [f"codebase_version {info.get('codebase_version')!r} is not supported; "
                                        f"this bridge reads {CODEBASE_VERSION} only"])
    for key, kind in (("fps", (int, float)), ("features", dict), ("data_path", str), ("total_episodes", int),
                      ("total_frames", int)):
        if not isinstance(info.get(key), kind) or isinstance(info.get(key), bool):
            problems.append(f"{key} is missing or not a {kind if isinstance(kind, type) else 'number'}")
    if problems:
        raise LeRobotFormatError(path, problems)
    if not info["fps"] > 0:
        problems.append(f"fps {info['fps']} is not positive")
    fields = {f for _, f, _, _ in string.Formatter().parse(info["data_path"]) if f is not None}
    if fields != {"chunk_index", "file_index"}:
        problems.append(f"data_path {info['data_path']!r} must use exactly the fields chunk_index and file_index")
    else:
        try:  # a representative formatting, so a malformed format specification is refused here, not later
            info["data_path"].format(chunk_index=0, file_index=0)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            problems.append(f"data_path {info['data_path']!r} cannot be formatted: {e}")
    for name, spec in info["features"].items():
        if not (isinstance(spec, dict) and isinstance(spec.get("dtype"), str) and isinstance(spec.get("shape"), list)
                and all(isinstance(d, int) and d > 0 for d in spec["shape"])):
            problems.append(f"feature {name!r} needs a dtype string and a shape list of positive integers")
    if problems:
        raise LeRobotFormatError(path, problems)
    return info


def _plan(info: dict, ignore=(), preserve=(), terminal_feature=None) -> tuple[dict, dict, list]:
    """Each feature's disposition, or every problem: (CCIL slots {feature: (slot, dim)}, dispositions, problems).

    A feature is mapped to a CCIL slot, checked (the index columns), the declared terminal indicator, preserved as
    source metadata (its schema, a summary and a hash, never its values), or ignored by the caller's rule. Any other
    feature fails closed."""
    ignore, preserve = set(ignore), set(preserve)
    plan, disp, problems = {}, {}, []
    for both in sorted(ignore & preserve):
        problems.append(f"feature {both!r} is both ignored and preserved")
    for name in sorted((ignore | preserve | ({terminal_feature} if terminal_feature else set())) - set(info["features"])):
        problems.append(f"feature {name!r} is named by the caller but is not in meta/info.json")
    for name, spec in info["features"].items():
        dtype, shape = spec["dtype"], tuple(spec["shape"])
        schema = {"dtype": dtype, "shape": list(shape)}
        if name == terminal_feature:
            if dtype != "bool" or shape != (1,):
                problems.append(f"terminal_feature {name!r} must be bool of shape [1], got {dtype} {list(shape)}")
            disp[name] = {"disposition": "terminal_indicator", **schema,
                          "rule": "exhaustive terminal-state indicator: true only on an episode's final frame; final "
                                  "true = terminated, final false = truncated; true earlier refuses the episode"}
        elif name in ignore:
            disp[name] = {"disposition": "ignored", **schema, "rule": "named in ignore by the caller"}
        elif dtype in DEFERRED or "depth" in name:
            problems.append(f"feature {name!r} ({dtype}) is an image, video, depth or audio modality: not supported "
                            f"until image CCIL (M5)")
        elif name in preserve:
            disp[name] = {"disposition": "preserved_as_source_metadata", **schema}
        elif name in INDEX_COLUMNS:
            if dtype != "int64" or shape != (1,):
                problems.append(f"index feature {name!r} must be int64 of shape [1], got {dtype} {list(shape)}")
            disp[name] = {"disposition": "checked_index", **schema}
        elif name == "timestamp":
            if dtype not in FLOATS or shape != (1,):
                problems.append(f"timestamp must be float32 or float64 of shape [1], got {dtype} {list(shape)}")
            disp[name] = {"disposition": "mapped", "to": "timestamps", **schema}
        elif name == "action" or name.startswith("observation."):
            if dtype not in FLOATS or len(shape) != 1:
                problems.append(f"feature {name!r} must be a float32 or float64 vector, got {dtype} {list(shape)}")
                continue
            if name == "action":
                slot = "action"
            else:
                suffix = name[len("observation."):]
                slot = "state" if suffix == "state" else "state." + re.sub(r"[^a-z0-9_]", "_", suffix.lower())
                if slot in {m for m, _ in plan.values()}:
                    problems.append(f"feature {name!r} maps to modality {slot!r}, which another feature has")
            plan[name] = (slot, shape[0])
            disp[name] = {"disposition": "mapped", "to": "actions" if slot == "action" else f"observations[{slot}]",
                          **schema}
        else:
            problems.append(f"feature {name!r} ({dtype}) has no CCIL slot; name it in ignore or preserve")
    for required in ("action", "timestamp", *INDEX_COLUMNS):
        if required not in info["features"]:
            problems.append(f"required feature {required!r} is missing")
    if not any(slot != "action" for slot, _ in plan.values()):
        problems.append("no observation.* feature: a CCIL dataset needs at least one state modality")
    return plan, disp, problems


def _episodes_table(root: Path, info: dict):
    """Every episode's metadata row, in episode order, from the episode files (checked for consistency)."""
    _, pq = _arrow()
    files = [_meta_file(root, f.relative_to(root).as_posix())
             for f in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))]
    if not files:
        raise LeRobotFormatError(root, ["no episode metadata under meta/episodes/"])
    cols = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index", "dataset_from_index",
            "dataset_to_index"]
    rows = []
    for f in files:
        schema = pq.read_schema(f)
        missing = [c for c in cols if c not in schema.names]
        if missing:
            raise LeRobotFormatError(f, [f"episode metadata is missing columns {missing}"])
        rel = f.relative_to(root).as_posix()
        rows += [{**r, "_source": f"{rel} row {i}"} for i, r in enumerate(pq.read_table(f, columns=cols).to_pylist())]
    seen, problems = {}, []
    for r in rows:
        if r["episode_index"] in seen:
            problems.append(f"episode {r['episode_index']} appears twice: {seen[r['episode_index']]} and {r['_source']}")
        seen[r["episode_index"]] = r["_source"]
    rows.sort(key=lambda r: r["episode_index"])
    if not problems and [r["episode_index"] for r in rows] != list(range(len(rows))):
        problems.append("episode indices are not 0..N-1")
    if len(rows) != info["total_episodes"]:
        problems.append(f"{len(rows)} episodes in meta/episodes, info.json says {info['total_episodes']}")
    start = 0
    for r in rows:
        e = r["episode_index"]
        if r["dataset_from_index"] != start or r["dataset_to_index"] - r["dataset_from_index"] != r["length"]:
            problems.append(f"{r['_source']} (episode {e}): offsets [{r['dataset_from_index']}, "
                            f"{r['dataset_to_index']}) are not contiguous from {start} with length {r['length']}")
        if r["length"] < 1:
            problems.append(f"{r['_source']} (episode {e}): length {r['length']}")
        start = r["dataset_to_index"]
    if start != info["total_frames"]:
        problems.append(f"episodes cover {start} frames, info.json says {info['total_frames']}")
    if problems:
        raise LeRobotFormatError(root / "meta" / "episodes", problems[:20])
    return rows


def _tasks(root: Path) -> dict:
    """task_index -> task string; the task string is the pandas index column the file's metadata names."""
    _, pq = _arrow()
    path = _meta_file(root, "meta/tasks.parquet")
    if not path.is_file():
        raise LeRobotFormatError(root, ["meta/tasks.parquet is missing"])
    table = pq.read_table(path)
    meta = (table.schema.metadata or {}).get(b"pandas")
    index_cols = json.loads(meta).get("index_columns", []) if meta else []
    names = [c for c in table.schema.names if c != "task_index"]
    text = next((c for c in index_cols if isinstance(c, str) and c in names), "task" if "task" in names else None)
    if "task_index" not in table.schema.names or text is None:
        raise LeRobotFormatError(path, [f"expected task_index and a task string column, got {table.schema.names}"])
    out = {}
    for i, r in enumerate(table.to_pylist()):
        if r["task_index"] in out:
            raise LeRobotFormatError(path, [f"row {i}: task_index {r['task_index']} appears twice"])
        out[r["task_index"]] = r[text]
    return out


def _column(table, name, dim):
    import pyarrow as pa
    col = table.column(name).combine_chunks()
    if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
        values = col.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(col), dim)
    return col.to_numpy(zero_copy_only=False).reshape(len(col), 1 if dim is None else dim)


class _Preserved:
    """A preserved feature's running summary and content hash, accumulated shard by shard."""

    def __init__(self):
        self.h, self.count, self.true, self.lo, self.hi, self.total = hashlib.sha256(), 0, 0, None, None, 0.0

    def add(self, col):
        import pyarrow as pa
        col = col.combine_chunks()
        self.count += len(col)
        flat = col.flatten() if (pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type)) else col
        # A canonical encoding, independent of Arrow's buffer layout: the values, then the null mask.
        values = flat.to_numpy(zero_copy_only=False)
        self.h.update(np.ascontiguousarray(values.astype(str) if values.dtype == object else values).tobytes())
        self.h.update(np.asarray(flat.is_null().to_numpy(zero_copy_only=False), dtype=bool).tobytes())
        if pa.types.is_boolean(flat.type):
            self.true += int(np.count_nonzero(flat.to_numpy(zero_copy_only=False)))
        elif pa.types.is_integer(flat.type) or pa.types.is_floating(flat.type):
            x = flat.to_numpy(zero_copy_only=False).astype(np.float64)
            if x.size:
                self.lo = float(x.min()) if self.lo is None else min(self.lo, float(x.min()))
                self.hi = float(x.max()) if self.hi is None else max(self.hi, float(x.max()))
                self.total += float(x.sum())

    def summary(self) -> dict:
        out = {"rows": self.count, "sha256_of_values": self.h.hexdigest()}
        if self.true:
            out["true_count"] = self.true
        if self.lo is not None:
            out.update(min=self.lo, max=self.hi)
        return out


def iter_episodes(root, ignore=(), files: dict | None = None, *, preserve=(), terminal_feature=None,
                  report: dict | None = None):
    """Yield the dataset's episodes in order, reading one data shard at a time.

    Endings: by default no next.* feature is interpreted, so every episode's ending is unknown
    (termination_known False, terminated False). terminal_feature names an exhaustive boolean terminal-state
    indicator instead (see _plan). files receives {relative path: SHA-256} of every file read; report receives the
    feature dispositions, the preserved features' summaries and the episodes' ending classification."""
    root = Path(root)
    info = read_info(root)
    plan, disp, problems = _plan(info, ignore, preserve, terminal_feature)
    if problems:
        raise LeRobotFormatError(root, problems)
    tasks = _tasks(root)
    rows = _episodes_table(root, info)
    if files is not None:
        for rel in ("meta/info.json", "meta/tasks.parquet"):
            files[rel] = _sha256(root / rel)
        for f in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
            files[f.relative_to(root).as_posix()] = _sha256(f)
    _, pq = _arrow()
    by_shard = {}
    for r in rows:
        by_shard.setdefault((r["data/chunk_index"], r["data/file_index"]), []).append(r)
    preserved = {name: _Preserved() for name, d in disp.items() if d["disposition"] == "preserved_as_source_metadata"}
    columns = [*plan, "timestamp", *INDEX_COLUMNS, *preserved, *([terminal_feature] if terminal_feature else [])]
    endings = {"terminated": 0, "truncated": 0, "unknown": 0}
    for (chunk, file), eps in sorted(by_shard.items(), key=lambda kv: kv[1][0]["episode_index"]):
        rel = info["data_path"].format(chunk_index=chunk, file_index=file)
        where = []
        path = _inside(root, rel, "data shard", where)
        if path is None or not path.is_file():
            raise LeRobotFormatError(root, where or [f"data shard {rel} (episodes {eps[0]['episode_index']}-"
                                                     f"{eps[-1]['episode_index']}) is missing"])
        missing = [c for c in columns if c not in pq.read_schema(path).names]
        if missing:
            raise LeRobotFormatError(path, [f"missing columns {missing}"])
        table = pq.read_table(path, columns=columns)
        if files is not None:
            files[rel] = _sha256(path)
        if terminal_feature:
            import pyarrow as pa
            if not pa.types.is_boolean(table.schema.field(terminal_feature).type):
                raise LeRobotFormatError(path, [f"terminal_feature {terminal_feature!r} is stored as "
                                                f"{table.schema.field(terminal_feature).type}, not as a boolean"])
        for name, p in preserved.items():
            p.add(table.column(name))
        index = table.column("index").to_numpy()
        step = np.flatnonzero(np.diff(index) != 1)
        if len(index) == 0:
            raise LeRobotFormatError(path, ["the shard has no rows"])
        if step.size:
            r = int(step[0]) + 1
            kind = "repeats an earlier index" if index[r] <= index[r - 1] else "skips indices"
            raise LeRobotFormatError(path, [f"{rel} row {r}: index {int(index[r])} after {int(index[r - 1])} {kind}; "
                                            f"indices must increase by one"])
        for r in eps:
            e, lo, hi = r["episode_index"], r["dataset_from_index"], r["dataset_to_index"]
            a, b = lo - int(index[0]), hi - int(index[0])
            if a < 0 or b > len(index):
                raise LeRobotFormatError(path, [f"{r['_source']} (episode {e}): rows [{lo}, {hi}) are not in {rel}"])
            part = table.slice(a, b - a)
            bad = []
            ep_col, fr_col = part.column("episode_index").to_numpy(), part.column("frame_index").to_numpy()
            off = np.flatnonzero(ep_col != e)
            if off.size:
                bad.append(f"{rel} row {a + int(off[0])}: episode_index {int(ep_col[off[0]])}, but the episode's "
                           f"metadata ({r['_source']}) places episode {e} there")
            off = np.flatnonzero(fr_col != np.arange(b - a))
            if off.size:
                f0 = int(off[0])
                bad.append(f"{rel} row {a + f0} (episode {e}, frame {f0}): frame_index {int(fr_col[f0])}; frames must "
                           f"be 0..{b - a - 1} in order, each once")
            task_ids = np.unique(part.column("task_index").to_numpy())
            if len(task_ids) != 1:
                bad.append(f"{rel} rows {a}-{b - 1} (episode {e}): {len(task_ids)} tasks within one episode; "
                           f"ccil.trajectories has one task per episode")
            elif int(task_ids[0]) not in tasks:
                bad.append(f"{rel} rows {a}-{b - 1} (episode {e}): task_index {int(task_ids[0])} is not in "
                           f"meta/tasks.parquet")
            elif r["tasks"] and tasks[int(task_ids[0])] not in r["tasks"]:
                bad.append(f"episode {e}: task {tasks[int(task_ids[0])]!r} is not among its tasks {r['tasks']} "
                           f"({r['_source']})")
            known, terminated = False, False
            if terminal_feature:
                flag = part.column(terminal_feature).to_numpy(zero_copy_only=False).astype(bool)
                early = np.flatnonzero(flag[:-1])
                if early.size:
                    f0 = int(early[0])
                    bad.append(f"{rel} row {a + f0} (episode {e}, frame {f0} of {b - a}): {terminal_feature} is true "
                               f"before the final frame; as an exhaustive terminal indicator it may be true only on "
                               f"the final frame")
                known, terminated = True, bool(flag[-1])
            if bad:
                raise LeRobotFormatError(path, bad)
            endings["terminated" if terminated else "truncated" if known else "unknown"] += 1
            obs = {slot: _column(part, name, dim) for name, (slot, dim) in plan.items() if slot != "action"}
            yield Episode(episode_id=str(e), observations=obs, actions=_column(part, "action", plan["action"][1]),
                          terminated=terminated, termination_known=known,
                          timestamps=part.column("timestamp").to_numpy(), task=tasks[int(task_ids[0])])
    if report is not None:
        for name, p in preserved.items():
            disp[name]["summary"] = p.summary()
        report.update(dispositions=disp, endings={
            "rule": (f"terminal_feature {terminal_feature!r}: an exhaustive terminal-state indicator"
                     if terminal_feature else "no next.* feature is interpreted: every ending is unknown"),
            **endings})


def read_dataset(root, ignore=(), *, preserve=(), terminal_feature=None) -> TrajectoryDataset:
    """The whole dataset as a validated ccil.trajectories dataset (iter_episodes streams it instead).

    Its meta records the source files' hashes, every feature's disposition and the episodes' ending
    classification, separately from any algorithm parameter."""
    root = Path(root)
    files, report = {}, {}
    episodes = list(iter_episodes(root, ignore, files, preserve=preserve, terminal_feature=terminal_feature,
                                  report=report))
    info = read_info(root)
    if (root / "meta" / "stats.json").is_file():
        files["meta/stats.json"] = _sha256(_meta_file(root, "meta/stats.json"))
    ds = TrajectoryDataset(episodes, {"dataset": {
        "format": f"lerobot.dataset/{CODEBASE_VERSION}", "fps": info["fps"], "files": dict(sorted(files.items())),
        "features": report["dispositions"], "endings": report["endings"]}})
    return ds.validate()


def convert(source, out, manifest, *, ignore=(), preserve=(), terminal_feature=None) -> dict:
    """A LeRobot v3.0 dataset to a ccil.trajectories file plus a conversion manifest: both, or neither.

    Both are staged under temporary names and then moved into place; if the second move fails, the first is
    removed again. The source is only read: its files' hashes are taken before and after, and must agree. Returns
    the manifest."""
    from ..schema import write_trajectories
    source, out, manifest = Path(source), Path(out), Path(manifest)
    for dest in (out, manifest):
        if dest.exists():
            raise FileExistsError(f"{dest} exists; conversions are written once")
    before = {p.relative_to(source).as_posix(): _sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    ds = read_dataset(source, ignore, preserve=preserve, terminal_feature=terminal_feature)
    after = {p.relative_to(source).as_posix(): _sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    if before != after:
        raise RuntimeError(f"{source} changed while it was read")
    pid = os.getpid()
    out_tmp = out.with_name(f".{out.name}.tmp-{pid}.npz")
    man_tmp = manifest.with_name(f".{manifest.name}.tmp-{pid}")
    try:
        write_trajectories(ds, out_tmp)
        doc = {"schema": "ccil.lerobot.conversion", "version": "1.0", "source": {"format": ds.meta["dataset"]["format"],
               "fps": ds.meta["dataset"]["fps"], "files": ds.meta["dataset"]["files"]},
               "features": ds.meta["dataset"]["features"], "endings": ds.meta["dataset"]["endings"],
               "episodes": len(ds.episodes), "frames": int(sum(ep.length for ep in ds.episodes)),
               "output": {"path": out.name, "sha256": _sha256(out_tmp)}, "source_unchanged": True}
        man_tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.replace(out_tmp, out)
        try:
            os.replace(man_tmp, manifest)
        except BaseException:
            out.unlink(missing_ok=True)  # the first output was moved into place: take it back
            raise
    finally:
        out_tmp.unlink(missing_ok=True)
        man_tmp.unlink(missing_ok=True)
    return doc


def read_stats(root) -> dict:
    """meta/stats.json as stored: {feature: {min, max, mean, std, count, q01, ...: lists}}."""
    path = _meta_file(Path(root), "meta/stats.json")
    try:
        stats = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LeRobotFormatError(path, [f"unreadable: {e}"]) from e
    if not isinstance(stats, dict) or not all(isinstance(v, dict) for v in stats.values()):
        raise LeRobotFormatError(path, ["expected {feature: {statistic: values}}"])
    return stats


def fetch(repo_id: str, revision: str, local_dir=None) -> tuple[Path, str]:
    """Download a dataset's tabular files from the Hugging Face Hub at a pinned revision.

    Returns (directory, resolved commit). Only meta/ and data/ are fetched: a dataset with video features is
    refused by read_info anyway. The bridge itself needs only the local directory."""
    if not revision:
        raise ValueError("fetch needs an explicit revision (a commit, tag or branch), recorded with the data")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise ImportError(INSTALL + "; Hub access needs Python 3.11+ (the bridge itself works on local directories "
                          "on 3.10)") from e
    sha = HfApi().dataset_info(repo_id, revision=revision).sha
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=sha, local_dir=local_dir,
                             allow_patterns=["meta/**", "data/**"])
    return Path(path), sha


# --- writing -------------------------------------------------------------------------------------------------------

def _feature_names(ep: Episode) -> list[tuple[str, str]]:
    """(CCIL modality, LeRobot feature) for each observation modality; images are refused."""
    out = []
    for modality in sorted(ep.observations, key=lambda m: (m != "state", m)):
        if not modality.startswith("state"):
            raise LeRobotFormatError("write", [f"modality {modality!r}: images are not supported until image CCIL (M5)"])
        out.append((modality, "observation.state" if modality == "state" else "observation." + modality[6:]))
    return out


def _value_feature(dtype: str, dim: int) -> dict:
    value = {"dtype": dtype, "_type": "Value"}
    return value if dim == 1 else {"feature": value, "length": dim, "_type": "List"}


def _episode_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    out = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    for k, q in QUANTILES.items():
        out[k] = np.quantile(x, q, axis=0)
    return out


def _aggregate(per_episode: list[dict]) -> dict:
    """Dataset statistics from per-episode ones: exact count, min, max, mean and std (pooled); quantiles as the
    count-weighted mean of the episodes' quantiles, the aggregation LeRobot itself uses."""
    n = np.array([s["count"][0] for s in per_episode], dtype=np.float64)
    means = np.stack([s["mean"] for s in per_episode])
    mean = (n[:, None] * means).sum(0) / n.sum()
    var = (n[:, None] * (np.stack([s["std"] for s in per_episode]) ** 2 + (means - mean) ** 2)).sum(0) / n.sum()
    out = {"min": np.min([s["min"] for s in per_episode], 0), "max": np.max([s["max"] for s in per_episode], 0),
           "mean": mean, "std": np.sqrt(var), "count": np.array([int(n.sum())])}
    for k in QUANTILES:
        out[k] = (n[:, None] * np.stack([s[k] for s in per_episode])).sum(0) / n.sum()
    return out


def write_dataset(episodes, root, fps: float, *, default_task: str = "", data_files_size_in_mb: int = 100,
                  chunks_size: int = 1000, robot_type: str | None = None, drop_endings: bool = False) -> Path:
    """Write episodes (a TrajectoryDataset or an iterable of Episode) as a LeRobotDataset v3.0 directory.

    Streams: each data shard is written when it reaches data_files_size_in_mb. Each streamed episode is validated
    as ccil.trajectories validates a dataset (shapes, finiteness, timestamps), and against the first episode's
    modalities, dimensions and dtypes, with unique IDs, before its rows are written. Endings: LeRobot v3.0 has no
    terminal flag, so an episode with a known ending (termination_known True, or a 1.0 terminated True) is refused
    unless drop_endings=True, the documented lossy mode; an unknown or unrecorded non-terminal ending is written. Timestamps must advance by 1 / fps
    (LeRobot checks this when it loads); episodes without timestamps get frame / fps. The directory is built
    under a temporary name and moved into place, so root exists only once complete; an existing root is refused.
    """
    pa, pq = _arrow()
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists; LeRobot datasets are written once")
    if isinstance(episodes, TrajectoryDataset):
        episodes = episodes.validate().episodes
    staging = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    cap = data_files_size_in_mb * 2**20
    try:
        features = schema = names = None
        tasks: dict = {}
        ep_rows, ep_stats, shard, shard_bytes = [], [], [], 0
        chunk = file = 0
        start = 0

        def flush():
            nonlocal shard, shard_bytes, chunk, file
            if not shard:
                return
            path = staging / DATA_PATH.format(chunk_index=chunk, file_index=file)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.concat_tables(shard), path)
            shard, shard_bytes = [], 0
            file += 1
            if file == chunks_size:
                chunk, file = chunk + 1, 0

        seen_ids = set()
        for e, ep in enumerate(episodes):
            # A streamed episode meets the same schema as a validated TrajectoryDataset, before its shard is written.
            if not isinstance(ep, Episode):
                raise LeRobotFormatError("write", [f"item {e} is a {type(ep).__name__}, not an Episode"])
            try:
                TrajectoryDataset([ep]).validate()
            except SchemaError as err:
                raise LeRobotFormatError("write", [f"episode {e} ({ep.episode_id!r}): {err}"]) from err
            if ep.episode_id in seen_ids:
                raise LeRobotFormatError("write", [f"episode {e}: duplicate episode_id {ep.episode_id!r}"])
            known = ep.termination_known is True or (ep.termination_known is None and ep.terminated)
            if known and not drop_endings:  # LeRobot v3.0 has no field for it: never lose it silently
                raise LeRobotFormatError("write", [
                    f"episode {e} ({ep.episode_id!r}) records a known ending ({'terminated' if ep.terminated else 'truncated'}); "
                    f"LeRobot v3.0 has no field for it, so writing would lose it; pass drop_endings=True to write "
                    f"anyway (the lossy mode), and its reader will see an unknown ending"])
            seen_ids.add(ep.episode_id)
            if names is None:
                names = _feature_names(ep)
                dims = {m: ep.observations[m].shape[1] for m, _ in names}
                adim = ep.actions.shape[1]
                dtypes = {m: str(ep.observations[m].dtype) for m, _ in names}
                dtypes["action"] = str(ep.actions.dtype)
                bad = [f"{k} has dtype {v}" for k, v in dtypes.items() if v not in FLOATS]
                if bad:
                    raise LeRobotFormatError("write", bad + ["states and actions must be float32 or float64"])
                features = {feat: {"dtype": dtypes[m], "shape": [dims[m]], "names": None} for m, feat in names}
                features["action"] = {"dtype": dtypes["action"], "shape": [adim], "names": None}
                for col in ("timestamp", *("frame_index", "episode_index", "index", "task_index")):
                    features[col] = {"dtype": "float32" if col == "timestamp" else "int64", "shape": [1], "names": None}
                hf = {k: _value_feature(v["dtype"], v["shape"][0]) for k, v in features.items()}
                fields = [pa.field(k, pa.list_(pa.from_numpy_dtype(np.dtype(v["dtype"])), v["shape"][0])
                                   if v["shape"][0] > 1 else pa.from_numpy_dtype(np.dtype(v["dtype"])))
                          for k, v in features.items()]
                schema = pa.schema(fields, metadata={"huggingface": json.dumps({"info": {"features": hf}})})
            if sorted(ep.observations) != sorted(m for m, _ in names):
                raise LeRobotFormatError("write", [f"episode {ep.episode_id}: modalities {sorted(ep.observations)} "
                                                   f"differ from the first episode's"])
            shape_dtype = {m: (ep.observations[m].shape[1], str(ep.observations[m].dtype)) for m, _ in names}
            shape_dtype["action"] = (ep.actions.shape[1], str(ep.actions.dtype))
            first = {m: (dims[m], dtypes[m]) for m, _ in names}
            first["action"] = (adim, dtypes["action"])
            if shape_dtype != first:
                raise LeRobotFormatError("write", [f"episode {ep.episode_id}: dimensions and dtypes {shape_dtype} "
                                                   f"differ from the first episode's {first}"])
            T = ep.length
            if ep.timestamps is None:
                ts = (np.arange(T) / fps).astype(np.float32)
            else:
                ts = np.asarray(ep.timestamps, dtype=np.float32)
                drift = np.abs(np.diff(ts.astype(np.float64)) - 1.0 / fps)
                if drift.size and drift.max() > TIMESTAMP_TOLERANCE_S:
                    raise LeRobotFormatError("write", [f"episode {ep.episode_id}: timestamps do not advance by "
                                                       f"1 / fps = {1.0 / fps} s (max error {drift.max():.3g} s)"])
            task = ep.task if ep.task is not None else default_task
            t_index = tasks.setdefault(task, len(tasks))
            cols = {feat: ep.observations[m] for m, feat in names}
            cols["action"] = ep.actions
            cols["timestamp"] = ts
            cols["frame_index"] = np.arange(T, dtype=np.int64)
            cols["episode_index"] = np.full(T, e, dtype=np.int64)
            cols["index"] = np.arange(start, start + T, dtype=np.int64)
            cols["task_index"] = np.full(T, t_index, dtype=np.int64)
            arrays = []
            for f in schema:
                x = np.ascontiguousarray(cols[f.name])
                if pa.types.is_fixed_size_list(f.type):
                    arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), f.type.value_type),
                                                                    f.type.list_size))
                else:
                    arrays.append(pa.array(x.reshape(-1), f.type))
            table = pa.Table.from_arrays(arrays, schema=schema)
            if shard and shard_bytes + table.nbytes > cap:
                flush()
            shard.append(table)
            shard_bytes += table.nbytes
            stats = {k: _episode_stats(cols[k]) for k in features}
            row = {"episode_index": e, "tasks": [task], "length": T, "data/chunk_index": chunk, "data/file_index": file,
                   "dataset_from_index": start, "dataset_to_index": start + T}
            for k, s in stats.items():
                for stat, v in s.items():
                    row[f"stats/{k}/{stat}"] = [int(c) for c in v] if stat == "count" else [float(c) for c in v]
            row["meta/episodes/chunk_index"], row["meta/episodes/file_index"] = 0, 0
            ep_rows.append(row)
            ep_stats.append(stats)
            start += T
        flush()
        if not ep_rows:
            raise LeRobotFormatError("write", ["no episodes"])
        pq.write_table(pa.Table.from_pylist(ep_rows), staging / EPISODES_PATH.format(chunk_index=0, file_index=0))
        pandas_meta = {"index_columns": ["task"], "column_indexes": [{"name": None, "field_name": None,
                       "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
                       "columns": [{"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                                    "numpy_type": "int64", "metadata": None},
                                   {"name": "task", "field_name": "task", "pandas_type": "unicode",
                                    "numpy_type": "object", "metadata": None}],
                       "attributes": {}, "creator": {"library": "pyarrow", "version": pa.__version__}}
        task_table = pa.table({"task_index": pa.array(list(tasks.values()), pa.int64()),
                               "task": pa.array(list(tasks), pa.string())},
                              metadata={"pandas": json.dumps(pandas_meta)})
        pq.write_table(task_table, staging / "meta" / "tasks.parquet")
        aggregate = {k: _aggregate([s[k] for s in ep_stats]) for k in features}
        (staging / "meta" / "stats.json").write_text(json.dumps(
            {k: {stat: ([int(c) for c in v] if stat == "count" else [float(c) for c in v]) for stat, v in s.items()}
             for k, s in aggregate.items()}, indent=4))
        info = {"codebase_version": CODEBASE_VERSION, "robot_type": robot_type, "total_episodes": len(ep_rows),
                "total_frames": start, "total_tasks": len(tasks), "chunks_size": chunks_size,
                "data_files_size_in_mb": data_files_size_in_mb, "video_files_size_in_mb": 200, "fps": fps,
                "splits": {"train": f"0:{len(ep_rows)}"}, "data_path": DATA_PATH, "video_path": None,
                "features": features}
        (staging / "meta" / "info.json").write_text(json.dumps(info, indent=4))
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root
