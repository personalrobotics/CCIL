"""The tabular subset of LeRobotDataset v3.0, read and written with PyArrow (#26).

Layout: meta/info.json declares the format version, the features and the data path template; data shards are
Parquet files at data_path.format(chunk_index=..., file_index=...); meta/episodes/chunk-XXX/file-YYY.parquet holds
one row per episode (its data shard, its global index range, its tasks and per-episode statistics);
meta/tasks.parquet maps task_index to the task string; meta/stats.json holds the dataset statistics.

Mapping to ccil.trajectories: observation.state -> observations["state"], any other observation.<name> ->
observations["state.<name>"]; action -> actions; timestamp -> timestamps; the episode's single task -> task.
LeRobot v3.0 records no terminal flag, so every episode reads as truncated (terminated=False), as the dataset's
meta records. Index columns (index, frame_index, episode_index, task_index) are checked, not copied.

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

from ..schema import Episode, TrajectoryDataset

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
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        problems.append(f"{where} {rel!r} is not a relative path inside the dataset")
        return None
    return root.joinpath(*p.parts)


# --- reading -------------------------------------------------------------------------------------------------------

def read_info(root) -> dict:
    """meta/info.json, validated: the format version, the fields the bridge relies on and the path template."""
    root = Path(root)
    path = root / "meta" / "info.json"
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
    for name, spec in info["features"].items():
        if not (isinstance(spec, dict) and isinstance(spec.get("dtype"), str) and isinstance(spec.get("shape"), list)
                and all(isinstance(d, int) and d > 0 for d in spec["shape"])):
            problems.append(f"feature {name!r} needs a dtype string and a shape list of positive integers")
    if problems:
        raise LeRobotFormatError(path, problems)
    return info


def _plan(info: dict, ignore) -> tuple[dict, list]:
    """Map each feature to its CCIL slot, or fail closed with every unsupported feature named."""
    ignore = set(ignore)
    plan, problems = {}, []
    for name, spec in info["features"].items():
        dtype, shape = spec["dtype"], tuple(spec["shape"])
        if name in ignore:
            continue
        if dtype in DEFERRED or "depth" in name:
            problems.append(f"feature {name!r} ({dtype}) is an image, video, depth or audio modality: not supported "
                            f"until image CCIL (M5)")
        elif name in INDEX_COLUMNS:
            if dtype != "int64" or shape != (1,):
                problems.append(f"index feature {name!r} must be int64 of shape [1], got {dtype} {list(shape)}")
        elif name == "timestamp":
            if dtype not in FLOATS or shape != (1,):
                problems.append(f"timestamp must be float32 or float64 of shape [1], got {dtype} {list(shape)}")
        elif name == "action" or name.startswith("observation."):
            if dtype not in FLOATS or len(shape) != 1:
                problems.append(f"feature {name!r} must be a float32 or float64 vector, got {dtype} {list(shape)}")
            elif name == "action":
                plan[name] = ("action", shape[0])
            else:
                suffix = name[len("observation."):]
                modality = "state" if suffix == "state" else "state." + re.sub(r"[^a-z0-9_]", "_", suffix.lower())
                if modality in {m for m, _ in plan.values()}:
                    problems.append(f"feature {name!r} maps to modality {modality!r}, which another feature has")
                plan[name] = (modality, shape[0])
        else:
            problems.append(f"feature {name!r} ({dtype}) has no CCIL slot; name it in ignore to read without it")
    for required in ("action", "timestamp", *INDEX_COLUMNS):
        if required not in info["features"]:
            problems.append(f"required feature {required!r} is missing")
    if not any(slot != "action" for slot, _ in plan.values()):
        problems.append("no observation.* feature: a CCIL dataset needs at least one state modality")
    return plan, problems


def _episodes_table(root: Path, info: dict):
    """Every episode's metadata row, in episode order, from the episode files (checked for consistency)."""
    _, pq = _arrow()
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
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
        rows += pq.read_table(f, columns=cols).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    problems = []
    if [r["episode_index"] for r in rows] != list(range(len(rows))):
        problems.append("episode indices are not 0..N-1, each once")
    if len(rows) != info["total_episodes"]:
        problems.append(f"{len(rows)} episodes in meta/episodes, info.json says {info['total_episodes']}")
    start = 0
    for r in rows:
        e = r["episode_index"]
        if r["dataset_from_index"] != start or r["dataset_to_index"] - r["dataset_from_index"] != r["length"]:
            problems.append(f"episode {e}: offsets [{r['dataset_from_index']}, {r['dataset_to_index']}) are not "
                            f"contiguous from {start} with length {r['length']}")
        if r["length"] < 1:
            problems.append(f"episode {e}: length {r['length']}")
        start = r["dataset_to_index"]
    if start != info["total_frames"]:
        problems.append(f"episodes cover {start} frames, info.json says {info['total_frames']}")
    if problems:
        raise LeRobotFormatError(root / "meta" / "episodes", problems[:20])
    return rows


def _tasks(root: Path) -> dict:
    """task_index -> task string; the task string is the pandas index column the file's metadata names."""
    _, pq = _arrow()
    path = root / "meta" / "tasks.parquet"
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
    for r in table.to_pylist():
        if r["task_index"] in out:
            raise LeRobotFormatError(path, [f"task_index {r['task_index']} appears twice"])
        out[r["task_index"]] = r[text]
    return out


def _column(table, name, dim):
    col = table.column(name).combine_chunks()
    import pyarrow as pa
    if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
        values = col.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(col), dim)
    return col.to_numpy(zero_copy_only=False).reshape(len(col), 1 if dim is None else dim)


def iter_episodes(root, ignore=(), files: dict | None = None):
    """Yield the dataset's episodes in order, reading one data shard at a time.

    files, when given, receives {relative path: SHA-256} of every file read."""
    root = Path(root)
    info = read_info(root)
    plan, problems = _plan(info, ignore)
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
    columns = [*plan, "timestamp", *INDEX_COLUMNS]
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
        index = table.column("index").to_numpy()
        if len(index) == 0 or np.any(np.diff(index) != 1):
            raise LeRobotFormatError(path, ["the index column is empty or not consecutive"])
        for r in eps:
            e, lo, hi = r["episode_index"], r["dataset_from_index"], r["dataset_to_index"]
            a, b = lo - int(index[0]), hi - int(index[0])
            if a < 0 or b > len(index):
                raise LeRobotFormatError(path, [f"episode {e}: rows [{lo}, {hi}) are not in this shard"])
            part = table.slice(a, b - a)
            bad = []
            if not np.array_equal(part.column("episode_index").to_numpy(), np.full(b - a, e)):
                bad.append("episode_index differs from the episode's metadata")
            if not np.array_equal(part.column("frame_index").to_numpy(), np.arange(b - a)):
                bad.append("frame_index is not 0..length-1")
            task_ids = np.unique(part.column("task_index").to_numpy())
            if len(task_ids) != 1:
                bad.append(f"{len(task_ids)} tasks within one episode; ccil.trajectories has one task per episode")
            elif int(task_ids[0]) not in tasks:
                bad.append(f"task_index {int(task_ids[0])} is not in meta/tasks.parquet")
            elif r["tasks"] and tasks[int(task_ids[0])] not in r["tasks"]:
                bad.append(f"task {tasks[int(task_ids[0])]!r} is not among the episode's tasks {r['tasks']}")
            if bad:
                raise LeRobotFormatError(path, [f"episode {e}: {m}" for m in bad])
            obs = {modality: _column(part, name, dim) for name, (modality, dim) in plan.items() if modality != "action"}
            yield Episode(episode_id=str(e), observations=obs, actions=_column(part, "action", plan["action"][1]),
                          terminated=False, timestamps=part.column("timestamp").to_numpy(),
                          task=tasks[int(task_ids[0])])


def read_dataset(root, ignore=()) -> TrajectoryDataset:
    """The whole dataset as a validated ccil.trajectories dataset (iter_episodes streams it instead)."""
    root = Path(root)
    files: dict = {}
    episodes = list(iter_episodes(root, ignore, files))
    info = read_info(root)
    plan, _ = _plan(info, ignore)
    if (root / "meta" / "stats.json").is_file():
        files["meta/stats.json"] = _sha256(root / "meta" / "stats.json")
    ds = TrajectoryDataset(episodes, {"dataset": {
        "format": f"lerobot.dataset/{CODEBASE_VERSION}", "fps": info["fps"],
        "feature_map": {name: slot for name, (slot, _) in plan.items()}, "ignored_features": sorted(ignore),
        "features": info["features"], "files": dict(sorted(files.items())),
        "terminated": "LeRobot v3.0 records no terminal flag: every episode reads as truncated"}})
    return ds.validate()


def read_stats(root) -> dict:
    """meta/stats.json as stored: {feature: {min, max, mean, std, count, q01, ...: lists}}."""
    path = Path(root) / "meta" / "stats.json"
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
                  chunks_size: int = 1000, robot_type: str | None = None) -> Path:
    """Write episodes (a TrajectoryDataset or an iterable of Episode) as a LeRobotDataset v3.0 directory.

    Streams: each data shard is written when it reaches data_files_size_in_mb. Timestamps must advance by 1 / fps
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

        for e, ep in enumerate(episodes):
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
