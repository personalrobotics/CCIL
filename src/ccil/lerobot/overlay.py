"""CCIL supervision as an overlay on an unchanged LeRobot dataset (#28).

    destination/
      data/ meta/ videos/        an unchanged copy of the source LeRobot dataset: stock readers see exactly it
      ccil/meta.json             schema, source identity, run, input hashes, feature mapping, generator, shard hashes
      ccil/labels/chunk-000/file-000.parquet ...   one row per usable synthetic label

A CCIL label is an independent counterfactual supervision record, not a frame of a trajectory, so labels are
never written as LeRobot episodes. The sidecar holds exactly the synthetic rows build_supervision returns
(accepted; and admissible and selected when certificates exist); rejected, inadmissible and unselected candidates
stay in the run's artifacts. Each row stores its synthetic observation (in the LeRobot feature its modality came
from), its action or action chunk and mask, its weight, its source episode, frame and global index, its candidate
row, its measurements and its classification. The run ID is recorded once, in ccil/meta.json; version 1 holds one
supervision run per overlay.

Provenance is verified before anything is written: the LeRobot source's files hash to the conversion manifest's,
and the supervision's dataset is the conversion's output, or the output of a data.prepare run whose input was that
conversion. Representations other than identity are refused: their labels are not in a LeRobot feature's space.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from .v3 import CODEBASE_VERSION, LeRobotFormatError, _arrow, _sha256, read_info

SCHEMA, VERSION = "ccil.lerobot.overlay", "1.0"
OVERLAY = "ccil"
LABELS_PATH = "ccil/labels/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _tree(root: Path, skip: str | None = None) -> dict:
    return {p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not (skip and p.relative_to(root).parts[0] == skip)}


def _dataset_link(conversion: dict, supervision_dataset: str, prepared_run) -> list[str]:
    """Why the supervision's dataset is not the conversion's output (empty when it is, directly or via data.prepare)."""
    out = conversion["output"]["sha256"]
    if supervision_dataset == out:
        return []
    if prepared_run is None:
        return [f"the supervision's dataset {supervision_dataset[:12]} is not the conversion's output {out[:12]}; pass "
                f"the data.prepare run that produced it"]
    manifest = json.loads((Path(prepared_run) / "manifest.json").read_text())
    problems = []
    if manifest.get("inputs", {}).get("dataset", {}).get("sha256") != out:
        problems.append("the data.prepare run's input is not the conversion's output")
    if manifest.get("files", {}).get("data/dataset.npz", {}).get("sha256") != supervision_dataset:
        problems.append("the data.prepare run's dataset is not the supervision's dataset")
    return problems


def export_overlay(source, destination, supervision, *, conversion_manifest, candidates, prepared_run=None,
                   rows_per_shard: int = 500_000) -> dict:
    """Write destination: an unchanged copy of the LeRobot dataset source plus the ccil/ supervision overlay.

    supervision is a ccil.supervision.Supervision (method ccil); candidates is the candidates file it was built
    from. Its SHA-256 must be the one the supervision recorded, and the run ID, generator and measurements are taken
    from it, never from the caller. The destination is built under a temporary name and moved into place, so it
    exists only once complete; an existing destination is refused. Returns ccil/meta.json."""
    from ..schema import read_candidates
    pa, pq = _arrow()
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} exists; overlays are written once")
    if isinstance(rows_per_shard, bool) or not isinstance(rows_per_shard, int) or rows_per_shard < 1:
        raise ValueError(f"rows_per_shard must be a positive integer, got {rows_per_shard!r}")
    info = read_info(source)
    conversion = json.loads(Path(conversion_manifest).read_text())
    man = supervision.manifest()
    problems = []
    if man.get("method") != "ccil":  # identified first: a non-CCIL supervision has no candidates to verify
        raise LeRobotFormatError("overlay", [f"the supervision's method is {man.get('method')!r}; an overlay carries "
                                             f"CCIL labels"])
    candidates = Path(candidates)
    recorded_cands = man.get("inputs", {}).get("candidates")
    if not candidates.is_file() or _sha256(candidates) != recorded_cands:
        raise LeRobotFormatError("overlay", [f"the candidates file {candidates.name} is not the one the supervision "
                                             f"was built from (SHA-256 {str(recorded_cands)[:12]})"])
    cands = read_candidates(candidates)
    run_id, generator, measurements = cands.run_id, cands.generator, cands.measurements
    rep = man.get("representation", {})
    if rep.get("kind") != "identity":
        problems.append(f"representation {rep.get('kind')!r}: only identity labels are in a LeRobot feature's space")
    modality = rep.get("modality")
    feature = next((name for name, d in conversion.get("features", {}).items()
                    if d.get("to") == f"observations[{modality}]"), None)
    if feature is None:
        problems.append(f"no LeRobot feature maps to the supervision's modality {modality!r}")
    files_now = _tree(source)
    recorded = conversion.get("source", {}).get("files", {})
    changed = sorted(k for k, v in recorded.items() if files_now.get(k) != v)
    if changed:
        problems.append(f"the source differs from the conversion manifest's files: {changed[:5]}")
    problems += _dataset_link(conversion, man["inputs"]["dataset"], prepared_run)
    synthetic = np.asarray(supervision.source) == "synthetic"  # possibly none: zero accepted labels is an outcome
    if problems:
        raise LeRobotFormatError("overlay", problems)

    # Map each label to its LeRobot source frame, from the source's own episode metadata.
    eps_files = sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    rows = [r for f in eps_files for r in pq.read_table(f, columns=["episode_index", "length", "dataset_from_index"])
            .to_pylist()]
    start = {str(r["episode_index"]): (r["dataset_from_index"], r["length"]) for r in rows}
    ep = np.asarray(supervision.episode_id)[synthetic]
    frame = np.asarray(supervision.frame)[synthetic].astype(np.int64)
    missing = sorted(set(ep) - set(start))
    if missing:
        raise LeRobotFormatError("overlay", [f"labels name episodes the source does not have: {missing[:5]}"])
    lo = np.array([start[e][0] for e in ep], dtype=np.int64)
    length = np.array([start[e][1] for e in ep], dtype=np.int64)
    if np.any(frame < 0) or np.any(frame >= length):
        bad = int(np.flatnonzero((frame < 0) | (frame >= length))[0])
        raise LeRobotFormatError("overlay", [f"label {bad}: frame {int(frame[bad])} is outside episode {ep[bad]}"])
    obs = np.asarray(supervision.observations)[synthetic]
    if obs.shape[1] != info["features"][feature]["shape"][0]:
        raise LeRobotFormatError("overlay", [f"labels have {obs.shape[1]} dimensions; {feature} has "
                                             f"{info['features'][feature]['shape'][0]}"])
    cand = np.asarray(supervision.candidate_row)[synthetic].astype(np.int64)
    cols = {"candidate_row": cand, "source_episode_index": ep.astype(np.int64),
            "source_frame_index": frame, "source_index": lo + frame,
            "weight": np.asarray(supervision.weights, dtype=np.float64)[synthetic],
            "classification": np.asarray(supervision.classification)[synthetic].astype(str),
            feature: obs.astype(np.dtype(info["features"][feature]["dtype"]))}
    chunked = supervision.action_chunks is not None
    if chunked:
        ch = np.asarray(supervision.action_chunks)[synthetic]
        cols["action_chunk"] = ch.reshape(len(ch), int(np.prod(ch.shape[1:]))).astype(
            np.dtype(info["features"]["action"]["dtype"]))  # explicit K * A: zero labels must still reshape
        cols["action_chunk_mask"] = np.asarray(supervision.action_chunk_mask)[synthetic].astype(bool)
    else:
        cols["action"] = np.asarray(supervision.actions)[synthetic].astype(np.dtype(info["features"]["action"]["dtype"]))
    for name, values in sorted((measurements or {}).items()):
        cols[f"measurement.{name}"] = np.asarray(values, dtype=np.float64)[cand]

    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(source, staging)  # the LeRobot portion: an unchanged copy
        shards = {}
        n = len(cand)
        for i, a in enumerate(range(0, max(n, 1), rows_per_shard)):  # zero labels: one empty shard with the schema
            b = min(a + rows_per_shard, n)
            arrays = []
            for name, x in cols.items():
                x = np.ascontiguousarray(x[a:b])
                if x.ndim == 2:
                    t = pa.from_numpy_dtype(x.dtype) if x.dtype != bool else pa.bool_()
                    arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), t), x.shape[1]))
                else:
                    arrays.append(pa.array(x))
            rel = LABELS_PATH.format(chunk_index=i // 1000, file_index=i % 1000)
            path = staging / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_arrays(arrays, names=list(cols)), path)
            shards[rel] = _sha256(path)
        lerobot_part = _tree(staging, skip=OVERLAY)
        if lerobot_part != files_now:
            raise LeRobotFormatError("overlay", ["the copied LeRobot portion differs from the source"])
        classes, counts = np.unique(cols["classification"], return_counts=True)
        meta = {"schema": SCHEMA, "version": VERSION,
                "source": {"format": f"lerobot.dataset/{CODEBASE_VERSION}", "files": lerobot_part,
                           "conversion_output_sha256": conversion["output"]["sha256"]},
                "run_id": run_id, "generator": generator,
                "inputs": {k: v for k, v in man["inputs"].items()},
                "supervision_manifest_sha256": _sha256_json(man),
                "supervision": {"method": man["method"], "certification": man.get("certification"),
                                "counts": man.get("counts"), "action_chunks": man.get("action_chunks"),
                                "demonstration_weight": man.get("demonstration_weight")},
                "feature_map": {"observation": {"modality": modality, "lerobot_feature": feature},
                                "action": "action_chunk" if chunked else "action"},
                "rows": n, "classifications": {str(c): int(k) for c, k in zip(classes, counts)},
                "columns": list(cols), "shards": shards,
                "demonstrations": {"episodes": sorted({int(e) for e in np.asarray(supervision.episode_id)[~synthetic]}),
                                   "frames": "t <= T - 2 of each listed episode (ccil.supervision)",
                                   "weight": man.get("demonstration_weight")},
                "endings": conversion.get("endings"),
                "temporal": "labels are independent records: they have no neighbours, and a label's future actions "
                            "are only its stored action chunk"}
        (staging / OVERLAY / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return meta


def _sha256_json(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


SHARD = re.compile(r"^ccil/labels/chunk-\d{3}/file-\d{3}\.parquet$")
META_KEYS = {"schema": str, "version": str, "source": dict, "run_id": str, "inputs": dict, "feature_map": dict,
             "rows": int, "columns": list, "shards": dict, "demonstrations": dict, "supervision": dict}


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def read_overlay(root) -> tuple[dict, dict]:
    """(ccil/meta.json, {column: array}) of an overlay, verified before anything it names is opened.

    The metadata's structure is checked first. Every shard path must match ccil/labels/chunk-NNN/file-NNN.parquet,
    appear once, and resolve (through any symlink) inside ccil/labels; no unlisted file may sit there. Every shard
    must have its recorded hash, and the LeRobot portion must be the source recorded at export. Any failure is one
    LeRobotFormatError."""
    _, pq = _arrow()
    root = Path(root)
    path = root / OVERLAY / "meta.json"
    if not path.is_file():
        raise LeRobotFormatError(root, ["no CCIL overlay (ccil/meta.json is missing)"])
    try:
        meta = json.loads(path.read_text(), object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise LeRobotFormatError(path, [f"unreadable: {e}"]) from e
    problems = []
    if not isinstance(meta, dict):
        raise LeRobotFormatError(path, ["not a JSON object"])
    for key, kind in META_KEYS.items():
        if not isinstance(meta.get(key), kind) or isinstance(meta.get(key), bool):
            problems.append(f"{key} is missing or not a {kind.__name__}")
    if not problems:
        if meta["schema"] != SCHEMA or meta["version"].split(".")[0] != VERSION.split(".")[0]:
            problems.append(f"schema {meta['schema']!r} {meta['version']!r} is not {SCHEMA} {VERSION}")
        if not isinstance(meta["source"].get("files"), dict):
            problems.append("source.files is missing")
        labels_root = (root / OVERLAY / "labels").resolve()
        for rel, digest in meta["shards"].items():
            if not isinstance(rel, str) or not SHARD.match(rel) or not isinstance(digest, str):
                problems.append(f"shard {rel!r} is not a ccil/labels/chunk-NNN/file-NNN.parquet path with a hash")
                continue
            try:
                (root / rel).resolve().relative_to(labels_root)
            except (ValueError, OSError):
                problems.append(f"shard {rel!r} resolves outside ccil/labels (through a symlink)")
        if not problems:
            listed = {(root / rel).resolve() for rel in meta["shards"]}
            present = {f.resolve() for f in (root / OVERLAY / "labels").rglob("*") if f.is_file() or f.is_symlink()}
            extra = sorted(str(f) for f in present - listed)
            if extra:
                problems.append(f"unlisted files under ccil/labels: {len(extra)}")
    if problems:
        raise LeRobotFormatError(path, problems)
    if _tree(root, skip=OVERLAY) != meta["source"]["files"]:
        problems.append("the LeRobot portion differs from the source recorded at export")
    tables = []
    for rel, digest in sorted(meta["shards"].items()):
        p = root / rel
        if not p.is_file() or _sha256(p) != digest:
            problems.append(f"{rel} is missing or does not have its recorded SHA-256")
        else:
            tables.append(pq.read_table(p))
    if problems:
        raise LeRobotFormatError(root, problems)
    import pyarrow as pa
    table = pa.concat_tables(tables)
    out = {}
    for name in table.schema.names:
        col = table.column(name).combine_chunks()
        if pa.types.is_fixed_size_list(col.type):
            out[name] = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), col.type.list_size)
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
    if sorted(out) != sorted(meta["columns"]) or len(out.get("candidate_row", ())) != meta["rows"]:
        raise LeRobotFormatError(root, ["the shards' columns or row count disagree with ccil/meta.json"])
    return meta, out
