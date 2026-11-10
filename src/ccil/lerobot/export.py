"""One augmented LeRobot v3.0 dataset that a stock LeRobot trainer consumes unchanged (#30).

    ccil export lerobot RESULT --conversion conversion.json --out DIR

From a curate result of a dataset converted from LeRobot (ccil.lerobot.convert), write one dataset: the training
episodes as they were converted, then one one-frame episode per usable selected synthetic row, holding the label's
observation (split back into its LeRobot features) and action at timestamp 0. Every frame carries int64 columns
ccil.synthetic, ccil.source_episode, ccil.source_frame, ccil.source_index (the source dataset's global index) and
ccil.candidate_row (-1 for demonstration frames), which LeRobot policies ignore. A stock trainer reads no row
weight, so every selected synthetic weight must be exactly 1. Held-out episodes are not written. The directory
holds ccil/export.json (ccil.lerobot-export 1.0): the inputs by hash, the episode split, the counts, the hash of
the synthetic rows' source indices (so a duplication control can match them exactly) and every file's SHA-256. With
a recorded split, held-out episodes and the labels of their transitions are never written. It is
staged under a temporary name and moved into place, so DIR exists only once complete.

A one-frame episode gives a temporal policy no real history or future: it is consumed correctly only by a policy
that reads one observation and predicts one action (for example ACT with n_obs_steps 1 and chunk_size 1).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from ..errors import InputError
from ..runs.provenance import sha256
from ..schema import Episode, read_candidates, read_trajectories

SCHEMA, VERSION = "ccil.lerobot-export", "1.0"
MANIFEST = Path("ccil") / "export.json"
COLUMNS = {"ccil.synthetic": "1 for an exported CCIL label, 0 for a demonstration frame",
           "ccil.source_episode": "the source LeRobot episode_index",
           "ccil.source_frame": "the source frame_index (a label's source transition frame t)",
           "ccil.source_index": "the source dataset's global index of that frame",
           "ccil.candidate_row": "the label's row in the candidates file, -1 for a demonstration frame"}


class ExportError(InputError):
    """The curate result, conversion, or selection cannot be exported for an unchanged trainer."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("cannot export a LeRobot dataset:\n  " + "\n  ".join(self.problems))


def export_dataset(result, conversion, out, *, revision: str | None = None) -> dict:
    """Write the augmented dataset at out from a curate result and the conversion manifest; return its manifest."""
    from ..api import CURATE, _generated, _verified
    from ..core import IdentityRepresentation
    from ..core.representation import parts
    from ..supervision import build_supervision
    from .v3 import write_dataset

    result, conversion, out = Path(result).resolve(), Path(conversion), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; exports are written once")
    doc = _verified(result, schema=None)
    if doc["schema"] != CURATE:
        raise ExportError([f"{result} is a {doc['schema']}; export reads a curate result (ccil curate writes one)"])
    base = result.parent
    gen = _generated(doc, base)
    art = {k: base / a["path"] for k, a in {**gen["artifacts"], **doc["artifacts"]}.items()}
    values = {k: e["value"] for k, e in gen["config"]["values"].items()}
    problems = []
    try:
        conv = json.loads(conversion.read_text())
    except (OSError, ValueError) as e:
        raise ExportError([f"{conversion}: not a readable conversion manifest ({e})"]) from None
    if conv.get("schema") != "ccil.lerobot.conversion":
        problems.append(f"{conversion} is not a ccil.lerobot.conversion manifest")
    prepare = next((s for s in gen["stages"] if s["stage"] == "data.prepare"), None)
    prepared_input = json.loads((base / prepare["run_dir"] / "manifest.json").read_text())["inputs"]["dataset"]["sha256"] \
        if prepare else None
    if prepared_input != conv.get("output", {}).get("sha256"):
        problems.append(f"the run's dataset (SHA-256 {str(prepared_input)[:12]}) is not this conversion's output "
                        f"({str(conv.get('output', {}).get('sha256'))[:12]})")
    if values.get("representation.kind") != "identity":
        problems.append("only identity labels are observations a LeRobot feature can hold")
    modality = values.get("representation.modality")
    ds = read_trajectories(art["dataset"])
    observed = sorted(ds.episodes[0].observations)
    if sorted(parts(modality)) != observed:
        problems.append(f"the labels hold {sorted(parts(modality))}, the dataset {observed}: a synthetic frame must "
                        "fill every observation feature")
    if not all(e.episode_id.isdigit() for e in ds.episodes):
        problems.append("episode IDs are not LeRobot episode indices: export needs a dataset converted from LeRobot")
    if problems:
        raise ExportError(problems)

    split = json.loads(art["split"].read_text()) if "split" in art else None
    # A recorded split is honored unconditionally: the export holds training episodes only, whatever an integrated
    # trainer's supervision.exclude_validation_episodes says.
    exclude = split is not None
    sup = build_supervision(art["dataset"], "demonstrations_plus_selection",
                            representation=IdentityRepresentation(modality), split=art["split"] if exclude else None,
                            candidates=art["candidates"], certificates=art.get("certificates"),
                            selection=art["selection"], calibration=art.get("calibration"))
    synthetic = sup.source == "synthetic"
    rows = np.asarray(sup.candidate_row)[synthetic].astype(np.int64)
    weights = np.asarray(sup.weights)[synthetic]
    if (weights != 1.0).any():
        raise ExportError([f"{int((weights != 1.0).sum())} selected synthetic rows have weights other than 1 "
                           f"(e.g. {float(weights[weights != 1.0][0])}); a stock trainer reads no row weight, so only "
                           "unit weights export"])
    cands = read_candidates(art["candidates"])
    train = set(split["train"]) if exclude else {e.episode_id for e in ds.episodes}
    episodes = {e.episode_id: e for e in ds.episodes}
    start, offset = {}, 0
    for e in ds.episodes:  # LeRobot v3.0 indices are contiguous in episode order
        start[e.episode_id] = offset
        offset += e.length
    demos = [e for e in ds.episodes if e.episode_id in train]
    src_ep = np.asarray(cands.source_episode_id).astype(str)[rows]
    src_fr = np.asarray(cands.source_frame).astype(np.int64)[rows]
    dtype = {m: ds.episodes[0].observations[m].dtype for m in observed}  # labels are stored in the dataset's dtype
    labels = [Episode(f"ccil.{int(r)}", {m: np.asarray(cands.observations[m][r:r + 1], dtype[m]) for m in observed},
                      np.asarray(cands.actions[r:r + 1], ds.episodes[0].actions.dtype), timestamps=np.zeros(1),
                      task=episodes[src_ep[i]].task) for i, r in enumerate(rows)]
    n_demo = len(demos)

    def columns(e, ep):
        if e < n_demo:
            T, eid = ep.length, int(ep.episode_id)
            frames = np.arange(T, dtype=np.int64)
            return {"ccil.synthetic": np.zeros(T), "ccil.source_episode": np.full(T, eid), "ccil.source_frame": frames,
                    "ccil.source_index": start[ep.episode_id] + frames, "ccil.candidate_row": np.full(T, -1)}
        i = e - n_demo
        return {"ccil.synthetic": np.ones(1), "ccil.source_episode": np.array([int(src_ep[i])]),
                "ccil.source_frame": np.array([src_fr[i]]), "ccil.source_index": np.array([start[src_ep[i]] + src_fr[i]]),
                "ccil.candidate_row": np.array([rows[i]])}

    staging = out.with_name(f".{out.name}.export-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        write_dataset(iter(demos + labels), staging, fps=float(conv["source"]["fps"]), columns=columns,
                      single_frame_episodes=True)
        source_index = np.sort(np.array([start[src_ep[i]] + src_fr[i] for i in range(len(rows))], np.int64))
        files = {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = {
            "schema": SCHEMA, "version": VERSION,
            "source": {"revision": revision, "fps": conv["source"]["fps"], "conversion_sha256": sha256(conversion),
                       "converted_dataset_sha256": conv["output"]["sha256"]},
            "inputs": {"curate_result": sha256(result), "generate_result": doc["generated"]["sha256"],
                       **{k: gen["artifacts"][k]["sha256"] for k in ("dataset", "split", "transition_model", "candidates",
                                                                     "certificates") if k in gen["artifacts"]},
                       "selection": doc["artifacts"]["selection"]["sha256"]},
            "representation": {"kind": "identity", "modality": modality},
            "episodes": {"train": [e.episode_id for e in demos],
                         "held_out": [e.episode_id for e in ds.episodes if e.episode_id not in train]},
            "counts": {"demonstration_episodes": n_demo, "demonstration_frames": int(sum(e.length for e in demos)),
                       "synthetic_rows": int(len(rows)), "episodes": n_demo + int(len(rows)),
                       "frames": int(sum(e.length for e in demos)) + int(len(rows))},
            "synthetic_source_index_sha256": hashlib.sha256(source_index.tobytes()).hexdigest(),
            "columns": COLUMNS,
            "consumption": "one observation, one action: a policy with no observation history and a one-step action "
                           "chunk (e.g. ACT n_obs_steps 1, chunk_size 1); every frame has weight 1",
            "storage": "labels are stored in the dataset's observation and action dtypes",
            "files": files}
        (staging / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_export(root) -> dict:
    """An export's manifest, after checking every recorded file's SHA-256; ExportError naming each change."""
    root = Path(root)
    try:
        manifest = json.loads((root / MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise ExportError([f"{root / MANIFEST}: not a readable export manifest ({e})"]) from None
    if (manifest.get("schema"), manifest.get("version")) != (SCHEMA, VERSION):
        raise ExportError([f"{root / MANIFEST} is not {SCHEMA} {VERSION}"])
    problems = [f"{rel} is missing" if not (root / rel).is_file() else f"{rel} changed since export"
                for rel, h in manifest["files"].items() if not (root / rel).is_file() or sha256(root / rel) != h]
    if problems:
        raise ExportError(problems[:20])
    return manifest
