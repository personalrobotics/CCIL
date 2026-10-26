"""The trainer-neutral supervision contract (#89): which rows reach a policy loss, and with which weights.

CCIL's ordinary workflow ends at immutable artifacts: the prepared
demonstrations, the candidate labels, and optionally their certificates. A
downstream trainer of the user's choice consumes them. build_supervision
resolves, from those files, exactly the rows a trainer may use, fails closed
when the artifacts do not belong together, and records a manifest that
reconciles every row count. It trains nothing and imports no trainer.

Arms (policy.method):
- bc: the demonstrations only.
- noise_bc: the demonstrations only; the trainer perturbs them online at every
  draw (ccil.training.noise_bc), so nothing synthetic is stored.
- ccil without certificates: the demonstrations and every accepted candidate,
  recorded as uncertified.
- ccil with certificates: the demonstrations and the candidates with
  accepted & admissible & selected.
A rejected, inadmissible, or unselected candidate never appears, whatever its
weight: there is no weight-only path into the loss.

Demonstration rows are the transition frames t <= T - 2 of the training
episodes (the legacy load_data rows), weighted by an explicit, recorded
demonstration_weight. Synthetic rows keep the weight recorded in the
candidates file. With a split, rows whose source episode is held out are
excluded and counted.

Action chunks are consumed or refused, never silently reduced to one action:
with action_chunks="refuse" (the default), chunked candidates raise; with
"consume", every row carries a chunk and mask. A synthetic row's chunk is the
stored one; a demonstration row's is a_t at position 0 and a_{t+k} at position
k, masked past the episode end (the label-chunk alignment, ccil.core.chunks,
with target frame t + 1).
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core.batch import InterfaceError
from .core.representation import transitions
from .data.split import SplitError, check_split
from .runs.provenance import sha256
from .schema import SchemaError, read_candidates, read_certificates, read_trajectories

METHODS = ("bc", "noise_bc", "ccil")
CHUNK_POLICIES = ("refuse", "consume")
DEMONSTRATION, SYNTHETIC = "demonstration", "synthetic"


class SupervisionError(ValueError):
    """The artifacts cannot form supervision as requested; .problems lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("supervision refused:\n  " + "\n  ".join(problems))


@dataclass
class Supervision:
    """The rows a policy loss may use, in a fixed order: demonstrations, then synthetic rows in candidate order."""

    observations: np.ndarray      # (N, D) in the representation's output modality
    actions: np.ndarray           # (N, A)
    weights: np.ndarray           # (N,) float64: demonstration_weight, or the candidate's recorded weight
    source: np.ndarray            # (N,) "demonstration" or "synthetic"
    episode_id: np.ndarray        # (N,) source episode
    frame: np.ndarray             # (N,) source frame t
    candidate_row: np.ndarray     # (N,) row in the candidates file, -1 for demonstrations
    classification: np.ndarray    # (N,) "demonstration", "uncertified", or the certificate's classification
    manifest_: dict = field(default_factory=dict)
    action_chunks: np.ndarray | None = None       # (N, K, A) with action_chunks="consume"
    action_chunk_mask: np.ndarray | None = None   # (N, K)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(observations, actions, weights): the single-action handoff; refused for chunked supervision."""
        if self.action_chunks is not None:
            raise SupervisionError(["this supervision consumes action chunks; train on chunk_arrays(), not on single "
                                    "actions"])
        return self.observations, self.actions, self.weights

    def chunk_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(observations, action_chunks (N, K, A), action_chunk_mask (N, K), weights): the chunked handoff."""
        if self.action_chunks is None:
            raise SupervisionError(["this supervision carries no action chunks; build it with action_chunks='consume'"])
        return self.observations, self.action_chunks, self.action_chunk_mask, self.weights

    def manifest(self) -> dict:
        return json.loads(json.dumps(self.manifest_))


def _demonstration_chunks(dataset, episode_ids, frames, K: int):
    """Position 0 = a_t, position k = a_{t+k}, masked and zero past the episode's last frame."""
    episodes = {ep.episode_id: ep for ep in dataset.episodes}
    A = dataset.episodes[0].actions.shape[1]
    chunks = np.zeros((len(frames), K, A), dtype=dataset.episodes[0].actions.dtype)
    mask = np.zeros((len(frames), K), bool)
    for i, (eid, t) in enumerate(zip(episode_ids, frames)):
        ep = episodes[str(eid)]
        pos = int(t) + np.arange(K)
        inside = pos <= ep.length - 1
        chunks[i, inside] = ep.actions[pos[inside]]
        mask[i] = inside
    return chunks, mask


def _representation_problems(representation, checkpoint, hashes) -> list[str]:
    """Bind the representation object to an immutable artifact: its checkpoint, for a trainable representation.

    A fitted representation (trainable, or of unknown kind) must come with the
    checkpoint it was loaded from. The checkpoint is hashed, and the object
    must serialize to exactly those bytes, so demonstrations are encoded in the
    same coordinates the candidates were generated in. A fixed representation
    (trainable False, such as identity) has no checkpoint.
    """
    if not getattr(representation, "trainable", True):
        return ["a fixed representation takes no checkpoint"] if checkpoint is not None else []
    if checkpoint is None:
        return [f"representation kind {getattr(representation, 'kind', '?')!r} is fitted; pass representation_checkpoint "
                "so supervision is bound to its exact parameters"]
    hashes["representation"] = sha256(checkpoint)
    if not hasattr(representation, "save"):
        return ["the representation cannot be compared with its checkpoint (it has no save)"]
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "representation.npz"
        representation.save(probe)
        if sha256(probe) != hashes["representation"]:
            return ["the representation object is not the checkpoint it was given with (its parameters differ)"]
    return []


def _valid_chunk_size(k) -> bool:
    return not isinstance(k, bool) and isinstance(k, (int, np.integer)) and k >= 1


def build_supervision(dataset, method: str, *, representation, candidates=None, certificates=None, split=None,
                      representation_checkpoint=None, demonstration_weight: float = 1.0, action_chunks: str = "refuse",
                      chunk_size: int | None = None) -> Supervision:
    """Resolve the usable rows for method from artifact files; raise SupervisionError on any inconsistency."""
    problems = []
    if method not in METHODS:
        raise SupervisionError([f"method must be one of {list(METHODS)}, got {method!r}"])
    if action_chunks not in CHUNK_POLICIES:
        raise SupervisionError([f"action_chunks must be one of {list(CHUNK_POLICIES)}, got {action_chunks!r}"])
    if isinstance(demonstration_weight, bool) or not isinstance(demonstration_weight, (int, float)) or not (
            np.isfinite(demonstration_weight) and demonstration_weight >= 0):
        raise SupervisionError([f"demonstration_weight must be finite and >= 0, got {demonstration_weight!r}"])
    if method != "ccil" and (candidates is not None or certificates is not None):
        raise SupervisionError([f"{method} trains on the demonstrations only; it takes no candidates or certificates"])
    if certificates is not None and candidates is None:
        raise SupervisionError(["certificates need the candidates file they certify"])
    if method == "ccil" and candidates is None:
        raise SupervisionError(["ccil needs a candidates file"])

    if chunk_size is not None and not _valid_chunk_size(chunk_size):
        raise SupervisionError([f"chunk_size must be an integer >= 1, got {chunk_size!r}"])
    hashes = {"dataset": sha256(dataset)}
    try:
        ds = read_trajectories(dataset)
    except SchemaError as e:
        raise SupervisionError([f"dataset: {e}"]) from None
    modality = representation.output_modality
    problems += _representation_problems(representation, representation_checkpoint, hashes)
    try:
        batch = transitions(ds, representation)
    except (KeyError, ValueError, InterfaceError) as e:
        raise SupervisionError(problems + [f"the representation cannot encode the dataset: {e}"]) from None
    obs_shape, action_width = tuple(np.asarray(batch.z).shape[1:]), int(np.asarray(batch.a).shape[1])

    # Split.
    train_eps = None
    if split is not None:
        hashes["split"] = sha256(split)
        try:
            record = json.loads(Path(split).read_text())
            check_split(record)
        except (ValueError, SplitError) as e:
            raise SupervisionError([f"split: {e}"]) from None
        if sorted(record["episode_ids"]) != sorted(ep.episode_id for ep in ds.episodes):
            raise SupervisionError(["split episode ids are not the dataset's episodes"])
        train_eps = set(record["train"])

    # Candidates and certificates, checked against each other and the dataset before any row is used.
    cands = cert = None
    if candidates is not None:
        hashes["candidates"] = sha256(candidates)
        try:
            cands = read_candidates(candidates, ds)
        except SchemaError as e:
            raise SupervisionError([f"candidates do not belong to this dataset: {e}"]) from None
        if cands.inputs.get("dataset") != hashes["dataset"]:
            problems.append(f"candidates were generated from dataset {cands.inputs.get('dataset')}, not this dataset "
                            f"({hashes['dataset']})")
        if cands.inputs.get("representation") != hashes.get("representation"):
            problems.append(f"candidates were generated in representation {cands.inputs.get('representation')}, not "
                            f"{hashes.get('representation')}")
        if modality not in cands.observations:
            problems.append(f"candidates hold modalities {sorted(cands.observations)}, not the representation's "
                            f"{modality!r}")
        elif tuple(cands.observations[modality].shape[1:]) != obs_shape:
            problems.append(f"candidate observations have shape {tuple(cands.observations[modality].shape[1:])}; the "
                            f"representation encodes demonstrations as {obs_shape}")
        if tuple(cands.actions.shape[1:]) != (action_width,):
            problems.append(f"candidate actions have shape {tuple(cands.actions.shape[1:])}; the demonstrations' are "
                            f"({action_width},)")
        if cands.action_chunks is not None and action_chunks == "refuse":
            problems.append("the candidates carry action chunks; a consumer must declare action_chunks='consume' "
                            "(and train on chunks) rather than use only their first action")
    if certificates is not None:
        hashes["certificates"] = sha256(certificates)
        try:
            cert = read_certificates(certificates)
        except SchemaError as e:
            raise SupervisionError([f"certificates: {e}"]) from None
        if cert.candidates_sha256 != hashes.get("candidates"):
            problems.append(f"certificates certify candidates {cert.candidates_sha256}, not this file "
                            f"({hashes.get('candidates')})")
        if cert.inputs.get("dataset") != hashes["dataset"]:
            problems.append(f"certificates were computed from dataset {cert.inputs.get('dataset')}, not this dataset")
        if cert.inputs.get("representation") != hashes.get("representation"):
            problems.append(f"certificates were computed in representation {cert.inputs.get('representation')}, not "
                            f"{hashes.get('representation')}")
        if cands is not None and not (
                len(cert.source_frame) == len(cands.source_frame)
                and np.array_equal(cert.source_episode_id.astype(str), cands.source_episode_id.astype(str))
                and np.array_equal(cert.source_frame, cands.source_frame)):
            problems.append("certificate rows do not match the candidate rows in identity and order")
        if not problems:
            from .core.certificates import verify_structure
            structural = verify_structure(cert, candidates)
            problems += [f"certificate structure: {p}" for p in structural]
    if action_chunks == "consume" and cands is None and chunk_size is None:
        problems.append("action_chunks='consume' without candidates needs chunk_size")
    if action_chunks == "consume" and cands is not None and cands.action_chunks is None:
        problems.append("action_chunks='consume' but the candidates carry no chunks")
    if chunk_size is not None and (cands is not None or action_chunks != "consume"):
        problems.append("chunk_size applies only to demonstration-only arms that consume chunks")
    if problems:
        raise SupervisionError(problems)

    # Demonstrations: transition frames of the training episodes.
    keep = np.ones(len(batch.frame), bool) if train_eps is None else np.isin(batch.episode_id, sorted(train_eps))
    n_demo = int(keep.sum())
    parts = {"observations": [np.asarray(batch.z[keep])], "actions": [np.asarray(batch.a[keep])],
             "weights": [np.full(n_demo, float(demonstration_weight))], "source": [np.full(n_demo, DEMONSTRATION)],
             "episode_id": [batch.episode_id[keep].astype(str)], "frame": [batch.frame[keep].astype(np.int64)],
             "candidate_row": [np.full(n_demo, -1, np.int64)], "classification": [np.full(n_demo, DEMONSTRATION)]}
    K = None
    if action_chunks == "consume":
        K = int(chunk_size) if cands is None else int(cands.action_chunks.shape[1])
        dc, dm = _demonstration_chunks(ds, parts["episode_id"][0], parts["frame"][0], K)
        parts["action_chunks"], parts["action_chunk_mask"] = [dc], [dm]

    counts = {"demonstration_transitions": int(len(batch.frame)), "demonstration_rows": n_demo,
              "demonstration_rows_held_out": int((~keep).sum())}
    certification = None
    if cands is not None:
        n = len(cands.accepted)
        accepted = np.asarray(cands.accepted, bool)
        usable = accepted.copy()
        counts.update(candidate_rows=n, accepted=int(accepted.sum()))
        if cert is not None:
            admissible, selected = np.asarray(cert.admissible, bool), np.asarray(cert.selected, bool)
            usable &= admissible & selected
            counts.update(accepted_and_admissible=int((accepted & admissible).sum()),
                          accepted_admissible_and_selected=int(usable.sum()))
            classification = cert.classification
            certification = {"profile": cert.descriptor["profile"], "version": cert.descriptor["version"],
                             "admission": cert.admission, "selection": cert.selection}
        else:
            classification = np.full(n, "uncertified")
            certification = "uncertified"
        in_train = np.ones(n, bool) if train_eps is None else np.isin(cands.source_episode_id.astype(str), sorted(train_eps))
        rows = np.flatnonzero(usable & in_train)
        counts.update(usable_rows_held_out=int((usable & ~in_train).sum()), synthetic_rows=int(len(rows)))
        parts["observations"].append(np.asarray(cands.observations[modality][rows]))
        parts["actions"].append(np.asarray(cands.actions[rows]))
        parts["weights"].append(np.asarray(cands.weight[rows], np.float64))
        parts["source"].append(np.full(len(rows), SYNTHETIC))
        parts["episode_id"].append(cands.source_episode_id[rows].astype(str))
        parts["frame"].append(np.asarray(cands.source_frame[rows], np.int64))
        parts["candidate_row"].append(rows.astype(np.int64))
        parts["classification"].append(np.asarray(classification)[rows].astype(str))
        if K is not None:
            parts["action_chunks"].append(cands.action_chunks[rows])
            parts["action_chunk_mask"].append(cands.action_chunk_mask[rows])

    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    arrays["weights"] = arrays["weights"].astype(np.float64)
    # Reconciliation: every count must close, or the supervision is not built.
    total = n_demo + counts.get("synthetic_rows", 0)
    closes = [len(v) == total for v in arrays.values()]
    closes.append(counts["demonstration_rows"] + counts["demonstration_rows_held_out"] == counts["demonstration_transitions"])
    if cands is not None:
        usable_total = counts.get("accepted_admissible_and_selected", counts["accepted"])
        closes.append(counts["synthetic_rows"] + counts["usable_rows_held_out"] == usable_total)
    if not all(closes):
        raise SupervisionError([f"row counts do not reconcile: {counts}"])
    counts["rows"] = total
    manifest = {"schema": "ccil.supervision-manifest", "version": "1.0", "method": method,
                "certification": certification, "demonstration_weight": float(demonstration_weight),
                "demonstration_frames": "transition frames t <= T - 2 of the training episodes",
                "action_chunks": {"policy": action_chunks, "chunk_size": K},
                "representation": {"kind": str(getattr(representation, "kind", "?")),
                                   "version": str(getattr(representation, "version", "?")), "modality": modality},
                "inputs": hashes, "counts": counts,
                "online_perturbation": "noise_bc: applied by the trainer at every draw" if method == "noise_bc" else None}
    return Supervision(arrays["observations"], arrays["actions"], arrays["weights"], arrays["source"], arrays["episode_id"],
                       arrays["frame"], arrays["candidate_row"], arrays["classification"], manifest,
                       arrays.get("action_chunks"), arrays.get("action_chunk_mask"))


__all__ = ["build_supervision", "Supervision", "SupervisionError", "METHODS", "CHUNK_POLICIES"]
