"""Candidate schema `ccil.candidates` 1.0.

One row per attempted candidate, accepted or rejected. A row is accepted
exactly when it carries no rejection reason; a rejected row may carry several
independent reasons. Source identity (episode_id, frame) and measurements are
required on every row so accepted labels can always be traced and replayed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .fields import ArraySpec, Violations, check_finite
from .trajectories import MODALITY, TrajectoryDataset

# Candidates may also live in a learned representation space (latent.<name>), so
# image methods emit labels without a new schema; trajectories never store latents.
CANDIDATE_MODALITY = re.compile(r"^(?:(state|image)(\.[a-z0-9_]+)?|latent\.[a-z0-9_]+)$")

SCHEMA = "ccil.candidates"
VERSION = (1, 0)
REASON_CODES = ("non_finite", "out_of_support", "generation_failed", "action_out_of_bounds")

SOURCE_EPISODE = ArraySpec("source_episode_id", "str", ("N",), "-", "Episode the candidate perturbs.")
SOURCE_FRAME = ArraySpec("source_frame", "int", ("N",), "frames", "Transition index t in the source episode (t <= T-2).")
ACTIONS = ArraySpec("actions", "float", ("N", "A"), "task-defined", "Candidate action.")
ACCEPTED = ArraySpec("accepted", "bool", ("N",), "-", "True exactly when the row has no rejection reason.")
REASONS = ArraySpec("reasons", "bool", ("N", "R"), "-", "reasons[i, r] is True when reason_codes[r] rejects row i.")
MEASUREMENT = ArraySpec("measurements[name]", "float", ("N",), "per measurement",
                        "Validity measurements (e.g. distance, residual); NaN allowed only on rejected rows.")
WEIGHT = ArraySpec("weight", "float", ("N",), "-", "Optional training weight; finite and >= 0, and 0 on rejected rows.",
                   required=False)
ARRAY_SPECS = [SOURCE_EPISODE, SOURCE_FRAME, ACTIONS, ACCEPTED, REASONS, MEASUREMENT, WEIGHT]
FIELDS = [
    ("run_id", "str", "Identifier of the run that produced the candidates."),
    ("generator", "dict", "name, version, and params_hash of the generator."),
    ("reason_codes", "tuple[str]", f"Column labels of reasons; each one of {list(REASON_CODES)}."),
    ("observations", "dict[str, array (N, ...)]", "Candidate observation per modality: a trajectory modality, or latent.<name> for labels in a learned representation."),
    ("target_observations", "dict[str, array (N, ...)]", "Observation the candidate action should reach, per modality."),
]


@dataclass
class CandidateSet:
    run_id: str
    generator: dict
    source_episode_id: np.ndarray
    source_frame: np.ndarray
    observations: dict
    actions: np.ndarray
    target_observations: dict
    accepted: np.ndarray
    reasons: np.ndarray
    reason_codes: tuple = REASON_CODES
    measurements: dict = field(default_factory=dict)
    weight: np.ndarray | None = None

    def validate(self, dataset: TrajectoryDataset | None = None) -> "CandidateSet":
        """Raise SchemaError listing every violation; with dataset, also check source identity."""
        out = Violations(SCHEMA)
        if not isinstance(self.run_id, str) or not self.run_id:
            out.add("run_id", "candidates", "run_id must be a non-empty string")
        if not isinstance(self.generator, dict) or any(
                not isinstance(self.generator.get(k), str) or not self.generator.get(k) for k in ("name", "version", "params_hash")):
            out.add("generator", "candidates", "generator needs non-empty string name, version, and params_hash")
        codes = tuple(self.reason_codes)
        unknown = [c for c in codes if c not in REASON_CODES]
        if unknown or len(set(codes)) != len(codes):
            out.add("reason_codes", "candidates", f"reason_codes must be distinct members of {list(REASON_CODES)}; got {list(codes)}")
        dims = {"R": len(codes)}
        ok = all([SOURCE_EPISODE.check(self.source_episode_id, "candidates", dims, out),
                  SOURCE_FRAME.check(self.source_frame, "candidates", dims, out),
                  ACCEPTED.check(self.accepted, "candidates", dims, out),
                  REASONS.check(self.reasons, "candidates", dims, out),
                  ACTIONS.check(self.actions, "candidates", dims, out, finite=False)])
        if not ok:
            out.raise_if_any()
        n = dims["N"]
        rejected = ~self.accepted
        if (self.source_frame < 0).any():
            out.add("source_frame", "candidates", "source_frame must be >= 0")
        if (self.source_episode_id == "").any():
            out.add("source_identity", "candidates", "every row needs a source_episode_id")
        mismatch = self.accepted == self.reasons.any(axis=1)
        for i in np.flatnonzero(mismatch)[:5]:
            out.add("accepted_reasons", f"row {int(i)}", "accepted must be True exactly when the row has no reason")
        if self.accepted.any():
            check_finite(self.actions, "actions", "accepted rows", out, rows=self.accepted)
        for label, mods in (("observations", self.observations), ("target_observations", self.target_observations)):
            if not isinstance(mods, dict) or not mods:
                out.add("structure", "candidates", f"{label} must be a non-empty dict of modalities")
                continue
            for name, x in mods.items():
                if not isinstance(name, str) or not CANDIDATE_MODALITY.match(name):
                    out.add("modality_name", "candidates", f"{label} modality {name!r} is not a valid modality name")
                elif not isinstance(x, np.ndarray) or len(x) != n:
                    out.add("shape", f"{label}[{name}]", f"expected an array with {n} rows")
                elif self.accepted.any() and x.dtype.kind == "f":
                    check_finite(x, f"{label}[{name}]", "accepted rows", out, rows=self.accepted)
        if isinstance(self.observations, dict) and isinstance(self.target_observations, dict) \
                and set(self.observations) != set(self.target_observations):
            out.add("modality_mismatch", "candidates", "observations and target_observations need the same modalities")
        if not isinstance(self.measurements, dict):
            out.add("structure", "candidates", "measurements must be a dict")
        else:
            for name, x in self.measurements.items():
                if MEASUREMENT.check(x, f"measurements[{name}]", {"N": n}, out, finite=False) and self.accepted.any():
                    check_finite(x[:, None], f"measurements[{name}]", "accepted rows", out, rows=self.accepted)
        if self.weight is not None and WEIGHT.check(self.weight, "weight", {"N": n}, out, finite=False):
            acc = self.weight[self.accepted]
            if not (np.isfinite(acc).all() and (acc >= 0).all()):
                out.add("weight", "accepted rows", "weights must be finite and >= 0")
            if (self.weight[rejected] != 0).any():
                out.add("weight", "rejected rows", "rejected rows must have weight 0")
        if dataset is not None:
            lengths = {ep.episode_id: ep.length for ep in dataset.episodes}
            for i in range(n):
                T = lengths.get(str(self.source_episode_id[i]))
                if T is None:
                    out.add("source_identity", f"row {i}", f"unknown source episode {self.source_episode_id[i]!r}")
                elif not 0 <= self.source_frame[i] <= T - 2:
                    out.add("source_identity", f"row {i}", f"source_frame {int(self.source_frame[i])} is not a transition of a {T}-frame episode")
        out.raise_if_any()
        return self

    def row_reasons(self, i: int) -> list[str]:
        return [c for c, on in zip(self.reason_codes, self.reasons[i]) if on]

    def accepted_labels(self) -> "CandidateSet":
        """The accepted rows, keeping source identity, generator, and measurements."""
        m = self.accepted
        return CandidateSet(
            run_id=self.run_id, generator=dict(self.generator), source_episode_id=self.source_episode_id[m],
            source_frame=self.source_frame[m], observations={k: v[m] for k, v in self.observations.items()},
            actions=self.actions[m], target_observations={k: v[m] for k, v in self.target_observations.items()},
            accepted=self.accepted[m], reasons=self.reasons[m], reason_codes=tuple(self.reason_codes),
            measurements={k: v[m] for k, v in self.measurements.items()},
            weight=None if self.weight is None else self.weight[m]).validate()
