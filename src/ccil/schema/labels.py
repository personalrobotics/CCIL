"""Candidate schema `ccil.candidates` 2.0.

One row per attempted candidate, accepted or rejected. A row is accepted
exactly when it carries no rejection reason; a rejected row may carry several
independent reasons. Source identity (episode_id, frame), measurements, and a
training weight are required on every row, and the file records the run, seed,
generator, input hashes (always including the transition model), and weighting
that produced it, so an accepted label can be traced and replayed from the
candidate file alone.

Reason columns are the built-in codes, whose meanings are fixed, and
namespaced extension codes <namespace>.<reason> that a method's own validators
declare (the ccil namespace is reserved). Built-ins come first in their
canonical order, then extensions sorted. A consumer that does not know an
extension's namespace still treats the row as rejected: accepted is True
exactly when no column, built-in or extension, rejects the row.

2.0 made weight, seed, inputs, and weighting required (1.0 had none of them).
2.1 added extension reason codes; a file declares 2.1 only when it uses one, so
files with built-in reasons alone stay 2.0. 2.2 added the optional
measurement_specs, the definition, norm, and units of every measurement, and the
generator's full params; a file declares 2.2 only when it carries specs.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np

from .fields import ArraySpec, Violations, check_finite
from .trajectories import MODALITY, TrajectoryDataset

# Candidates may also live in a learned representation space (latent.<name>), so
# image methods emit labels without a new schema; trajectories never store latents.
CANDIDATE_MODALITY = re.compile(r"^(?:(state|image)(\.[a-z0-9_]+)?|latent\.[a-z0-9_]+)$")

SCHEMA = "ccil.candidates"
VERSION = (2, 2)
REASON_CODES = ("non_finite", "out_of_support", "generation_failed", "action_out_of_bounds")
EXTENSION_REASON = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
RESERVED_NAMESPACES = ("ccil",)


def reason_code_problem(code) -> str | None:
    """Why code is neither a built-in reason nor a valid extension, or None."""
    if not isinstance(code, str):
        return f"{code!r} is not a string"
    if code in REASON_CODES:
        return None
    if not EXTENSION_REASON.match(code):
        return (f"{code!r} is neither a built-in reason {list(REASON_CODES)} nor an extension "
                f"<namespace>.<reason> matching {EXTENSION_REASON.pattern}")
    if code.split(".")[0] in RESERVED_NAMESPACES:
        return f"{code!r} uses the reserved namespace {code.split('.')[0]!r}"
    return None


def canonical_reason_order(codes) -> tuple:
    """Built-in codes in REASON_CODES order, then extension codes sorted."""
    return tuple(c for c in REASON_CODES if c in codes) + tuple(sorted(c for c in codes if c not in REASON_CODES))


def file_version(codes, measurement_specs=None) -> tuple:
    """The lowest version that can hold these reason codes and measurement specs."""
    if measurement_specs is not None:
        return (2, 2)
    return (2, 1) if any(c not in REASON_CODES for c in codes) else (2, 0)

SOURCE_EPISODE = ArraySpec("source_episode_id", "str", ("N",), "-", "Episode the candidate perturbs.")
SOURCE_FRAME = ArraySpec("source_frame", "int", ("N",), "frames", "Transition index t in the source episode (t <= T-2).")
ACTIONS = ArraySpec("actions", "float", ("N", "A"), "task-defined", "Candidate action.")
ACCEPTED = ArraySpec("accepted", "bool", ("N",), "-", "True exactly when the row has no rejection reason.")
REASONS = ArraySpec("reasons", "bool", ("N", "R"), "-", "reasons[i, r] is True when reason_codes[r] rejects row i.")
MEASUREMENT = ArraySpec("measurements[name]", "float", ("N",), "per measurement",
                        "Validity measurements (e.g. distance, residual); NaN allowed only on rejected rows.")
WEIGHT = ArraySpec("weight", "float", ("N",), "-", "Training weight; finite and >= 0 on accepted rows, 0 on rejected rows.")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ARRAY_SPECS = [SOURCE_EPISODE, SOURCE_FRAME, ACTIONS, ACCEPTED, REASONS, MEASUREMENT, WEIGHT]
FIELDS = [
    ("run_id", "str", "Identifier of the run that produced the candidates."),
    ("generator", "dict", "name, version, and params_hash of the generator; optionally params, its full parameters."),
    ("seed", "int", "Seed the generator drew its randomness from."),
    ("inputs", "dict[str, str]", "SHA-256 (64 lowercase hex) of each input by name; always includes transition_model, the model checkpoint the labels came from."),
    ("weighting", "dict", "name, version, and params of the weighting that produced weight."),
    ("measurement_specs", "dict | None", "Optional (2.2): for every measurement name, its definition, norm (or null), and units."),
    ("reason_codes", "tuple[str]", f"Column labels of reasons, distinct and in canonical order: built-ins {list(REASON_CODES)} first, then extension codes <namespace>.<reason> sorted."),
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
    weight: np.ndarray
    seed: int
    inputs: dict
    weighting: dict
    reason_codes: tuple = REASON_CODES
    measurements: dict = field(default_factory=dict)
    measurement_specs: dict | None = None

    def validate(self, dataset: TrajectoryDataset | None = None) -> "CandidateSet":
        """Raise SchemaError listing every violation; with dataset, also check source identity."""
        out = Violations(SCHEMA)
        if not isinstance(self.run_id, str) or not self.run_id:
            out.add("run_id", "candidates", "run_id must be a non-empty string")
        if not isinstance(self.generator, dict) or any(
                not isinstance(self.generator.get(k), str) or not self.generator.get(k) for k in ("name", "version", "params_hash")):
            out.add("generator", "candidates", "generator needs non-empty string name, version, and params_hash")
        if isinstance(self.generator, dict) and "params" in self.generator and not isinstance(self.generator["params"], dict):
            out.add("generator", "candidates", "generator params, when present, must be a dict")
        if type(self.seed) is not int:
            out.add("seed", "candidates", f"seed must be an int, got {type(self.seed).__name__}")
        if not isinstance(self.inputs, dict) or "transition_model" not in self.inputs or any(
                not isinstance(k, str) or not k or not isinstance(h, str) or not SHA256.match(h) for k, h in self.inputs.items()):
            out.add("inputs", "candidates", "inputs must map non-empty names, including transition_model, to SHA-256 hex digests")
        if not isinstance(self.weighting, dict) or not isinstance(self.weighting.get("params"), dict) or any(
                not isinstance(self.weighting.get(k), str) or not self.weighting.get(k) for k in ("name", "version")):
            out.add("weighting", "candidates", "weighting needs non-empty string name and version and a params dict")
        codes = tuple(self.reason_codes) if isinstance(self.reason_codes, (tuple, list)) else ()
        if not isinstance(self.reason_codes, (tuple, list)):
            out.add("reason_codes", "candidates", f"reason_codes must be a sequence, got {type(self.reason_codes).__name__}")
        bad = [why for why in map(reason_code_problem, codes) if why]
        for why in bad:
            out.add("reason_codes", "candidates", why)
        if not bad:
            if len(set(codes)) != len(codes):
                out.add("reason_codes", "candidates", f"reason_codes must be distinct; got {list(codes)}")
            elif codes != canonical_reason_order(codes):
                out.add("reason_codes", "candidates", f"reason_codes must be in canonical order "
                                                      f"{list(canonical_reason_order(codes))}; got {list(codes)}")
        dims = {"R": len(codes)}
        ok = all([SOURCE_EPISODE.check(self.source_episode_id, "candidates", dims, out),
                  SOURCE_FRAME.check(self.source_frame, "candidates", dims, out),
                  ACCEPTED.check(self.accepted, "candidates", dims, out),
                  REASONS.check(self.reasons, "candidates", dims, out),
                  ACTIONS.check(self.actions, "candidates", dims, out, finite=False),
                  WEIGHT.check(self.weight, "candidates", dims, out, finite=False)])
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
                elif not isinstance(x, np.ndarray) or x.ndim == 0 or len(x) != n:
                    out.add("shape", f"{label}[{name}]", f"expected an array with {n} rows")
                elif self.accepted.any() and x.dtype.kind == "f":
                    check_finite(x, f"{label}[{name}]", "accepted rows", out, rows=self.accepted)
        if isinstance(self.observations, dict) and isinstance(self.target_observations, dict) \
                and set(self.observations) != set(self.target_observations):
            out.add("modality_mismatch", "candidates", "observations and target_observations need the same modalities")
        if self.measurement_specs is not None:
            specs = self.measurement_specs
            names = set(self.measurements) if isinstance(self.measurements, dict) else set()
            if not isinstance(specs, dict) or set(specs) != names:
                out.add("measurement_specs", "candidates", "measurement_specs must name exactly the measurements "
                                                           f"{sorted(names)}")
            else:
                for name, spec in specs.items():
                    if not (isinstance(spec, dict) and set(spec) == {"definition", "norm", "units"}
                            and isinstance(spec["definition"], str) and spec["definition"]
                            and isinstance(spec["units"], str) and spec["units"]
                            and (spec["norm"] is None or isinstance(spec["norm"], str))):
                        out.add("measurement_specs", f"measurement_specs[{name}]",
                                "needs a non-empty definition and units, and a norm string or null")
        if not isinstance(self.measurements, dict):
            out.add("structure", "candidates", "measurements must be a dict")
        else:
            for name, x in self.measurements.items():
                if MEASUREMENT.check(x, f"measurements[{name}]", {"N": n}, out, finite=False) and self.accepted.any():
                    check_finite(x[:, None], f"measurements[{name}]", "accepted rows", out, rows=self.accepted)
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
        """The accepted rows, keeping source identity, provenance, measurements, and weights."""
        m = self.accepted
        return CandidateSet(
            run_id=self.run_id, generator=dict(self.generator), source_episode_id=self.source_episode_id[m],
            source_frame=self.source_frame[m], observations={k: v[m] for k, v in self.observations.items()},
            actions=self.actions[m], target_observations={k: v[m] for k, v in self.target_observations.items()},
            accepted=self.accepted[m], reasons=self.reasons[m], weight=self.weight[m], seed=self.seed,
            inputs=dict(self.inputs), weighting=copy.deepcopy(self.weighting), reason_codes=tuple(self.reason_codes),
            measurements={k: v[m] for k, v in self.measurements.items()},
            measurement_specs=copy.deepcopy(self.measurement_specs)).validate()
