"""Selection schema `ccil.selection` 1.0: which candidate rows a consumer may use, and why.

A selection is the output of one curation policy applied to one candidate set
and, optionally, its certificates (#167). It never modifies either: changing
a tolerance writes a new selection and leaves both hashes unchanged. One row
per candidate row, in the same order, with the source identity repeated for
joining.

Per row it records the mask and weight, the raw error the policy compared
(the certificate's total_error_value, in the certificate's units), the
error-to-tolerance ratio, the relative rank of that error within the named
population, and every reason the row was not selected. A ratio of at most one
means exactly that the recorded error is within the declared tolerance, which
is strictly positive; it is a bound only when the error is (`error.is_bound`).
Ranks are relative to the named population of this candidate set, not
universal quantities.

evidence_mode is provenance: reference_assisted exactly when reference
dynamics computed a constant or residual of the certificate the selection
used. evidence_allowed is what the curation run permitted; a data_only
allowance never records reference-assisted evidence.

Version 1.1 adds calibration: a selection curated with a ccil.calibration
records its SHA-256 and which rows it calibrated. A calibrated row's error is
its calibrated bound, a coverage statement at the calibration's level rather
than a worst-case bound, so such a selection is always reference_assisted and
admits bounds only. Without a calibration a selection is written as 1.0, and
1.0 files are read with no calibrated rows.

A consumer trains on candidate.accepted & mask, with weight.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .fields import ArraySpec, Violations

SCHEMA = "ccil.selection"
VERSION = (1, 1)
BASE_VERSION = (1, 0)  # written when no calibration is used
ADMIT = ("accepted", "bound", "score")
POPULATIONS = ("admitted", "produced")
EVIDENCE_MODES = ("data_only", "reference_assisted")
USES = ("fidelity", "novelty")
REASON_CODES = ("selection.not_accepted", "selection.unsupported", "selection.not_bound", "selection.exceeds_tolerance",
                "selection.not_selected")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

SOURCE_EPISODE = ArraySpec("source_episode_id", "str", ("N",), "-", "Episode of the candidate row; equals the candidates file's.")
SOURCE_FRAME = ArraySpec("source_frame", "int", ("N",), "frames", "Transition index of the candidate row; equals the candidates file's.")
MASK = ArraySpec("mask", "bool", ("N",), "-", "The row may be used: accepted, and passed every rule of the policy.")
WEIGHT = ArraySpec("weight", "float", ("N",), "-", "Loss weight: the candidate's weight on selected rows, 0 elsewhere.")
ERROR = ArraySpec("error", "float", ("N",), "per error.units",
                  "The raw error the policy compared: the certificate's total_error_value. NaN without certificates and "
                  "on unsupported rows.")
RATIO = ArraySpec("ratio", "float", ("N",), "-", "error / tolerance (tolerance > 0); NaN exactly without a tolerance or a finite error.")
RANK = ArraySpec("rank", "float", ("N",), "-",
                 "Relative rank of error within the named population, in [0, 1] (the fraction of that population with "
                 "error at most this row's); NaN outside it.")
REASONS = ArraySpec("reasons", "bool", ("N", "R"), "-", "reasons[i, r] is True when reason_codes[r] applies to row i.")
CALIBRATED = ArraySpec("calibrated", "bool", ("N",), "-",
                       "1.1, with a calibration: the row's error is its calibrated bound (an empirical row in the "
                       "calibration's regime). Absent in 1.0.")
ARRAY_SPECS = [SOURCE_EPISODE, SOURCE_FRAME, MASK, WEIGHT, ERROR, RATIO, RANK, REASONS, CALIBRATED]
FIELDS = [
    ("policy", "dict", f"admit {list(ADMIT)}, tolerance (null or finite > 0, in error.units), quantile (null or in "
                       f"(0, 1]), and population {list(POPULATIONS)} the quantile is taken over."),
    ("policy_version", "str", "Version of the curation rules that produced the selection."),
    ("uses", "list[str]", f"Quality terms the policy compares, from {list(USES)}."),
    ("caveats", "list[str]", "What the policy does not establish, e.g. that a fidelity-only policy favours labels "
                             "close to the demonstrations."),
    ("error", "dict | null", "definition, units, and is_bound of the error column; null without certificates."),
    ("threshold", "float | null", "The error threshold the quantile resolved to; null without a quantile or a population."),
    ("evidence_mode", "str", f"{list(EVIDENCE_MODES)}: whether reference dynamics shaped the evidence used (provenance)."),
    ("evidence_allowed", "str", f"{list(EVIDENCE_MODES)}: the evidence the curation run permitted; data_only permits no "
                                "reference-assisted evidence."),
    ("candidates_sha256", "str", "SHA-256 of the candidates file."),
    ("certificates_sha256", "str | null", "SHA-256 of the certificates file; null for admit accepted."),
    ("calibration_sha256", "str | null", "1.1: SHA-256 of the ccil.calibration used; null or absent without one."),
    ("reason_codes", "list[str]", f"Exactly {list(REASON_CODES)}."),
]


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class SelectionSet:
    source_episode_id: np.ndarray
    source_frame: np.ndarray
    mask: np.ndarray
    weight: np.ndarray
    error: np.ndarray
    ratio: np.ndarray
    rank: np.ndarray
    reasons: np.ndarray               # (N, len(REASON_CODES)) bool
    policy: dict
    policy_version: str
    uses: list
    caveats: list
    error_spec: dict | None
    threshold: float | None
    evidence_mode: str
    evidence_allowed: str
    candidates_sha256: str
    certificates_sha256: str | None
    calibrated: np.ndarray | None = None  # None: no calibration (all False)
    calibration_sha256: str | None = None

    @property
    def version(self) -> tuple:
        return VERSION if self.calibration_sha256 is not None else BASE_VERSION

    def validate(self) -> "SelectionSet":
        """Raise SchemaError listing every violation of the structure and its invariants."""
        out = Violations(SCHEMA)
        w = "selection"
        dims: dict = {}
        ok = all([SOURCE_EPISODE.check(self.source_episode_id, w, dims, out),
                  SOURCE_FRAME.check(self.source_frame, w, dims, out), MASK.check(self.mask, w, dims, out),
                  WEIGHT.check(self.weight, w, dims, out),
                  *(s.check(x, w, dims, out, finite=False) for s, x in
                    ((ERROR, self.error), (RATIO, self.ratio), (RANK, self.rank)))])
        ok = REASONS.check(self.reasons, w, {**dims, "R": len(REASON_CODES)}, out) and ok
        if self.calibrated is None and "N" in dims:
            self.calibrated = np.zeros(dims["N"], bool)
        ok = CALIBRATED.check(self.calibrated, w, dims, out) and ok
        self._check_meta(out)
        if ok and not out.items:
            self._check_rows(out)
        out.raise_if_any()
        return self

    def _check_meta(self, out):
        w = "selection"
        p = self.policy
        if not (isinstance(p, dict) and set(p) == {"admit", "tolerance", "quantile", "population"} and p["admit"] in ADMIT
                and (p["tolerance"] is None or (_number(p["tolerance"]) and p["tolerance"] > 0))
                and (p["quantile"] is None or (_number(p["quantile"]) and 0 < p["quantile"] <= 1))
                and p["population"] in POPULATIONS and not (p["quantile"] is None and p["population"] != "admitted")):
            out.add("policy", w, f"policy must be {{admit: {list(ADMIT)}, tolerance: null or finite > 0, quantile: null or "
                                 f"in (0, 1], population: {list(POPULATIONS)}}}, with a non-default population only "
                                 "with a quantile")
            return
        accepted_only = p["admit"] == "accepted"
        if accepted_only and (p["tolerance"] is not None or p["quantile"] is not None):
            out.add("policy", w, "admit accepted compares no error: it takes no tolerance or quantile")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            out.add("policy_version", w, "policy_version must be a non-empty string")
        if not (isinstance(self.uses, list) and all(u in USES for u in self.uses) and len(set(self.uses)) == len(self.uses)):
            out.add("uses", w, f"uses must be distinct names from {list(USES)}")
        if not (isinstance(self.caveats, list) and all(isinstance(c, str) and c for c in self.caveats)):
            out.add("caveats", w, "caveats must be a list of non-empty strings")
        e = self.error_spec
        if accepted_only:
            if e is not None:
                out.add("error", w, "admit accepted has no error column")
        elif not (isinstance(e, dict) and set(e) == {"definition", "units", "is_bound"}
                  and isinstance(e["definition"], str) and e["definition"] and isinstance(e["units"], str) and e["units"]
                  and isinstance(e["is_bound"], bool)):
            out.add("error", w, "error must be {definition, units, is_bound}")
        elif e["is_bound"] != (p["admit"] == "bound"):
            out.add("error", w, "error.is_bound is true exactly when the policy admits bounds only")
        t = self.threshold
        if not (t is None or (_number(t) and t >= 0)) or (p["quantile"] is None and t is not None):
            out.add("threshold", w, "threshold must be null, or finite >= 0 with a quantile")
        if self.evidence_mode not in EVIDENCE_MODES or self.evidence_allowed not in EVIDENCE_MODES:
            out.add("evidence_mode", w, f"evidence_mode and evidence_allowed must be one of {list(EVIDENCE_MODES)}")
        elif self.evidence_mode == "reference_assisted" and (self.evidence_allowed == "data_only" or accepted_only):
            out.add("evidence_mode", w, "reference-assisted evidence needs a certificate and a reference_assisted allowance")
        if not (isinstance(self.candidates_sha256, str) and SHA256.match(self.candidates_sha256)):
            out.add("inputs", w, "candidates_sha256 must be 64 lowercase hex")
        c = self.certificates_sha256
        if accepted_only != (c is None) or not (c is None or (isinstance(c, str) and SHA256.match(c))):
            out.add("inputs", w, "certificates_sha256 is 64 lowercase hex for admit bound or score, and null for accepted")
        k = self.calibration_sha256
        if k is not None and not (isinstance(k, str) and SHA256.match(k)):
            out.add("calibration", w, "calibration_sha256 must be null or 64 lowercase hex")
        elif k is not None and (p["admit"] != "bound" or self.evidence_mode != "reference_assisted"):
            out.add("calibration", w, "a calibrated selection admits bounds only and is reference_assisted")

    def _check_rows(self, out):
        """Invariants between the arrays; recomputing them from the inputs is verify_selection's job."""
        w = "selection"
        rule = lambda name, bad, detail: [out.add(name, f"{w}, row {int(i)}", detail) for i in np.flatnonzero(bad)]  # noqa: E731
        rule("mask", self.mask & self.reasons.any(axis=1), "a selected row has a reason against it")
        rule("mask", ~self.mask & ~self.reasons.any(axis=1), "an unselected row needs a reason")
        rule("weight", ~(np.isfinite(self.weight) & (self.weight >= 0)), "weight must be finite and >= 0")
        rule("weight", ~self.mask & (self.weight != 0), "weight must be 0 on unselected rows")
        if self.calibration_sha256 is None:
            rule("calibrated", self.calibrated, "a row is calibrated only through a recorded calibration")
        rule("calibrated", self.calibrated & self.reasons[:, 1:3].any(axis=1), "a calibrated row is a supported bound")
        with np.errstate(invalid="ignore"):
            rule("rank", ~np.isnan(self.rank) & ~((self.rank >= 0) & (self.rank <= 1)), "rank must be in [0, 1] or NaN")
            rule("error", ~np.isnan(self.error) & ~(np.isfinite(self.error) & (self.error >= 0)),
                 "error must be finite and >= 0, or NaN")
        tol = self.policy["tolerance"]
        with np.errstate(invalid="ignore"):
            rule("ratio", ~np.isnan(self.ratio) & ~(np.isfinite(self.ratio) & (self.ratio >= 0)),
                 "ratio must be finite and >= 0, or NaN")
        rule("ratio", np.isnan(self.ratio) != (np.isnan(self.error) | (tol is None)),
             "ratio is NaN exactly without a tolerance or a finite error")
        if self.policy["admit"] == "accepted":
            rule("error", ~np.isnan(self.error) | ~np.isnan(self.ratio) | ~np.isnan(self.rank),
                 "admit accepted records no error, ratio, or rank")
            rule("reasons", self.reasons[:, 1:].any(axis=1), "admit accepted can only reject unaccepted rows")
