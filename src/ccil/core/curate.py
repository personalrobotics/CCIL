"""Curation: a selection over a candidate set and its certificates (#167).

    generate candidates -> certify candidates -> curate a selection -> train on accepted & mask

A curation policy reads the immutable candidates and, optionally, their
certificates, and writes a separate ccil.selection (schema/selection.py). It
never modifies either input, so changing a tolerance or policy writes a new
selection while the candidate and certificate hashes stay the same.

Policies (CurationPolicy.admit):
- accepted: every accepted row; no certificates, no error compared.
- bound: rows whose certificate is an upper bound (certified, conditional,
  calibrated) and whose bound is at most the tolerance.
- score: an explicit opt-in that admits supported rows whose empirical score
  is at most the tolerance; its rows are never presented as bounds.
A quantile then keeps the admitted rows whose error is at most that quantile
of a declared population: the accepted, admitted rows (admitted), or every
supported row a legacy run would have generated (produced), as the
label-filtering branch did. It is a data budget, not a quality claim.

Unsupported rows (outside the certificate's neighborhood or without a finite
total) are never selected. Selection is monotone: a lower tolerance or
quantile selects a subset. Every policy here compares one quality term, the
certificate's error, so it records the caveat that fidelity alone favours
labels that stay close to the demonstrations.

Evidence: a certificate whose constants or demonstration residuals come from
reference dynamics is reference_assisted evidence. curate's allowance
(allow_evidence) says what it may use; data_only, the default, refuses
reference-assisted evidence. The selection records both the allowance and the
provenance of the evidence it actually used, which is data_only whenever the
certificate is data-only or there is none.

Calibration: a ccil.calibration of the same certificate file
(core/calibrate.py) turns the empirical rows in its regime into calibrated
rows, whose error is their calibrated bound, for a bound policy. Its
observed errors come from reference dynamics, so it needs a reference_assisted
allowance and the selection records it (ccil.selection 1.1).

Before deciding anything, curate verifies the certificate's structure against
the candidates (core/certificates.py verify_structure): a certificate whose
totals, classification, or reasons do not follow from its persisted terms
and descriptor is refused. The source audit (audit_sources) remains the
stronger, separate check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..runs.provenance import sha256
from ..schema import SchemaError, read_candidates, read_certificates
from ..schema.certificates import BOUND_CLASSES
from ..schema.selection import ADMIT, EVIDENCE_MODES, POPULATIONS, REASON_CODES, SelectionSet
from .batch import InterfaceError

POLICY_VERSION = "1"
FIDELITY_CAVEAT = ("fidelity only: the policy compares label error and no novelty term, so it favours labels close to "
                   "the demonstrations")
DATA_ONLY_CAVEAT = "data-only evidence supports curation; it does not establish downstream policy improvement"
# Candidate reasons that mean a row was never produced in the legacy sense: legacy generated no label for it.
NOT_PRODUCED = ("generation_failed", "non_finite", "label_filter.model_misfit")


@dataclass(frozen=True)
class CurationPolicy:
    admit: str = "bound"
    tolerance: float | None = None
    quantile: float | None = None
    population: str = "admitted"

    def __post_init__(self):
        problems = []
        if self.admit not in ADMIT:
            problems.append(f"admit must be one of {list(ADMIT)}, got {self.admit!r}")
        if self.tolerance is not None and not (_number(self.tolerance) and self.tolerance > 0):
            problems.append(f"tolerance must be finite and > 0 or None, got {self.tolerance!r}")
        if self.quantile is not None and not (_number(self.quantile) and 0 < self.quantile <= 1):
            problems.append(f"quantile must be in (0, 1] or None, got {self.quantile!r}")
        if self.population not in POPULATIONS:
            problems.append(f"population must be one of {list(POPULATIONS)}, got {self.population!r}")
        elif self.population != "admitted" and self.quantile is None:
            problems.append("a population other than admitted applies to a quantile")
        if self.admit == "accepted" and (self.tolerance is not None or self.quantile is not None):
            problems.append("admit accepted compares no error: it takes no tolerance or quantile")
        if problems:
            raise InterfaceError("; ".join(problems))

    def spec(self) -> dict:
        return {"admit": self.admit, "tolerance": None if self.tolerance is None else float(self.tolerance),
                "quantile": None if self.quantile is None else float(self.quantile), "population": self.population}


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def produced_rows(candidates) -> np.ndarray:
    """Rows a legacy run would have generated: not rejected for generation failure, non-finite values, or model misfit."""
    cols = [i for i, c in enumerate(candidates.reason_codes) if c in NOT_PRODUCED]
    reasons = np.asarray(candidates.reasons, bool)
    return ~reasons[:, cols].any(axis=1) if cols else np.ones(len(reasons), bool)


def admit_rows(classification, total, accepted, admit: str, tolerance, quantile, population=None) -> dict:
    """The masks of one policy from a classification and its totals.

    Shared by curate() and the verification of legacy 1.0 certificates, which
    stored the same decision. population is a row mask (the produced
    population) or None (the accepted, admitted rows).
    """
    cls = np.asarray(classification)
    total = np.asarray(total, np.float64)
    supported = cls != "unsupported"
    bound = np.isin(cls, BOUND_CLASSES)
    candidate = bound if admit == "bound" else supported
    with np.errstate(invalid="ignore"):
        within = np.ones(len(total), bool) if tolerance is None else total <= tolerance
    admissible = candidate & within
    eligible = np.asarray(accepted, bool) & admissible
    pool = eligible if population is None else np.asarray(population, bool) & supported
    threshold = None
    if quantile is None:
        selected = np.ones(len(total), bool)
    elif pool.any():
        threshold = float(np.quantile(total[pool], quantile))
        selected = eligible & (total <= threshold)
    else:
        selected = np.zeros(len(total), bool)
    return {"supported": supported, "bound": bound, "candidate": candidate, "within": within, "admissible": admissible,
            "eligible": eligible, "pool": pool, "selected": selected, "threshold": threshold}


def _ranks(total, pool) -> np.ndarray:
    """Fraction of the pool with total at most each pool row's; NaN outside the pool."""
    rank = np.full(len(total), np.nan)
    ref = np.sort(total[pool])
    if len(ref):
        rank[pool] = np.searchsorted(ref, total[pool], side="right") / len(ref)
    return rank


def reference_assisted(cert) -> bool:
    """Whether reference dynamics computed any of the certificate's constants or demonstration residuals."""
    from .certificates import PROVIDERS
    d = cert.descriptor
    entries = list(d.get("constants", {}).values()) + ([d["demonstration"]] if "demonstration" in d else [])
    for e in entries:
        p = e.get("provenance")
        if p and p.get("provider") not in PROVIDERS:
            raise InterfaceError(f"the certificate names an unregistered provider {p.get('provider')!r}")
    return any(e.get("provenance") and PROVIDERS[e["provenance"]["provider"]].needs_reference for e in entries)


def decide_selection(candidates, cert, policy: CurationPolicy, candidates_sha256: str, certificates_sha256: str | None,
                     allow_evidence: str = "data_only", calibration: dict | None = None,
                     calibration_sha256: str | None = None) -> SelectionSet:
    """The selection one policy makes, from loaded candidates, certificates, and calibration; pure arithmetic.

    The recorded evidence mode is the evidence's provenance; reference-assisted
    evidence (certificate constants, or any calibration) under a data_only
    allowance raises InterfaceError.
    """
    from .calibrate import calibrated_rows, matching_calibration, structure_problems
    if allow_evidence not in EVIDENCE_MODES:
        raise InterfaceError(f"allow_evidence must be one of {list(EVIDENCE_MODES)}, got {allow_evidence!r}")
    if calibration is not None:
        if policy.admit != "bound":
            raise InterfaceError(f"a calibration gives bounds; it applies to admit bound, not {policy.admit}")
        matching_calibration(calibration, candidates_sha256, certificates_sha256)
        problems = structure_problems(calibration, candidates, cert)
        if problems:
            raise InterfaceError("the calibration does not verify against its certificate: " + "; ".join(problems))
        if allow_evidence == "data_only":
            raise InterfaceError("a calibration's observed errors come from reference dynamics, which data_only curation "
                                 "cannot use; allow reference_assisted evidence (generation.evidence_mode)")
    evidence_mode = ("reference_assisted" if calibration is not None or (cert is not None and reference_assisted(cert))
                     else "data_only")
    if evidence_mode == "reference_assisted" and allow_evidence == "data_only":
        raise InterfaceError("the certificate's constants come from reference dynamics, which data_only curation cannot "
                             "use; allow reference_assisted evidence (generation.evidence_mode)")
    accepted = np.asarray(candidates.accepted, bool)
    n = len(accepted)
    reasons = np.zeros((n, len(REASON_CODES)), bool)
    reasons[:, 0] = ~accepted
    nan = np.full(n, np.nan)
    calibrated = np.zeros(n, bool)
    if policy.admit == "accepted":
        mask = accepted
        error, ratio, rank, threshold, error_spec, uses = nan, nan.copy(), nan.copy(), None, None, []
        caveats = ["uncertified: rows are selected by acceptance alone; no error is compared"]
    else:
        total = np.asarray(cert.total_error_value, np.float64)
        classification = np.asarray(cert.classification)
        if calibration is not None:
            calibrated, bound = calibrated_rows(calibration, candidates, cert)
            total = np.where(calibrated, bound, total)
            classification = np.where(calibrated, "calibrated", classification)
        rows = admit_rows(classification, total, accepted, policy.admit, policy.tolerance, policy.quantile,
                          produced_rows(candidates) if policy.population == "produced" else None)
        supported, threshold = rows["supported"], rows["threshold"]
        mask = rows["eligible"] & rows["selected"]
        reasons[:, 1] = ~supported
        reasons[:, 2] = supported & ~rows["candidate"]
        reasons[:, 3] = rows["candidate"] & ~rows["within"]
        reasons[:, 4] = rows["eligible"] & ~rows["selected"]
        error = np.where(supported, total, np.nan)
        ratio = nan.copy() if policy.tolerance is None else np.where(supported, total / policy.tolerance, np.nan)
        rank = _ranks(total, rows["pool"] if policy.quantile is not None else rows["eligible"])
        units = cert.descriptor["units"]["error"]
        definition = "the certificate's total_error_value: " + ("an upper bound on ||T(q) - y||" if policy.admit == "bound"
                                                                 else "an empirical score, not a bound")
        if calibration is not None:
            definition += (f"; on calibrated rows, the calibration's bound, which covers ||T(q) - y|| with probability at "
                           f"least {calibration['level']} (not a worst-case bound)")
        error_spec = {"definition": definition, "units": units, "is_bound": policy.admit == "bound"}
        uses, caveats = ["fidelity"], [FIDELITY_CAVEAT]
    if evidence_mode == "data_only":
        caveats = caveats + [DATA_ONLY_CAVEAT]
    weight = np.where(mask, np.asarray(candidates.weight, np.float64), 0.0)
    return SelectionSet(
        source_episode_id=np.asarray(candidates.source_episode_id), source_frame=np.asarray(candidates.source_frame),
        mask=mask, weight=weight, error=error, ratio=ratio, rank=rank, reasons=reasons, policy=policy.spec(),
        policy_version=POLICY_VERSION, uses=uses, caveats=caveats, error_spec=error_spec, threshold=threshold,
        evidence_mode=evidence_mode, evidence_allowed=allow_evidence, candidates_sha256=candidates_sha256,
        certificates_sha256=certificates_sha256, calibrated=calibrated, calibration_sha256=calibration_sha256).validate()


def curate(candidates, certificates=None, policy: CurationPolicy = CurationPolicy(admit="accepted"),
           allow_evidence: str = "data_only", calibration=None) -> SelectionSet:
    """Select rows of the candidates file with policy; no input is modified.

    candidates, certificates, and calibration are file paths. Raises
    InterfaceError when the policy and inputs do not fit: a certificate policy
    without certificates, certificates with admit accepted, a legacy 1.0
    certificate, a certificate that fails structural verification against these
    candidates, a calibration of another certificate file or without a bound
    policy, or reference-assisted evidence under a data_only allowance.
    """
    cands = read_candidates(candidates)
    cand_sha = sha256(Path(candidates))
    if policy.admit == "accepted":
        if certificates is not None or calibration is not None:
            raise InterfaceError("admit accepted selects by acceptance alone; it takes no certificates or calibration")
        return decide_selection(cands, None, policy, cand_sha, None, allow_evidence)
    if certificates is None:
        raise InterfaceError(f"admit {policy.admit} compares certificate errors; it needs certificates")
    cert = read_certificates(certificates)
    if cert.legacy:
        raise InterfaceError("a 1.0 certificate already carries its own selection; certify the candidates again (2.0)")
    problems = _certificate_problems(cert, candidates)
    if problems:
        raise InterfaceError("the certificate does not verify against these candidates: " + "; ".join(problems))
    cal, cal_sha = _calibration(calibration)
    return decide_selection(cands, cert, policy, cand_sha, sha256(Path(certificates)), allow_evidence, cal, cal_sha)


def _calibration(path):
    """(document, SHA-256) of a calibration file, or (None, None)."""
    if path is None:
        return None, None
    from ..schema.calibration import read_calibration
    return read_calibration(path), sha256(Path(path))


def _certificate_problems(cert, candidates_path) -> list[str]:
    """Structural verification of the certificate against the candidates file (identity, profile, arithmetic)."""
    from .certificates import verify_structure
    return verify_structure(cert, candidates_path)


def verify_selection(sel: SelectionSet, candidates_path, certificates_path=None, calibration_path=None) -> list[str]:
    """Problems found by recomputing the selection from its inputs (empty when consistent)."""
    problems = []
    cand_sha = sha256(Path(candidates_path))
    if cand_sha != sel.candidates_sha256:
        problems.append(f"candidates SHA-256 {cand_sha} != recorded {sel.candidates_sha256}")
    cert_sha = None if certificates_path is None else sha256(Path(certificates_path))
    if cert_sha != sel.certificates_sha256:
        problems.append(f"certificates SHA-256 {cert_sha} != recorded {sel.certificates_sha256}")
    cal_sha = None if calibration_path is None else sha256(Path(calibration_path))
    if cal_sha != sel.calibration_sha256:
        problems.append(f"calibration SHA-256 {cal_sha} != recorded {sel.calibration_sha256}")
    if problems:
        return problems
    try:
        sel.validate()
        cands = read_candidates(candidates_path)
        cert = None if certificates_path is None else read_certificates(certificates_path)
        if cert is not None:
            structural = _certificate_problems(cert, candidates_path)
            if structural:
                return [f"certificate: {p}" for p in structural]
        cal, _ = _calibration(calibration_path)
        again = decide_selection(cands, cert, CurationPolicy(**sel.policy), cand_sha, cert_sha, sel.evidence_allowed,
                                 cal, cal_sha)
    except (SchemaError, InterfaceError) as e:
        return [str(e)]
    if sel.evidence_mode != again.evidence_mode:
        problems.append(f"evidence_mode {sel.evidence_mode!r} is not the certificate's provenance ({again.evidence_mode!r})")
    for name in ("source_episode_id", "source_frame", "mask", "weight", "reasons", "calibrated"):
        if not np.array_equal(np.asarray(getattr(sel, name)), np.asarray(getattr(again, name))):
            problems.append(f"{name} does not follow from the inputs and the policy")
    for name in ("error", "ratio", "rank"):
        if not np.allclose(getattr(sel, name), getattr(again, name), rtol=1e-12, atol=0.0, equal_nan=True):
            problems.append(f"{name} does not follow from the inputs and the policy")
    for name in ("uses", "caveats", "error_spec", "policy_version"):
        if getattr(sel, name) != getattr(again, name):
            problems.append(f"{name} is not what this policy version records")
    if (sel.threshold is None) != (again.threshold is None) or (
            sel.threshold is not None and not np.isclose(sel.threshold, again.threshold, rtol=1e-12, atol=0.0)):
        problems.append(f"threshold {sel.threshold!r} does not follow from the inputs ({again.threshold!r})")
    return problems


__all__ = ["CurationPolicy", "curate", "decide_selection", "verify_selection", "admit_rows", "produced_rows",
           "reference_assisted", "POLICY_VERSION"]
