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

Evidence modes: a certificate whose constants or demonstration residuals come
from reference dynamics is reference_assisted evidence. curate refuses it in
the default data_only mode, and records the mode it ran in.
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
        if self.tolerance is not None and not _nonneg(self.tolerance):
            problems.append(f"tolerance must be finite and >= 0 or None, got {self.tolerance!r}")
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


def _nonneg(x) -> bool:
    return _number(x) and x >= 0


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
    return any(e.get("provenance") and PROVIDERS[e["provenance"]["provider"]].needs_reference for e in entries)


def decide_selection(candidates, cert, policy: CurationPolicy, candidates_sha256: str, certificates_sha256: str | None,
                     evidence_mode: str) -> SelectionSet:
    """The selection one policy makes, from loaded candidates and certificates; pure arithmetic over the files."""
    accepted = np.asarray(candidates.accepted, bool)
    n = len(accepted)
    reasons = np.zeros((n, len(REASON_CODES)), bool)
    reasons[:, 0] = ~accepted
    nan = np.full(n, np.nan)
    if policy.admit == "accepted":
        mask = accepted
        error, ratio, rank, threshold, error_spec, uses = nan, nan.copy(), nan.copy(), None, None, []
        caveats = ["uncertified: rows are selected by acceptance alone; no error is compared"]
    else:
        total = np.asarray(cert.total_error_value, np.float64)
        rows = admit_rows(cert.classification, total, accepted, policy.admit, policy.tolerance, policy.quantile,
                          produced_rows(candidates) if policy.population == "produced" else None)
        supported, threshold = rows["supported"], rows["threshold"]
        mask = rows["eligible"] & rows["selected"]
        reasons[:, 1] = ~supported
        reasons[:, 2] = supported & ~rows["candidate"]
        reasons[:, 3] = rows["candidate"] & ~rows["within"]
        reasons[:, 4] = rows["eligible"] & ~rows["selected"]
        error = np.where(supported, total, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = nan.copy() if policy.tolerance is None else np.where(supported, total / policy.tolerance, np.nan)
        rank = _ranks(total, rows["pool"] if policy.quantile is not None else rows["eligible"])
        units = cert.descriptor["units"]["error"]
        error_spec = {"definition": "the certificate's total_error_value: "
                                    + ("an upper bound on ||T(q) - y||" if policy.admit == "bound"
                                       else "an empirical score, not a bound"),
                      "units": units, "is_bound": policy.admit == "bound"}
        uses, caveats = ["fidelity"], [FIDELITY_CAVEAT]
    if evidence_mode == "data_only":
        caveats = caveats + [DATA_ONLY_CAVEAT]
    weight = np.where(mask, np.asarray(candidates.weight, np.float64), 0.0)
    return SelectionSet(
        source_episode_id=np.asarray(candidates.source_episode_id), source_frame=np.asarray(candidates.source_frame),
        mask=mask, weight=weight, error=error, ratio=ratio, rank=rank, reasons=reasons, policy=policy.spec(),
        policy_version=POLICY_VERSION, uses=uses, caveats=caveats, error_spec=error_spec, threshold=threshold,
        evidence_mode=evidence_mode, candidates_sha256=candidates_sha256, certificates_sha256=certificates_sha256).validate()


def curate(candidates, certificates=None, policy: CurationPolicy = CurationPolicy(admit="accepted"),
           evidence_mode: str = "data_only") -> SelectionSet:
    """Select rows of the candidates file with policy; neither input is modified.

    candidates and certificates are file paths. Raises InterfaceError when the
    policy and inputs do not fit: a certificate policy without certificates,
    certificates with admit accepted, certificates of other candidates, a
    legacy 1.0 certificate, or reference-assisted evidence in data_only mode.
    """
    if evidence_mode not in EVIDENCE_MODES:
        raise InterfaceError(f"evidence_mode must be one of {list(EVIDENCE_MODES)}, got {evidence_mode!r}")
    cands = read_candidates(candidates)
    cand_sha = sha256(Path(candidates))
    if policy.admit == "accepted":
        if certificates is not None:
            raise InterfaceError("admit accepted selects by acceptance alone; it takes no certificates")
        return decide_selection(cands, None, policy, cand_sha, None, evidence_mode)
    if certificates is None:
        raise InterfaceError(f"admit {policy.admit} compares certificate errors; it needs certificates")
    cert = read_certificates(certificates)
    problems = []
    if cert.legacy:
        problems.append("a 1.0 certificate already carries its own selection; certify the candidates again (2.0)")
    if cert.candidates_sha256 != cand_sha:
        problems.append(f"certificates certify candidates {cert.candidates_sha256}, not this file ({cand_sha})")
    elif not (np.array_equal(cert.source_episode_id.astype(str), np.asarray(cands.source_episode_id).astype(str))
              and np.array_equal(cert.source_frame, cands.source_frame)):
        problems.append("certificate rows do not match the candidate rows")
    if evidence_mode == "data_only" and reference_assisted(cert):
        problems.append("the certificate's constants come from reference dynamics, which data_only curation cannot "
                        "use; set evidence_mode reference_assisted")
    if problems:
        raise InterfaceError("; ".join(problems))
    return decide_selection(cands, cert, policy, cand_sha, sha256(Path(certificates)), evidence_mode)


def verify_selection(sel: SelectionSet, candidates_path, certificates_path=None) -> list[str]:
    """Problems found by recomputing the selection from its inputs (empty when consistent)."""
    problems = []
    cand_sha = sha256(Path(candidates_path))
    if cand_sha != sel.candidates_sha256:
        problems.append(f"candidates SHA-256 {cand_sha} != recorded {sel.candidates_sha256}")
    cert_sha = None if certificates_path is None else sha256(Path(certificates_path))
    if cert_sha != sel.certificates_sha256:
        problems.append(f"certificates SHA-256 {cert_sha} != recorded {sel.certificates_sha256}")
    if problems:
        return problems
    try:
        sel.validate()
        cands = read_candidates(candidates_path)
        cert = None if certificates_path is None else read_certificates(certificates_path)
        again = decide_selection(cands, cert, CurationPolicy(**sel.policy), cand_sha, cert_sha, sel.evidence_mode)
    except (SchemaError, InterfaceError) as e:
        return [str(e)]
    if sel.evidence_mode == "data_only" and cert is not None and reference_assisted(cert):
        problems.append("a data_only selection uses reference-assisted certificates")
    for name in ("source_episode_id", "source_frame", "mask", "weight", "reasons"):
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
