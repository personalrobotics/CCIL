"""Calibration and the reference audit: reference dynamics as evidence, declared as such (#167).

    certify (score) -> calibrate on held-out rows -> curate (reference_assisted) -> audit (optional)

calibrate() maps one certificate file's empirical scores to the label error
observed under reference dynamics, ||(T(z_G, a_G) - y) / state_scale|| with
T(z, a) = z + A z + B a, on the accepted, supported rows of declared episodes
(the held-out ones). It writes ccil.calibration (schema/calibration.py): bins
of equal calibration count, each with a split-conformal bound at the declared
level. calibrated_rows() is the only way a row becomes calibrated: a row of
that same certificate file, outside the calibration population, empirical,
with its score in a bin that has a bound. Curation uses it only under a
reference_assisted allowance (core/curate.py).

reference_audit() annotates a decided selection with the observed error. It
reads the selection and never feeds back into it: an audited data_only result
stays data_only.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..runs.provenance import sha256
from ..schema import SchemaError, read_candidates, read_certificates, read_selection
from ..schema.audit import SCHEMA as AUDIT_SCHEMA, STATEMENT as AUDIT_STATEMENT, VERSION as AUDIT_VERSION
from ..schema.calibration import METHOD, SCHEMA, STATEMENT, VERSION, check
from .batch import InterfaceError

POPULATION_RULE = "accepted, supported rows whose source episode is one of the calibration episodes"
ERROR_DEFINITION = "||(T(z_G, a_G) - y) / state_scale|| with T(z, a) = z + A z + B a, the reference dynamics"


def observed_error(candidates, cert, reference) -> np.ndarray:
    """The label error of every candidate row under the reference dynamics, in the certificate's state scale."""
    from .certificates import read_reference_dynamics
    A, B = read_reference_dynamics(reference)
    co = cert.descriptor["coordinates"]
    z = np.asarray(candidates.observations[co["modality"]], np.float64)
    a = np.asarray(candidates.actions, np.float64)
    y = np.asarray(candidates.target_observations[co["modality"]], np.float64)
    if A.shape != (z.shape[1], z.shape[1]) or B.shape != (z.shape[1], a.shape[1]):
        raise InterfaceError(f"reference dynamics {A.shape}, {B.shape} do not match the certificate's coordinates "
                             f"({z.shape[1]} state, {a.shape[1]} action)")
    return np.linalg.norm((z + z @ A.T + a @ B.T - y) / np.asarray(co["state_scale"], np.float64), axis=1)


def _verified(candidates_path, certificates_path):
    """(candidates, certificate), refusing a certificate that does not verify against the candidates or is not a score."""
    from .certificates import verify_structure
    cands, cert = read_candidates(candidates_path), read_certificates(certificates_path)
    if cert.legacy:
        raise InterfaceError("calibration needs a 2.0 certificate; certify the candidates again")
    problems = verify_structure(cert, candidates_path)
    if problems:
        raise InterfaceError("the certificate does not verify against these candidates: " + "; ".join(problems))
    if cert.descriptor["form"] != "score":
        raise InterfaceError(f"profile {cert.descriptor['profile']} gives bounds already; calibration maps an "
                             "empirical score to observed error")
    return cands, cert


def _bin_of(edges, score) -> np.ndarray:
    """Bin index of each score in [edges[0], edges[-1]] (last bin closed), -1 outside."""
    i = np.searchsorted(edges, score, side="right") - 1
    i = np.where(score == edges[-1], len(edges) - 2, i)
    with np.errstate(invalid="ignore"):
        inside = (score >= edges[0]) & (score <= edges[-1])
    return np.where(inside, i, -1)


def build_calibration(cands, cert, error, episodes, level: float, bins: int, inputs: dict) -> dict:
    """The calibration document from loaded candidates, certificate, and observed errors; pure arithmetic."""
    if not (isinstance(level, float) and 0 < level < 1):
        raise InterfaceError(f"level must be in (0, 1), got {level!r}")
    if not (type(bins) is int and bins >= 1):
        raise InterfaceError(f"bins must be an int >= 1, got {bins!r}")
    episodes = sorted({str(e) for e in episodes})
    pop = population(cands, cert, episodes)
    if not pop.any():
        raise InterfaceError("no accepted, supported rows come from the calibration episodes")
    score, err = np.asarray(cert.total_error_value, np.float64)[pop], error[pop]
    edges = np.unique(np.quantile(score, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0]])
    which = _bin_of(edges, score)
    out = []
    for b in range(len(edges) - 1):
        e = np.sort(err[which == b])
        if not len(e):
            continue  # tied quantile edges; a later bin starts at the same edge
        k = math.ceil((len(e) + 1) * level)
        out.append({"lower": float(edges[b]), "upper": float(edges[b + 1]), "count": int(len(e)), "k": int(k),
                    "bound": float(e[k - 1]) if k <= len(e) else None,
                    "observed": {"median": float(np.median(e)), "at_level": float(np.quantile(e, level)),
                                 "max": float(e[-1])}})
    for prev, b in zip(out, out[1:]):
        b["lower"] = prev["upper"]
    d = cert.descriptor
    doc = {"schema": SCHEMA, "version": f"{VERSION[0]}.{VERSION[1]}", "evidence_mode": "reference_assisted",
           "method": {**METHOD, "bins": bins}, "level": float(level),
           "score": {"profile": d["profile"], "version": d["version"], "units": d["units"]["error"],
                     "definition": "the certificate's total_error_value, an empirical score"},
           "error": {"definition": ERROR_DEFINITION, "units": d["units"]["error"]},
           "population": {"rule": POPULATION_RULE, "episodes": episodes, "rows": int(pop.sum())},
           "regime": {"score_min": out[0]["lower"], "score_max": out[-1]["upper"], "neighborhood": d["neighborhood"]},
           "bins": out, "statement": STATEMENT, "inputs": dict(inputs)}
    return check(doc)


def population(cands, cert, episodes) -> np.ndarray:
    """The calibration rows: accepted and supported, from the calibration episodes."""
    return (np.asarray(cands.accepted, bool) & (np.asarray(cert.classification) != "unsupported")
            & np.isin(np.asarray(cands.source_episode_id).astype(str), list(episodes)))


def calibrate(candidates, certificates, reference, episodes, level: float = 0.9, bins: int = 4) -> dict:
    """Calibrate the score of certificates (paths) on the rows of episodes, against reference dynamics."""
    cands, cert = _verified(candidates, certificates)
    inputs = {"candidates": sha256(Path(candidates)), "certificates": sha256(Path(certificates)),
              "reference": sha256(Path(reference))}
    return build_calibration(cands, cert, observed_error(cands, cert, reference), episodes, level, bins, inputs)


def calibrated_rows(cal: dict, cands, cert) -> tuple[np.ndarray, np.ndarray]:
    """(calibrated mask, calibrated bound or NaN) per row of the certificate the calibration names; pure arithmetic."""
    score = np.asarray(cert.total_error_value, np.float64)
    bins = cal["bins"]
    edges = np.array([bins[0]["lower"]] + [b["upper"] for b in bins])
    bound = np.array([np.nan if b["bound"] is None else b["bound"] for b in bins])
    which = _bin_of(edges, score)
    value = np.where(which >= 0, bound[np.clip(which, 0, len(bins) - 1)], np.nan)
    eligible = ((np.asarray(cert.classification) == "empirical")
                & ~np.isin(np.asarray(cands.source_episode_id).astype(str), cal["population"]["episodes"]))
    mask = eligible & np.isfinite(value)
    return mask, np.where(mask, value, np.nan)


def matching_calibration(cal: dict, candidates_sha256: str, certificates_sha256: str | None) -> None:
    """Refuse a calibration of other candidates or another certificate file."""
    if certificates_sha256 is None:
        raise InterfaceError("a calibration maps a certificate's scores; it needs that certificate")
    for name, sha in (("candidates", candidates_sha256), ("certificates", certificates_sha256)):
        if cal["inputs"][name] != sha:
            raise InterfaceError(f"the calibration was made from {name} {cal['inputs'][name]}, not {sha}; a "
                                 "calibration applies to its own certificate file only")


def structure_problems(cal: dict, cands, cert) -> list[str]:
    """What the files alone contradict in a calibration: its population, score, regime, and bins (not the observed errors).

    Recomputed with the observed errors set to zero, so every recorded field
    that does not depend on the reference dynamics must match.
    """
    try:
        again = build_calibration(cands, cert, np.zeros(len(cert.classification)), cal["population"]["episodes"],
                                  cal["level"], cal["method"]["bins"], cal["inputs"])
    except InterfaceError as e:
        return [str(e)]
    problems = [f"{key} does not follow from the certificate" for key in ("score", "population", "regime", "method")
                if cal[key] != again[key]]
    if [(b["lower"], b["upper"], b["count"], b["k"]) for b in cal["bins"]] != [
            (b["lower"], b["upper"], b["count"], b["k"]) for b in again["bins"]]:
        problems.append("the bins' score ranges and counts do not follow from the certificate")
    return problems


def verify_calibration(cal: dict, candidates, certificates, reference) -> list[str]:
    """Problems found by recomputing the calibration from its inputs (empty when consistent)."""
    given = {"candidates": sha256(Path(candidates)), "certificates": sha256(Path(certificates)),
             "reference": sha256(Path(reference))}
    problems = [f"input {k} SHA-256 {given[k]} != recorded {cal['inputs'][k]}" for k in given if given[k] != cal["inputs"][k]]
    if problems:
        return problems
    try:
        check(cal)
        cands, cert = _verified(candidates, certificates)
        again = build_calibration(cands, cert, observed_error(cands, cert, reference), cal["population"]["episodes"],
                                  cal["level"], cal["method"]["bins"], given)
    except (SchemaError, InterfaceError) as e:
        return [str(e)]
    for key in ("score", "error", "population", "regime", "statement", "method"):
        if cal[key] != again[key]:
            problems.append(f"{key} does not follow from the inputs")
    if len(cal["bins"]) != len(again["bins"]):
        return problems + [f"{len(cal['bins'])} bins; the inputs give {len(again['bins'])}"]
    for i, (b, c) in enumerate(zip(cal["bins"], again["bins"])):
        same = all(b[k] == c[k] for k in ("count", "k")) and all(
            (b[k] is None and c[k] is None) or (b[k] is not None and c[k] is not None and math.isclose(b[k], c[k], rel_tol=1e-12))
            for k in ("lower", "upper", "bound"))
        if not same:
            problems.append(f"bin {i} does not follow from the inputs")
    return problems


def _summary(x) -> dict:
    if not len(x):
        return {"rows": 0, "median": None, "q90": None, "max": None}
    return {"rows": int(len(x)), "median": float(np.median(x)), "q90": float(np.quantile(x, 0.9)), "max": float(x.max())}


def reference_audit(selection, candidates, reference, certificates=None, calibration=None, modality="state") -> dict:
    """The ccil.reference-audit of a selection (paths): observed error beside the decision, never feeding it.

    The selection must verify against its inputs first. Without certificates
    the error is measured in the unit state scale of modality.
    """
    from ..schema.audit import check as check_audit
    from ..schema.calibration import read_calibration
    from .curate import verify_selection
    problems = verify_selection(read_selection(selection), candidates, certificates, calibration)
    if problems:
        raise InterfaceError("the selection does not verify against its inputs: " + "; ".join(problems))
    sel, cands = read_selection(selection), read_candidates(candidates)
    cert = None if certificates is None else read_certificates(certificates)
    if cert is None:  # an uncertified selection: the unit state scale
        from .certificates import read_reference_dynamics
        A, B = read_reference_dynamics(reference)
        z, a, y = (np.asarray(x, np.float64) for x in (cands.observations[modality], cands.actions,
                                                         cands.target_observations[modality]))
        err, units = np.linalg.norm(z + z @ A.T + a @ B.T - y, axis=1), "state units"
    else:
        err, units = observed_error(cands, cert, reference), cert.descriptor["units"]["error"]
    accepted, mask = np.asarray(cands.accepted, bool), np.asarray(sel.mask, bool)
    bounds = None
    if sel.error_spec is not None and sel.error_spec["is_bound"]:
        e, b = err[mask], sel.error[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(b > 0, e / b, np.where(e > 0, np.inf, 0.0))
        bounds = {"rows": int(mask.sum()), "violations": int((e > b).sum()),
                  "max_ratio": float(ratio.max()) if len(ratio) else None}
    tol = sel.policy["tolerance"]
    tolerance = None if tol is None else {"value": float(tol), "rows": int(mask.sum()), "within": int((err[mask] <= tol).sum())}
    calibrated = None
    if calibration is not None:
        cal = read_calibration(calibration)
        rows = mask & np.asarray(sel.calibrated, bool)
        calibrated = {"level": cal["level"], "rows": int(rows.sum()), "covered": int((err[rows] <= sel.error[rows]).sum())}
    doc = {"schema": AUDIT_SCHEMA, "version": f"{AUDIT_VERSION[0]}.{AUDIT_VERSION[1]}",
           "selection": {"policy": sel.policy, "evidence_mode": sel.evidence_mode, "evidence_allowed": sel.evidence_allowed},
           "error": {"definition": ERROR_DEFINITION, "units": units},
           "groups": {"selected": _summary(err[mask]), "accepted_unselected": _summary(err[accepted & ~mask])},
           "bounds": bounds, "tolerance": tolerance, "calibrated": calibrated, "statement": AUDIT_STATEMENT,
           "inputs": {"selection": sha256(Path(selection)), "candidates": sha256(Path(candidates)),
                      "certificates": None if certificates is None else sha256(Path(certificates)),
                      "calibration": None if calibration is None else sha256(Path(calibration)),
                      "reference": sha256(Path(reference))}}
    return check_audit(doc)


__all__ = ["calibrate", "build_calibration", "calibrated_rows", "matching_calibration", "structure_problems",
           "verify_calibration",
           "observed_error", "population", "reference_audit"]
