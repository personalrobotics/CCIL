"""Calibration and the reference audit: reference dynamics as evidence, declared as such (#167).

    certify (score) -> calibrate on held-out episodes -> curate (reference_assisted) -> audit (optional)

calibrate() bounds the label error observed under reference dynamics,
||(T(z_G, a_G) - y) / state_scale|| with T(z, a) = z + A z + B a, of one
certificate file's eligible rows (accepted and empirical). The unit is the
episode: each calibration episode is scored by the largest observed error over
its eligible rows, and the bound is the split-conformal order statistic of
those scores (schema/calibration.py states what it guarantees).
calibrated_rows() is the only way a row becomes calibrated: an eligible row of
that same certificate file whose episode is not a calibration episode.
Curation uses it only under a reference_assisted allowance (core/curate.py).

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
from ..schema.calibration import METHOD, SCHEMA, STATEMENT, UNIT, VERSION, check
from .batch import RequestError
from .representation import observation

POPULATION_RULE = "eligible (accepted, empirical) rows whose source episode is one of the calibration episodes"
ERROR_DEFINITION = "||(T(z_G, a_G) - y) / state_scale|| with T(z, a) = z + A z + B a, the reference dynamics"


def observed_error(candidates, cert, reference) -> np.ndarray:
    """The label error of every candidate row under the reference dynamics, in the certificate's state scale."""
    from .certificates import read_reference_dynamics
    A, B = read_reference_dynamics(reference)
    co = cert.descriptor["coordinates"]
    z = np.asarray(observation(candidates.observations, co["modality"]), np.float64)
    a = np.asarray(candidates.actions, np.float64)
    y = np.asarray(observation(candidates.target_observations, co["modality"]), np.float64)
    if A.shape != (z.shape[1], z.shape[1]) or B.shape != (z.shape[1], a.shape[1]):
        raise RequestError(f"reference dynamics {A.shape}, {B.shape} do not match the certificate's coordinates "
                             f"({z.shape[1]} state, {a.shape[1]} action)")
    return np.linalg.norm((z + z @ A.T + a @ B.T - y) / np.asarray(co["state_scale"], np.float64), axis=1)


def _verified(candidates_path, certificates_path):
    """(candidates, certificate), refusing a certificate that does not verify against the candidates or is not a score."""
    from .certificates import verify_structure
    cands, cert = read_candidates(candidates_path), read_certificates(certificates_path)
    if cert.legacy:
        raise RequestError("calibration needs a 2.0 certificate; certify the candidates again")
    problems = verify_structure(cert, candidates_path)
    if problems:
        raise RequestError("the certificate does not verify against these candidates: " + "; ".join(problems))
    if cert.descriptor["form"] != "score":
        raise RequestError(f"profile {cert.descriptor['profile']} gives bounds already; calibration bounds the "
                             "observed error of an empirical profile's rows")
    return cands, cert


def eligible(cands, cert) -> np.ndarray:
    """The rows a calibration scores and applies to: accepted, and classified empirical by the certificate."""
    return np.asarray(cands.accepted, bool) & (np.asarray(cert.classification) == "empirical")


def build_calibration(cands, cert, error, episodes, level: float, inputs: dict) -> dict:
    """The calibration document from loaded candidates, certificate, and observed errors; pure arithmetic."""
    if not (isinstance(level, float) and 0 < level < 1):
        raise RequestError(f"level must be in (0, 1), got {level!r}")
    episodes = sorted({str(e) for e in episodes})
    ep = np.asarray(cands.source_episode_id).astype(str)
    pop = eligible(cands, cert) & np.isin(ep, episodes)
    if not pop.any():
        raise RequestError("no accepted, empirical rows come from the calibration episodes")
    scores = sorted(float(error[pop & (ep == e)].max()) for e in np.unique(ep[pop]))
    n = len(scores)
    k = math.ceil((n + 1) * level)
    d = cert.descriptor
    doc = {"schema": SCHEMA, "version": f"{VERSION[0]}.{VERSION[1]}", "evidence_mode": "reference_assisted",
           "method": dict(METHOD), "level": float(level), "unit": dict(UNIT),
           "certificate": {"profile": d["profile"], "version": d["version"], "neighborhood": d["neighborhood"]},
           "error": {"definition": ERROR_DEFINITION, "units": d["units"]["error"]},
           "population": {"rule": POPULATION_RULE, "episodes": episodes, "rows": int(pop.sum())},
           "bound": {"n": n, "k": k, "scores": scores, "value": scores[k - 1] if k <= n else None},
           "statement": STATEMENT, "inputs": dict(inputs)}
    return check(doc)


def calibrate(candidates, certificates, reference, episodes, level: float = 0.9) -> dict:
    """Calibrate the eligible rows of certificates (paths) on the rows of episodes, against reference dynamics."""
    cands, cert = _verified(candidates, certificates)
    inputs = {"candidates": sha256(Path(candidates)), "certificates": sha256(Path(certificates)),
              "reference": sha256(Path(reference))}
    return build_calibration(cands, cert, observed_error(cands, cert, reference), episodes, level, inputs)


def calibrated_rows(cal: dict, cands, cert) -> tuple[np.ndarray, np.ndarray]:
    """(calibrated mask, calibrated bound or NaN) per row of the certificate the calibration names; pure arithmetic."""
    value = cal["bound"]["value"]
    mask = (eligible(cands, cert) & (value is not None)
            & ~np.isin(np.asarray(cands.source_episode_id).astype(str), cal["population"]["episodes"]))
    return mask, np.where(mask, np.nan if value is None else value, np.nan)


def matching_calibration(cal: dict, candidates_sha256: str, certificates_sha256: str | None) -> None:
    """Refuse a calibration of other candidates or another certificate file."""
    if certificates_sha256 is None:
        raise RequestError("a calibration maps a certificate's scores; it needs that certificate")
    for name, sha in (("candidates", candidates_sha256), ("certificates", certificates_sha256)):
        if cal["inputs"][name] != sha:
            raise RequestError(f"the calibration was made from {name} {cal['inputs'][name]}, not {sha}; a "
                                 "calibration applies to its own certificate file only")


def structure_problems(cal: dict, cands, cert) -> list[str]:
    """What the files alone contradict in a calibration: its certificate, population, n and k (not the observed errors).

    Recomputed with the observed errors set to zero, so every recorded field
    that does not depend on the reference dynamics must match.
    """
    try:
        again = build_calibration(cands, cert, np.zeros(len(cert.classification)), cal["population"]["episodes"],
                                  cal["level"], cal["inputs"])
    except RequestError as e:
        return [str(e)]
    problems = [f"{key} does not follow from the certificate" for key in ("certificate", "population")
                if cal[key] != again[key]]
    if (cal["bound"]["n"], cal["bound"]["k"]) != (again["bound"]["n"], again["bound"]["k"]):
        problems.append("the bound's n and k do not follow from the certificate")
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
                                  cal["level"], given)
    except (SchemaError, RequestError) as e:
        return [str(e)]
    for key in ("method", "unit", "certificate", "error", "population", "statement"):
        if cal[key] != again[key]:
            problems.append(f"{key} does not follow from the inputs")
    b, c = cal["bound"], again["bound"]
    close = lambda x, y: (x is None and y is None) or (x is not None and y is not None and math.isclose(x, y, rel_tol=1e-12))  # noqa: E731
    if not ((b["n"], b["k"]) == (c["n"], c["k"]) and close(b["value"], c["value"])
            and all(close(x, y) for x, y in zip(b["scores"], c["scores"]))):
        problems.append("the bound does not follow from the inputs")
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
        raise RequestError("the selection does not verify against its inputs: " + "; ".join(problems))
    sel, cands = read_selection(selection), read_candidates(candidates)
    cert = None if certificates is None else read_certificates(certificates)
    if cert is None:  # an uncertified selection: the unit state scale
        from .certificates import read_reference_dynamics
        A, B = read_reference_dynamics(reference)
        z, a, y = (np.asarray(x, np.float64) for x in (observation(cands.observations, modality), cands.actions,
                                                         observation(cands.target_observations, modality)))
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
    if calibration is not None:  # every calibrated row, the population the bound covers, by row and by episode
        cal = read_calibration(calibration)
        rows = np.asarray(sel.calibrated, bool)
        covered = err <= np.where(rows, sel.error, np.inf)
        ep = np.asarray(cands.source_episode_id).astype(str)
        episodes = np.unique(ep[rows])
        calibrated = {"level": cal["level"], "rows": int(rows.sum()), "covered": int((rows & covered).sum()),
                      "episodes": int(len(episodes)),
                      "episodes_covered": int(sum(covered[rows & (ep == e)].all() for e in episodes))}
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


__all__ = ["calibrate", "build_calibration", "calibrated_rows", "eligible", "matching_calibration", "structure_problems",
           "verify_calibration", "observed_error", "reference_audit"]
