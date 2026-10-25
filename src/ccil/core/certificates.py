"""Label-error certificates: a post-generation layer over candidate files (#88).

    generate candidates -> certify candidates -> select / admit labels -> train policy

Candidate files are never modified. A certificate set has one row per candidate
(joined by source identity and position) and records, per row:

- the terms it used: state and action displacement from the anchor, the
  anchor's observed model error, and the solver error ||T_hat(q) - y|| of the
  candidate under the learned model T_hat(z, a) = z + f_hat(z, a);
- model_error_value and total_error_value. An additive profile gives
  total = solver_error + model_error; a direct profile gives the total itself
  (and leaves model_error_value NaN). solver_error is always a diagnostic and is
  never added twice;
- classification (certified, conditional, calibrated, empirical, unsupported)
  and is_upper_bound, which is true only for certified, conditional, and
  calibrated rows: a score is never presented as a bound;
- in_neighborhood, admissible, and selected, and the reasons a row is not.

Consumers train on usable = candidate.accepted & admissible & selected.
Selection is a data-budget choice over eligible rows and never changes a
candidate's acceptance or a row's classification.

A certificate can be verified at two levels: structurally, from the
certificate and its candidates (identity, formulas, classifications, masks),
and by source audit, which recomputes the source terms from the dataset,
representation, and model. Hashes establish artifact identity, not correctness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema.fields import SchemaError
from ..schema.io import _atomic_savez, _format_error, _load_npz
from .batch import InterfaceError
from .representation import transitions

SCHEMA, VERSION = "ccil.certificates", (1, 0)
# Strongest first; a row's classification is the weakest of what it relies on.
CLASSIFICATIONS = ("certified", "conditional", "calibrated", "empirical", "unsupported")
BOUND_CLASSES = ("certified", "conditional", "calibrated")
SOURCE_CLASS = {"analytic_bound": "certified", "model_bound": "certified", "declared": "conditional",
                "point_jacobian": "empirical"}
REASON_CODES = ("certificate.unsupported", "certificate.exceeds_bound", "certificate.not_selected")
TERMS = ("state_displacement", "action_displacement", "anchor_model_error", "solver_error")
REP = "representation units (state units for representation.kind=identity)"


def _weakest(classes) -> str:
    return max(classes, key=CLASSIFICATIONS.index)


@dataclass(frozen=True)
class Constant:
    """A constant a profile relies on, with where its value comes from."""

    value: float
    source: str
    rationale: str
    units: str

    def __post_init__(self):
        if self.source not in SOURCE_CLASS:
            raise InterfaceError(f"constant source must be one of {sorted(SOURCE_CLASS)}, got {self.source!r}")
        if isinstance(self.value, bool) or not (np.isfinite(self.value) and self.value >= 0):
            raise InterfaceError(f"constant value must be finite and >= 0, got {self.value!r}")
        if not (isinstance(self.rationale, str) and self.rationale and isinstance(self.units, str) and self.units):
            raise InterfaceError("a constant needs a rationale and units")

    def to_dict(self) -> dict:
        return {"value": float(self.value), "source": self.source, "rationale": self.rationale, "units": self.units}


class ContinuityProfile:
    """Continuity / discrepancy profile: an additive certificate.

    model_error_value = anchor_model_error + C_z ||z_G - z_t|| + C_a ||a_G - a_t||,
    where C_z and C_a bound the local variation of f - f_hat with respect to state
    and action over the neighborhood (for example C = K_true + K_hat when derived
    from separate bounds). By the triangle inequality the true label error
    ||T(q) - y|| is at most solver_error + model_error_value whenever the
    constants hold. Rows outside the neighborhood are unsupported.
    """

    name, version, form = "continuity", "1", "additive"

    def __init__(self, state_constant: Constant, action_constant: Constant, neighborhood_state: float = np.inf,
                 neighborhood_action: float = np.inf):
        for c in (state_constant, action_constant):
            if c.source == "point_jacobian":
                raise InterfaceError("continuity needs constants over the neighborhood; a point Jacobian is a score, "
                                     "not a constant (the paper-proxy profiles use it)")
        for r in (neighborhood_state, neighborhood_action):
            if not r > 0:
                raise InterfaceError(f"neighborhood radii must be > 0 (inf for global), got {r!r}")
        self.state_constant, self.action_constant = state_constant, action_constant
        self.neighborhood = {"state": float(neighborhood_state), "action": float(neighborhood_action)}

    def descriptor(self) -> dict:
        return {"profile": self.name, "version": self.version, "form": self.form,
                "constants": {"C_z": self.state_constant.to_dict(), "C_a": self.action_constant.to_dict()},
                "neighborhood": {k: (None if np.isinf(v) else v) for k, v in self.neighborhood.items()},
                "norms": {"state": "l2", "action": "l2", "output": "l2"},
                "metric": "separate state and action displacements; no joint metric",
                "units": {"state": REP, "action": "action units", "error": REP},
                "assumptions": ["C_z and C_a bound the variation of f - f_hat over the neighborhood",
                                "the demonstration transition (z_t, a_t, z_{t+1}) is exact"]}

    @classmethod
    def from_descriptor(cls, d: dict) -> "ContinuityProfile":
        c = d["constants"]
        nb = d["neighborhood"]
        return cls(Constant(**c["C_z"]), Constant(**c["C_a"]),
                   np.inf if nb["state"] is None else nb["state"], np.inf if nb["action"] is None else nb["action"])

    def evaluate(self, terms: dict) -> tuple[np.ndarray, np.ndarray]:
        """(model_error_value, total_error_value) from persisted terms."""
        model = (terms["anchor_model_error"] + self.state_constant.value * terms["state_displacement"]
                 + self.action_constant.value * terms["action_displacement"])
        return model, terms["solver_error"] + model

    def in_neighborhood(self, terms: dict) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return ((terms["state_displacement"] <= self.neighborhood["state"])
                    & (terms["action_displacement"] <= self.neighborhood["action"]))

    def base_classification(self) -> str:
        return _weakest([SOURCE_CLASS[self.state_constant.source], SOURCE_CLASS[self.action_constant.source]])


PROFILES = {ContinuityProfile.name: ContinuityProfile}


def source_terms(dataset, representation, candidates, model) -> dict:
    """The per-candidate terms every profile draws on, from the dataset, representation, and model."""
    batch = transitions(dataset, representation)
    index = {(str(e), int(f)): i for i, (e, f) in enumerate(zip(batch.episode_id, batch.frame))}
    try:
        src = np.array([index[(str(e), int(f))] for e, f in zip(candidates.source_episode_id, candidates.source_frame)])
    except KeyError as e:
        raise InterfaceError(f"candidate source {e} is not a transition of the dataset") from None
    modality = representation.output_modality
    z_g = np.asarray(candidates.observations[modality], np.float64)
    a_g = np.asarray(candidates.actions, np.float64)
    target = np.asarray(candidates.target_observations[modality], np.float64)
    z, a, zn = (np.asarray(x, np.float64) for x in (batch.z, batch.a, batch.z_next))
    anchor_error = np.linalg.norm(z + model.delta(z, a) - zn, axis=1)
    return {"state_displacement": np.linalg.norm(z_g - z[src], axis=1),
            "action_displacement": np.linalg.norm(a_g - a[src], axis=1),
            "anchor_model_error": anchor_error[src],
            "solver_error": np.linalg.norm(z_g + model.delta(z_g, a_g) - target, axis=1)}


@dataclass
class CertificateSet:
    source_episode_id: np.ndarray
    source_frame: np.ndarray
    terms: dict
    model_error_value: np.ndarray
    total_error_value: np.ndarray
    classification: np.ndarray        # (N,) str, one of CLASSIFICATIONS
    is_upper_bound: np.ndarray
    in_neighborhood: np.ndarray
    admissible: np.ndarray
    selected: np.ndarray
    reasons: np.ndarray               # (N, len(REASON_CODES)) bool
    descriptor: dict
    thresholds: dict                  # max_total_error, selection_quantile, selection_threshold
    candidates_sha256: str
    inputs: dict = field(default_factory=dict)

    @property
    def usable_given(self):
        """admissible & selected; combine with candidate.accepted for training rows."""
        return self.admissible & self.selected


def _decide(profile, terms, accepted, max_total_error, selection_quantile):
    """Values, classification, and masks from persisted terms: shared by certify and structural verification."""
    model, total = profile.evaluate(terms)
    finite = np.isfinite(total) & np.all([np.isfinite(terms[k]) for k in TERMS], axis=0)
    inside = profile.in_neighborhood(terms)
    supported = finite & inside
    cls = np.where(supported, profile.base_classification(), "unsupported").astype("<U12")
    bound = np.isin(cls, BOUND_CLASSES)
    with np.errstate(invalid="ignore"):
        within = np.ones(len(total), bool) if max_total_error is None else (total <= max_total_error)
    admissible = bound & within
    eligible = accepted & admissible
    threshold = None
    if selection_quantile is not None and eligible.any():
        threshold = float(np.quantile(total[eligible], selection_quantile))
        selected = eligible & (total <= threshold)
    else:
        selected = np.ones(len(total), bool)
    reasons = np.stack([~supported, supported & bound & ~within, eligible & ~selected], axis=1)
    return model, total, cls, bound, inside, admissible, selected, reasons, threshold


def certify(candidates, dataset, representation, model, profile, candidates_sha256: str,
            max_total_error: float | None = None, selection_quantile: float | None = None,
            inputs: dict | None = None) -> CertificateSet:
    """Certify every candidate row with profile; the candidates are not modified."""
    if selection_quantile is not None and not 0 < selection_quantile <= 1:
        raise InterfaceError(f"selection_quantile must be in (0, 1], got {selection_quantile!r}")
    terms = source_terms(dataset, representation, candidates, model)
    model_v, total, cls, bound, inside, admissible, selected, reasons, thr = _decide(
        profile, terms, np.asarray(candidates.accepted), max_total_error, selection_quantile)
    return CertificateSet(
        source_episode_id=np.asarray(candidates.source_episode_id), source_frame=np.asarray(candidates.source_frame),
        terms=terms, model_error_value=model_v, total_error_value=total, classification=cls, is_upper_bound=bound,
        in_neighborhood=inside, admissible=admissible, selected=selected, reasons=reasons,
        descriptor=profile.descriptor(), candidates_sha256=candidates_sha256, inputs=dict(inputs or {}),
        thresholds={"max_total_error": max_total_error, "selection_quantile": selection_quantile,
                    "selection_threshold": thr})


# --- persistence ---------------------------------------------------------------

def write_certificates(cert: CertificateSet, path) -> None:
    arrays = {"source_episode_id": cert.source_episode_id, "source_frame": cert.source_frame,
              "model_error_value": cert.model_error_value, "total_error_value": cert.total_error_value,
              "classification": np.array([CLASSIFICATIONS.index(c) for c in cert.classification], np.int8),
              "is_upper_bound": cert.is_upper_bound, "in_neighborhood": cert.in_neighborhood,
              "admissible": cert.admissible, "selected": cert.selected, "reasons": cert.reasons}
    arrays.update({f"term/{k}": v for k, v in cert.terms.items()})
    meta = {"schema": SCHEMA, "version": f"{VERSION[0]}.{VERSION[1]}", "optional_fields": [],
            "descriptor": cert.descriptor, "thresholds": cert.thresholds, "candidates_sha256": cert.candidates_sha256,
            "inputs": cert.inputs, "classification_codes": list(CLASSIFICATIONS), "reason_codes": list(REASON_CODES)}
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    _atomic_savez(path, arrays)


def read_certificates(path) -> CertificateSet:
    arrays, meta = _load_npz(path, SCHEMA)
    if meta.get("schema") != SCHEMA or str(meta.get("version", "")).split(".")[0] != str(VERSION[0]):
        raise _format_error(SCHEMA, path, [f"not a {SCHEMA} {VERSION[0]}.x file"])
    need = {"source_episode_id", "source_frame", "model_error_value", "total_error_value", "classification",
            "is_upper_bound", "in_neighborhood", "admissible", "selected", "reasons"} | {f"term/{k}" for k in TERMS}
    missing = sorted(need - set(arrays))
    if missing:
        raise _format_error(SCHEMA, path, [f"missing fields {missing}"])
    if meta.get("classification_codes") != list(CLASSIFICATIONS) or meta.get("reason_codes") != list(REASON_CODES):
        raise _format_error(SCHEMA, path, ["unknown classification or reason codes"])
    n = len(arrays["source_episode_id"])
    if any(np.asarray(v).shape[:1] != (n,) for v in arrays.values()):
        raise _format_error(SCHEMA, path, ["arrays do not share one row count"])
    codes = arrays["classification"]
    if codes.dtype.kind not in "iu" or ((codes < 0) | (codes >= len(CLASSIFICATIONS))).any():
        raise _format_error(SCHEMA, path, ["classification codes out of range"])
    return CertificateSet(
        source_episode_id=arrays["source_episode_id"], source_frame=arrays["source_frame"],
        terms={k: arrays[f"term/{k}"] for k in TERMS}, model_error_value=arrays["model_error_value"],
        total_error_value=arrays["total_error_value"],
        classification=np.array([CLASSIFICATIONS[c] for c in codes], dtype="<U12"),
        is_upper_bound=arrays["is_upper_bound"], in_neighborhood=arrays["in_neighborhood"],
        admissible=arrays["admissible"], selected=arrays["selected"], reasons=arrays["reasons"],
        descriptor=meta.get("descriptor"), thresholds=meta.get("thresholds"),
        candidates_sha256=meta.get("candidates_sha256"), inputs=meta.get("inputs") or {})


# --- verification --------------------------------------------------------------

def verify_structure(cert: CertificateSet, candidates, candidates_path=None) -> list[str]:
    """Problems found from the certificate and its candidates alone (empty when consistent).

    Checks the candidate hash and row identity, then recomputes every value,
    classification, mask, and threshold from the persisted terms and descriptor.
    It cannot tell whether the persisted terms themselves are right: that is
    the source audit's job.
    """
    problems = []
    if candidates_path is not None:
        digest = hashlib.sha256(Path(candidates_path).read_bytes()).hexdigest()
        if digest != cert.candidates_sha256:
            problems.append(f"candidates SHA-256 {digest} != recorded {cert.candidates_sha256}")
    if not (np.array_equal(cert.source_episode_id.astype(str), np.asarray(candidates.source_episode_id).astype(str))
            and np.array_equal(cert.source_frame, candidates.source_frame)):
        return problems + ["certificate rows do not match the candidate rows"]
    try:
        profile = PROFILES[cert.descriptor["profile"]].from_descriptor(cert.descriptor)
    except (KeyError, TypeError, InterfaceError) as e:
        return problems + [f"descriptor does not define a known profile: {e}"]
    if json.dumps(profile.descriptor(), sort_keys=True) != json.dumps(cert.descriptor, sort_keys=True):
        problems.append("descriptor is not the one this version of the profile writes")
    t = cert.thresholds
    model, total, cls, bound, inside, admissible, selected, reasons, thr = _decide(
        profile, cert.terms, np.asarray(candidates.accepted), t.get("max_total_error"), t.get("selection_quantile"))
    checks = {"model_error_value": (model, cert.model_error_value), "total_error_value": (total, cert.total_error_value)}
    for name, (want, got) in checks.items():
        if not np.allclose(got, want, rtol=1e-12, atol=0.0, equal_nan=True):
            problems.append(f"{name} does not follow from the persisted terms on {int((~np.isclose(got, want, rtol=1e-12, atol=0.0, equal_nan=True)).sum())} rows")
    for name, want, got in (("classification", cls, cert.classification), ("is_upper_bound", bound, cert.is_upper_bound),
                            ("in_neighborhood", inside, cert.in_neighborhood), ("admissible", admissible, cert.admissible),
                            ("selected", selected, cert.selected), ("reasons", reasons, cert.reasons)):
        if not np.array_equal(np.asarray(got), np.asarray(want)):
            problems.append(f"{name} does not follow from the persisted terms and descriptor")
    if (thr is None) != (t.get("selection_threshold") is None) or (thr is not None and not np.isclose(thr, t["selection_threshold"], rtol=1e-12)):
        problems.append("selection_threshold does not follow from the persisted totals")
    return problems


def audit_sources(cert: CertificateSet, candidates, dataset, representation, model, rtol: float = 1e-9) -> list[str]:
    """Problems found by recomputing every source term independently (per episode, per row)."""
    episodes = {ep.episode_id: ep for ep in dataset.episodes}
    modality = representation.output_modality
    encoded = {eid: np.asarray(representation.encode(ep.observations), np.float64) for eid, ep in episodes.items()}
    problems = []
    for i in range(len(cert.source_frame)):
        eid, t = str(cert.source_episode_id[i]), int(cert.source_frame[i])
        ep, zs = episodes[eid], encoded[eid]
        z_t, z_n, a_t = zs[t:t + 1], zs[t + 1:t + 2], np.asarray(ep.actions[t:t + 1], np.float64)
        z_g = np.asarray(candidates.observations[modality][i:i + 1], np.float64)
        a_g = np.asarray(candidates.actions[i:i + 1], np.float64)
        y = np.asarray(candidates.target_observations[modality][i:i + 1], np.float64)
        expected = {"state_displacement": float(np.linalg.norm(z_g - z_t)),
                    "action_displacement": float(np.linalg.norm(a_g - a_t)),
                    "anchor_model_error": float(np.linalg.norm(z_t + model.delta(z_t, a_t) - z_n)),
                    "solver_error": float(np.linalg.norm(z_g + model.delta(z_g, a_g) - y))}
        for name, want in expected.items():
            got = float(cert.terms[name][i])
            if not np.isclose(got, want, rtol=rtol, atol=1e-12):
                problems.append(f"row {i}: persisted {name} {got!r} != recomputed {want!r}")
    return problems


def require_upper_bound(cert: CertificateSet) -> None:
    """Raise unless every row the caller would use is an upper bound (certified, conditional, or calibrated)."""
    if not cert.is_upper_bound[cert.admissible].all() or not np.isin(cert.classification[cert.admissible], BOUND_CLASSES).all():
        raise InterfaceError(f"profile {cert.descriptor.get('profile')!r} does not give upper bounds")


__all__ = ["Constant", "ContinuityProfile", "CertificateSet", "certify", "write_certificates", "read_certificates",
           "verify_structure", "audit_sources", "require_upper_bound", "source_terms", "PROFILES", "SCHEMA",
           "VERSION", "CLASSIFICATIONS", "BOUND_CLASSES", "SOURCE_CLASS", "REASON_CODES", "SchemaError"]
