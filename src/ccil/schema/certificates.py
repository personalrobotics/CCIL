"""Certificate schema `ccil.certificates` 1.0.

One row per candidate row of a ccil.candidates file, in the same order, with
the source identity repeated for joining; the candidates file is never
modified. A row records the terms its profile used (every term the descriptor
declares is persisted), model_error_value and total_error_value, a
classification, is_upper_bound (true exactly for certified, conditional, and
calibrated rows), whether the label lies in the profile's neighborhood, and
the admissible and selected masks with independent certificate reasons. The
file records the profile descriptor (form, terms, constants with their
sources and provenance, neighborhood, coordinates and scales, derived units,
assumptions), the admission policy, the selection, the SHA-256 of every input
(always the candidates, dataset, and transition model), and the identity of
the representation and transition model. Training uses
candidate.accepted & admissible & selected.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .fields import ArraySpec, Violations

SCHEMA = "ccil.certificates"
VERSION = (1, 0)
CLASSIFICATIONS = ("certified", "conditional", "calibrated", "empirical", "unsupported")
BOUND_CLASSES = ("certified", "conditional", "calibrated")
REASON_CODES = ("certificate.unsupported", "certificate.not_upper_bound", "certificate.exceeds_threshold",
                "certificate.not_selected")
FORMS = ("additive", "direct", "score")
SOURCES = ("analytic_bound", "model_bound", "declared")
ADMISSION_MODES = ("upper_bound", "score")
# Persisted by every profile, as diagnostics at least.
GLOBAL_TERMS = ("state_displacement", "action_displacement", "anchor_model_error", "solver_error")
REQUIRED_INPUTS = ("candidates", "dataset", "transition_model")
TERM_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

SOURCE_EPISODE = ArraySpec("source_episode_id", "str", ("N",), "-", "Episode of the candidate row; equals the candidates file's.")
SOURCE_FRAME = ArraySpec("source_frame", "int", ("N",), "frames", "Transition index of the candidate row; equals the candidates file's.")
TERM = ArraySpec("term/<name>", "float", ("N",), "per descriptor.terms",
                 "Every term the descriptor declares; NaN allowed only where the row is unsupported or the term is not required.")
MODEL_ERROR = ArraySpec("model_error_value", "float", ("N",), "error",
                        "Additive profiles: the bound on ||f(q) - f_hat(q)||. NaN for direct and score profiles.")
TOTAL_ERROR = ArraySpec("total_error_value", "float", ("N",), "error",
                        "The bound (bound classifications) or score (empirical) on ||T(q) - y||; finite and >= 0 unless unsupported.")
CLASSIFICATION = ArraySpec("classification", "int", ("N",), "-", "Index into classification_codes.")
IS_UPPER_BOUND = ArraySpec("is_upper_bound", "bool", ("N",), "-", "True exactly for certified, conditional, and calibrated rows.")
IN_NEIGHBORHOOD = ArraySpec("in_neighborhood", "bool", ("N",), "-", "The label lies in the profile's neighborhood.")
ADMISSIBLE = ArraySpec("admissible", "bool", ("N",), "-", "Passes the admission policy; never set on unsupported rows.")
SELECTED = ArraySpec("selected", "bool", ("N",), "-",
                     "Budget selection; with a quantile, a subset of accepted & admissible rows, otherwise every row.")
REASONS = ArraySpec("reasons", "bool", ("N", "R"), "-", "reasons[i, r] is True when reason_codes[r] applies to row i.")
ARRAY_SPECS = [SOURCE_EPISODE, SOURCE_FRAME, TERM, MODEL_ERROR, TOTAL_ERROR, CLASSIFICATION, IS_UPPER_BOUND,
               IN_NEIGHBORHOOD, ADMISSIBLE, SELECTED, REASONS]
FIELDS = [
    ("descriptor", "dict", "profile, version, form (additive, direct, score), terms {name: {definition, units, required}}, "
                           "constants {role: {value, source, maps, rationale, provenance}}, neighborhood {state, action} "
                           "(null = unbounded), coordinates {modality, state_dim, action_dim, state_scale, action_scale}, "
                           "units, norms, assumptions."),
    ("admission", "dict", f"mode {list(ADMISSION_MODES)} and threshold (null or finite >= 0) on total_error_value."),
    ("selection", "dict", "quantile (null or in (0, 1]) and the resulting threshold (null when no quantile or no eligible row)."),
    ("candidates_sha256", "str", "SHA-256 of the candidates file."),
    ("inputs", "dict[str, str]", f"SHA-256 of each input by name; always {list(REQUIRED_INPUTS)}."),
    ("components", "dict", "representation {kind, version, modality} and transition_model {kind, version}."),
    ("classification_codes", "list[str]", f"Exactly {list(CLASSIFICATIONS)}."),
    ("reason_codes", "list[str]", f"Exactly {list(REASON_CODES)}."),
]


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sha_map(x) -> bool:
    return isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) and SHA256.match(v) for k, v in x.items())


def _check_descriptor(d, out: Violations) -> dict | None:
    """Structural rules of the descriptor; returns the terms spec when usable."""
    w = "descriptor"
    if not isinstance(d, dict):
        out.add("descriptor", w, "descriptor must be an object")
        return None
    for k in ("profile", "version"):
        if not isinstance(d.get(k), str) or not d.get(k):
            out.add("descriptor", w, f"{k} must be a non-empty string")
    if d.get("form") not in FORMS:
        out.add("descriptor", w, f"form must be one of {list(FORMS)}, got {d.get('form')!r}")
    terms = d.get("terms")
    if not isinstance(terms, dict) or any(
            not (isinstance(n, str) and TERM_NAME.match(n) and isinstance(s, dict) and isinstance(s.get("definition"), str)
                 and isinstance(s.get("units"), str) and isinstance(s.get("required"), bool)) for n, s in terms.items()):
        out.add("descriptor", w, "terms must map names to {definition: str, units: str, required: bool}")
        terms = None
    elif not set(GLOBAL_TERMS) <= set(terms):
        out.add("descriptor", w, f"terms must include the global terms {list(GLOBAL_TERMS)}")
    constants = d.get("constants")
    if not isinstance(constants, dict):
        out.add("descriptor", w, "constants must be an object")
    else:
        for role, c in constants.items():
            where = f"{w}.constants.{role}"
            if not isinstance(c, dict) or set(c) != {"value", "source", "maps", "rationale", "provenance"}:
                out.add("descriptor", where, "a constant has exactly value, source, maps, rationale, provenance")
                continue
            if not (_number(c["value"]) and c["value"] >= 0):
                out.add("descriptor", where, f"value must be finite and >= 0, got {c['value']!r}")
            if c["source"] not in SOURCES:
                out.add("descriptor", where, f"source must be one of {list(SOURCES)}, got {c['source']!r}")
            if c["maps"] not in ("state", "action"):
                out.add("descriptor", where, "maps must be 'state' or 'action'")
            if not isinstance(c["rationale"], str) or not c["rationale"]:
                out.add("descriptor", where, "rationale must be a non-empty string")
            p = c["provenance"]
            if c["source"] == "declared":
                if p is not None:
                    out.add("descriptor", where, "a declared constant has no provenance")
            elif not (isinstance(p, dict) and set(p) == {"provider", "version", "params", "inputs"}
                      and isinstance(p["provider"], str) and isinstance(p["version"], str)
                      and isinstance(p["params"], dict) and _sha_map(p["inputs"]) and p["inputs"]):
                out.add("descriptor", where, "a provided constant needs provenance {provider, version, params, inputs (SHA-256)}")
    nb = d.get("neighborhood")
    if not (isinstance(nb, dict) and set(nb) == {"state", "action"}
            and all(v is None or (_number(v) and v > 0) for v in nb.values())):
        out.add("descriptor", w, "neighborhood must be {state, action}, each null or finite > 0")
    co = d.get("coordinates")
    if not (isinstance(co, dict) and set(co) == {"modality", "state_dim", "action_dim", "state_scale", "action_scale"}
            and isinstance(co["modality"], str) and all(type(co[k]) is int and co[k] > 0 for k in ("state_dim", "action_dim"))
            and all(isinstance(co[s], list) and len(co[s]) == co[n] and all(_number(x) and x > 0 for x in co[s])
                    for s, n in (("state_scale", "state_dim"), ("action_scale", "action_dim")))):
        out.add("descriptor", w, "coordinates must be {modality, state_dim, action_dim, state_scale, action_scale} "
                                 "with positive finite scales of the stated dimensions")
    if not (isinstance(d.get("units"), dict) and all(isinstance(v, str) for v in d["units"].values())):
        out.add("descriptor", w, "units must map names to strings")
    if not (isinstance(d.get("assumptions"), list) and all(isinstance(a, str) for a in d["assumptions"])):
        out.add("descriptor", w, "assumptions must be a list of strings")
    return terms


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
    admission: dict
    selection: dict
    candidates_sha256: str
    inputs: dict
    components: dict = field(default_factory=dict)

    def validate(self) -> "CertificateSet":
        """Raise SchemaError listing every violation of the structure and its invariants."""
        out = Violations(SCHEMA)
        w = "certificates"
        terms_spec = _check_descriptor(self.descriptor, out)
        dims: dict = {}
        ok = all([SOURCE_EPISODE.check(self.source_episode_id, w, dims, out),
                  SOURCE_FRAME.check(self.source_frame, w, dims, out),
                  *(s.check(x, w, dims, out, finite=False) for s, x in (
                      (MODEL_ERROR, self.model_error_value), (TOTAL_ERROR, self.total_error_value))),
                  *(s.check(x, w, dims, out) for s, x in (
                      (IS_UPPER_BOUND, self.is_upper_bound), (IN_NEIGHBORHOOD, self.in_neighborhood),
                      (ADMISSIBLE, self.admissible), (SELECTED, self.selected)))])
        ok = REASONS.check(self.reasons, w, {**dims, "R": len(REASON_CODES)}, out) and ok
        cls = np.asarray(self.classification)
        if cls.shape != (dims.get("N", -1),) or not np.isin(cls, CLASSIFICATIONS).all():
            out.add("classification", w, f"classification must be (N,) codes from {list(CLASSIFICATIONS)}")
            ok = False
        if not isinstance(self.terms, dict):
            out.add("terms", w, "terms must be a dict")
            ok = False
        else:
            for name, x in self.terms.items():
                ok = ArraySpec(f"term/{name}", "float", ("N",), "-", "").check(x, w, dims, out, finite=False) and ok
            if terms_spec is not None and set(self.terms) != set(terms_spec):
                out.add("terms", w, f"persisted terms {sorted(self.terms)} != declared {sorted(terms_spec)}")
                ok = False
        self._check_meta(out)
        if ok and terms_spec is not None and not out.items:
            self._check_rows(terms_spec, out)
        out.raise_if_any()
        return self

    def _check_meta(self, out):
        w = "certificates"
        a = self.admission
        if not (isinstance(a, dict) and set(a) == {"mode", "threshold"} and a["mode"] in ADMISSION_MODES
                and (a["threshold"] is None or (_number(a["threshold"]) and a["threshold"] >= 0))):
            out.add("admission", w, f"admission must be {{mode: {list(ADMISSION_MODES)}, threshold: null or finite >= 0}}")
        s = self.selection
        if not (isinstance(s, dict) and set(s) == {"quantile", "threshold"}
                and (s["quantile"] is None or (_number(s["quantile"]) and 0 < s["quantile"] <= 1))
                and (s["threshold"] is None or (_number(s["threshold"]) and s["threshold"] >= 0))
                and not (s["quantile"] is None and s["threshold"] is not None)):
            out.add("selection", w, "selection must be {quantile: null or in (0, 1], threshold: null or finite >= 0}, "
                                    "with no threshold without a quantile")
        if not (isinstance(self.candidates_sha256, str) and SHA256.match(self.candidates_sha256)):
            out.add("inputs", w, "candidates_sha256 must be 64 lowercase hex")
        if not _sha_map(self.inputs) or not set(REQUIRED_INPUTS) <= set(self.inputs):
            out.add("inputs", w, f"inputs must map names to SHA-256 and include {list(REQUIRED_INPUTS)}")
        elif self.inputs["candidates"] != self.candidates_sha256:
            out.add("inputs", w, "inputs.candidates must equal candidates_sha256")
        c = self.components
        if not (isinstance(c, dict) and set(c) == {"representation", "transition_model"}
                and isinstance(c["representation"], dict) and set(c["representation"]) == {"kind", "version", "modality"}
                and isinstance(c["transition_model"], dict) and set(c["transition_model"]) == {"kind", "version"}
                and all(isinstance(v, str) for part in c.values() for v in part.values())):
            out.add("components", w, "components must be representation {kind, version, modality} and "
                                     "transition_model {kind, version}, all strings")

    def _check_rows(self, terms_spec, out):
        """Invariants between the arrays; the profile-specific arithmetic is verify_structure's job."""
        w = "certificates"
        cls = np.asarray(self.classification)
        unsupported = cls == "unsupported"
        rule = lambda name, bad, detail: [out.add(name, f"{w}, row {int(i)}", detail) for i in np.flatnonzero(bad)]  # noqa: E731
        rule("is_upper_bound", self.is_upper_bound != np.isin(cls, BOUND_CLASSES),
             "is_upper_bound must be true exactly for certified, conditional, and calibrated rows")
        rule("in_neighborhood", ~self.in_neighborhood & ~unsupported, "a row outside the neighborhood must be unsupported")
        total = self.total_error_value
        with np.errstate(invalid="ignore"):
            rule("total_error_value", ~unsupported & ~(np.isfinite(total) & (total >= 0)),
                 "total_error_value must be finite and >= 0 on supported rows")
            for name, spec in terms_spec.items():
                if spec["required"]:
                    rule("terms", ~unsupported & ~np.isfinite(self.terms[name]), f"required term {name} is not finite on a supported row")
        additive = self.descriptor["form"] == "additive"
        model = self.model_error_value
        with np.errstate(invalid="ignore"):
            rule("model_error_value", (~unsupported & ~(np.isfinite(model) & (model >= 0))) if additive else ~np.isnan(model),
                 "model_error_value must be finite and >= 0 on supported rows of an additive profile, NaN otherwise")
        rule("admissible", self.admissible & unsupported, "an unsupported row cannot be admissible")
        if self.admission.get("mode") == "upper_bound":
            rule("admissible", self.admissible & ~self.is_upper_bound, "upper_bound admission admits only bounds")
        if self.selection.get("quantile") is not None:
            rule("selected", self.selected & ~self.admissible, "with a quantile, selected rows must be admissible")
        rule("reasons", self.reasons[:, 0] != unsupported, "certificate.unsupported must be set exactly on unsupported rows")
        rule("reasons", self.reasons[:, 3] & self.selected, "certificate.not_selected on a selected row")


def code_classifications(classification) -> np.ndarray:
    return np.array([CLASSIFICATIONS.index(c) for c in classification], np.int8).reshape(len(classification))


def decode_classifications(codes) -> np.ndarray:
    return np.array([CLASSIFICATIONS[int(c)] for c in codes], dtype="<U12").reshape(len(codes))
