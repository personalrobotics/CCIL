"""Label-error certificates: a post-generation layer over candidate files (#88).

    generate candidates -> certify candidates -> select / admit labels -> train policy

Candidate files are never modified; certify() writes a ccil.certificates set
(src/ccil/schema/certificates.py) with one row per candidate row. Consumers
train on candidate.accepted & admissible & selected.

Constants. A profile's constants are specified, never asserted: Declared(value,
rationale) is an assumption and always conditional, whatever its value;
Provided(provider[, reference]) is computed during certification by a registered
bound provider from artifacts whose SHA-256 the certificate records, and the
source audit recomputes it. Only provided constants can be certified.

Terms. Every certificate persists the four global terms (displacements,
anchor model error, solver error); a profile may declare more, with a
computation and a per-row recomputation hook, and says which terms it
requires. Only required terms gate support.

Coordinates. Norms are taken in scaled coordinates, z / state_scale and
a / action_scale, and errors in the scaled state coordinates; the scales and
dimensions are part of the descriptor and the units are derived from them.

Admission is a policy separate from classification: "upper_bound" admits only
bound rows (certified, conditional, calibrated) within the threshold; "score"
is an explicit opt-in that admits any supported row whose score is within the
threshold, and its rows stay visibly non-bound. Selection is a budget quantile
over accepted, admissible rows and never changes a candidate's acceptance.

Verification has two levels. verify_structure uses the certificate and the
candidates file: identity, the descriptor, and every value, classification,
mask, and threshold recomputed from the persisted terms. audit_sources also
takes the source artifacts: it checks each against its recorded SHA-256 and
component identity, then recomputes every term and provided constant.
Hashes establish artifact identity, not mathematical correctness.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..runs.provenance import sha256
from ..schema import read_candidates, read_trajectories
from ..schema.certificates import (BOUND_CLASSES, CLASSIFICATIONS, GLOBAL_TERMS, REASON_CODES, SCHEMA,
                                   VERSION, CertificateSet)
from ..schema.fields import SchemaError
from ..schema.io import _atomic_savez, _format_error, _open
from .batch import InterfaceError
from .generators import BackwardEuler, ExactPreimage, NoisyAction, NoisyActionPreimage
from .representation import transitions
from .transition import load_transition_model

SOURCE_CLASS = {"analytic_bound": "certified", "model_bound": "certified", "declared": "conditional"}
STATE_UNITS = "scaled state (z / state_scale)"
ACTION_UNITS = "scaled action (a / action_scale)"
GLOBAL_TERM_SPECS = {
    "state_displacement": {"definition": "||(z_G - z_t) / state_scale||_2", "units": STATE_UNITS},
    "action_displacement": {"definition": "||(a_G - a_t) / action_scale||_2", "units": ACTION_UNITS},
    "anchor_model_error": {"definition": "||(z_t + f_hat(z_t, a_t) - z_{t+1}) / state_scale||_2", "units": STATE_UNITS},
    "solver_error": {"definition": "||(z_G + f_hat(z_G, a_G) - y) / state_scale||_2", "units": STATE_UNITS},
}


def _weakest(classes) -> str:
    return max(classes, key=CLASSIFICATIONS.index)


def _nonneg(x, name, allow_none=True):
    if x is None and allow_none:
        return
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise InterfaceError(f"{name} must be finite and >= 0{' or None' if allow_none else ''}, got {x!r}")


# --- constants -------------------------------------------------------------

@dataclass(frozen=True)
class Declared:
    """An assumed constant: conditional, whatever its value."""

    value: float
    rationale: str

    def __post_init__(self):
        _nonneg(self.value, "a declared constant", allow_none=False)
        if not isinstance(self.rationale, str) or not self.rationale:
            raise InterfaceError("a declared constant needs a rationale")

    def spec(self) -> dict:
        return {"declared": float(self.value), "rationale": self.rationale}


@dataclass(frozen=True)
class Provided:
    """A constant a registered bound provider computes, from a named reference artifact when it needs one."""

    provider: str
    reference: str | None = None

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise InterfaceError(f"unknown bound provider {self.provider!r}; registered: {sorted(PROVIDERS)}")
        needs = PROVIDERS[self.provider].needs_reference
        if needs and (not isinstance(self.reference, str) or not self.reference):
            raise InterfaceError(f"provider {self.provider!r} needs a reference artifact name")
        if not needs and self.reference is not None:
            raise InterfaceError(f"provider {self.provider!r} takes no reference artifact")

    def spec(self) -> dict:
        return {"provider": self.provider, "version": PROVIDERS[self.provider].version, "reference": self.reference}


REFERENCE_SCHEMA, REFERENCE_VERSION = "ccil.reference-dynamics", (1, 0)


def write_reference_dynamics(path, A, B, description: str) -> None:
    """A known linear system f(z, a) = A z + B a, the true dynamics an analytic bound is computed against."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    _check_reference(A, B, path)
    meta = {"schema": REFERENCE_SCHEMA, "version": f"{REFERENCE_VERSION[0]}.{REFERENCE_VERSION[1]}",
            "optional_fields": [], "kind": "linear", "description": description}
    _atomic_savez(path, {"A": A, "B": B, "meta": np.array(json.dumps(meta, sort_keys=True))})


def _check_reference(A, B, path):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1] and B.ndim == 2 and B.shape[0] == A.shape[0]
            and A.dtype.kind == "f" and B.dtype.kind == "f" and np.isfinite(A).all() and np.isfinite(B).all()):
        raise _format_error(REFERENCE_SCHEMA, path, [f"A must be finite float (D, D) and B finite float (D, A); "
                                                     f"got {A.shape} {A.dtype}, {B.shape} {B.dtype}"])


def read_reference_dynamics(path) -> tuple[np.ndarray, np.ndarray]:
    z, meta = _open(path, REFERENCE_SCHEMA, REFERENCE_VERSION, {"A", "B"}, ())
    if meta.get("kind") != "linear" or set(z) != {"A", "B"}:
        raise _format_error(REFERENCE_SCHEMA, path, ["a linear reference holds exactly A and B, kind 'linear'"])
    _check_reference(z["A"], z["B"], path)
    return z["A"], z["B"]


def _reference(provider, reference, ctx):
    path = ctx.references.get(reference)
    if path is None:
        raise InterfaceError(f"{provider} needs reference artifact {reference!r}; given {sorted(ctx.references)}")
    A, B = read_reference_dynamics(path)
    sz, sa = ctx.state_scale, ctx.action_scale
    if len(sz) != len(A) or len(sa) != B.shape[1]:
        raise InterfaceError(f"reference dynamics {A.shape}, {B.shape} do not match the coordinates ({len(sz)}, {len(sa)})")
    return A, B


def _scaled(M, maps, ctx) -> np.ndarray:
    """S_z^-1 M S_z for state maps, S_z^-1 M S_a for action maps."""
    sz = ctx.state_scale
    return M / sz[:, None] * (sz if maps == "state" else ctx.action_scale)[None, :]


class LinearDiscrepancy:
    """analytic_bound: the exact variation of f - f_hat when both are linear.

    With f = A z + B a and f_hat = A_hat z + B_hat a + c, f - f_hat changes by
    (A - A_hat) dz + (B - B_hat) da, so in scaled coordinates the state constant
    is ||S_z^-1 (A - A_hat) S_z||_2 and the action constant ||S_z^-1 (B - B_hat) S_a||_2,
    over any neighborhood.
    """

    name, version, source, bounds, needs_reference = "linear_discrepancy", "1", "analytic_bound", "discrepancy", True

    def resolve(self, maps: str, reference: str, ctx: "Context") -> dict:
        A, B = _reference(self.name, reference, ctx)
        model = ctx.model
        if getattr(model, "kind", None) != "linear":
            raise InterfaceError(f"{self.name} needs a linear transition model, got {getattr(model, 'kind', None)!r}")
        if A.shape != model.A.shape or B.shape != model.B.shape:
            raise InterfaceError(f"reference dynamics {A.shape}, {B.shape} do not match the model {model.A.shape}, {model.B.shape}")
        scaled = _scaled((A - model.A) if maps == "state" else (B - model.B), maps, ctx)
        return {"value": float(np.linalg.norm(scaled, 2)), "source": self.source, "maps": maps,
                "rationale": f"exact operator norm of the scaled {'A - A_hat' if maps == 'state' else 'B - B_hat'} "
                             f"against reference {reference!r}",
                "provenance": {"provider": self.name, "version": self.version, "params": {"reference": reference},
                               "inputs": {"reference": ctx.hashes[f"reference.{reference}"],
                                          "transition_model": ctx.hashes["transition_model"]}}}


class ReferenceLipschitz:
    """analytic_bound: the exact Lipschitz constant of the true f when it is a known linear system.

    ||S_z^-1 A S_z||_2 in state and ||S_z^-1 B S_a||_2 in action.
    """

    name, version, source, bounds, needs_reference = "reference_lipschitz", "1", "analytic_bound", "true", True

    def resolve(self, maps: str, reference: str, ctx: "Context") -> dict:
        A, B = _reference(self.name, reference, ctx)
        return {"value": float(np.linalg.norm(_scaled(A if maps == "state" else B, maps, ctx), 2)),
                "source": self.source, "maps": maps,
                "rationale": f"exact operator norm of the scaled {'A' if maps == 'state' else 'B'} of reference {reference!r}",
                "provenance": {"provider": self.name, "version": self.version, "params": {"reference": reference},
                               "inputs": {"reference": ctx.hashes[f"reference.{reference}"]}}}


class ModelLipschitz:
    """model_bound: a global Lipschitz bound of the frozen transition model, from its own lipschitz_bound().

    The model computes it from its weights with exact operator norms (see
    LinearTransitionModel.lipschitz_bound and MLPTransitionModel.lipschitz_bound);
    a model without one is refused.
    """

    name, version, source, bounds, needs_reference = "model_lipschitz", "1", "model_bound", "learned", False

    def resolve(self, maps: str, reference, ctx: "Context") -> dict:
        fn = getattr(ctx.model, "lipschitz_bound", None)
        if fn is None:
            raise InterfaceError(f"transition model kind {getattr(ctx.model, 'kind', None)!r} has no lipschitz_bound")
        try:
            b = fn(maps, ctx.state_scale, ctx.action_scale)
        except ValueError as e:
            raise InterfaceError(f"{self.name}: {e}") from None
        factors = ", ".join(f"{f:.6g}" for f in b["factors"])
        return {"value": float(b["value"]), "source": self.source, "maps": maps,
                "rationale": f"product of exact layer operator norms [{factors}] of the {ctx.model.kind} model, with "
                             f"normalization and output scales folded in; activation {b['activation']}",
                "provenance": {"provider": self.name, "version": self.version, "params": {},
                               "inputs": {"transition_model": ctx.hashes["transition_model"]}}}


PROVIDERS = {p.name: p for p in (LinearDiscrepancy(), ReferenceLipschitz(), ModelLipschitz())}


def resolve_constant(spec, role: tuple, ctx: "Context") -> dict:
    """Resolve a role's (maps, bounds) constant from its specification."""
    maps, bounds = role
    if isinstance(spec, Declared):
        return {"value": float(spec.value), "source": "declared", "maps": maps, "rationale": spec.rationale,
                "provenance": None}
    if isinstance(spec, Provided):
        provider = PROVIDERS[spec.provider]
        if provider.bounds != bounds:
            raise InterfaceError(f"provider {spec.provider!r} bounds the {provider.bounds} dynamics; this constant "
                                 f"needs a bound on the {bounds} dynamics")
        return provider.resolve(maps, spec.reference, ctx)
    raise InterfaceError(f"a constant must be Declared(...) or Provided(...), got {type(spec).__name__}; "
                         "a value cannot be asserted as a bound")


# --- profiles --------------------------------------------------------------

class Profile:
    """A certificate profile. Subclasses set the class attributes and evaluate().

    roles maps each constant to (maps, bounds): the input it maps from ("state"
    or "action") and what it bounds ("true" f, "learned" f_hat, or the
    "discrepancy" f - f_hat); a provided constant must come from a provider
    of that kind. applies_to names the generator label-error profiles
    (generators.label_error_profile) whose candidates the profile can certify,
    or None for any;
    extra_terms declares terms beyond the global ones, which compute_terms()
    computes for every row and audit_term() recomputes for one; required names
    the terms a row needs finite to be supported; intrinsic is the strongest
    classification the profile can give regardless of constants (None, or
    "empirical" for a score).
    """

    name = version = form = ""
    roles: dict = {}
    extra_terms: dict = {}
    required: tuple = GLOBAL_TERMS
    intrinsic: str | None = None
    assumptions: tuple = ()
    applies_to: tuple | None = None

    def __init__(self, constants: dict, neighborhood_state=None, neighborhood_action=None):
        if set(constants) != set(self.roles):
            raise InterfaceError(f"{self.name} needs constants {sorted(self.roles)}, got {sorted(constants)}")
        for role, c in constants.items():
            if not isinstance(c, (Declared, Provided)):
                raise InterfaceError(f"constant {role} must be Declared(...) or Provided(...), got {type(c).__name__}; "
                                     "a value cannot be asserted as a bound")
        for r in (neighborhood_state, neighborhood_action):
            if r is not None and (isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0):
                raise InterfaceError(f"neighborhood radii must be finite and > 0, or None for unbounded; got {r!r}")
        self.constants = dict(constants)
        self.neighborhood = {"state": neighborhood_state, "action": neighborhood_action}

    def spec(self) -> dict:
        """The unresolved profile, for stage identity."""
        return {"profile": self.name, "version": self.version, "neighborhood": self.neighborhood,
                "constants": {r: c.spec() for r, c in sorted(self.constants.items())}}

    def term_specs(self) -> dict:
        specs = {**GLOBAL_TERM_SPECS, **self.extra_terms}
        return {n: {**s, "required": n in self.required} for n, s in specs.items()}

    def descriptor(self, constants: dict, coordinates: dict) -> dict:
        return {"profile": self.name, "version": self.version, "form": self.form, "terms": self.term_specs(),
                "constants": constants, "neighborhood": dict(self.neighborhood), "coordinates": coordinates,
                "norms": {"state": "l2 of scaled coordinates", "action": "l2 of scaled coordinates",
                          "error": "l2 of scaled state coordinates"},
                "units": {"state": STATE_UNITS, "action": ACTION_UNITS, "error": STATE_UNITS,
                          **{r: f"error per {m}" for r, (m, _) in self.roles.items()}},
                "assumptions": list(self.assumptions)}

    def compute_terms(self, ctx: "Context") -> dict:
        return {}

    def audit_term(self, name: str, row: dict) -> float:
        raise InterfaceError(f"{self.name} declares no term {name!r}")

    def evaluate(self, terms: dict, values: dict) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def in_neighborhood(self, terms: dict) -> np.ndarray:
        inside = np.ones(len(terms["state_displacement"]), bool)
        with np.errstate(invalid="ignore"):
            for key, term in (("state", "state_displacement"), ("action", "action_displacement")):
                if self.neighborhood[key] is not None:
                    inside &= terms[term] <= self.neighborhood[key]
        return inside

    def base_classification(self, constants: dict) -> str:
        return _weakest([SOURCE_CLASS[c["source"]] for c in constants.values()]
                        + ([self.intrinsic] if self.intrinsic else []) + ["certified"])

    @classmethod
    def from_descriptor(cls, d: dict) -> "Profile":
        """The profile a descriptor names; its constants come from the descriptor, already resolved."""
        profile = cls.__new__(cls)
        profile.constants = None
        profile.neighborhood = dict(d["neighborhood"])
        return profile


class ContinuityProfile(Profile):
    """Additive: model_error = anchor_model_error + C_z ||dz|| + C_a ||da||.

    C_z and C_a bound the variation of f - f_hat with respect to state and action
    over the neighborhood (C = K_true + K_learned when derived from separate
    bounds); with the demonstration transition exact,
    ||T(q) - y|| <= solver_error + model_error.
    """

    name, version, form = "continuity", "1", "additive"
    roles = {"C_z": ("state", "discrepancy"), "C_a": ("action", "discrepancy")}
    assumptions = ("C_z and C_a bound the variation of f - f_hat over the neighborhood",
                   "the demonstration transition (z_t, a_t, z_{t+1}) is exact")

    def __init__(self, C_z, C_a, neighborhood_state=None, neighborhood_action=None):
        super().__init__({"C_z": C_z, "C_a": C_a}, neighborhood_state, neighborhood_action)

    def evaluate(self, terms, values):
        model = (terms["anchor_model_error"] + values["C_z"] * terms["state_displacement"]
                 + values["C_a"] * terms["action_displacement"])
        return model, terms["solver_error"] + model


class BackTrackProfile(Profile):
    """Ke et al., Thm 4.2 (additive): model_error = anchor_model_error + (K_hat_z + K_z) ||dz||.

    A BackTrack label keeps the anchor's action, so ||f(q) - f_hat(q)|| <=
    ||(f - f_hat)(z_t, a_t)|| + (K_hat_z + K_z) ||z_G - z_t||, and with the
    demonstration transition exact the first term is the anchor model error.
    Then ||T(q) - y|| <= solver_error + model_error. K_hat_z bounds the learned
    f_hat and K_z the true f, both in state. No action constant enters; a row
    whose action differs from its anchor's is outside the profile's
    neighborhood and unsupported.
    """

    name, version, form = "backtrack", "1", "additive"
    roles = {"K_hat_z": ("state", "learned"), "K_z": ("state", "true")}
    applies_to = ("backtrack",)
    assumptions = ("K_hat_z bounds the state Lipschitz constant of f_hat and K_z that of the true f over the neighborhood",
                   "the label keeps the anchor's action",
                   "the demonstration transition (z_t, a_t, z_{t+1}) is exact")

    def __init__(self, K_hat_z, K_z, neighborhood_state=None):
        super().__init__({"K_hat_z": K_hat_z, "K_z": K_z}, neighborhood_state, None)

    def in_neighborhood(self, terms):
        with np.errstate(invalid="ignore"):
            return super().in_neighborhood(terms) & (terms["action_displacement"] == 0)

    def evaluate(self, terms, values):
        model = terms["anchor_model_error"] + (values["K_hat_z"] + values["K_z"]) * terms["state_displacement"]
        return model, terms["solver_error"] + model


class DisturbedActionProfile(Profile):
    """Ke et al., Thm 4.3 (direct): total = K_a ||da|| + (1 + K_z) ||dz|| + ||y - z_{t+1}||.

    With the demonstration transition exact, z_{t+1} = z_t + f(z_t, a_t), so
    T(q) - y = dz + f(z_G, a_G) - f(z_t, a_t) + (z_{t+1} - y), bounded by the
    true constants alone. The learned model enters only through which labels
    were generated: solver_error is recorded as a diagnostic and never added.
    The target offset is 0 for disturbed-action generators, whose target is
    z_{t+1}.
    """

    name, version, form = "disturbed_action", "1", "direct"
    roles = {"K_z": ("state", "true"), "K_a": ("action", "true")}
    applies_to = ("disturbed_action",)
    extra_terms = {"target_offset": {"definition": "||(y - z_{t+1}) / state_scale||_2", "units": STATE_UNITS}}
    required = ("state_displacement", "action_displacement", "target_offset")
    assumptions = ("K_z and K_a bound the state and action Lipschitz constants of the true f over the neighborhood",
                   "the demonstration transition (z_t, a_t, z_{t+1}) is exact")

    def __init__(self, K_z, K_a, neighborhood_state=None, neighborhood_action=None):
        super().__init__({"K_z": K_z, "K_a": K_a}, neighborhood_state, neighborhood_action)

    def compute_terms(self, ctx):
        _, _, zn = _anchor_rows(ctx)
        y = np.asarray(ctx.candidates.target_observations[ctx.modality], np.float64)
        return {"target_offset": np.linalg.norm((y - zn) / ctx.state_scale, axis=1)}

    def audit_term(self, name, row):
        if name != "target_offset":
            return super().audit_term(name, row)
        return float(np.linalg.norm((row["y"] - row["z_next"]) / row["state_scale"]))

    def evaluate(self, terms, values):
        total = (values["K_a"] * terms["action_displacement"] + (1 + values["K_z"]) * terms["state_displacement"]
                 + terms["target_offset"])
        return np.full(len(total), np.nan), total


PROFILES: dict = {}


def register_profile(cls) -> type:
    if PROFILES.get(cls.name, cls) is not cls:
        raise InterfaceError(f"profile name {cls.name!r} is already registered")
    PROFILES[cls.name] = cls
    return cls


for _profile in (ContinuityProfile, BackTrackProfile, DisturbedActionProfile):
    register_profile(_profile)

# Generator name -> the CCIL label-error profile its labels follow (generators.label_error_profile).
GENERATOR_PROFILES = {g.name: g.label_error_profile for g in (BackwardEuler, ExactPreimage, NoisyAction, NoisyActionPreimage)}


def check_applies(profile, candidates) -> None:
    """Raise unless profile can certify these candidates' generator."""
    if profile.applies_to is None:
        return
    name = (candidates.generator or {}).get("name")
    if GENERATOR_PROFILES.get(name) not in profile.applies_to:
        raise InterfaceError(f"profile {profile.name} certifies {list(profile.applies_to)} labels; generator {name!r} "
                             f"produces {GENERATOR_PROFILES.get(name)!r}")


# --- sources and computation ---------------------------------------------

@dataclass
class Sources:
    """The artifacts a certificate is computed from, by path; their SHA-256 are recorded and audited."""

    candidates: Path
    dataset: Path
    transition_model: Path
    representation: object
    extra: dict = field(default_factory=dict)        # e.g. {"representation": checkpoint} for a learned one
    references: dict = field(default_factory=dict)   # name -> path, for bound providers
    device: str = "cpu"

    def paths(self) -> dict:
        return {"candidates": Path(self.candidates), "dataset": Path(self.dataset),
                "transition_model": Path(self.transition_model), **{k: Path(v) for k, v in self.extra.items()},
                **{f"reference.{k}": Path(v) for k, v in self.references.items()}}

    def hashes(self) -> dict:
        return {k: sha256(p) for k, p in sorted(self.paths().items())}


@dataclass
class Context:
    """Loaded sources, in the coordinates of one certificate."""

    candidates: object
    dataset: object
    representation: object
    model: object
    state_scale: np.ndarray
    action_scale: np.ndarray
    references: dict
    hashes: dict

    @property
    def modality(self) -> str:
        return self.representation.output_modality


def _components(representation, model) -> dict:
    return {"representation": {"kind": str(representation.kind), "version": str(representation.version),
                               "modality": str(representation.output_modality)},
            "transition_model": {"kind": str(model.kind), "version": str(model.version)}}


def _scales(value, dim, name) -> np.ndarray:
    s = np.ones(dim) if value is None else np.asarray(value, np.float64)
    if s.shape != (dim,) or not (np.isfinite(s).all() and (s > 0).all()):
        raise InterfaceError(f"{name} must be {dim} finite positive values, got {value!r}")
    return s


def _anchor_rows(ctx: Context):
    """(z_t, a_t, z_{t+1}) of each candidate row's source transition."""
    batch = transitions(ctx.dataset, ctx.representation)
    index = {(str(e), int(f)): i for i, (e, f) in enumerate(zip(batch.episode_id, batch.frame))}
    c = ctx.candidates
    src = np.array([index[(str(e), int(f))] for e, f in zip(c.source_episode_id, c.source_frame)], dtype=np.int64)
    return tuple(np.asarray(x, np.float64)[src] for x in (batch.z, batch.a, batch.z_next))


def global_terms(ctx: Context) -> dict:
    """The global terms for every candidate row, in scaled coordinates."""
    c = ctx.candidates
    z_g = np.asarray(c.observations[ctx.modality], np.float64)
    a_g = np.asarray(c.actions, np.float64)
    y = np.asarray(c.target_observations[ctx.modality], np.float64)
    z, a, zn = _anchor_rows(ctx)
    sz, sa = ctx.state_scale, ctx.action_scale
    return {"state_displacement": np.linalg.norm((z_g - z) / sz, axis=1),
            "action_displacement": np.linalg.norm((a_g - a) / sa, axis=1),
            "anchor_model_error": np.linalg.norm((z + ctx.model.delta(z, a) - zn) / sz, axis=1),
            "solver_error": np.linalg.norm((z_g + ctx.model.delta(z_g, a_g) - y) / sz, axis=1)}


def decide(profile: Profile, constants: dict, terms: dict, accepted, admission: dict, quantile):
    """Values, classification, masks, reasons, and selection threshold from persisted terms.

    Shared by certify() and verify_structure(), so the structural check recomputes
    exactly what certification decided.
    """
    model, total = profile.evaluate(terms, {r: c["value"] for r, c in constants.items()})
    required = [n for n, s in profile.term_specs().items() if s["required"]]
    inside = profile.in_neighborhood(terms)
    supported = inside & np.isfinite(total) & np.all([np.isfinite(terms[n]) for n in required], axis=0)
    cls = np.where(supported, profile.base_classification(constants), "unsupported").astype("<U12")
    bound = np.isin(cls, BOUND_CLASSES)
    candidate = bound if admission["mode"] == "upper_bound" else supported
    with np.errstate(invalid="ignore"):
        within = np.ones(len(total), bool) if admission["threshold"] is None else total <= admission["threshold"]
    admissible = candidate & within
    eligible = np.asarray(accepted, bool) & admissible
    threshold = None
    if quantile is None:
        selected = np.ones(len(total), bool)
    elif eligible.any():
        threshold = float(np.quantile(total[eligible], quantile))
        selected = eligible & (total <= threshold)
    else:
        selected = np.zeros(len(total), bool)
    reasons = np.stack([~supported, (admission["mode"] == "upper_bound") & supported & ~bound,
                        candidate & ~within, eligible & ~selected], axis=1)
    return {"model_error_value": model, "total_error_value": total, "classification": cls, "is_upper_bound": bound,
            "in_neighborhood": inside, "admissible": admissible, "selected": selected, "reasons": reasons}, threshold


def _admission(mode, threshold) -> dict:
    if mode not in ("upper_bound", "score"):
        raise InterfaceError(f"admission must be 'upper_bound' or 'score', got {mode!r}")
    _nonneg(threshold, "the admission threshold")
    return {"mode": mode, "threshold": None if threshold is None else float(threshold)}


def certify(sources: Sources, profile: Profile, admission: str = "upper_bound", threshold: float | None = None,
            selection_quantile: float | None = None, state_scale=None, action_scale=None) -> CertificateSet:
    """Certify every candidate row of sources.candidates with profile; the candidates are not modified."""
    if not isinstance(profile, Profile) or profile.constants is None:
        raise InterfaceError("certify needs a profile built from Declared/Provided constants")
    adm = _admission(admission, threshold)
    if selection_quantile is not None and (isinstance(selection_quantile, bool) or not isinstance(selection_quantile, (int, float))
                                           or not 0 < selection_quantile <= 1):
        raise InterfaceError(f"selection_quantile must be in (0, 1] or None, got {selection_quantile!r}")
    hashes = sources.hashes()
    dataset = read_trajectories(sources.dataset)
    candidates = read_candidates(sources.candidates, dataset)
    check_applies(profile, candidates)
    model = load_transition_model(sources.transition_model, device=sources.device)
    rep = sources.representation
    probe = transitions(dataset, rep)
    ctx = Context(candidates, dataset, rep, model, _scales(state_scale, probe.z.shape[1], "state_scale"),
                  _scales(action_scale, probe.a.shape[1], "action_scale"),
                  {k: Path(v) for k, v in sources.references.items()}, hashes)
    terms = {**global_terms(ctx), **profile.compute_terms(ctx)}
    if set(terms) != set(profile.term_specs()):
        raise InterfaceError(f"{profile.name} computed terms {sorted(terms)}, declares {sorted(profile.term_specs())}")
    constants = {role: resolve_constant(spec, profile.roles[role], ctx) for role, spec in sorted(profile.constants.items())}
    coordinates = {"modality": ctx.modality, "state_dim": int(probe.z.shape[1]), "action_dim": int(probe.a.shape[1]),
                   "state_scale": ctx.state_scale.tolist(), "action_scale": ctx.action_scale.tolist()}
    rows, thr = decide(profile, constants, terms, candidates.accepted, adm, selection_quantile)
    return CertificateSet(
        source_episode_id=np.asarray(candidates.source_episode_id), source_frame=np.asarray(candidates.source_frame),
        terms=terms, descriptor=profile.descriptor(constants, coordinates), admission=adm,
        selection={"quantile": None if selection_quantile is None else float(selection_quantile), "threshold": thr},
        candidates_sha256=hashes["candidates"], inputs=hashes, components=_components(rep, model), **rows).validate()


# --- verification --------------------------------------------------------------

def _profile_of(cert: CertificateSet):
    cls = PROFILES.get(cert.descriptor.get("profile"))
    if cls is None:
        raise InterfaceError(f"profile {cert.descriptor.get('profile')!r} is not registered")
    if cert.descriptor.get("version") != cls.version:
        raise InterfaceError(f"profile {cls.name} version {cert.descriptor.get('version')!r} is not {cls.version!r}")
    return cls.from_descriptor(cert.descriptor)


def verify_structure(cert: CertificateSet, candidates_path) -> list[str]:
    """Problems found from the certificate and its candidates file alone (empty when consistent).

    It cannot tell whether the persisted terms or provided constants are right:
    that is the source audit's job.
    """
    problems = []
    digest = sha256(Path(candidates_path))
    if digest != cert.candidates_sha256:
        problems.append(f"candidates SHA-256 {digest} != recorded {cert.candidates_sha256}")
    try:
        cert.validate()
        candidates = read_candidates(candidates_path)
    except SchemaError as e:
        return problems + [f"schema: {e}"]
    if not (np.array_equal(cert.source_episode_id.astype(str), np.asarray(candidates.source_episode_id).astype(str))
            and np.array_equal(cert.source_frame, candidates.source_frame)):
        return problems + ["certificate rows do not match the candidate rows"]
    try:
        profile = _profile_of(cert)
    except InterfaceError as e:
        return problems + [str(e)]
    d = cert.descriptor
    if set(d["constants"]) != set(profile.roles):
        return problems + [f"descriptor constants {sorted(d['constants'])} != profile roles {sorted(profile.roles)}"]
    try:
        check_applies(profile, candidates)
    except InterfaceError as e:
        problems.append(str(e))
    for role, c in d["constants"].items():
        p = c["provenance"]
        provider = None if p is None else PROVIDERS.get(p["provider"])
        if p is not None and (provider is None or provider.source != c["source"] or provider.version != p["version"]
                              or provider.bounds != profile.roles[role][1]):
            problems.append(f"constant {role} claims {c['source']} from an unregistered or mismatched provider {p}")
    if json.dumps(profile.descriptor(d["constants"], d["coordinates"]), sort_keys=True) != json.dumps(d, sort_keys=True):
        problems.append("descriptor is not the one this version of the profile writes")
    rows, thr = decide(profile, d["constants"], cert.terms, candidates.accepted, cert.admission, cert.selection["quantile"])
    for name in ("model_error_value", "total_error_value"):
        close = np.isclose(getattr(cert, name), rows[name], rtol=1e-12, atol=0.0, equal_nan=True)
        if not close.all():
            problems.append(f"{name} does not follow from the persisted terms on {int((~close).sum())} rows")
    for name in ("classification", "is_upper_bound", "in_neighborhood", "admissible", "selected", "reasons"):
        if not np.array_equal(np.asarray(getattr(cert, name)), rows[name]):
            problems.append(f"{name} does not follow from the persisted terms, descriptor, and policy")
    rec = cert.selection["threshold"]
    if (thr is None) != (rec is None) or (thr is not None and not np.isclose(thr, rec, rtol=1e-12, atol=0.0)):
        problems.append(f"selection threshold {rec!r} does not follow from the persisted totals ({thr!r})")
    return problems


def audit_sources(cert: CertificateSet, sources: Sources, rtol: float = 1e-5) -> list[str]:
    """Problems found by checking each source against the certificate's record and recomputing every term.

    Terms are recomputed one row at a time, independently of certify's batched
    path. A float32 network rounds z + f_hat(z, a) differently in a batch and
    alone, by a few ulps of the state, so terms match to rtol relative plus
    1e-6 * (1 + max |scaled z|) absolute (about 16 float32 ulps of the state).
    Provided constants must match exactly.
    """
    problems = []
    given = sources.hashes()
    for name, recorded in sorted(cert.inputs.items()):
        if name not in given:
            problems.append(f"input {name!r} was recorded but not supplied")
        elif given[name] != recorded:
            problems.append(f"input {name!r} SHA-256 {given[name]} != recorded {recorded}")
    for name in sorted(set(given) - set(cert.inputs)):
        problems.append(f"input {name!r} was supplied but is not recorded")
    if problems:
        return problems  # recomputing from other artifacts would audit something else
    dataset = read_trajectories(sources.dataset)
    candidates = read_candidates(sources.candidates, dataset)
    model = load_transition_model(sources.transition_model, device=sources.device)
    rep = sources.representation
    if _components(rep, model) != cert.components:
        return [f"components {_components(rep, model)} != recorded {cert.components}"]
    try:
        profile = _profile_of(cert)
    except InterfaceError as e:
        return [str(e)]
    co = cert.descriptor["coordinates"]
    sz, sa = np.asarray(co["state_scale"], np.float64), np.asarray(co["action_scale"], np.float64)
    episodes = {ep.episode_id: ep for ep in dataset.episodes}
    encoded = {eid: np.asarray(rep.encode(ep.observations), np.float64) for eid, ep in episodes.items()}
    dims = (next(iter(encoded.values())).shape[1], dataset.episodes[0].actions.shape[1])
    if dims != (co["state_dim"], co["action_dim"]):
        return [f"sources have state and action dims {dims}; recorded ({co['state_dim']}, {co['action_dim']})"]
    m = co["modality"]
    extra = [n for n in cert.terms if n not in GLOBAL_TERMS]
    atol = 1e-6 * (1 + max(float(np.abs(zs / sz).max()) for zs in encoded.values()))
    for i in range(len(cert.source_frame)):
        eid, t = str(cert.source_episode_id[i]), int(cert.source_frame[i])
        ep, zs = episodes[eid], encoded[eid]
        row = {"z_t": zs[t:t + 1], "z_next": zs[t + 1:t + 2], "a_t": np.asarray(ep.actions[t:t + 1], np.float64),
               "z_g": np.asarray(candidates.observations[m][i:i + 1], np.float64),
               "a_g": np.asarray(candidates.actions[i:i + 1], np.float64),
               "y": np.asarray(candidates.target_observations[m][i:i + 1], np.float64),
               "model": model, "state_scale": sz, "action_scale": sa}
        expected = {
            "state_displacement": float(np.linalg.norm((row["z_g"] - row["z_t"]) / sz)),
            "action_displacement": float(np.linalg.norm((row["a_g"] - row["a_t"]) / sa)),
            "anchor_model_error": float(np.linalg.norm((row["z_t"] + model.delta(row["z_t"], row["a_t"]) - row["z_next"]) / sz)),
            "solver_error": float(np.linalg.norm((row["z_g"] + model.delta(row["z_g"], row["a_g"]) - row["y"]) / sz))}
        expected.update({n: float(profile.audit_term(n, row)) for n in extra})
        for name, want in expected.items():
            got = float(cert.terms[name][i])
            if not (np.isclose(got, want, rtol=rtol, atol=atol) or (np.isnan(got) and np.isnan(want))):
                problems.append(f"row {i}: persisted {name} {got!r} != recomputed {want!r}")
    ctx = Context(candidates, dataset, rep, model, sz, sa, {k: Path(v) for k, v in sources.references.items()}, given)
    for role, c in cert.descriptor["constants"].items():
        p = c["provenance"]
        if p is None:
            continue
        again = PROVIDERS[p["provider"]].resolve(c["maps"], p["params"].get("reference"), ctx)
        if again != c:
            problems.append(f"constant {role} {c['value']!r} does not match its provider's recomputation {again['value']!r}")
    return problems


def require_upper_bound(cert: CertificateSet) -> None:
    """Raise unless every supported row is an upper bound and admission admits only bounds."""
    supported = cert.classification != "unsupported"
    if cert.admission["mode"] != "upper_bound" or not cert.is_upper_bound[supported].all():
        raise InterfaceError(f"certificate ({cert.descriptor.get('profile')}, admission {cert.admission['mode']!r}) "
                             "does not admit rows as upper bounds")


__all__ = ["Declared", "Provided", "Profile", "ContinuityProfile", "BackTrackProfile", "DisturbedActionProfile", "Sources", "Context", "certify", "decide",
           "verify_structure", "audit_sources", "require_upper_bound", "register_profile", "write_reference_dynamics",
           "read_reference_dynamics", "check_applies", "PROFILES", "PROVIDERS", "GENERATOR_PROFILES", "SCHEMA", "VERSION", "REASON_CODES"]
