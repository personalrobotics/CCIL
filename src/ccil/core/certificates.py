"""Label-error certificates: a post-generation layer over candidate files (#88).

    generate candidates -> certify candidates -> select / admit labels -> train policy

Candidate files are never modified; certify() writes a ccil.certificates set
(src/ccil/schema/certificates.py) with one row per candidate row. Consumers
train on candidate.accepted & admissible & selected.

Constants. A profile's constants are specified, never asserted: Declared(value,
rationale) is an assumption and always conditional, whatever its value;
Provided(provider[, reference]) is computed during certification by a
registered bound provider from artifacts whose SHA-256 the certificate records,
and the source audit recomputes it. A provider's analytic_bound is an upper
bound verified in exact arithmetic (core/verified.py); its model_bound is a
floating-point bound that holds for the model in exact real arithmetic and is
therefore conditional.

Demonstrations. Every bound relies on the demonstration transition
(z_t, a_t, z_{t+1}); stored transitions are exact only up to their rounding. A
profile that uses one adds the demonstration residual
rho_t = ||S_z^-1 (T(z_t, a_t) - z_{t+1})|| once to its total: computed per row
as a verified upper bound from reference dynamics (Provided("reference_residual",
name)), or a declared bound on every rho_t (Declared), which is conditional.

Terms. Every certificate persists the four global terms (displacements,
anchor model error, solver error); a profile may declare more, with a
computation and a per-row recomputation hook, and says which terms it
requires. Only required terms gate support. Displacements are verified upper
bounds; the model terms are too when the model bounds its own evaluation error
(delta_error_bound, as the float64 linear model does); otherwise the
descriptor records model_evaluation "assumed_exact" and a profile whose total
uses them is at most conditional.

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
component identity, recomputes every term one row at a time to a row-local
tolerance, recomputes every provided constant, and re-verifies every
analytic_bound exactly. Hashes establish artifact identity, not mathematical
correctness.
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
                                   SELECTION_POPULATIONS, VERSION, CertificateSet)
from ..schema.fields import SchemaError
from ..schema.io import _atomic_savez, _format_error, _open
from .batch import InterfaceError
from .generators import BackwardEuler, ExactPreimage, NoisyAction, NoisyActionPreimage
from .representation import transitions
from .transition import load_transition_model
from .verified import U32, U64, gamma, is_spectral_upper_bound, sum_upper, up, upper_norm, verified_spectral_norm

SOURCE_CLASS = {"analytic_bound": "certified", "model_bound": "conditional", "declared": "conditional"}
STATE_UNITS = "scaled state (z / state_scale)"
ACTION_UNITS = "scaled action (a / action_scale)"
GLOBAL_TERM_SPECS = {
    "state_displacement": {"definition": "upper bound on ||(z_G - z_t) / state_scale||_2", "units": STATE_UNITS},
    "action_displacement": {"definition": "upper bound on ||(a_G - a_t) / action_scale||_2", "units": ACTION_UNITS},
    "anchor_model_error": {"definition": "||(z_t + f_hat(z_t, a_t) - z_{t+1}) / state_scale||_2 (an upper bound "
                                         "when model_evaluation is bounded)", "units": STATE_UNITS},
    "solver_error": {"definition": "||(z_G + f_hat(z_G, a_G) - y) / state_scale||_2 (an upper bound when "
                                   "model_evaluation is bounded)", "units": STATE_UNITS},
}
DEMONSTRATION_TERM = {"definition": "upper bound on ||(T(z_t, a_t) - z_{t+1}) / state_scale||_2, the demonstration "
                                    "residual under the true dynamics T", "units": STATE_UNITS}
# Row-local audit tolerance: ulps of the operands' magnitude, in the precision the term was evaluated in.
AUDIT_ULPS = 64


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
    """An assumed constant (or demonstration-residual bound): conditional, whatever its value."""

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
    """A quantity a registered bound provider computes, from a named reference artifact when it needs one."""

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
    return z["A"].astype(np.float64), z["B"].astype(np.float64)


def _reference(provider, reference, ctx):
    path = ctx.references.get(reference)
    if path is None:
        raise InterfaceError(f"{provider} needs reference artifact {reference!r}; given {sorted(ctx.references)}")
    A, B = read_reference_dynamics(path)
    if len(ctx.state_scale) != len(A) or len(ctx.action_scale) != B.shape[1]:
        raise InterfaceError(f"reference dynamics {A.shape}, {B.shape} do not match the coordinates "
                             f"({len(ctx.state_scale)}, {len(ctx.action_scale)})")
    return A, B


def _cols(maps, ctx):
    return ctx.state_scale if maps == "state" else ctx.action_scale


def _provenance(provider, params, inputs):
    return {"provider": provider.name, "version": provider.version, "params": params, "inputs": inputs}


class LinearDiscrepancy:
    """analytic_bound: the variation of f - f_hat when both are linear, verified exactly.

    With f = A z + B a and f_hat = A_hat z + B_hat a + c, f - f_hat changes by
    (A - A_hat) dz + (B - B_hat) da, so in scaled coordinates the state constant
    is ||S_z^-1 (A - A_hat) S_z||_2 and the action constant ||S_z^-1 (B - B_hat) S_a||_2,
    over any neighborhood.
    """

    name, version, sources, bounds, needs_reference = "linear_discrepancy", "2", ("analytic_bound",), "discrepancy", True

    def _operands(self, maps, reference, ctx):
        A, B = _reference(self.name, reference, ctx)
        model = ctx.model
        if getattr(model, "kind", None) != "linear":
            raise InterfaceError(f"{self.name} needs a linear transition model, got {getattr(model, 'kind', None)!r}")
        if A.shape != model.A.shape or B.shape != model.B.shape:
            raise InterfaceError(f"reference dynamics {A.shape}, {B.shape} do not match the model {model.A.shape}, {model.B.shape}")
        return (A, model.A) if maps == "state" else (B, model.B)

    def resolve(self, maps: str, reference: str, ctx: "Context") -> dict:
        M, M_hat = self._operands(maps, reference, ctx)
        return {"value": verified_spectral_norm(M, row_div=ctx.state_scale, col_mul=_cols(maps, ctx), minus=M_hat),
                "source": "analytic_bound", "maps": maps,
                "rationale": f"verified operator norm of the scaled {'A - A_hat' if maps == 'state' else 'B - B_hat'} "
                             f"against reference {reference!r}",
                "provenance": _provenance(self, {"reference": reference},
                                          {"reference": ctx.hashes[f"reference.{reference}"],
                                           "transition_model": ctx.hashes["transition_model"]})}

    def verify(self, value, maps, reference, ctx) -> bool:
        M, M_hat = self._operands(maps, reference, ctx)
        return is_spectral_upper_bound(M, value, row_div=ctx.state_scale, col_mul=_cols(maps, ctx), minus=M_hat)


class ReferenceLipschitz:
    """analytic_bound: the Lipschitz constant of the true f when it is a known linear system, verified exactly.

    ||S_z^-1 A S_z||_2 in state and ||S_z^-1 B S_a||_2 in action.
    """

    name, version, sources, bounds, needs_reference = "reference_lipschitz", "2", ("analytic_bound",), "true", True

    def resolve(self, maps: str, reference: str, ctx: "Context") -> dict:
        A, B = _reference(self.name, reference, ctx)
        return {"value": verified_spectral_norm(A if maps == "state" else B, row_div=ctx.state_scale, col_mul=_cols(maps, ctx)),
                "source": "analytic_bound", "maps": maps,
                "rationale": f"verified operator norm of the scaled {'A' if maps == 'state' else 'B'} of reference {reference!r}",
                "provenance": _provenance(self, {"reference": reference},
                                          {"reference": ctx.hashes[f"reference.{reference}"]})}

    def verify(self, value, maps, reference, ctx) -> bool:
        A, B = _reference(self.name, reference, ctx)
        return is_spectral_upper_bound(A if maps == "state" else B, value, row_div=ctx.state_scale, col_mul=_cols(maps, ctx))


class ModelLipschitz:
    """A Lipschitz bound of the frozen transition model, from its own lipschitz_bound().

    A model whose bound is verified exactly (the linear model) gives an
    analytic_bound; any other (the MLP: exact layer norms in floating point, for
    the network in exact real arithmetic) gives a model_bound, which is
    conditional. A model without lipschitz_bound is refused.
    """

    name, version, sources, bounds, needs_reference = "model_lipschitz", "2", ("analytic_bound", "model_bound"), "learned", False

    def resolve(self, maps: str, reference, ctx: "Context") -> dict:
        fn = getattr(ctx.model, "lipschitz_bound", None)
        if fn is None:
            raise InterfaceError(f"transition model kind {getattr(ctx.model, 'kind', None)!r} has no lipschitz_bound")
        try:
            b = fn(maps, ctx.state_scale, ctx.action_scale)
        except ValueError as e:
            raise InterfaceError(f"{self.name}: {e}") from None
        verified = bool(b.get("verified"))
        rationale = ("verified operator norm of the linear model's scaled map" if verified else
                     f"product of layer operator norms of the {ctx.model.kind} model with normalization and output "
                     f"scales folded in (activation {b['activation']}); holds for the network in exact real "
                     "arithmetic, so float32 evaluation and float64 norm rounding are assumptions")
        return {"value": float(b["value"]), "source": "analytic_bound" if verified else "model_bound", "maps": maps,
                "rationale": rationale,
                "provenance": _provenance(self, {}, {"transition_model": ctx.hashes["transition_model"]})}

    def verify(self, value, maps, reference, ctx) -> bool:
        fn = getattr(ctx.model, "verify_lipschitz_bound", None)
        return bool(fn and fn(maps, value, ctx.state_scale, ctx.action_scale))


class ReferenceResidual:
    """analytic_bound per row: rho_t = ||S_z^-1 (z_t + A z_t + B a_t - z_{t+1})|| under reference dynamics.

    Evaluated in float64 with its forward error bound added, so each value is
    an upper bound on the exact residual of the stored transition.
    """

    name, version, sources, bounds, needs_reference = "reference_residual", "1", ("analytic_bound",), "demonstration", True

    def residuals(self, reference, z, a, zn, ctx) -> tuple[np.ndarray, np.ndarray]:
        """(upper bounds, operand magnitudes) per row."""
        A, B = _reference(self.name, reference, ctx)
        Tz = z + z @ A.T + a @ B.T
        n = A.shape[1] + B.shape[1] + 2
        mag = np.abs(z) + np.abs(z) @ np.abs(A).T + np.abs(a) @ np.abs(B).T + np.abs(zn)
        rho = upper_norm(Tz - zn, up(gamma(n) * mag * (1 + gamma(n + 2))), ctx.state_scale)
        return rho, np.linalg.norm(mag / ctx.state_scale, axis=1)

    def describe(self, reference, ctx) -> dict:
        return {"source": "analytic_bound", "value": None,
                "rationale": f"per-row residual of the stored transition under reference {reference!r}, with its "
                             "float64 forward error bound",
                "provenance": _provenance(self, {"reference": reference},
                                          {"reference": ctx.hashes[f"reference.{reference}"]})}


PROVIDERS = {p.name: p for p in (LinearDiscrepancy(), ReferenceLipschitz(), ModelLipschitz(), ReferenceResidual())}


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
    or None for any; extra_terms declares terms beyond the global ones, which
    compute_terms() computes for every row and audit_term() recomputes for one;
    required names the terms a row needs finite to be supported; intrinsic is
    the strongest classification the profile can give regardless of constants
    (None, or "empirical" for a score). demonstration_term adds the required
    term demonstration_residual, specified by the demonstration argument;
    uses_model_terms says the total uses anchor_model_error or solver_error.
    """

    name = version = form = ""
    roles: dict = {}
    extra_terms: dict = {}
    required: tuple = GLOBAL_TERMS
    intrinsic: str | None = None
    assumptions: tuple = ()
    applies_to: tuple | None = None
    demonstration_term = False
    uses_model_terms = False

    def __init__(self, constants: dict, neighborhood_state=None, neighborhood_action=None, demonstration=None):
        if set(constants) != set(self.roles):
            raise InterfaceError(f"{self.name} needs constants {sorted(self.roles)}, got {sorted(constants)}")
        for role, c in constants.items():
            if not isinstance(c, (Declared, Provided)):
                raise InterfaceError(f"constant {role} must be Declared(...) or Provided(...), got {type(c).__name__}; "
                                     "a value cannot be asserted as a bound")
        for r in (neighborhood_state, neighborhood_action):
            if r is not None and (isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0):
                raise InterfaceError(f"neighborhood radii must be finite and > 0, or None for unbounded; got {r!r}")
        if self.demonstration_term:
            if not (isinstance(demonstration, Declared)
                    or (isinstance(demonstration, Provided) and PROVIDERS[demonstration.provider].bounds == "demonstration")):
                raise InterfaceError(f"{self.name} needs demonstration=Provided('reference_residual', <reference>) or "
                                     "Declared(<bound on every demonstration residual>, <rationale>)")
        elif demonstration is not None:
            raise InterfaceError(f"{self.name} takes no demonstration specification")
        self.constants = dict(constants)
        self.demonstration = demonstration
        self.neighborhood = {"state": neighborhood_state, "action": neighborhood_action}

    def spec(self) -> dict:
        """The unresolved profile, for stage identity."""
        out = {"profile": self.name, "version": self.version, "neighborhood": self.neighborhood,
               "constants": {r: c.spec() for r, c in sorted(self.constants.items())}}
        if self.demonstration_term:
            out["demonstration"] = self.demonstration.spec()
        return out

    def term_specs(self) -> dict:
        specs = {**GLOBAL_TERM_SPECS, **({"demonstration_residual": DEMONSTRATION_TERM} if self.demonstration_term else {}),
                 **self.extra_terms}
        required = set(self.required) | ({"demonstration_residual"} if self.demonstration_term else set())
        return {n: {**s, "required": n in required} for n, s in specs.items()}

    def descriptor(self, resolved: dict, coordinates: dict) -> dict:
        d = {"profile": self.name, "version": self.version, "form": self.form, "terms": self.term_specs(),
             "constants": resolved["constants"], "neighborhood": dict(self.neighborhood), "coordinates": coordinates,
             "model_evaluation": resolved["model_evaluation"],
             "norms": {"state": "l2 of scaled coordinates", "action": "l2 of scaled coordinates",
                       "error": "l2 of scaled state coordinates"},
             "units": {"state": STATE_UNITS, "action": ACTION_UNITS, "error": STATE_UNITS,
                       **{r: f"error per {m}" for r, (m, _) in self.roles.items()}},
             "assumptions": list(self.assumptions)}
        if self.demonstration_term:
            d["demonstration"] = resolved["demonstration"]
        return d

    def compute_terms(self, ctx: "Context") -> dict:
        return {}

    def audit_term(self, name: str, row: dict) -> tuple[float, float]:
        """(recomputed value, absolute tolerance) of a profile term for one row."""
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

    def base_classification(self, resolved: dict) -> str:
        classes = [SOURCE_CLASS[c["source"]] for c in resolved["constants"].values()]
        if self.demonstration_term:
            classes.append(SOURCE_CLASS[resolved["demonstration"]["source"]])
        if self.uses_model_terms and resolved["model_evaluation"] != "bounded":
            classes.append("conditional")
        return _weakest(classes + ([self.intrinsic] if self.intrinsic else []) + ["certified"])

    @classmethod
    def from_descriptor(cls, d: dict) -> "Profile":
        """The profile a descriptor names; its constants come from the descriptor, already resolved."""
        profile = cls.__new__(cls)
        profile.constants = profile.demonstration = None
        profile.neighborhood = dict(d["neighborhood"])
        return profile


def resolved_of(d: dict) -> dict:
    """The parts of a descriptor decide() needs: constants, demonstration, and model_evaluation."""
    return {"constants": d["constants"], "demonstration": d.get("demonstration"),
            "model_evaluation": d.get("model_evaluation")}


class ContinuityProfile(Profile):
    """Additive: model_error = anchor_model_error + demonstration_residual + C_z ||dz|| + C_a ||da||.

    C_z and C_a bound the variation of f - f_hat with respect to state and action
    over the neighborhood (C = K_true + K_learned when derived from separate
    bounds). ||(f - f_hat)(z_t, a_t)|| <= anchor_model_error + rho_t, so
    ||T(q) - y|| <= solver_error + model_error.
    """

    name, version, form = "continuity", "2", "additive"
    roles = {"C_z": ("state", "discrepancy"), "C_a": ("action", "discrepancy")}
    demonstration_term = uses_model_terms = True
    assumptions = ("C_z and C_a bound the variation of f - f_hat over the neighborhood",)

    def __init__(self, C_z, C_a, demonstration, neighborhood_state=None, neighborhood_action=None):
        super().__init__({"C_z": C_z, "C_a": C_a}, neighborhood_state, neighborhood_action, demonstration)

    def evaluate(self, terms, values):
        model = (terms["anchor_model_error"] + terms["demonstration_residual"]
                 + values["C_z"] * terms["state_displacement"] + values["C_a"] * terms["action_displacement"])
        return model, terms["solver_error"] + model


class BackTrackProfile(Profile):
    """Ke et al., Thm 4.2 (additive): model_error = anchor_model_error + demonstration_residual + (K_hat_z + K_z) ||dz||.

    A BackTrack label keeps the anchor's action, so ||f(q) - f_hat(q)|| <=
    ||(f - f_hat)(z_t, a_t)|| + (K_hat_z + K_z) ||z_G - z_t||, and
    ||(f - f_hat)(z_t, a_t)|| <= anchor_model_error + rho_t. Then
    ||T(q) - y|| <= solver_error + model_error. K_hat_z bounds the learned
    f_hat and K_z the true f, both in state. No action constant enters; a row
    whose action differs from its anchor's is outside the profile's
    neighborhood and unsupported.
    """

    name, version, form = "backtrack", "2", "additive"
    roles = {"K_hat_z": ("state", "learned"), "K_z": ("state", "true")}
    applies_to = ("backtrack",)
    demonstration_term = uses_model_terms = True
    assumptions = ("K_hat_z bounds the state Lipschitz constant of f_hat and K_z that of the true f over the neighborhood",
                   "the label keeps the anchor's action")

    def __init__(self, K_hat_z, K_z, demonstration, neighborhood_state=None):
        super().__init__({"K_hat_z": K_hat_z, "K_z": K_z}, neighborhood_state, None, demonstration)

    def in_neighborhood(self, terms):
        with np.errstate(invalid="ignore"):
            return super().in_neighborhood(terms) & (terms["action_displacement"] == 0)

    def evaluate(self, terms, values):
        model = (terms["anchor_model_error"] + terms["demonstration_residual"]
                 + (values["K_hat_z"] + values["K_z"]) * terms["state_displacement"])
        return model, terms["solver_error"] + model


class DisturbedActionProfile(Profile):
    """Direct label-error bound for DisturbedAction labels, a corollary of the proof of Ke et al., Thm 4.3.

        total = K_a ||da|| + (1 + K_z) ||dz|| + demonstration_residual + target_offset

    Thm 4.3 is stated for the model discrepancy; its proof gives the label
    error directly: T(q) - y = dz + f(z_G, a_G) - f(z_t, a_t) + (T(z_t, a_t) -
    z_{t+1}) + (z_{t+1} - y), bounded by the true constants, rho_t, and the
    target offset (0 for disturbed-action generators, whose target is z_{t+1}).
    No learned quantity enters: solver_error is recorded as a diagnostic and
    never added.
    """

    name, version, form = "disturbed_action", "2", "direct"
    roles = {"K_z": ("state", "true"), "K_a": ("action", "true")}
    applies_to = ("disturbed_action",)
    demonstration_term = True
    extra_terms = {"target_offset": {"definition": "upper bound on ||(y - z_{t+1}) / state_scale||_2", "units": STATE_UNITS}}
    required = ("state_displacement", "action_displacement", "target_offset")
    assumptions = ("K_z and K_a bound the state and action Lipschitz constants of the true f over the neighborhood",)

    def __init__(self, K_z, K_a, demonstration, neighborhood_state=None, neighborhood_action=None):
        super().__init__({"K_z": K_z, "K_a": K_a}, neighborhood_state, neighborhood_action, demonstration)

    def compute_terms(self, ctx):
        _, _, zn = _anchor_rows(ctx)
        y = np.asarray(ctx.candidates.target_observations[ctx.modality], np.float64)
        return {"target_offset": _difference_norm(y, zn, ctx.state_scale)[0]}

    def audit_term(self, name, row):
        if name != "target_offset":
            return super().audit_term(name, row)
        value, mag = _difference_norm(row["y"], row["z_next"], row["state_scale"])
        return float(value[0]), AUDIT_ULPS * U64 * float(mag[0])

    def evaluate(self, terms, values):
        total = (values["K_a"] * terms["action_displacement"] + (1 + values["K_z"]) * terms["state_displacement"]
                 + terms["demonstration_residual"] + terms["target_offset"])
        return np.full(len(total), np.nan), total


JACOBIAN_TERMS = {
    "state_jacobian_norm": {"definition": "||S_z^-1 J_S f_hat(z_t, a_t) S_z||_2: spectral norm of the exact state "
                                          "Jacobian at the anchor", "units": "error per scaled state"},
    "action_jacobian_norm": {"definition": "||S_z^-1 J_A f_hat(z_t, a_t) S_a||_2: spectral norm of the exact action "
                                           "Jacobian at the anchor", "units": "error per scaled action"},
    "next_state_jacobian_norm": {"definition": "||[I + S_z^-1 J_S f_hat S_z, S_z^-1 J_A f_hat S_a]||_2 at the anchor: the "
                                               "next-state map's Jacobian in (z, a)", "units": "error per scaled (state, action)"},
}


def jacobian_terms(model, z, a, state_scale, action_scale) -> dict:
    """Point Jacobian norms of f_hat at the given anchors, in scaled coordinates.

    They need the model's exact jacobian_za; without one every value is NaN
    (so a row that requires it is unsupported), never a finite-difference
    estimate, which vanishes at float32 for large inputs.
    """
    n = len(z)
    fn = getattr(model, "jacobian_za", None)
    if fn is None:
        return {k: np.full(n, np.nan) for k in JACOBIAN_TERMS}
    sz, sa = np.asarray(state_scale, np.float64), np.asarray(action_scale, np.float64)
    J = np.asarray(fn(z, a), np.float64)
    d = z.shape[1]
    Js = J[:, :, :d] * sz[None, None, :] / sz[None, :, None]
    Ja = J[:, :, d:] * sa[None, None, :] / sz[None, :, None]
    Jn = np.concatenate([np.eye(d)[None] + Js, Ja], axis=2)
    return {"state_jacobian_norm": np.linalg.norm(Js, 2, axis=(1, 2)), "action_jacobian_norm": np.linalg.norm(Ja, 2, axis=(1, 2)),
            "next_state_jacobian_norm": np.linalg.norm(Jn, 2, axis=(1, 2))}


class _PointJacobianScore(Profile):
    """An empirical score from point Jacobians of f_hat at the anchor: not a bound, whatever its value.

    The Jacobians are evaluated at the anchor only and stand in for constants
    over the neighborhood, and the anchor model error is taken as 0 (the paper
    restricts labels to anchors with a small model error, which
    generation.model_error_threshold enforces at generation). Rows are
    classified empirical, with is_upper_bound false.
    """

    form, intrinsic = "score", "empirical"

    def __init__(self, neighborhood_state=None, neighborhood_action=None):
        super().__init__({}, neighborhood_state, neighborhood_action)

    def compute_terms(self, ctx):
        z, a, _ = _anchor_rows(ctx)
        terms = jacobian_terms(ctx.model, z, a, ctx.state_scale, ctx.action_scale)
        return {k: terms[k] for k in self.extra_terms}

    def audit_term(self, name, row):
        value = float(jacobian_terms(row["model"], row["z_t"], row["a_t"], row["state_scale"], row["action_scale"])[name][0])
        u = U32 if getattr(row["model"], "evaluation_dtype", "float32") == "float32" else U64
        d = row["z_t"].shape[1] + row["a_t"].shape[1]
        # A Jacobian entry accumulates over the network's width; d ulps per entry, AUDIT_ULPS of headroom.
        return value, AUDIT_ULPS * u * d * abs(value) if np.isfinite(value) else 0.0


class BackTrackPaperProxy(_PointJacobianScore):
    """Ke et al., Def. 4.5 (Eq. 11): E_BT = ||J_S f_hat(z_t, a_t)|| ||z_G - z_t||, an empirical score."""

    name, version = "backtrack_paper_proxy", "2"
    applies_to = ("backtrack",)
    extra_terms = {"state_jacobian_norm": JACOBIAN_TERMS["state_jacobian_norm"]}
    required = ("state_displacement", "state_jacobian_norm")
    assumptions = ("the point Jacobian at the anchor stands in for the Lipschitz constants", "anchor model error ~ 0",
                   "the label keeps the anchor's action")

    def __init__(self, neighborhood_state=None):
        super().__init__(neighborhood_state, None)

    def in_neighborhood(self, terms):
        """The definition has no action term: a row whose action differs from its anchor's is unsupported."""
        with np.errstate(invalid="ignore"):
            return super().in_neighborhood(terms) & (terms["action_displacement"] == 0)

    def evaluate(self, terms, values):
        total = terms["state_jacobian_norm"] * terms["state_displacement"]
        return np.full(len(total), np.nan), total


class DisturbedActionPaperProxy(_PointJacobianScore):
    """Ke et al., Def. 4.6 (Eq. 12): E_DA = ||J_A f_hat|| ||a_G - a_t|| + (1 + ||J_S f_hat||) ||z_G - z_t||, a score."""

    name, version = "disturbed_action_paper_proxy", "1"
    applies_to = ("disturbed_action",)
    extra_terms = {k: JACOBIAN_TERMS[k] for k in ("state_jacobian_norm", "action_jacobian_norm")}
    required = ("state_displacement", "action_displacement", "state_jacobian_norm", "action_jacobian_norm")
    assumptions = ("the point Jacobians at the anchor stand in for the Lipschitz constants", "anchor model error ~ 0")

    def evaluate(self, terms, values):
        total = (terms["action_jacobian_norm"] * terms["action_displacement"]
                 + (1 + terms["state_jacobian_norm"]) * terms["state_displacement"])
        return np.full(len(total), np.nan), total


class LegacyLabelFiltering(_PointJacobianScore):
    """The label-filtering branch's score (label_error at 73d187d), an opt-in for parity.

        score = ||J_{z + f_hat}(z_t, a_t)||_2 * ||(z_G - z_t, a_G - a_t)||_2

    with J the Jacobian of the next-state map in (z, a) and one joint norm of
    the state and action displacement. It matches neither paper definition
    and applies to every generator, as the branch did.
    """

    name, version = "legacy_label_filtering", "1"
    extra_terms = {"next_state_jacobian_norm": JACOBIAN_TERMS["next_state_jacobian_norm"]}
    required = ("state_displacement", "action_displacement", "next_state_jacobian_norm")
    assumptions = ("the next-state Jacobian at the anchor stands in for a Lipschitz constant over a joint "
                   "state-action metric",)

    def evaluate(self, terms, values):
        total = terms["next_state_jacobian_norm"] * np.hypot(terms["state_displacement"], terms["action_displacement"])
        return np.full(len(total), np.nan), total


PROFILES: dict = {}


def register_profile(cls) -> type:
    if PROFILES.get(cls.name, cls) is not cls:
        raise InterfaceError(f"profile name {cls.name!r} is already registered")
    PROFILES[cls.name] = cls
    return cls


for _profile in (ContinuityProfile, BackTrackProfile, DisturbedActionProfile, BackTrackPaperProxy,
                 DisturbedActionPaperProxy, LegacyLabelFiltering):
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


def _difference_norm(x, y, scale):
    """(upper bound on ||(x - y) / scale||, operand magnitude ||(|x| + |y|) / scale||) per row."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    d = x - y
    return upper_norm(d, U64 * np.abs(d), scale), np.linalg.norm((np.abs(x) + np.abs(y)) / scale, axis=1)


def _model_residual(model, z, a, target, scale):
    """(||(z + f_hat(z, a) - target) / scale||, operand magnitude) per row; an upper bound when the model bounds its error."""
    delta = np.asarray(model.delta(z, a), np.float64)
    r = z + delta - target
    mag = np.abs(z) + np.abs(delta) + np.abs(target)
    bound = getattr(model, "delta_error_bound", None)
    if bound is None:
        value = np.linalg.norm(r / scale, axis=1)
    else:
        value = upper_norm(r, up((bound(z, a) + gamma(2) * mag) * (1 + gamma(4))), scale)
    return value, np.linalg.norm(mag / scale, axis=1)


def model_evaluation(model) -> str:
    return "bounded" if hasattr(model, "delta_error_bound") else "assumed_exact"


def global_terms(ctx: Context) -> dict:
    """The global terms for every candidate row, in scaled coordinates."""
    c = ctx.candidates
    z_g = np.asarray(c.observations[ctx.modality], np.float64)
    a_g = np.asarray(c.actions, np.float64)
    y = np.asarray(c.target_observations[ctx.modality], np.float64)
    z, a, zn = _anchor_rows(ctx)
    sz, sa = ctx.state_scale, ctx.action_scale
    return {"state_displacement": _difference_norm(z_g, z, sz)[0], "action_displacement": _difference_norm(a_g, a, sa)[0],
            "anchor_model_error": _model_residual(ctx.model, z, a, zn, sz)[0],
            "solver_error": _model_residual(ctx.model, z_g, a_g, y, sz)[0]}


def resolve_demonstration(spec, ctx: Context) -> tuple[dict, np.ndarray]:
    """(descriptor entry, per-row demonstration_residual) from a demonstration specification."""
    n = len(ctx.candidates.source_frame)
    if isinstance(spec, Declared):
        return ({"source": "declared", "value": float(spec.value), "rationale": spec.rationale, "provenance": None},
                np.full(n, float(spec.value)))
    provider = PROVIDERS[spec.provider]
    z, a, zn = _anchor_rows(ctx)
    return provider.describe(spec.reference, ctx), provider.residuals(spec.reference, z, a, zn, ctx)[0]


# Candidate reasons that mean a row was never produced in the legacy sense: legacy generated no label for it.
NOT_PRODUCED = ("generation_failed", "non_finite", "label_filter.model_misfit")


def produced_rows(candidates) -> np.ndarray:
    """Rows a legacy run would have generated: not rejected for generation failure, non-finite values, or model misfit."""
    cols = [i for i, c in enumerate(candidates.reason_codes) if c in NOT_PRODUCED]
    reasons = np.asarray(candidates.reasons, bool)
    return ~reasons[:, cols].any(axis=1) if cols else np.ones(len(reasons), bool)


def decide(profile: Profile, resolved: dict, terms: dict, accepted, admission: dict, quantile, population=None):
    """Values, classification, masks, reasons, and selection threshold from persisted terms.

    resolved holds the descriptor's constants, demonstration, and
    model_evaluation. Shared by certify() and verify_structure(), so the
    structural check recomputes exactly what certification decided. Totals are
    rounded up to enclose the float64 arithmetic of the formula.

    The selection quantile is taken over the accepted, admissible rows, or,
    with population (a row mask: the "produced" selection population), over
    the supported rows of that mask, independently of acceptance and the
    admission threshold, as the label-filtering branch did. Either way only
    accepted, admissible rows are selected.
    """
    model, total = profile.evaluate(terms, {r: c["value"] for r, c in resolved["constants"].items()})
    model, total = sum_upper(model), sum_upper(total)
    required = [n for n, s in profile.term_specs().items() if s["required"]]
    inside = profile.in_neighborhood(terms)
    supported = inside & np.isfinite(total) & np.all([np.isfinite(terms[n]) for n in required], axis=0)
    cls = np.where(supported, profile.base_classification(resolved), "unsupported").astype("<U12")
    bound = np.isin(cls, BOUND_CLASSES)
    candidate = bound if admission["mode"] == "upper_bound" else supported
    with np.errstate(invalid="ignore"):
        within = np.ones(len(total), bool) if admission["threshold"] is None else total <= admission["threshold"]
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
            selection_quantile: float | None = None, state_scale=None, action_scale=None,
            selection_population: str = "admitted") -> CertificateSet:
    """Certify every candidate row of sources.candidates with profile; the candidates are not modified."""
    if not isinstance(profile, Profile) or profile.constants is None:
        raise InterfaceError("certify needs a profile built from Declared/Provided constants")
    adm = _admission(admission, threshold)
    if selection_quantile is not None and (isinstance(selection_quantile, bool) or not isinstance(selection_quantile, (int, float))
                                           or not 0 < selection_quantile <= 1):
        raise InterfaceError(f"selection_quantile must be in (0, 1] or None, got {selection_quantile!r}")
    if selection_population not in SELECTION_POPULATIONS or (selection_population != "admitted" and selection_quantile is None):
        raise InterfaceError(f"selection_population must be one of {list(SELECTION_POPULATIONS)}, and a population other "
                             f"than admitted needs a selection_quantile; got {selection_population!r}")
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
    resolved = {"constants": {role: resolve_constant(spec, profile.roles[role], ctx)
                              for role, spec in sorted(profile.constants.items())},
                "demonstration": None, "model_evaluation": model_evaluation(model)}
    if profile.demonstration_term:
        resolved["demonstration"], terms["demonstration_residual"] = resolve_demonstration(profile.demonstration, ctx)
    if set(terms) != set(profile.term_specs()):
        raise InterfaceError(f"{profile.name} computed terms {sorted(terms)}, declares {sorted(profile.term_specs())}")
    coordinates = {"modality": ctx.modality, "state_dim": int(probe.z.shape[1]), "action_dim": int(probe.a.shape[1]),
                   "state_scale": ctx.state_scale.tolist(), "action_scale": ctx.action_scale.tolist()}
    rows, thr = decide(profile, resolved, terms, candidates.accepted, adm, selection_quantile,
                       produced_rows(candidates) if selection_population == "produced" else None)
    return CertificateSet(
        source_episode_id=np.asarray(candidates.source_episode_id), source_frame=np.asarray(candidates.source_frame),
        terms=terms, descriptor=profile.descriptor(resolved, coordinates), admission=adm,
        selection={"quantile": None if selection_quantile is None else float(selection_quantile), "threshold": thr,
                   "population": selection_population},
        candidates_sha256=hashes["candidates"], inputs=hashes, components=_components(rep, model), **rows).validate()


# --- verification --------------------------------------------------------------

def _profile_of(cert: CertificateSet):
    cls = PROFILES.get(cert.descriptor.get("profile"))
    if cls is None:
        raise InterfaceError(f"profile {cert.descriptor.get('profile')!r} is not registered")
    if cert.descriptor.get("version") != cls.version:
        raise InterfaceError(f"profile {cls.name} version {cert.descriptor.get('version')!r} is not {cls.version!r}")
    return cls.from_descriptor(cert.descriptor)


def _provider_problem(what, entry, bounds) -> str | None:
    p = entry["provenance"]
    if p is None:
        return None
    provider = PROVIDERS.get(p["provider"])
    if (provider is None or entry["source"] not in provider.sources or provider.version != p["version"]
            or provider.bounds != bounds):
        return f"{what} claims {entry['source']} from an unregistered or mismatched provider {p}"
    return None


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
    if profile.demonstration_term != ("demonstration" in d):
        return problems + [f"descriptor {'lacks' if profile.demonstration_term else 'has'} a demonstration entry"]
    try:
        check_applies(profile, candidates)
    except InterfaceError as e:
        problems.append(str(e))
    for role, c in d["constants"].items():
        problems.append(_provider_problem(f"constant {role}", c, profile.roles[role][1]))
    if profile.demonstration_term:
        problems.append(_provider_problem("demonstration", d["demonstration"], "demonstration"))
    problems = [p for p in problems if p]
    resolved = resolved_of(d)
    if json.dumps(profile.descriptor(resolved, d["coordinates"]), sort_keys=True) != json.dumps(d, sort_keys=True):
        problems.append("descriptor is not the one this version of the profile writes")
    rows, thr = decide(profile, resolved, cert.terms, candidates.accepted, cert.admission, cert.selection["quantile"],
                       produced_rows(candidates) if cert.selection["population"] == "produced" else None)
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


def _same_provided(persisted: dict, again: dict) -> bool:
    """Identity fields equal and the value equal to 1e-9 relative: float64 norms differ by a few ulps across BLAS."""
    keys = set(persisted) - {"value", "rationale"}
    if any(persisted[k] != again.get(k) for k in keys):
        return False
    a, b = persisted["value"], again["value"]
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-9 * max(abs(a), abs(b)))


def audit_sources(cert: CertificateSet, sources: Sources) -> list[str]:
    """Problems found by checking each source against the certificate's record and recomputing every term.

    Terms are recomputed one row at a time, independently of certify's batched
    path, and compared to a row-local tolerance: AUDIT_ULPS ulps, in the
    precision the term was evaluated in (the model's evaluation_dtype for the
    model terms, float64 otherwise), of that row's operand magnitude. Provided
    constants are recomputed and must match (values to 1e-9 relative), and
    every analytic_bound value must be verified again in exact arithmetic.
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
    d = cert.descriptor
    if d["model_evaluation"] != model_evaluation(model):
        return [f"model_evaluation {d['model_evaluation']!r} != {model_evaluation(model)!r} for this model"]
    co = d["coordinates"]
    sz, sa = np.asarray(co["state_scale"], np.float64), np.asarray(co["action_scale"], np.float64)
    episodes = {ep.episode_id: ep for ep in dataset.episodes}
    encoded = {eid: np.asarray(rep.encode(ep.observations), np.float64) for eid, ep in episodes.items()}
    dims = (next(iter(encoded.values())).shape[1], dataset.episodes[0].actions.shape[1])
    if dims != (co["state_dim"], co["action_dim"]):
        return [f"sources have state and action dims {dims}; recorded ({co['state_dim']}, {co['action_dim']})"]
    ctx = Context(candidates, dataset, rep, model, sz, sa, {k: Path(v) for k, v in sources.references.items()}, given)
    demo = d.get("demonstration")
    u_model = U32 if getattr(model, "evaluation_dtype", "float32") == "float32" else U64
    m = co["modality"]
    extra = [n for n in cert.terms if n not in GLOBAL_TERMS and n != "demonstration_residual"]
    for i in range(len(cert.source_frame)):
        eid, t = str(cert.source_episode_id[i]), int(cert.source_frame[i])
        ep, zs = episodes[eid], encoded[eid]
        row = {"z_t": zs[t:t + 1], "z_next": zs[t + 1:t + 2], "a_t": np.asarray(ep.actions[t:t + 1], np.float64),
               "z_g": np.asarray(candidates.observations[m][i:i + 1], np.float64),
               "a_g": np.asarray(candidates.actions[i:i + 1], np.float64),
               "y": np.asarray(candidates.target_observations[m][i:i + 1], np.float64),
               "model": model, "state_scale": sz, "action_scale": sa}
        expected = {}
        for name, (x, y_, s) in (("state_displacement", (row["z_g"], row["z_t"], sz)),
                                 ("action_displacement", (row["a_g"], row["a_t"], sa))):
            v, mag = _difference_norm(x, y_, s)
            expected[name] = (float(v[0]), AUDIT_ULPS * U64 * float(mag[0]))
        for name, (zz, aa, tgt) in (("anchor_model_error", (row["z_t"], row["a_t"], row["z_next"])),
                                    ("solver_error", (row["z_g"], row["a_g"], row["y"]))):
            v, mag = _model_residual(model, zz, aa, tgt, sz)
            expected[name] = (float(v[0]), AUDIT_ULPS * u_model * float(mag[0]))
        if demo is not None:
            if demo["provenance"] is None:
                expected["demonstration_residual"] = (demo["value"], 0.0)
            else:
                v, mag = PROVIDERS[demo["provenance"]["provider"]].residuals(
                    demo["provenance"]["params"]["reference"], row["z_t"], row["a_t"], row["z_next"], ctx)
                expected["demonstration_residual"] = (float(v[0]), AUDIT_ULPS * U64 * float(mag[0]))
        expected.update({n: profile.audit_term(n, row) for n in extra})
        for name, (want, tol) in expected.items():
            got = float(cert.terms[name][i])
            if not (abs(got - want) <= tol or (np.isnan(got) and np.isnan(want))):
                problems.append(f"row {i}: persisted {name} {got!r} != recomputed {want!r} (tolerance {tol:.3g})")
    for role, c in d["constants"].items():
        p = c["provenance"]
        if p is None:
            continue
        provider = PROVIDERS[p["provider"]]
        again = provider.resolve(c["maps"], p["params"].get("reference"), ctx)
        if not _same_provided(c, again):
            problems.append(f"constant {role} {c['value']!r} does not match its provider's recomputation {again['value']!r}")
        elif c["source"] == "analytic_bound" and not provider.verify(c["value"], c["maps"], p["params"].get("reference"), ctx):
            problems.append(f"constant {role} {c['value']!r} is not verified as an upper bound")
    if demo is not None and demo["provenance"] is not None:
        again = PROVIDERS[demo["provenance"]["provider"]].describe(demo["provenance"]["params"]["reference"], ctx)
        if not _same_provided(demo, again):
            problems.append("demonstration provenance does not match its provider's")
    return problems


def require_upper_bound(cert: CertificateSet) -> None:
    """Raise unless every supported row is an upper bound and admission admits only bounds."""
    supported = cert.classification != "unsupported"
    if cert.admission["mode"] != "upper_bound" or not cert.is_upper_bound[supported].all():
        raise InterfaceError(f"certificate ({cert.descriptor.get('profile')}, admission {cert.admission['mode']!r}) "
                             "does not admit rows as upper bounds")


__all__ = ["Declared", "Provided", "Profile", "ContinuityProfile", "BackTrackProfile", "DisturbedActionProfile",
           "BackTrackPaperProxy", "DisturbedActionPaperProxy", "LegacyLabelFiltering", "jacobian_terms", "Sources",
           "Context", "certify", "decide", "verify_structure", "audit_sources", "require_upper_bound", "register_profile",
           "write_reference_dynamics", "read_reference_dynamics", "check_applies", "model_evaluation", "resolved_of", "PROFILES",
           "PROVIDERS", "GENERATOR_PROFILES", "SCHEMA", "VERSION", "REASON_CODES"]
