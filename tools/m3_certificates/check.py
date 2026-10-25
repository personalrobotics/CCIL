#!/usr/bin/env python3
"""Evidence for #88 parts 1 and 2: the certificate contract, continuity, and the CCIL theorem profiles.

    uv run --group evidence python tools/m3_certificates/check.py --out DIR

On the analytic double integrator (true dynamics f(z, a) = A z + B a, so the
true label error ||T(q) - y|| is computed independently of the certificate):
- constants come from a registered provider (linear_discrepancy, against a
  reference-dynamics file) or are declared: with the exact model the provider
  gives C = 0 and every row is certified and tight; with a deliberately wrong
  linear model it gives ||A - A_hat||, ||B - B_hat|| and every row of both
  generators is covered, also in scaled coordinates;
- a raw value cannot be asserted as a bound: it is refused, a declared zero is
  conditional and fails coverage, and a forged certified constant passes the
  structural check but not the source audit;
- labels outside the neighborhood are unsupported and not admissible;
- admission is a policy, not a classification: an empirical profile admits
  nothing under upper_bound admission and its supported rows under explicit
  score admission, staying non-bound, and require_upper_bound refuses it;
- the masks follow their rules, a quantile with no eligible row selects none,
  and candidate.accepted never changes;
- profile-declared terms round-trip, are audited, and only required terms gate
  support;
- structural recomputation and the source audit detect different corruptions,
  and the audit binds every input hash and component identity;
- the reader and writer enforce the schema: unknown fields, wrong dtypes and
  ranks, malformed metadata, and broken invariants raise SchemaError;
- the API refuses malformed constants, thresholds, and scales;
- the pipeline writes a complete labels.certify run next to an unchanged
  candidates file, covered against the true dynamics; a rerun reuses it, and a
  changed reference or threshold is a new run (the inspectable artifact).
Part 2, the backtrack and disturbed_action theorem profiles and MLP
model_bound extraction, is described in theorems.py.
Writes DIR/report.json.
"""
import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
sys.path.insert(0, str(REPO / "tools" / "m2_interfaces"))
from check import fixture  # noqa: E402  (tools/m2_interfaces/check.py: the #10 fixture as ccil.trajectories)
from system import DoubleIntegrator  # noqa: E402

from ccil.core import (BackwardEuler, Finite, IdentityRepresentation, InterfaceError,  # noqa: E402
                       LinearTransitionModel, NoisyAction, generate_labels)
from ccil.core.certificates import (GLOBAL_TERM_SPECS, ContinuityProfile, Declared, Profile, Provided,  # noqa: E402
                                    Sources, audit_sources, certify, decide, register_profile, require_upper_bound,
                                    resolved_of, verify_structure, write_reference_dynamics)
from ccil.runs import validate_run  # noqa: E402
from ccil.schema import (Episode, SchemaError, TrajectoryDataset, read_candidates, read_certificates,  # noqa: E402
                         write_candidates, write_certificates, write_trajectories)
from ccil.schema.io import _atomic_savez  # noqa: E402

SEED, K, SIGMA = 20261025, 4, 0.05
DEMO = Provided("reference_residual", "true")


@register_profile
class AnchorActionScore(Profile):
    """A test score profile: solver_error + 0.1 ||a_t||, with its own term; empirical by construction."""

    name, version, form, intrinsic = "evidence_anchor_action_score", "1", "score", "empirical"
    extra_terms = {"anchor_action_norm": {"definition": "||a_t / action_scale||_2", "units": "scaled action"}}
    required = ("solver_error", "anchor_action_norm")

    def __init__(self):
        super().__init__({})

    def compute_terms(self, ctx):
        idx = {(ep.episode_id, t): ep.actions[t] for ep in ctx.dataset.episodes for t in range(ep.length - 1)}
        a = np.array([idx[(str(e), int(f))] for e, f in zip(ctx.candidates.source_episode_id, ctx.candidates.source_frame)],
                     np.float64)
        return {"anchor_action_norm": np.linalg.norm(a / ctx.action_scale, axis=1)}

    def audit_term(self, name, row):
        return float(np.linalg.norm(row["a_t"] / row["action_scale"])), 1e-12

    def evaluate(self, terms, values):
        return np.full(len(terms["solver_error"]), np.nan), terms["solver_error"] + 0.1 * terms["anchor_action_norm"]


def write_bundle(root, bundle):
    """Each certificate with copies of its sources and a manifest; audit_bundle.py audits it on any host."""
    root.mkdir()
    manifest = {}
    for name, (cert, src) in sorted(bundle.items()):
        d = root / name
        d.mkdir()
        files = {}
        for key, p in sorted(src.paths().items()):
            shutil.copyfile(p, d / f"{key}.npz")
            files[key] = f"{key}.npz"
        write_certificates(cert, d / "certificates.npz")
        manifest[name] = {"files": files, "certificates": "certificates.npz",
                          "modality": cert.components["representation"]["modality"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_label_error(c, sys_, scale=None):
    """||(T(q) - y) / scale|| with the true dynamics, from the stored candidate arrays only."""
    z, a = c.observations["state"].astype(np.float64), c.actions.astype(np.float64)
    e = sys_.step_np(z, a) - c.target_observations["state"].astype(np.float64)
    return np.linalg.norm(e / (1.0 if scale is None else scale), axis=1)


def outcome(fn):
    try:
        fn()
        return "accepted"
    except SchemaError as e:
        return f"SchemaError: rules={sorted(e.counts)}: {e.violations[0]['detail'][:140]}"
    except InterfaceError as e:
        return f"InterfaceError: {str(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    bundle = {}  # name -> (certificate, sources), written for audit on another host

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:900]}", flush=True)

    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    config["episodes"] = [e for e in config["episodes"] if not e["corrupt_actions"]]
    dataset = fixture(config).validate()
    data_path = out / "fixture.npz"
    write_trajectories(dataset, data_path)
    sys_ = DoubleIntegrator(config["h"])
    truth = out / "true-dynamics.npz"
    write_reference_dynamics(truth, sys_.A, sys_.B, "analytic double integrator, the fixture's true dynamics")
    rep = IdentityRepresentation("state")
    A_hat, B_hat = sys_.A + np.array([[0.05, 0.0], [-0.08, 0.04]]), sys_.B * 1.6
    models = {"exact": LinearTransitionModel(sys_.A, sys_.B), "wrong": LinearTransitionModel(A_hat, B_hat)}
    for name, m in models.items():
        m.save(out / f"model-{name}.npz")
    generators = {"backward_euler": BackwardEuler(), "noisy_action": NoisyAction(K, SIGMA)}
    provided = ContinuityProfile(Provided("linear_discrepancy", "true"), Provided("linear_discrepancy", "true"), DEMO)

    def sources(model, gen):
        c = generate_labels(dataset, rep, models[model], generators[gen], [Finite()], run_id=f"{model}-{gen}", seed=SEED)
        path = out / f"candidates-{model}-{gen}.npz"
        write_candidates(c, path)
        return Sources(path, data_path, out / f"model-{model}.npz", rep, references={"true": truth})

    # 1-2. Provider constants: exact model certified and tight; wrong model covered (also scaled).
    src = {(m, g): sources(m, g) for m in models for g in generators}
    for g in generators:
        s = src[("exact", g)]
        cert = certify(s, provided)
        err = true_label_error(read_candidates(s.candidates), sys_)
        gap = cert.total_error_value - err
        record(f"exact_model_certified_and_tight[{g}]",
               (cert.classification == "certified").all() and (gap >= 0).all() and float(np.abs(gap).max()) <= 1e-5
               and all(c["value"] == 0.0 and c["provenance"]["provider"] == "linear_discrepancy" for c in cert.descriptor["constants"].values()),
               rows=int(len(err)), constants={r: c["value"] for r, c in cert.descriptor["constants"].items()},
               max_true_error=float(err.max()), max_abs_gap=float(np.abs(gap).max()))
    Cz, Ca = float(np.linalg.norm(sys_.A - A_hat, 2)), float(np.linalg.norm(sys_.B - B_hat, 2))
    for g in generators:
        for scale in (None, (2.0, 0.5)):
            s = src[("wrong", g)]
            sa = None if scale is None else (4.0,)
            cert = certify(s, provided, state_scale=scale, action_scale=sa)
            err = true_label_error(read_candidates(s.candidates), sys_, None if scale is None else np.array(scale))
            slack = cert.total_error_value - err
            values = {r: c["value"] for r, c in cert.descriptor["constants"].items()}
            record(f"wrong_model_covered_by_provider_constants[{g}, scale={scale}]",
                   (cert.classification == "certified").all() and (slack >= 0).all()
                   and (scale is not None or (np.isclose(values["C_z"], Cz) and np.isclose(values["C_a"], Ca)))
                   and float(cert.terms["anchor_model_error"].max()) > 1e-3,
                   constants=values, unscaled_norms={"A - A_hat": Cz, "B - B_hat": Ca},
                   coordinates=cert.descriptor["coordinates"], rows=int(len(err)), min_slack=float(slack.min()),
                   max_true_error=float(err.max()), max_bound=float(cert.total_error_value.max()))

    # 3. A value cannot be asserted as a bound.
    s = src[("wrong", "noisy_action")]
    cands = read_candidates(s.candidates)
    err = true_label_error(cands, sys_)
    refused = {"raw_float": outcome(lambda: ContinuityProfile(0.0, 0.0, DEMO)),
               "raw_dict_claiming_analytic": outcome(lambda: ContinuityProfile(
                   {"value": 0.0, "source": "analytic_bound"}, Declared(0.0, "x"), DEMO))}
    declared = certify(s, ContinuityProfile(Declared(0.0, "assumed (wrongly) that the model is exact"),
                                            Declared(0.0, "assumed (wrongly) that the model is exact"), DEMO))
    violated = declared.total_error_value < err
    good = certify(s, provided)
    forged = copy.deepcopy(good)
    for c in forged.descriptor["constants"].values():
        c["value"] = 0.0
    rows, thr = decide(provided, resolved_of(forged.descriptor), forged.terms, cands.accepted, forged.admission, None)
    for k, v in rows.items():
        setattr(forged, k, v)
    record("values_cannot_be_asserted_as_bounds",
           all(v.startswith("InterfaceError") for v in refused.values())
           and (declared.classification == "conditional").all() and violated.any()
           and (forged.classification == "certified").all() and (forged.total_error_value < err).any()
           and verify_structure(forged, s.candidates) == [] and any("recomputation" in p for p in audit_sources(forged, s)),
           refused=refused, declared_zero={"classification": "conditional", "violated_rows": int(violated.sum()),
                                           "rows": int(len(err)), "worst_shortfall": float((err - declared.total_error_value).max())},
           forged_certified_zero={"violated_rows": int((forged.total_error_value < err).sum()),
                                  "structure": verify_structure(forged, s.candidates),
                                  "audit": [p for p in audit_sources(forged, s) if "constant" in p]})

    # 4. Outside the neighborhood: unsupported, not a bound, not admissible.
    radius = float(np.median(good.terms["state_displacement"]))
    nb = ContinuityProfile(Provided("linear_discrepancy", "true"), Provided("linear_discrepancy", "true"), DEMO, neighborhood_state=radius)
    cn = certify(s, nb)
    outside = cn.terms["state_displacement"] > radius
    record("outside_neighborhood_unsupported",
           outside.any() and (~outside).any() and (cn.classification[outside] == "unsupported").all()
           and not cn.is_upper_bound[outside].any() and not cn.admissible[outside].any()
           and cn.reasons[outside, 0].all() and not cn.reasons[~outside, 0].any() and (cn.classification[~outside] == "certified").all(),
           radius=radius, outside=int(outside.sum()), inside=int((~outside).sum()))

    # 5. Admission is a policy: an empirical profile under each mode.
    score = AnchorActionScore()
    ub = certify(s, score)
    limit = float(np.quantile(certify(s, score, admission="score").total_error_value, 0.75))
    sc = certify(s, score, admission="score", threshold=limit)
    cont = certify(s, provided)
    record("admission_is_separate_from_classification",
           (ub.classification == "empirical").all() and not ub.admissible.any() and ub.reasons[:, 1].all()
           and sc.admissible.any() and not sc.is_upper_bound.any() and not sc.reasons[:, 1].any()
           and np.array_equal(sc.admissible, sc.total_error_value <= limit)
           and outcome(lambda: require_upper_bound(ub)).startswith("InterfaceError")
           and outcome(lambda: require_upper_bound(sc)).startswith("InterfaceError")
           and outcome(lambda: require_upper_bound(cont)) == "accepted",
           upper_bound_mode={"admissible": int(ub.admissible.sum()), "not_upper_bound": int(ub.reasons[:, 1].sum())},
           score_mode={"threshold": limit, "admissible": int(sc.admissible.sum()), "is_upper_bound": int(sc.is_upper_bound.sum())},
           require_upper_bound={"empirical_upper_bound_mode": outcome(lambda: require_upper_bound(ub)),
                                "empirical_score_mode": outcome(lambda: require_upper_bound(sc)),
                                "continuity_provider": outcome(lambda: require_upper_bound(cont))})

    # 6. Masks; a quantile with no eligible row selects none; acceptance never changes.
    before = sha(s.candidates)
    limit = float(np.quantile(cn.total_error_value[~outside], 0.8))
    cm = certify(s, nb, threshold=limit, selection_quantile=0.5)
    eligible = cands.accepted & cm.admissible
    thr = cm.selection["threshold"]
    empty = certify(s, provided, threshold=0.0, selection_quantile=0.5)
    record("masks_follow_rules",
           sha(s.candidates) == before and np.array_equal(read_candidates(s.candidates).accepted, cands.accepted)
           and np.array_equal(cm.admissible, cm.is_upper_bound & (cm.total_error_value <= limit))
           and np.array_equal(cm.selected, eligible & (cm.total_error_value <= thr))
           and thr == float(np.quantile(cm.total_error_value[eligible], 0.5))
           and np.array_equal(cm.reasons[:, 2], cm.is_upper_bound & (cm.total_error_value > limit))
           and np.array_equal(cm.reasons[:, 3], eligible & ~cm.selected)
           and not empty.admissible.any() and not empty.selected.any() and empty.selection["threshold"] is None,
           threshold=limit, selection_threshold=thr, accepted=int(cands.accepted.sum()), admissible=int(cm.admissible.sum()),
           selected=int(cm.selected.sum()), no_eligible={"admissible": int(empty.admissible.sum()), "selected": int(empty.selected.sum()),
                                                        "threshold": empty.selection["threshold"]})

    # 7. Profile terms round-trip, are audited, and only required terms gate support.
    spath = out / "score-certificates.npz"
    write_certificates(sc, spath)
    back = read_certificates(spath)
    i = int(np.flatnonzero(sc.admissible)[0])
    falsified = copy.deepcopy(back)
    falsified.terms["anchor_action_norm"][i] *= 0.5
    rows, _ = decide(score, resolved_of(falsified.descriptor), falsified.terms, cands.accepted, falsified.admission, None)
    for k, v in rows.items():
        setattr(falsified, k, v)
    nan_terms = {k: v.copy() for k, v in back.terms.items()}
    nan_terms["anchor_model_error"][:] = np.nan
    adm = {"mode": "score", "threshold": None}
    score_rows, _ = decide(score, resolved_of(back.descriptor), nan_terms, cands.accepted, adm, None)
    cont_terms = {k: v.copy() for k, v in cont.terms.items()}
    cont_terms["anchor_model_error"][:] = np.nan
    cont_rows, _ = decide(provided, resolved_of(cont.descriptor), cont_terms, cands.accepted,
                          {"mode": "upper_bound", "threshold": None}, None)
    record("profile_terms_round_trip_and_audit",
           set(back.terms) == set(GLOBAL_TERM_SPECS) | {"anchor_action_norm"}
           and np.array_equal(back.terms["anchor_action_norm"], sc.terms["anchor_action_norm"])
           and verify_structure(back, s.candidates) == [] and audit_sources(back, s) == []
           and verify_structure(falsified, s.candidates) == [] and any("anchor_action_norm" in p for p in audit_sources(falsified, s))
           and (score_rows["classification"] == "empirical").all() and (cont_rows["classification"] == "unsupported").all(),
           terms=sorted(back.terms), required=[n for n, sp in back.descriptor["terms"].items() if sp["required"]],
           non_required_nan={"score_profile_supported": int((score_rows["classification"] != "unsupported").sum()),
                             "continuity_supported": int((cont_rows["classification"] != "unsupported").sum())},
           falsified_term_audit=[p for p in audit_sources(falsified, s)][:2])

    # 8. Two verification levels, and the audit's input binding.
    cpath = out / "certificates.npz"
    write_certificates(cm, cpath)
    base = read_certificates(cpath)
    clean = {"structure": verify_structure(base, s.candidates), "audit": audit_sources(base, s)}
    j = int(np.flatnonzero(base.is_upper_bound)[0])

    def mutate(fn):
        cs = copy.deepcopy(base)
        fn(cs)
        return cs

    def consistent(cs, term, delta):
        """Falsify one term at row j and rederive every value and mask, as a careful forger would."""
        cs = copy.deepcopy(cs)
        cs.terms[term][j] += delta
        rows, thr = decide(nb, resolved_of(cs.descriptor), cs.terms, cands.accepted, cs.admission, cs.selection["quantile"])
        for k, v in rows.items():
            setattr(cs, k, v)
        cs.selection = {**cs.selection, "threshold": thr}
        return cs

    tampered = out / "tampered-candidates.npz"
    raw = Path(s.candidates).read_bytes()
    tampered.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    other_model = Sources(s.candidates, data_path, out / "model-exact.npz", rep, references={"true": truth})
    corruptions = {
        "total_halved": (mutate(lambda cs: cs.total_error_value.__setitem__(j, cs.total_error_value[j] / 2)), s.candidates, s),
        "unsupported_marked_certified": (mutate(lambda cs: cs.classification.__setitem__(
            int(np.flatnonzero(cs.classification == "unsupported")[0]), "certified")), s.candidates, s),
        "selected_flipped": (mutate(lambda cs: cs.selected.__setitem__(j, ~cs.selected[j])), s.candidates, s),
        "unit_string_edited": (mutate(lambda cs: cs.descriptor["units"].__setitem__("C_z", "dimensionless")), s.candidates, s),
        "constant_lowered_without_rederiving": (mutate(lambda cs: cs.descriptor["constants"]["C_z"].__setitem__("value", 0.0)), s.candidates, s),
        "candidates_file_changed": (base, tampered, s),
        "anchor_error_falsified_consistently": (consistent(base, "anchor_model_error", -float(base.terms["anchor_model_error"][j])), s.candidates, s),
        "solver_error_falsified_consistently": (consistent(base, "solver_error", -float(base.terms["solver_error"][j])), s.candidates, s),
        "recorded_model_hash_zeroed": (mutate(lambda cs: cs.inputs.__setitem__("transition_model", "0" * 64)), s.candidates, s),
        "audited_against_another_model": (base, s.candidates, other_model),
        "representation_identity_changed": (mutate(lambda cs: cs.components["representation"].__setitem__("version", "2")), s.candidates, s),
    }
    detected = {n: {"structure": bool(verify_structure(cs, cf)), "audit": bool(audit_sources(cs, srcs))}
                for n, (cs, cf, srcs) in corruptions.items()}
    structural = ("total_halved", "unsupported_marked_certified", "selected_flipped", "unit_string_edited",
                  "candidates_file_changed")
    sourced = ("anchor_error_falsified_consistently", "solver_error_falsified_consistently", "recorded_model_hash_zeroed",
               "audited_against_another_model", "representation_identity_changed")
    record("structure_and_audit_detect_different_corruptions",
           clean == {"structure": [], "audit": []}
           and all(detected[n] == {"structure": True, "audit": False} for n in structural)
           and all(detected[n] == {"structure": False, "audit": True} for n in sourced)
           # The totals no longer follow from it, and the provider recomputes it.
           and detected["constant_lowered_without_rederiving"] == {"structure": True, "audit": True},
           clean=clean, detected=detected)

    # 9. Schema ingress: the reader and the writer.
    with np.load(cpath, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    def file_with(edit):
        a = copy.deepcopy(arrays)
        edit(a)
        p = out / "malformed.npz"
        _atomic_savez(p, a)
        return outcome(lambda: read_certificates(p))

    def meta(fn):
        def edit(a):
            m = json.loads(str(a["meta"]))
            fn(m)
            a["meta"] = np.array(json.dumps(m))
        return edit
    malformed = {
        "unknown_array_field": file_with(lambda a: a.__setitem__("extra", np.zeros(len(a["selected"])))),
        "integer_selected_mask": file_with(lambda a: a.__setitem__("selected", a["selected"].astype(np.int64))),
        "rank_2_total": file_with(lambda a: a.__setitem__("total_error_value", a["total_error_value"][:, None])),
        "admission_null": file_with(meta(lambda m: m.update(admission=None))),
        "negative_admission_threshold": file_with(meta(lambda m: m["admission"].update(threshold=-1.0))),
        "selection_threshold_without_quantile": file_with(meta(lambda m: m.update(selection={"quantile": None, "threshold": 0.1}))),
        "bad_sha256": file_with(meta(lambda m: m.update(candidates_sha256="abc"))),
        "unknown_meta_field": file_with(meta(lambda m: m.update(note="x"))),
        "missing_term": file_with(lambda a: a.pop("term/anchor_model_error")),
        "undeclared_term": file_with(lambda a: a.__setitem__("term/extra", a["term/solver_error"])),
        "analytic_constant_without_provenance": file_with(meta(lambda m: m["descriptor"]["constants"]["C_z"].update(provenance=None))),
        "declared_constant_with_provenance": file_with(meta(lambda m: m["descriptor"]["constants"]["C_z"].update(
            source="declared"))),
        "scale_of_wrong_length": file_with(meta(lambda m: m["descriptor"]["coordinates"].update(state_scale=[1.0]))),
        "next_major_version": file_with(meta(lambda m: m.update(version="2.0"))),
        "classification_out_of_range": file_with(lambda a: a["classification"].__setitem__(0, 9)),
        "row_count_mismatch": file_with(lambda a: a.__setitem__("selected", a["selected"][:-1])),
        "admissible_unsupported_row": file_with(lambda a: a["admissible"].__setitem__(
            int(np.flatnonzero(a["classification"] == 4)[0]), True)),
        "is_upper_bound_inconsistent": file_with(lambda a: a["is_upper_bound"].__setitem__(j, False)),
    }
    bad = copy.deepcopy(base)
    bad.is_upper_bound = bad.is_upper_bound.astype(np.int64)
    malformed["writer_integer_mask"] = outcome(lambda: write_certificates(bad, out / "never.npz"))
    rule = {"unknown_array_field": "format", "integer_selected_mask": "dtype", "rank_2_total": "rank",
            "admission_null": "admission", "negative_admission_threshold": "admission",
            "selection_threshold_without_quantile": "selection", "bad_sha256": "inputs", "unknown_meta_field": "format",
            "missing_term": "terms", "undeclared_term": "terms", "analytic_constant_without_provenance": "descriptor",
            "declared_constant_with_provenance": "descriptor", "scale_of_wrong_length": "descriptor",
            "next_major_version": "format", "classification_out_of_range": "format", "row_count_mismatch": "shape",
            "admissible_unsupported_row": "admissible", "is_upper_bound_inconsistent": "is_upper_bound",
            "writer_integer_mask": "dtype"}
    record("schema_ingress_enforced",
           all(v.startswith("SchemaError") and f"'{rule[k]}'" in v.split(":")[1] for k, v in malformed.items())
           and set(rule) == set(malformed) and not (out / "never.npz").exists(),
           malformed=malformed)

    # 10. The API refuses malformed constants, policies, and scales.
    api = {
        "declared_negative": outcome(lambda: Declared(-1.0, "x")),
        "declared_no_rationale": outcome(lambda: Declared(1.0, "")),
        "unknown_provider": outcome(lambda: Provided("guess", "true")),
        "neighborhood_zero": outcome(lambda: ContinuityProfile(Declared(0.0, "x"), Declared(0.0, "x"), DEMO, neighborhood_state=0.0)),
        "threshold_nan": outcome(lambda: certify(s, provided, threshold=float("nan"))),
        "threshold_negative": outcome(lambda: certify(s, provided, threshold=-1.0)),
        "threshold_bool": outcome(lambda: certify(s, provided, threshold=True)),
        "quantile_zero": outcome(lambda: certify(s, provided, selection_quantile=0.0)),
        "unknown_admission": outcome(lambda: certify(s, provided, admission="maybe")),
        "scale_wrong_length": outcome(lambda: certify(s, provided, state_scale=(1.0,))),
        "scale_zero": outcome(lambda: certify(s, provided, action_scale=(0.0,))),
        "missing_reference": outcome(lambda: certify(Sources(s.candidates, data_path, s.transition_model, rep), provided)),
    }
    record("api_validation", all(v.startswith("InterfaceError") for v in api.values()), refused=api)

    # 11. The pipeline: labels.certify next to an unchanged candidates file.
    from ccil.config import resolve
    from ccil.pipelines.offline import certify_stage, generate_offline
    work = out / "pipeline"
    work.mkdir()
    write_trajectories(dataset, work / "fixture.npz")
    cfg = resolve(recipe="analytic_double_integrator",
                  sets=[f"data.path={work / 'fixture.npz'}", f"run.output_dir={work / 'runs'}",
                        "generation.method=noisy_action", f"generation.labels_per_transition={K}",
                        f"generation.action_noise_std={SIGMA}"])
    prepared, fitted, labeled = generate_offline(cfg)
    cand_file = labeled.out("labels/candidates.npz")
    cand_sha = sha(cand_file)
    first_run = certify_stage(cfg, prepared, fitted, labeled, rep, provided, selection_quantile=0.75, references={"true": truth})
    again = certify_stage(cfg, prepared, fitted, labeled, rep, provided, selection_quantile=0.75, references={"true": truth})
    other_truth = work / "perturbed-dynamics.npz"
    write_reference_dynamics(other_truth, sys_.A * 1.01, sys_.B, "a perturbed reference")
    changed_ref = certify_stage(cfg, prepared, fitted, labeled, rep, provided, selection_quantile=0.75, references={"true": other_truth})
    changed_thr = certify_stage(cfg, prepared, fitted, labeled, rep, provided, threshold=1.0, selection_quantile=0.75,
                                references={"true": truth})
    status = {p.name: validate_run(p)[0] for p in sorted((work / "runs").iterdir()) if p.is_dir()}
    pc = read_candidates(cand_file)
    pcert = read_certificates(first_run.out("labels/certificates.npz"))
    perr = true_label_error(pc, sys_)
    psrc = Sources(cand_file, prepared.out("data/dataset.npz"), fitted.out("checkpoints/transition_model.npz"), rep,
                   references={"true": truth})
    record("pipeline_certify_stage",
           set(status.values()) == {"complete"} and len(status) == 6 and sha(cand_file) == cand_sha
           and pcert.candidates_sha256 == cand_sha and again.reused and again.run_dir == first_run.run_dir
           and len({first_run.run_dir, changed_ref.run_dir, changed_thr.run_dir}) == 3
           and (pcert.classification == "certified").all() and (pcert.total_error_value >= perr).all()
           and verify_structure(pcert, cand_file) == [] and audit_sources(pcert, psrc) == [],
           stages=status, candidates_sha256=cand_sha, reused=again.reused,
           constants={r: c["value"] for r, c in pcert.descriptor["constants"].items()},
           min_slack_vs_true=float((pcert.total_error_value - perr).min()),
           usable=int((pc.accepted & pcert.admissible & pcert.selected).sum()), rows=int(len(pc.accepted)),
           artifact=f"<out>/pipeline/runs/{first_run.run_dir.name}/labels/certificates.npz")

    # 12. The audit's tolerance is row-local: one extreme state does not loosen any other row.
    big = np.array([[1e6, 0.0]])
    acts = np.array([[0.01], [-0.02], [0.03]])
    for t in range(len(acts)):
        big = np.vstack([big, sys_.step_np(big[-1:], acts[t:t + 1])])
    extreme = TrajectoryDataset(list(dataset.episodes) + [Episode("extreme", {"state": big.astype(np.float32)},
                                                                  np.vstack([acts, [[0.0]]]).astype(np.float32))]).validate()
    xpath = out / "fixture-extreme.npz"
    write_trajectories(extreme, xpath)
    xc = generate_labels(extreme, rep, models["wrong"], generators["backward_euler"], [Finite()], run_id="extreme", seed=SEED)
    write_candidates(xc, out / "candidates-extreme.npz")
    xs = Sources(out / "candidates-extreme.npz", xpath, out / "model-wrong.npz", rep, references={"true": truth})
    xcert = certify(xs, provided)
    k = int(np.flatnonzero(xcert.source_episode_id != "extreme")[0])
    delta = 1e-6
    xforged = copy.deepcopy(xcert)
    xforged.terms["anchor_model_error"][k] += delta
    rows, thr = decide(provided, resolved_of(xforged.descriptor), xforged.terms, xc.accepted, xforged.admission, None)
    for name, v in rows.items():
        setattr(xforged, name, v)
    xaudit = audit_sources(xforged, xs)
    old_tol = 1e-6 * (1 + float(np.abs(big).max())) + 1e-5 * float(xcert.terms["anchor_model_error"][k])
    record("audit_tolerance_is_row_local",
           audit_sources(xcert, xs) == [] and verify_structure(xforged, xs.candidates) == []
           and any(f"row {k}: persisted anchor_model_error" in p for p in xaudit) and delta < old_tol,
           extreme_state=float(np.abs(big).max()), corrupted_row=k, corruption=delta,
           term=float(xcert.terms["anchor_model_error"][k]), previous_global_tolerance=old_tol,
           audit=[p for p in xaudit if p.startswith(f"row {k}:")])
    bundle["continuity_linear"] = (cm, s)

    # Part 2: the theorem profiles and model bounds (tools/m3_certificates/theorems.py).
    import theorems
    theorems.run(out, record, dataset, data_path, sys_, truth, bundle)

    # A bundle of certificates with their sources, for audit on another host (tools/m3_certificates/audit_bundle.py).
    bundle_sha256 = write_bundle(out / "bundle", bundle)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases, "bundle_sha256": bundle_sha256,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
