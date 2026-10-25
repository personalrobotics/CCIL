#!/usr/bin/env python3
"""Evidence for #88 part 1: the label-error certificate contract and the continuity profile.

    uv run --group evidence python tools/m3_certificates/check.py --out DIR

On the analytic double integrator (true dynamics f(z, a) = A z + B a, so the
true label error ||T(q) - y|| is computed independently of the certificate):
- with the exact model and analytic constants C = 0, every row is certified,
  covered, and tight;
- with a deliberately wrong linear model f_hat = A_hat z + B_hat a, the
  analytic constants C_z = ||A - A_hat||, C_a = ||B - B_hat|| cover every row
  of both generators (state-only and state-and-action displacements);
- the same model with an underestimated declared constant (C = 0) is
  conditional and fails coverage on some rows: a bound is only as good as its
  constants;
- labels outside the declared neighborhood are unsupported, not upper bounds,
  and not admissible;
- the masks follow their rules, and candidate.accepted never changes;
- structural recomputation detects arithmetic, classification, mask,
  descriptor, and candidate-file corruptions; only the source audit detects a
  falsified source term that is consistent with the totals, and the audit does
  not detect arithmetic corruptions: the two levels check different things;
- the reader rejects malformed certificate files;
- the pipeline writes a complete labels.certify run next to an unchanged
  candidates file; a rerun reuses it, and a changed constant is a new run
  (the inspectable artifact).
Writes DIR/report.json.
"""
import argparse
import copy
import hashlib
import json
import platform
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
from ccil.core.certificates import (Constant, ContinuityProfile, _decide, audit_sources, certify,  # noqa: E402
                                    read_certificates, verify_structure, write_certificates)
from ccil.runs import validate_run  # noqa: E402
from ccil.schema import read_candidates, write_candidates, write_trajectories  # noqa: E402
from ccil.schema.io import _atomic_savez  # noqa: E402

SEED, K, SIGMA = 20261025, 4, 0.05
# Stored candidates and states are float32; the demonstration transition is exact only up to that rounding.
COVER_TOL = 1e-6


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_label_error(c, sys_):
    """||T(q) - y|| with the true dynamics, from the stored candidate arrays only."""
    z, a = c.observations["state"].astype(np.float64), c.actions.astype(np.float64)
    return np.linalg.norm(sys_.step_np(z, a) - c.target_observations["state"].astype(np.float64), axis=1)


def analytic(value, why):
    return Constant(value, "analytic_bound", why, "representation units per input unit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:700]}", flush=True)

    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    config["episodes"] = [e for e in config["episodes"] if not e["corrupt_actions"]]
    dataset = fixture(config).validate()
    sys_ = DoubleIntegrator(config["h"])
    exact = LinearTransitionModel(sys_.A, sys_.B)
    A_hat, B_hat = sys_.A + np.array([[0.05, 0.0], [-0.08, 0.04]]), sys_.B * 1.6
    wrong = LinearTransitionModel(A_hat, B_hat)
    Cz, Ca = float(np.linalg.norm(sys_.A - A_hat, 2)), float(np.linalg.norm(sys_.B - B_hat, 2))
    rep = IdentityRepresentation("state")
    generators = {"backward_euler": BackwardEuler(), "noisy_action": NoisyAction(K, SIGMA)}

    def candidates(model, gen, name):
        c = generate_labels(dataset, rep, model, gen, [Finite()], run_id=name, seed=SEED)
        path = out / f"{name}.npz"
        write_candidates(c, path)
        return read_candidates(path, dataset), path

    # 1. Exact model, analytic C = 0: certified, covered, tight.
    for gname, gen in generators.items():
        c, path = candidates(exact, gen, f"exact-{gname}")
        prof = ContinuityProfile(analytic(0.0, "the model is the true dynamics"), analytic(0.0, "the model is the true dynamics"))
        cert = certify(c, dataset, rep, exact, prof, sha(path))
        err = true_label_error(c, sys_)
        gap = cert.total_error_value - err
        record(f"exact_model_certified_and_tight[{gname}]",
               (cert.classification == "certified").all() and cert.is_upper_bound.all()
               and (gap >= -COVER_TOL).all() and float(np.abs(gap).max()) <= 1e-5,
               rows=int(len(err)), max_true_error=float(err.max()), max_bound=float(cert.total_error_value.max()),
               max_abs_gap=float(np.abs(gap).max()), max_anchor_model_error=float(cert.terms["anchor_model_error"].max()))

    # 2. Wrong linear model, analytic C = ||A - A_hat||, ||B - B_hat||: every row covered.
    wrong_sets = {}
    for gname, gen in generators.items():
        c, path = candidates(wrong, gen, f"wrong-{gname}")
        prof = ContinuityProfile(analytic(Cz, "||A - A_hat||_2 of the true and learned linear maps"),
                                 analytic(Ca, "||B - B_hat||_2 of the true and learned linear maps"))
        cert = certify(c, dataset, rep, wrong, prof, sha(path))
        err = true_label_error(c, sys_)
        slack = cert.total_error_value - err
        wrong_sets[gname] = (c, path, cert)
        record(f"wrong_model_covered_with_analytic_constants[{gname}]",
               (cert.classification == "certified").all() and (slack >= -COVER_TOL).all()
               and float(cert.terms["anchor_model_error"].max()) > 1e-3
               and (gname != "noisy_action" or float(cert.terms["action_displacement"].max()) > 0),
               C_z=Cz, C_a=Ca, rows=int(len(err)), min_slack=float(slack.min()),
               max_true_error=float(err.max()), max_solver_error=float(cert.terms["solver_error"].max()),
               max_bound=float(cert.total_error_value.max()),
               max_bound_over_true=float(np.max(cert.total_error_value / np.maximum(err, 1e-12))))

    # 3. Underestimated declared constant: conditional, and coverage fails.
    c, path, _ = wrong_sets["noisy_action"]
    under = ContinuityProfile(Constant(0.0, "declared", "assumed (wrongly) that the model is exact", "1"),
                              Constant(0.0, "declared", "assumed (wrongly) that the model is exact", "1"))
    cu = certify(c, dataset, rep, wrong, under, sha(path))
    err = true_label_error(c, sys_)
    violated = cu.total_error_value < err - COVER_TOL
    record("underestimated_constant_fails_coverage",
           (cu.classification == "conditional").all() and cu.is_upper_bound.all() and violated.any(),
           classification="conditional", rows=int(len(err)), violated_rows=int(violated.sum()),
           worst_shortfall=float((err - cu.total_error_value).max()))

    # 4. Outside the neighborhood: unsupported, not a bound, not admissible.
    c, path, cert = wrong_sets["noisy_action"]
    radius = float(np.median(cert.terms["state_displacement"]))
    nb = ContinuityProfile(analytic(Cz, "||A - A_hat||_2"), analytic(Ca, "||B - B_hat||_2"), neighborhood_state=radius)
    cn = certify(c, dataset, rep, wrong, nb, sha(path))
    outside = cn.terms["state_displacement"] > radius
    record("outside_neighborhood_unsupported",
           outside.any() and (~outside).any() and (cn.classification[outside] == "unsupported").all()
           and not cn.is_upper_bound[outside].any() and not cn.admissible[outside].any()
           and cn.reasons[outside, 0].all() and not cn.reasons[~outside, 0].any()
           and (cn.classification[~outside] == "certified").all(),
           radius=radius, outside=int(outside.sum()), inside=int((~outside).sum()))

    # 5. Masks follow their rules; candidate acceptance never changes.
    before = c.accepted.copy()
    limit = float(np.quantile(cert.total_error_value, 0.8))
    cm = certify(c, dataset, rep, wrong, nb, sha(path), max_total_error=limit, selection_quantile=0.5)
    eligible = c.accepted & cm.admissible
    thr = cm.thresholds["selection_threshold"]
    record("masks_follow_rules_and_acceptance_unchanged",
           np.array_equal(c.accepted, before)
           and np.array_equal(cm.admissible, cm.is_upper_bound & (cm.total_error_value <= limit))
           and np.array_equal(cm.selected, eligible & (cm.total_error_value <= thr))
           and thr == float(np.quantile(cm.total_error_value[eligible], 0.5))
           and np.array_equal(cm.reasons[:, 1], cm.is_upper_bound & (cm.total_error_value > limit))
           and np.array_equal(cm.reasons[:, 2], eligible & ~cm.selected),
           max_total_error=limit, selection_threshold=thr, accepted=int(c.accepted.sum()),
           admissible=int(cm.admissible.sum()), selected=int(cm.selected.sum()),
           usable=int((c.accepted & cm.admissible & cm.selected).sum()))

    # 6. Two verification levels detect different corruptions.
    cpath = out / "certificates.npz"
    write_certificates(cm, cpath)
    base = read_certificates(cpath)
    clean = {"structure": verify_structure(base, c, path), "audit": audit_sources(base, c, dataset, rep, wrong)}

    def forged_consistent(cs, term, i, delta):
        """Falsify one source term and rederive every value and mask from it, as a careful forger would."""
        cs = copy.deepcopy(cs)
        cs.terms[term][i] += delta
        t = cs.thresholds
        (cs.model_error_value, cs.total_error_value, cs.classification, cs.is_upper_bound, cs.in_neighborhood,
         cs.admissible, cs.selected, cs.reasons, thr) = _decide(nb, cs.terms, c.accepted, t["max_total_error"],
                                                                 t["selection_quantile"])
        cs.thresholds = {**t, "selection_threshold": thr}
        return cs

    def mutate(fn):
        cs = copy.deepcopy(base)
        fn(cs)
        return cs

    i = int(np.flatnonzero(base.is_upper_bound)[0])
    tampered = out / "tampered-candidates.npz"
    tampered.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))
    corruptions = {
        "total_halved": (mutate(lambda cs: cs.total_error_value.__setitem__(i, cs.total_error_value[i] / 2)), path),
        "unsupported_marked_certified": (mutate(lambda cs: cs.classification.__setitem__(
            int(np.flatnonzero(cs.classification == "unsupported")[0]), "certified")), path),
        "admissible_flipped": (mutate(lambda cs: cs.admissible.__setitem__(i, ~cs.admissible[i])), path),
        "descriptor_constant_lowered": (mutate(lambda cs: cs.descriptor["constants"]["C_z"].__setitem__("value", 0.0)), path),
        "candidates_file_changed": (base, tampered),
        "anchor_error_falsified_consistently": (forged_consistent(base, "anchor_model_error", i, -float(base.terms["anchor_model_error"][i])), path),
        "solver_error_falsified_consistently": (forged_consistent(base, "solver_error", i, -float(base.terms["solver_error"][i])), path),
    }
    detected = {}
    for name, (cs, cfile) in corruptions.items():
        detected[name] = {"structure": bool(verify_structure(cs, c, cfile)), "audit": bool(audit_sources(cs, c, dataset, rep, wrong))}
    structural = ("total_halved", "unsupported_marked_certified", "admissible_flipped", "descriptor_constant_lowered",
                  "candidates_file_changed")
    sourced = ("anchor_error_falsified_consistently", "solver_error_falsified_consistently")
    record("structure_and_audit_detect_different_corruptions",
           clean == {"structure": [], "audit": []}
           and all(detected[n] == {"structure": True, "audit": False} for n in structural)
           and all(detected[n] == {"structure": False, "audit": True} for n in sourced),
           clean=clean, detected=detected)

    # 7. The reader rejects malformed certificate files.
    with np.load(cpath, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    def rejects(edit):
        a = copy.deepcopy(arrays)
        edit(a)
        p = out / "malformed.npz"
        _atomic_savez(p, a)
        try:
            read_certificates(p)
            return "accepted"
        except Exception as e:  # noqa: BLE001 (the evidence records which error, if any)
            return f"{type(e).__name__}: {str(e).replace(str(out), '<out>')[:200]}"

    def meta_edit(fn):
        def edit(a):
            m = json.loads(str(a["meta"]))
            fn(m)
            a["meta"] = np.array(json.dumps(m))
        return edit
    malformed = {
        "next_major_version": rejects(meta_edit(lambda m: m.update(version="2.0"))),
        "missing_term": rejects(lambda a: a.pop("term/anchor_model_error")),
        "classification_code_out_of_range": rejects(lambda a: a["classification"].__setitem__(0, 9)),
        "row_count_mismatch": rejects(lambda a: a.__setitem__("selected", a["selected"][:-1])),
        "unknown_reason_codes": rejects(meta_edit(lambda m: m.update(reason_codes=["certificate.other"]))),
    }
    record("malformed_certificate_files_rejected", all(v.startswith("SchemaError") for v in malformed.values()),
           malformed=malformed)

    # 8. Constant sources are validated, and continuity refuses a point Jacobian.
    api = {}
    for label, make in {"unknown_source": lambda: Constant(1.0, "guess", "x", "1"),
                        "negative_value": lambda: Constant(-1.0, "declared", "x", "1"),
                        "no_rationale": lambda: Constant(1.0, "declared", "", "1"),
                        "point_jacobian_in_continuity": lambda: ContinuityProfile(
                            Constant(1.0, "point_jacobian", "J at the anchor", "1"), analytic(0.0, "x"))}.items():
        try:
            make()
            api[label] = "accepted"
        except InterfaceError as e:
            api[label] = f"InterfaceError: {e}"
    record("constants_validated", all(v.startswith("InterfaceError") for v in api.values()), refused=api)

    # 9. The pipeline: labels.certify next to an unchanged candidates file.
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
    declared = ContinuityProfile(Constant(0.1, "declared", "assumed variation of f - f_hat on this fixture", "1"),
                                 Constant(0.1, "declared", "assumed variation of f - f_hat on this fixture", "1"))
    first = certify_stage(cfg, prepared, fitted, labeled, rep, declared, selection_quantile=0.75)
    again = certify_stage(cfg, prepared, fitted, labeled, rep, declared, selection_quantile=0.75)
    changed = certify_stage(cfg, prepared, fitted, labeled, rep, ContinuityProfile(
        Constant(0.2, "declared", "assumed variation of f - f_hat on this fixture", "1"),
        Constant(0.1, "declared", "assumed variation of f - f_hat on this fixture", "1")), selection_quantile=0.75)
    status = {p.name: validate_run(p)[0] for p in sorted((work / "runs").iterdir()) if p.is_dir()}
    pc = read_candidates(cand_file)
    pcert = read_certificates(first.out("labels/certificates.npz"))
    stage = json.loads(first.out("stage.json").read_text())
    record("pipeline_certify_stage",
           set(status.values()) == {"complete"} and len(status) == 5 and sha(cand_file) == cand_sha
           and pcert.candidates_sha256 == cand_sha and pcert.inputs.get("candidates") == cand_sha
           and again.reused and again.run_dir == first.run_dir and changed.run_dir != first.run_dir
           and stage["parameters"]["profile"]["constants"]["C_z"]["value"] == 0.1
           and verify_structure(pcert, pc, cand_file) == [],
           stages=status, candidates_sha256=cand_sha, reused=again.reused,
           changed_constant_new_run=changed.run_dir != first.run_dir,
           classification=sorted(set(pcert.classification.tolist())),
           usable=int((pc.accepted & pcert.admissible & pcert.selected).sum()), rows=int(len(pc.accepted)),
           artifact=f"<out>/pipeline/runs/{first.run_dir.name}/labels/certificates.npz")

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
