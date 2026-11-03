#!/usr/bin/env python3
"""Evidence for #167: certificates are evidence, selections are separate decisions.

    uv run --group evidence python tools/m45_quality/check.py --out DIR

On the analytic double integrator (true dynamics f(z, a) = A z + B a, so the
true label error ||T(q) - y|| is computed independently of every artifact):
- soundness: with a deliberately wrong linear model and reference-dynamics
  constants, every row a bound policy selects at tolerance tol has true error
  <= its certified bound <= tol, and a ratio <= 1 exactly on the selected rows;
- arithmetic recomputation: a consumer that reads the three .npz files with
  numpy alone recomputes the mask, ratio, and weight of a bound selection;
- source recomputation: the certificate's terms are re-derived from their
  sources (audit_sources) and the selection from its inputs (verify_selection);
- tolerance change: through the pipeline, a new tolerance or quantile writes a
  new labels.curate run, reuses the candidate and certificate runs with their
  hashes unchanged, and selects nested rows;
- fail closed: unsupported rows are never selected, a bound policy selects no
  empirical row, and data_only curation refuses reference-assisted evidence,
  in the library and in the configuration before any run is written;
- boundaries: a certificate whose totals do not follow from its terms is
  refused before any selection exists; a selection records the provenance of
  the evidence it used, not the allowance it ran under, and a selection that
  claims either wrongly fails verification; a zero tolerance is refused;
- corruption refusal: a corrupted selection file, a forged mask, a selection
  of other certificates, a 2.0 certificate without a selection, and a 1.0
  certificate with one are refused by build_supervision;
- legacy: a 1.0 certificate still yields accepted & admissible & selected, and
  those are exactly the rows the same policy selects through ccil.selection.
Writes DIR/report.json (the inspectable artifact) beside the selections it
describes; the report records each selection's SHA-256.
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

from ccil.config import ConfigError, resolve  # noqa: E402
from ccil.core import Finite, IdentityRepresentation, InterfaceError, LinearTransitionModel, NoisyAction, generate_labels  # noqa: E402
from ccil.core.certificates import (ContinuityProfile, DisturbedActionPaperProxy, Provided, Sources,  # noqa: E402
                                    audit_sources, certify, decide, resolved_of, write_reference_dynamics)
from ccil.core.curate import CurationPolicy, curate, verify_selection  # noqa: E402
from ccil.pipelines.offline import generate_offline, stage  # noqa: E402
from ccil.runs import layout  # noqa: E402
from ccil.schema import (SchemaError, read_candidates, read_selection, write_candidates,  # noqa: E402
                         write_certificates, write_selection, write_trajectories)
from ccil.supervision import SupervisionError, build_supervision  # noqa: E402

SEED, K, SIGMA = 20261103, 6, 0.08
REF = Provided("linear_discrepancy", "true")
DEMO = Provided("reference_residual", "true")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_label_error(c, sys_):
    """||T(q) - y|| with the true dynamics, from the stored candidate arrays only."""
    z, a = c.observations["state"].astype(np.float64), c.actions.astype(np.float64)
    return np.linalg.norm(sys_.step_np(z, a) - c.target_observations["state"].astype(np.float64), axis=1)


def outcome(fn):
    try:
        fn()
        return "accepted"
    except (SchemaError, InterfaceError, SupervisionError, ConfigError) as e:
        return f"{type(e).__name__}: " + " | ".join(line.strip() for line in str(e).splitlines()[1:] or [str(e)])[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    artifacts = {}

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

    # A deliberately wrong linear model: label errors are real, and reference constants cover them.
    wrong = LinearTransitionModel(sys_.A + np.array([[0.05, 0.0], [-0.08, 0.04]]), sys_.B * 1.6)
    wrong.save(out / "model-wrong.npz")
    cands = generate_labels(dataset, rep, wrong, NoisyAction(K, SIGMA), [Finite()], run_id="wrong-noisy", seed=SEED,
                            inputs={"dataset": sha(data_path), "transition_model": sha(out / "model-wrong.npz")})
    cpath = out / "candidates.npz"
    write_candidates(cands, cpath)
    src = Sources(cpath, data_path, out / "model-wrong.npz", rep, references={"true": truth})
    cert = certify(src, ContinuityProfile(REF, REF, DEMO))
    kpath = out / "certificates.npz"
    write_certificates(cert, kpath)
    err = true_label_error(cands, sys_)
    tol = float(np.quantile(cert.total_error_value, 0.5))
    sel = curate(cpath, kpath, CurationPolicy(admit="bound", tolerance=tol), allow_evidence="reference_assisted")
    spath = out / "selection-bound.npz"
    write_selection(sel, spath)
    artifacts["selection-bound.npz"] = sha(spath)

    # 1. Soundness against the true dynamics.
    chosen = sel.mask
    record("selected_bounds_cover_true_error",
           chosen.any() and (~chosen).any() and (cert.classification == "certified").all()
           and (err[chosen] <= cert.total_error_value[chosen]).all() and (cert.total_error_value[chosen] <= tol).all()
           and np.array_equal(chosen, cands.accepted & (sel.ratio <= 1)) and sel.error_spec["is_bound"]
           and sel.evidence_mode == "reference_assisted",
           rows=int(len(chosen)), selected=int(chosen.sum()), tolerance=tol,
           max_true_error_selected=float(err[chosen].max()), max_bound_selected=float(cert.total_error_value[chosen].max()),
           max_true_error_unselected=float(err[~chosen].max()))

    # 2. A consumer recomputes the selection with numpy alone.
    with np.load(cpath, allow_pickle=False) as c, np.load(kpath, allow_pickle=False) as k, \
            np.load(spath, allow_pickle=False) as s:
        meta = json.loads(str(s["meta"]))
        t = meta["policy"]["tolerance"]
        mask = c["accepted"] & k["is_upper_bound"] & (k["total_error_value"] <= t)
        ratio = np.where(k["classification"] != 4, k["total_error_value"] / t, np.nan)
        weight = np.where(mask, c["weight"], 0.0)
        consumer = {"mask": np.array_equal(mask, s["mask"]), "ratio": np.allclose(ratio, s["ratio"], equal_nan=True),
                    "weight": np.array_equal(weight, s["weight"]),
                    "inputs": meta["candidates_sha256"] == sha(cpath) and meta["certificates_sha256"] == sha(kpath)}
    record("consumer_recomputes_selection_with_numpy", all(consumer.values()), checks=consumer,
           error_units=meta["error"]["units"], caveats=meta["caveats"])

    # 3. Source recomputation of both layers.
    record("sources_recompute_certificate_and_selection",
           audit_sources(cert, src) == [] and verify_selection(read_selection(spath), cpath, kpath) == [],
           audit=audit_sources(cert, src), verify=verify_selection(read_selection(spath), cpath, kpath))

    # 4. Fail closed: unsupported rows, empirical rows under a bound policy, data_only with reference evidence.
    radius = float(np.median(cert.terms["state_displacement"]))
    nb = certify(src, ContinuityProfile(REF, REF, DEMO, neighborhood_state=radius))
    nbpath = out / "certificates-neighborhood.npz"
    write_certificates(nb, nbpath)
    snb = curate(cpath, nbpath, CurationPolicy(admit="bound"), allow_evidence="reference_assisted")
    outside = nb.classification == "unsupported"
    score = certify(src, DisturbedActionPaperProxy())
    scpath = out / "certificates-score.npz"
    write_certificates(score, scpath)
    as_bound = curate(cpath, scpath, CurationPolicy(admit="bound"))
    limit = float(np.quantile(score.total_error_value, 0.75))
    as_score = curate(cpath, scpath, CurationPolicy(admit="score", tolerance=limit))
    data_only = outcome(lambda: curate(cpath, kpath, CurationPolicy(admit="bound", tolerance=tol)))
    record("fail_closed",
           outside.any() and not snb.mask[outside].any() and snb.reasons[outside, 1].all() and snb.mask[~outside].any()
           and (score.classification == "empirical").all() and not as_bound.mask.any()
           and as_bound.reasons[cands.accepted, 2].all()
           and as_score.mask.any() and not as_score.error_spec["is_bound"]
           and np.array_equal(as_score.mask, cands.accepted & (score.total_error_value <= limit))
           and data_only.startswith("InterfaceError") and "reference_assisted" in data_only,
           unsupported_rows=int(outside.sum()), unsupported_selected=int(snb.mask[outside].sum()),
           empirical_under_bound_policy=int(as_bound.mask.sum()), empirical_under_score_policy=int(as_score.mask.sum()),
           data_only_refusal=data_only)

    # 4b. Boundaries: forged certificates, recorded provenance, zero tolerance.
    forged_cert = copy.deepcopy(cert)
    forged_cert.total_error_value = forged_cert.total_error_value * 0.5
    fcpath = out / "certificates-forged-totals.npz"
    write_certificates(forged_cert, fcpath)
    forged_refusal = outcome(lambda: curate(cpath, fcpath, CurationPolicy(admit="bound", tolerance=tol),
                                            allow_evidence="reference_assisted"))
    permitted = curate(cpath, scpath, CurationPolicy(admit="score", tolerance=limit), allow_evidence="reference_assisted")
    uncertified = curate(cpath, None, CurationPolicy(admit="accepted"), allow_evidence="reference_assisted")
    claimed = copy.deepcopy(permitted)
    claimed.evidence_mode = "reference_assisted"
    disclaimed = copy.deepcopy(sel)
    disclaimed.evidence_mode = "data_only"
    zero = {"policy": outcome(lambda: CurationPolicy(admit="bound", tolerance=0.0)),
            "config": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=[
                f"data.path={data_path}", "certificate.profile=disturbed_action_paper_proxy", "curate.tolerance=0"]))}
    record("boundaries_hold",
           forged_refusal.startswith("InterfaceError") and "does not verify" in forged_refusal
           and permitted.evidence_mode == "data_only" and permitted.evidence_allowed == "reference_assisted"
           and uncertified.evidence_mode == "data_only"
           and any("provenance" in q for q in verify_selection(claimed, cpath, scpath))
           and any("provenance" in q for q in verify_selection(disclaimed, cpath, kpath))
           and all(v.startswith(("InterfaceError", "ConfigError")) for v in zero.values())
           and np.isnan(sel.ratio).sum() == int((cert.classification == "unsupported").sum()),
           forged_certificate=forged_refusal,
           data_only_certificate_under_reference_allowance={"evidence_mode": permitted.evidence_mode,
                                                             "evidence_allowed": permitted.evidence_allowed},
           wrong_claims={"reference_claimed": verify_selection(claimed, cpath, scpath),
                         "data_only_claimed": verify_selection(disclaimed, cpath, kpath)},
           zero_tolerance=zero)

    # 5. Tolerance changes through the pipeline: new selections, unchanged candidates and certificates.
    work = out / "pipeline"
    work.mkdir()
    write_trajectories(dataset, work / "fixture.npz")
    base = [f"data.path={work / 'fixture.npz'}", f"run.output_dir={work / 'runs'}", "compute.device=cpu",
            "generation.method=noisy_action", f"generation.labels_per_transition={K}",
            f"generation.action_noise_std={SIGMA}", "generation.support_radius=null",
            "certificate.profile=disturbed_action", "certificate.true_state_lipschitz=1.0",
            "certificate.true_action_lipschitz=0.2", "certificate.demonstration_residual_bound=0.0"]
    runs = {}
    for name, extra in (("loose", ["curate.tolerance=0.02"]), ("tight", ["curate.tolerance=0.01"]),
                        ("budget", ["curate.tolerance=0.02", "curate.quantile=0.5"]), ("loose_again", ["curate.tolerance=0.02"])):
        res = generate_offline(resolve(recipe="analytic_double_integrator", sets=base + extra))
        lab, cer, cur = stage(res, "labels.generate"), stage(res, "labels.certify"), stage(res, "labels.curate")
        runs[name] = {"labels": lab.run_dir.name, "certify": cer.run_dir.name, "curate": cur.run_dir.name,
                      "curate_reused": cur.reused, "candidates_sha256": sha(lab.out(layout.CANDIDATES)),
                      "certificates_sha256": sha(cer.out(layout.CERTIFICATES)),
                      "selection_sha256": sha(cur.out(layout.SELECTION)),
                      "mask": read_selection(cur.out(layout.SELECTION)).mask,
                      "evidence_mode": read_selection(cur.out(layout.SELECTION)).evidence_mode}
    loose, tight, budget = runs["loose"]["mask"], runs["tight"]["mask"], runs["budget"]["mask"]
    same = lambda key: len({r[key] for r in runs.values()}) == 1  # noqa: E731
    record("tolerance_change_is_a_new_selection",
           same("labels") and same("certify") and same("candidates_sha256") and same("certificates_sha256")
           and len({runs[n]["curate"] for n in ("loose", "tight", "budget")}) == 3
           and runs["loose_again"]["curate_reused"] and runs["loose_again"]["curate"] == runs["loose"]["curate"]
           and not (tight & ~loose).any() and not (budget & ~loose).any() and tight.sum() < loose.sum()
           and budget.sum() < loose.sum() and {r["evidence_mode"] for r in runs.values()} == {"data_only"},
           runs={n: {k: v for k, v in r.items() if k != "mask"} | {"selected": int(r["mask"].sum())} for n, r in runs.items()})
    for n in ("loose", "tight", "budget"):
        artifacts[f"pipeline/runs/{runs[n]['curate']}/{layout.SELECTION}"] = runs[n]["selection_sha256"]

    # 6. Configuration refuses reference evidence in data_only mode before any run is written.
    probe = work / "probe-runs"
    refused = outcome(lambda: resolve(recipe="analytic_double_integrator", sets=[
        f"data.path={work / 'fixture.npz'}", f"run.output_dir={probe}", "certificate.profile=continuity",
        f"certificate.reference_dynamics={truth}"]))
    moved = outcome(lambda: resolve(recipe="analytic_double_integrator", sets=[
        f"data.path={work / 'fixture.npz'}", "certificate.max_total_error=0.01"]))
    record("configuration_refuses_before_running",
           refused.startswith("ConfigError") and "reference_assisted" in refused and not probe.exists()
           and moved.startswith("ConfigError") and "curate.tolerance" in moved,
           reference_in_data_only=refused, moved_key=moved)

    # 7. Corruption refusal at the consumer.
    sup = lambda **kw: build_supervision(data_path, "ccil", representation=rep, candidates=cpath, **kw)  # noqa: E731
    flipped = out / "selection-flipped.npz"
    raw = spath.read_bytes()
    flipped.write_bytes(raw[:len(raw) // 2] + bytes([raw[len(raw) // 2] ^ 1]) + raw[len(raw) // 2 + 1:])
    forged = copy.deepcopy(read_selection(spath))
    i = int(np.flatnonzero(cands.accepted & ~forged.mask)[0])
    forged.mask[i], forged.reasons[i] = True, False
    forged.weight[i] = float(cands.weight[i])
    fpath = out / "selection-forged.npz"
    write_selection(forged, fpath)
    other = curate(cpath, nbpath, CurationPolicy(admit="bound"), allow_evidence="reference_assisted")
    opath = out / "selection-other-certificates.npz"
    write_selection(other, opath)
    rows, thr = decide(ContinuityProfile(REF, REF, DEMO), resolved_of(cert.descriptor), cert.terms, cands.accepted,
                       {"mode": "upper_bound", "threshold": tol}, None)
    legacy = copy.deepcopy(cert)
    for name, v in rows.items():
        setattr(legacy, name, v)
    legacy.admission, legacy.selection = {"mode": "upper_bound", "threshold": tol}, {"quantile": None, "threshold": thr,
                                                                                    "population": "admitted"}
    lpath = out / "certificates-legacy-1.0.npz"
    write_certificates(legacy, lpath)
    refusals = {"selection_byte_flipped": outcome(lambda: sup(certificates=kpath, selection=flipped)),
                "forged_mask": outcome(lambda: sup(certificates=kpath, selection=fpath)),
                "selection_of_other_certificates": outcome(lambda: sup(certificates=kpath, selection=opath)),
                "certificate_2_0_without_selection": outcome(lambda: sup(certificates=kpath)),
                "certificate_1_0_with_selection": outcome(lambda: sup(certificates=lpath, selection=spath)),
                "clean": outcome(lambda: sup(certificates=kpath, selection=spath))}
    expected = {"selection_byte_flipped": "cannot be read", "forged_mask": "does not follow from the inputs",
                "selection_of_other_certificates": "the selection was curated from certificates",
                "certificate_2_0_without_selection": "selects nothing by itself",
                "certificate_1_0_with_selection": "carries its own selection"}
    record("corruption_refused",
           refusals["clean"] == "accepted" and forged.validate() is not None
           and all(refusals[k].startswith("SupervisionError") and phrase in refusals[k] for k, phrase in expected.items()),
           refusals=refusals)

    # 8. Legacy 1.0 certificates keep their rows, and they equal the same policy's selection.
    old = sup(certificates=lpath)
    new = sup(certificates=kpath, selection=spath)
    record("legacy_certificate_rows_equal_selection",
           old.manifest()["certification"].get("legacy") is True and np.array_equal(old.candidate_row, new.candidate_row)
           and np.array_equal(old.weights, new.weights) and new.manifest()["version"] == "1.1"
           and new.manifest()["selection"]["policy"]["tolerance"] == tol,
           legacy_rows=int((old.source == "synthetic").sum()), selection_rows=int((new.source == "synthetic").sum()),
           legacy_version=".".join(map(str, legacy.version)))

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases, "artifacts": artifacts,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
