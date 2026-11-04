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
The diagnosis (part 2), on double-integrator demonstrations with exploration
noise (dense, so tau_demo is small) under exact or nonlinear dynamics, each
gate recomputed independently with numpy from the dataset, the split, the
candidates, and the evaluation checkpoint:
- admission: with exact dynamics, M = 1 passes every gate and labels are
  generated, with d_source and d_demo on every row;
- refusal: a tight support radius fails reach (eta_reach < 1), a linear model
  of nonlinear dynamics fails trust, and no held-out episodes leave tau_demo
  undefined; each names its gate and a next step;
- the empty result is complete: zero-row candidates, certificates, and
  selection, every run directory complete, no synthetic supervision row;
- a declared recovery length generates and still reports failing gates, and
  a configuration written before version 3 keeps generating at 1.
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
from ccil.schema import (Episode, SchemaError, TrajectoryDataset, read_candidates, read_certificates,  # noqa: E402
                         read_selection, write_candidates, write_certificates, write_selection, write_trajectories)
from ccil.schema.diagnosis import read_diagnosis  # noqa: E402
from ccil.runs import validate_run  # noqa: E402
from ccil.supervision import SupervisionError, build_supervision  # noqa: E402

SEED, K, SIGMA = 20261103, 6, 0.08
DT, PROBE = 0.05, 256
REF = Provided("linear_discrepancy", "true")
DEMO = Provided("reference_residual", "true")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_label_error(c, sys_):
    """||T(q) - y|| with the true dynamics, from the stored candidate arrays only."""
    z, a = c.observations["state"].astype(np.float64), c.actions.astype(np.float64)
    return np.linalg.norm(sys_.step_np(z, a) - c.target_observations["state"].astype(np.float64), axis=1)


def demonstrations(path, k, n=20, spread=0.05, noise=0.3, T=41, seed=0):
    """Double-integrator demonstrations with exploration noise; k > 0 adds a -k sin(3x) force the linear model cannot fit."""
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        x = np.zeros((T, 2), np.float32)
        x[0] = np.array([1.5, 0.0]) + rng.normal(0, spread, 2)
        u = np.zeros((T, 1), np.float32)
        for s in range(T - 1):
            u[s] = -1.0 * x[s, 0] - 1.5 * x[s, 1] + rng.normal(0, noise)
            x[s + 1] = x[s] + DT * np.array([x[s, 1], u[s, 0] - k * np.sin(3 * x[s, 0])])
        eps.append(Episode(f"ep{i:02d}", {"state": x}, u, timestamps=np.arange(T) * DT, termination_known=True))
    write_trajectories(TrajectoryDataset(eps), path)


def independent_gates(dataset_path, split_path, cands, eval_ckpt):
    """tau_demo, r_reach, median reach, and the held-out error, recomputed with numpy alone."""
    with np.load(dataset_path, allow_pickle=False) as z:
        ids, lengths, states, actions = z["episode_ids"], z["episode_lengths"], z["obs/state"].astype(np.float64), z["actions"]
    split = json.loads(Path(split_path).read_text()) if split_path else None
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    per = {str(e): (states[s:s + n], actions[s:s + n].astype(np.float64)) for e, s, n in zip(ids, starts, lengths)}
    train = [str(e) for e in ids] if split is None else split["train"]
    held = [] if split is None else split["validation"]
    tr = np.vstack([per[e][0] for e in train])
    scale = np.where(tr.std(axis=0) > 0, tr.std(axis=0), 1.0)
    near = lambda p, ref: np.sqrt(((p[:, None, :] / scale - ref[None, :, :] / scale) ** 2).sum(-1)).min(axis=1)  # noqa: E731
    tau = float(np.quantile(near(np.vstack([per[e][0] for e in held]), tr), 0.95)) if held else None
    train_tr = [(e, f) for e in [str(x) for x in ids] if e in set(train) for f in range(len(per[e][0]) - 1)]
    idx = np.arange(len(train_tr)) if len(train_tr) <= PROBE else np.unique(np.round(np.linspace(0, len(train_tr) - 1, PROBE)).astype(int))
    probe = {train_tr[i] for i in idx}
    in_probe = np.array([(str(e), int(f)) in probe for e, f in zip(cands.source_episode_id, cands.source_frame)])
    src = np.array([per[str(e)][0][int(f)] for e, f in zip(cands.source_episode_id, cands.source_frame)])
    d_source = np.linalg.norm((cands.observations["state"].astype(np.float64) - src) / scale, axis=1)
    d = d_source[in_probe & cands.accepted]
    eps = None
    if held:
        with np.load(eval_ckpt, allow_pickle=False) as m:
            A, B, c = m["A"], m["B"], m["c"]
        z = np.vstack([per[e][0][:-1] for e in held])
        a = np.vstack([per[e][1][:-1] for e in held])
        zn = np.vstack([per[e][0][1:] for e in held])
        eps = float(np.quantile(np.linalg.norm((z + z @ A.T + a @ B.T + c - zn) / scale, axis=1), 0.95))
    return {"tau": tau, "r_reach": float(np.quantile(d, 0.9)) if len(d) else None,
            "median_reach": float(np.quantile(d, 0.5)) if len(d) else None, "epsilon": eps, "scale": scale,
            "d_source": d_source, "d_demo": near(cands.observations["state"].astype(np.float64), tr)}


def close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and np.isclose(a, b, rtol=1e-9, atol=1e-12))


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

    # Part 2: the diagnosis.
    diag = out / "diagnosis"
    diag.mkdir()
    exact, nonlinear = diag / "demos-exact.npz", diag / "demos-nonlinear.npz"
    demonstrations(exact, 0.0)
    demonstrations(nonlinear, 4.0)
    common = ["compute.device=cpu", "transition_model.kind=linear", "generation.method=noisy_action",
              "generation.labels_per_transition=4", "generation.action_noise_std=1.0"]

    def pipeline(name, data, extra=()):
        res = generate_offline(resolve(sets=[f"data.path={data}", f"run.output_dir={diag / 'runs'}", *common, *extra]))
        prep, fit, lab, dia = (stage(res, s) for s in ("data.prepare", "transition.fit", "labels.generate", "labels.diagnose"))
        split = prep.out(layout.SPLIT) if prep.out(layout.SPLIT).exists() else None
        ev = fit.out("checkpoints/evaluation_model.npz")
        return {"results": res, "diagnosis": read_diagnosis(dia.out(layout.DIAGNOSIS)), "split": split,
                "dataset": prep.out(layout.DATASET), "candidates": read_candidates(lab.out(layout.CANDIDATES)),
                "eval": ev if ev.exists() else fit.out("checkpoints/transition_model.npz")}

    def matches(doc, ind):
        g = doc["gates"][0]
        return {"tau_demo": close(doc["tau_demo"]["value"], ind["tau"]), "r_reach": close(g["reach"]["r_reach"], ind["r_reach"]),
                "median_reach": close(g["trust"]["median_reach"], ind["median_reach"]),
                "epsilon": close(g["trust"]["epsilon"], ind["epsilon"]),
                "eta_reach": close(g["reach"]["eta_reach"], None if not ind["tau"] or ind["r_reach"] is None
                                   else ind["r_reach"] / ind["tau"])}

    admit = pipeline("admit", exact)
    ind = independent_gates(admit["dataset"], admit["split"], admit["candidates"], admit["eval"])
    doc, c = admit["diagnosis"], admit["candidates"]
    agree = matches(doc, ind)
    train_rows = np.isin(c.source_episode_id.astype(str), json.loads(Path(admit["split"]).read_text())["train"])
    rows_agree = {
        "d_source": bool(np.allclose(c.measurements["quality.d_source"], ind["d_source"], rtol=1e-6, atol=1e-9)),
        "d_demo": bool(np.allclose(c.measurements["quality.d_demo"], ind["d_demo"], rtol=1e-6, atol=1e-9)),
        # The nearest training state is at most as far as the source, when the source is a training state.
        "d_demo_le_d_source": bool((c.measurements["quality.d_demo"] <= c.measurements["quality.d_source"] + 1e-12)[train_rows].all())}
    record("diagnosis_admits_when_gates_pass",
           doc["admitted"] == 1 and doc["admitted_by"] == "gates" and doc["gates"][0]["pass"] and len(c.accepted) > 0
           and all(agree.values()) and all(rows_agree.values()) and np.isclose(doc["timing"]["dt_ctrl"], DT, rtol=1e-6)
           and np.isclose(doc["gates"][0]["horizon_s"], DT, rtol=1e-6),
           independent=agree, rows=rows_agree, gates=doc["gates"][0], timing=doc["timing"], candidates=int(len(c.accepted)),
           bands=doc["bands"], nested_views=doc["nested_views"])

    # Each refusal recomputed from a declared run of the same configuration, whose candidates are persisted.
    refusals = {"reach": (exact, ["generation.support_radius=0.01"], "eta_reach"),
                "trust": (nonlinear, [], "held-out error"),
                "no_held_out": (exact, ["data.validation_fraction=0"], "held-out")}
    refused = {}
    for name, (data, extra, phrase) in refusals.items():
        auto = pipeline(name, data, extra)
        declared = pipeline(name, data, [*extra, "generation.recovery_steps=1"])
        ind = independent_gates(declared["dataset"], declared["split"], declared["candidates"], declared["eval"])
        d_auto, d_decl = auto["diagnosis"], declared["diagnosis"]
        refused[name] = {"admitted": d_auto["admitted"], "reason": d_auto["reason"], "next_step": d_auto["next_step"],
                         "independent": matches(d_auto, ind), "candidates_auto": int(len(auto["candidates"].accepted)),
                         "candidates_declared": int(len(declared["candidates"].accepted)),
                         "declared": {"admitted": d_decl["admitted"], "admitted_by": d_decl["admitted_by"],
                                      "gates_pass": d_decl["gates"][0]["pass"]}, "phrase": phrase}
    record("diagnosis_refuses_each_gate",
           all(r["admitted"] is None and r["phrase"] in r["reason"] and r["next_step"] and all(r["independent"].values())
               and r["candidates_auto"] == 0 and r["candidates_declared"] > 0
               and r["declared"] == {"admitted": 1, "admitted_by": "declared", "gates_pass": False}
               for r in refused.values()),
           refusals=refused)

    # The empty result, with a certificate and a selection, is complete and gives no synthetic row.
    cert_sets = ["certificate.profile=disturbed_action", "certificate.true_state_lipschitz=1.0",
                 "certificate.true_action_lipschitz=0.2", "certificate.demonstration_residual_bound=0.0",
                 "curate.tolerance=0.05"]
    empty = pipeline("empty", nonlinear, cert_sets)
    res = empty["results"]
    cer, cur = stage(res, "labels.certify"), stage(res, "labels.curate")
    ecert, esel = read_certificates(cer.out(layout.CERTIFICATES)), read_selection(cur.out(layout.SELECTION))
    status = {r.stage: validate_run(r.run_dir)[0] for r in res}
    esup = build_supervision(empty["dataset"], "ccil", representation=rep, candidates=stage(res, "labels.generate").out(layout.CANDIDATES),
                             certificates=cer.out(layout.CERTIFICATES), selection=cur.out(layout.SELECTION), split=empty["split"])
    record("empty_result_is_complete",
           empty["diagnosis"]["admitted"] is None and len(empty["candidates"].accepted) == 0
           and len(ecert.classification) == 0 and len(esel.mask) == 0 and set(status.values()) == {"complete"}
           and esup.manifest()["counts"]["synthetic_rows"] == 0 and esup.manifest()["counts"]["demonstration_rows"] > 0,
           stages=status, counts=esup.manifest()["counts"], reason=empty["diagnosis"]["reason"])

    # A configuration written before version 3 keeps generating at 1, with a note.
    old = diag / "version-2.yaml"
    old.write_text("config_version: 2\ngeneration: {method: backward_euler_fast}\ntransition_model: {kind: linear}\n")
    r2 = resolve(config=old, sets=[f"data.path={exact}"])
    record("older_configuration_keeps_declared_length",
           r2.values["generation.recovery_steps"] == "1" and any("recovery_steps" in n for n in r2.notes)
           and resolve(sets=[f"data.path={exact}", "generation.method=backward_euler_fast"]).values["generation.recovery_steps"] == "auto",
           notes=[n for n in r2.notes if "recovery_steps" in n])

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
