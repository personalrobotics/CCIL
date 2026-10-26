"""Evidence for #88 part 3: score profiles, selection, configuration, and migration (called by check.py).

- the paper proxies equal Defs. 4.5 and 4.6 from the model's point Jacobians,
  a BackTrack row whose action moved is unsupported whatever its generator,
  and legacy_label_filtering equals an independent transcription of the
  label-filtering branch's label_error (73d187d), on a linear model and on an
  MLP;
- scores are never bounds: every row is empirical with is_upper_bound false,
  upper_bound admission admits none, and require_upper_bound refuses them;
  score admission filters by score_threshold;
- selection never changes acceptance: the candidates file is byte-identical,
  rows rejected by the model-error gate are never selected, and the quantile
  is taken over accepted, admissible rows only (the label-filtering
  population), which differs from the quantile over all rows; with both
  legacy keys, selection_population produced reproduces the branch's
  selection (quantile over every generated label, then max_label_err and
  epsilon) exactly, where the default admitted population does not;
- a model without exact Jacobians, at 1e4 in float32 where a finite
  difference reads a derivative of 19999 as 0, gives unsupported rows with NaN
  scores, never 0;
- the branch's keys migrate to certificate.*, the withdrawn generation keys
  fail with guidance, inconsistent certificate settings (including a declared
  learned constant outside the backtrack bound profile and an action
  neighborhood for either BackTrack profile) are refused, and the integrated
  pipeline refuses certificate.profile;
- ccil labels generate with certificate settings writes complete runs, and
  the candidates it certifies are read unchanged by the reader merged before
  #88 (8663348).
"""
import json
import os
import subprocess
import sys

import numpy as np

from ccil.core import (BackwardEuler, Finite, GenerationFailed, IdentityRepresentation, InterfaceError,
                       LinearTransitionModel, ModelMisfit, NoisyAction, SupportRadius, generate_labels, jacobian_fd)
from ccil.core import transition as transition_module
from ccil.core.certificates import (BackTrackPaperProxy, DisturbedActionPaperProxy, LegacyLabelFiltering, Sources,
                                    audit_sources, certify, require_upper_bound, verify_structure)
from ccil.schema import Episode, SchemaError, TrajectoryDataset, read_candidates, write_candidates, write_trajectories
from ccil.schema.io import _atomic_savez

SEED, K, SIGMA = 20261025, 4, 0.05
PRE_88 = "8663348"  # main before #88: the candidates reader that predates certificates


def outcome(fn):
    try:
        fn()
        return "accepted"
    except (InterfaceError, SchemaError, ValueError) as e:
        return f"{type(e).__name__}: {str(e).splitlines()[-1].strip()[:200]}"


class Float32Quadratic:
    """f(z, a) = z^2 - z in float32, with no exact Jacobian: df/dz = 2z - 1 (19999 at 1e4)."""

    kind, version, evaluation_dtype = "evidence_float32_quadratic", "1", "float32"

    def delta(self, z, a):
        z32 = np.asarray(z, np.float32)
        return (z32 * z32 - z32).astype(np.float64)

    def save(self, path):
        meta = {"component": "transition_model", "kind": self.kind, "version": self.version}
        _atomic_savez(path, {"meta": np.array(json.dumps(meta))})

    @classmethod
    def load(cls, path, device="cpu"):
        return cls()


def run(out, record, dataset, data_path, sys_, truth, repo, bundle):
    import torch
    from theorems import synthetic

    from ccil.models.mlp import MLPTransitionModel

    rep = IdentityRepresentation("state")
    A_hat, B_hat = sys_.A + np.array([[0.05, 0.0], [-0.08, 0.04]]), sys_.B * 1.6
    lin = LinearTransitionModel(A_hat, B_hat)
    lin_path = out / "model-proxy-linear.npz"
    lin.save(lin_path)

    def sources(model, model_path, gen, data=dataset, dpath=data_path, validators=(Finite(),), tag=""):
        c = generate_labels(data, rep, model, gen, list(validators), run_id=f"proxy-{gen.name}{tag}", seed=SEED)
        path = out / f"candidates-proxy-{gen.name}{tag}-{model_path.stem}.npz"
        write_candidates(c, path)
        return Sources(path, dpath, model_path, rep)

    # P1. The paper proxies are Defs. 4.5 and 4.6, from point Jacobians (here the linear model's A_hat, B_hat).
    nA, nB = float(np.linalg.norm(A_hat, 2)), float(np.linalg.norm(B_hat, 2))
    s_bt = sources(lin, lin_path, BackwardEuler())
    s_da = sources(lin, lin_path, NoisyAction(K, SIGMA))
    c_bt, c_da = certify(s_bt, BackTrackPaperProxy(), admission="score"), certify(s_da, DisturbedActionPaperProxy(), admission="score")
    cb, cd = read_candidates(s_bt.candidates), read_candidates(s_da.candidates)
    z, a = (np.concatenate([ep.observations["state"][:-1] for ep in dataset.episodes]).astype(np.float64),
            np.concatenate([ep.actions[:-1] for ep in dataset.episodes]).astype(np.float64))
    src_da = np.repeat(np.arange(len(z)), K)
    ds_bt = np.linalg.norm(cb.observations["state"].astype(np.float64) - z, axis=1)
    ds_da = np.linalg.norm(cd.observations["state"].astype(np.float64) - z[src_da], axis=1)
    da = np.linalg.norm(cd.actions.astype(np.float64) - a[src_da], axis=1)
    e_bt, e_da = nA * ds_bt, nB * da + (1 + nA) * ds_da
    # A BackTrack proxy has no action term, so a row whose action moved is unsupported, whatever its generator says.
    moved = read_candidates(s_bt.candidates)
    moved.actions = moved.actions.copy()
    moved.actions[0] += np.float32(0.25)
    moved_path = out / "candidates-proxy-backtrack-moved-action.npz"
    write_candidates(moved, moved_path)
    s_moved = Sources(moved_path, data_path, lin_path, rep)
    c_moved = certify(s_moved, BackTrackPaperProxy(), admission="score")
    moved_ok = (c_moved.classification[0] == "unsupported" and not c_moved.admissible[0] and c_moved.reasons[0, 0]
                and (c_moved.classification[1:] == "empirical").all() and moved.generator == cb.generator)
    record("paper_proxies_equal_definitions",
           np.allclose(c_bt.total_error_value, e_bt, rtol=1e-12, atol=1e-15)
           and np.allclose(c_da.total_error_value, e_da, rtol=1e-12, atol=1e-15)
           and np.allclose(c_bt.terms["state_jacobian_norm"], nA, rtol=1e-12)
           and np.allclose(c_da.terms["action_jacobian_norm"], nB, rtol=1e-12)
           and verify_structure(c_bt, s_bt.candidates) == [] and audit_sources(c_bt, s_bt) == []
           and verify_structure(c_da, s_da.candidates) == [] and audit_sources(c_da, s_da) == [] and moved_ok,
           backtrack_moved_action={"row": 0, "action_change": 0.25, "classification": str(c_moved.classification[0]),
                                   "admissible": bool(c_moved.admissible[0]), "generator": moved.generator["name"]},
           definitions={"backtrack_paper_proxy": "||J_S f_hat|| ||z_G - z_t|| (Def. 4.5)",
                        "disturbed_action_paper_proxy": "||J_A f_hat|| ||a_G - a_t|| + (1 + ||J_S f_hat||) ||z_G - z_t|| (Def. 4.6)"},
           state_jacobian_norm=nA, action_jacobian_norm=nB,
           max_relative_difference={"backtrack": float(np.max(np.abs(c_bt.total_error_value / e_bt - 1))),
                                    "disturbed_action": float(np.max(np.abs(c_da.total_error_value / e_da - 1)))})

    # P2. legacy_label_filtering equals the branch's label_error: ||J_{s + f}(s_t, a_t)|| * ||(s_G, a_G) - (s_t, a_t)||.
    syn = synthetic(sys_, SEED)
    syn_path = out / "synthetic-proxy.npz"
    write_trajectories(syn, syn_path)
    zs = np.concatenate([ep.observations["state"][:-1] for ep in syn.episodes]).astype(np.float64)
    as_ = np.concatenate([ep.actions[:-1] for ep in syn.episodes]).astype(np.float64)
    zn = np.concatenate([ep.observations["state"][1:] for ep in syn.episodes]).astype(np.float64)
    mlp = MLPTransitionModel([32, 32], epochs=40, batch_size=64, device="cpu", deterministic=True)
    mlp.fit(zs, as_, zn, seed=SEED)
    mlp_path = out / "model-proxy-mlp.npz"
    mlp.save(mlp_path)

    def branch_label_error(model_next, s_new, a_new, s_orig, a_orig):
        """Transcribed from label-filtering gen_aug_label.label_error and WorldModel.local_lipschitz_coeff."""
        d = s_orig.shape[1]
        sa = torch.as_tensor(np.concatenate([s_orig, a_orig], -1), dtype=torch.float32)
        jacs = torch.func.vmap(torch.func.jacrev(lambda x: model_next(x[..., :d], x[..., d:])))(sa)
        local_L = torch.linalg.norm(jacs, ord=2, dim=(-2, -1)).detach().numpy().astype(np.float64)
        dist = np.linalg.norm(np.concatenate([s_orig, a_orig], -1) - np.concatenate([s_new, a_new], -1), axis=-1)
        return local_L * dist

    parity = {}
    for name, (model, path, data, dpath, next_fn) in {
            "linear": (lin, lin_path, dataset, data_path,
                       lambda s_, a_: s_ + s_ @ torch.as_tensor(A_hat.T, dtype=torch.float32) + a_ @ torch.as_tensor(B_hat.T, dtype=torch.float32)),
            "mlp": (mlp, mlp_path, syn, syn_path, None)}.items():
        src = sources(model, path, NoisyAction(K, SIGMA), data=data, dpath=dpath, tag=f"-{name}")
        cert = certify(src, LegacyLabelFiltering(), admission="score")
        c = read_candidates(src.candidates)
        zz = np.concatenate([ep.observations["state"][:-1] for ep in data.episodes]).astype(np.float64)
        aa = np.concatenate([ep.actions[:-1] for ep in data.episodes]).astype(np.float64)
        idx = np.repeat(np.arange(len(zz)), K)
        if next_fn is None:
            mlp.net.eval()
            next_fn = lambda s_, a_: s_ + mlp.net.predict(s_, a_)  # noqa: E731  (WorldModel.predict: s + f)
        branch = branch_label_error(next_fn, c.observations["state"].astype(np.float64), c.actions.astype(np.float64),
                                    zz[idx], aa[idx])
        rel = np.abs(cert.total_error_value / branch - 1)
        parity[name] = {"rows": int(len(branch)), "max_relative_difference_vs_branch_float32": float(rel.max()),
                        "structure": verify_structure(cert, src.candidates), "audit": audit_sources(cert, src)[:2],
                        "ok": bool(rel.max() < 1e-5)}
        if name == "linear":  # and the exact float64 closed form ||[I + A_hat, B_hat]|| * ||(dz, da)||
            exact = float(np.linalg.norm(np.hstack([np.eye(2) + A_hat, B_hat]), 2)) * np.hypot(
                np.linalg.norm(c.observations["state"].astype(np.float64) - zz[idx], axis=1),
                np.linalg.norm(c.actions.astype(np.float64) - aa[idx], axis=1))
            parity[name]["max_relative_difference_vs_closed_form"] = float(np.max(np.abs(cert.total_error_value / exact - 1)))
            parity[name]["ok"] = parity[name]["ok"] and parity[name]["max_relative_difference_vs_closed_form"] < 1e-12
        parity[name]["ok"] = parity[name]["ok"] and not parity[name]["structure"] and not parity[name]["audit"]
        if name == "mlp":
            bundle["legacy_label_filtering_mlp"] = (cert, src)
    record("legacy_label_filtering_matches_the_branch", all(v["ok"] for v in parity.values()),
           branch="label-filtering 73d187d gen_aug_label.label_error", parity=parity,
           note="the branch transcription evaluates in float32, as the branch did (tolerance 1e-5); the linear case is "
                "also compared with its float64 closed form (1e-12)")

    # P3. Scores are never bounds.
    ub = certify(s_da, DisturbedActionPaperProxy())
    limit = float(np.quantile(c_da.total_error_value, 0.6))
    sc = certify(s_da, DisturbedActionPaperProxy(), admission="score", threshold=limit)
    record("scores_are_never_bounds",
           all((x.classification == "empirical").all() and not x.is_upper_bound.any() for x in (c_bt, c_da, ub, sc))
           and not ub.admissible.any() and ub.reasons[:, 1].all()
           and np.array_equal(sc.admissible, sc.total_error_value <= limit)
           and outcome(lambda: require_upper_bound(ub)).startswith("InterfaceError")
           and outcome(lambda: require_upper_bound(sc)).startswith("InterfaceError"),
           upper_bound_admission={"admissible": int(ub.admissible.sum()), "not_upper_bound": int(ub.reasons[:, 1].sum())},
           score_admission={"threshold": limit, "admissible": int(sc.admissible.sum()), "rows": int(len(sc.admissible))},
           require_upper_bound=outcome(lambda: require_upper_bound(sc)))

    # P4. Selection never changes acceptance; misfit anchors are rejected at generation and never selected.
    wrong = TrajectoryDataset([Episode(ep.episode_id, {"state": ep.observations["state"].copy()}, ep.actions.copy())
                               for ep in dataset.episodes])
    for ep in wrong.episodes:  # perturb the stored state at frame 3, so the model misfits transitions 2 and 3
        if ep.length > 3:
            ep.observations["state"][3] += np.float32(1.0)
    wrong.validate()
    wpath = out / "fixture-misfit.npz"
    write_trajectories(wrong, wpath)
    gate = 0.5
    sm = sources(lin, lin_path, NoisyAction(K, SIGMA), data=wrong, dpath=wpath,
                 validators=(Finite(), GenerationFailed(), ModelMisfit(gate)), tag="-misfit")
    before = open(sm.candidates, "rb").read()
    cm = read_candidates(sm.candidates)
    sel = certify(sm, LegacyLabelFiltering(), admission="score", selection_quantile=0.3)
    misfit = ~cm.accepted
    eligible = cm.accepted & sel.admissible
    q_eligible, q_all = float(np.quantile(sel.total_error_value[eligible], 0.3)), float(np.quantile(sel.total_error_value, 0.3))
    record("selection_never_changes_acceptance",
           open(sm.candidates, "rb").read() == before and np.array_equal(read_candidates(sm.candidates).accepted, cm.accepted)
           and misfit.any() and not sel.selected[misfit].any() and sel.selection["threshold"] == q_eligible
           and q_eligible != q_all and np.array_equal(sel.selected, eligible & (sel.total_error_value <= q_eligible)),
           rows=int(len(cm.accepted)), rejected_by_model_gate=int(misfit.sum()), selected=int(sel.selected.sum()),
           threshold_over_accepted_admissible=q_eligible, threshold_if_rejected_rows_were_included=q_all)

    # P4b. Both legacy keys at once: the branch's selection (utils.py at 73d187d) takes the quantile over every generated
    # label, then intersects with max_label_err and epsilon. selection_population produced reproduces it.
    def branch_selection(err, produced_mask, q, max_err, eps_ok):
        sel = np.ones(len(err), bool)  # transcribed: sel_bounded over the labels the branch generated
        thresh = np.quantile(err[produced_mask], q)
        sel &= err <= thresh
        sel &= err <= max_err
        sel &= eps_ok
        return produced_mask & sel, float(thresh)

    both = {}
    for name, (src, eps, gate_on) in {"fixture": (s_da, None, False), "misfit_and_support": (None, 0.06, True)}.items():
        if src is None:
            src = sources(lin, lin_path, NoisyAction(K, SIGMA), data=wrong, dpath=wpath,
                          validators=(Finite(), GenerationFailed(), ModelMisfit(gate), SupportRadius(eps)), tag="-both")
        c = read_candidates(src.candidates)
        probe = certify(src, LegacyLabelFiltering(), admission="score")
        meas = c.measurements
        produced_mask = np.isfinite(probe.total_error_value)
        if gate_on:  # the branch generated nothing for anchors its model misfit
            produced_mask &= meas["anchor_model_error"] < gate
        max_err = float(np.median(probe.total_error_value[produced_mask]))
        data_ = dataset if gate_on is False else wrong
        anchors = {ep.episode_id: ep.observations["state"].astype(np.float64) for ep in data_.episodes}
        z_t = np.array([anchors[str(e)][int(f)] for e, f in zip(c.source_episode_id, c.source_frame)])
        dist = np.linalg.norm(c.observations["state"].astype(np.float64) - z_t, axis=1)  # the branch's aug.epsilon test
        eps_ok = np.ones(len(c.accepted), bool) if eps is None else dist < eps
        legacy, legacy_thr = branch_selection(probe.total_error_value, produced_mask, 0.3, max_err, eps_ok)
        cert = certify(src, LegacyLabelFiltering(), admission="score", threshold=max_err, selection_quantile=0.3,
                       selection_population="produced")
        admitted = certify(src, LegacyLabelFiltering(), admission="score", threshold=max_err, selection_quantile=0.3)
        usable = c.accepted & cert.admissible & cert.selected
        usable_admitted = c.accepted & admitted.admissible & admitted.selected
        both[name] = {"rows": int(len(c.accepted)), "max_label_err": max_err,
                      "branch": {"selected": int(legacy.sum()), "threshold": legacy_thr},
                      "produced": {"selected": int(usable.sum()), "threshold": cert.selection["threshold"]},
                      "admitted": {"selected": int(usable_admitted.sum()), "threshold": admitted.selection["threshold"]},
                      "ok": bool(np.array_equal(usable, legacy) and np.isclose(cert.selection["threshold"], legacy_thr, rtol=0, atol=0)
                                 and verify_structure(cert, src.candidates) == [])}
    record("legacy_selection_with_both_keys_matches_the_branch", all(v["ok"] for v in both.values()), cases=both,
           source="label-filtering 73d187d correct_il/utils.py lines 89-99")

    # P5. Without exact Jacobians the score is unresolved: unsupported and NaN, never 0.
    transition_module.TRANSITION_MODELS[Float32Quadratic.kind] = Float32Quadratic  # evidence-only kind, this process
    quad_path = out / "model-float32-quadratic.npz"
    Float32Quadratic().save(quad_path)
    one = TrajectoryDataset([Episode("q", {"state": np.full((3, 1), 1e4, np.float32)}, np.zeros((3, 1), np.float32))])
    qpath = out / "fixture-quadratic.npz"
    write_trajectories(one, qpath)
    fd = float(jacobian_fd(Float32Quadratic(), np.full((1, 1), 1e4), np.zeros((1, 1))).ravel()[0])
    sq = sources(Float32Quadratic(), quad_path, BackwardEuler(), data=one, dpath=qpath, tag="-quadratic")
    cq = certify(sq, BackTrackPaperProxy(), admission="score")
    record("unresolved_derivatives_are_unsupported",
           fd == 0.0 and (cq.classification == "unsupported").all() and np.isnan(cq.terms["state_jacobian_norm"]).all()
           and np.isnan(cq.total_error_value).all() and cq.reasons[:, 0].all() and not cq.admissible.any(),
           float32_finite_difference_at_1e4=fd, true_derivative=2e4 - 1,
           classification=sorted(set(cq.classification.tolist())), state_jacobian_norm=cq.terms["state_jacobian_norm"].tolist())
    del transition_module.TRANSITION_MODELS[Float32Quadratic.kind]

    # P6. Migration and configuration.
    from ccil.config import resolve
    legacy_yml = out / "legacy_filtering.yml"
    legacy_yml.write_text((repo / "config/pendulum_cont.yml").read_text().replace(
        "  epsilon:", "  label_err_quantile: 0.3\n  max_label_err: 0.5\n  model_err_thresh: 0.1\n  epsilon:"))
    rc = resolve(config=legacy_yml)
    migrated = {k: rc[k] for k in ("certificate.profile", "certificate.score_threshold", "certificate.selection_quantile",
                                   "certificate.selection_population", "generation.model_error_threshold")}
    base = ["data.path=x.npz"]
    refused = {
        "withdrawn_max_label_error": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + ["generation.max_label_error=0.5"])),
        "withdrawn_label_error_quantile": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + ["generation.label_error_quantile=0.3"])),
        "score_profile_with_bound_threshold": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "certificate.profile=legacy_label_filtering", "certificate.max_total_error=1"])),
        "bound_profile_with_score_threshold": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "generation.method=noisy_action", "certificate.profile=disturbed_action", "certificate.true_state_lipschitz=0.1",
            "certificate.true_action_lipschitz=0.1", "certificate.demonstration_residual_bound=0", "certificate.score_threshold=1"])),
        "profile_for_the_other_family": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "generation.method=noisy_action", "certificate.profile=backtrack_paper_proxy"])),
        "bound_profile_without_demonstration": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "generation.method=backward_euler", "certificate.profile=backtrack", "certificate.true_state_lipschitz=0.1"])),
        "certificate_settings_without_profile": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "certificate.selection_quantile=0.3"])),
        "declared_learned_source_without_profile": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "certificate.learned_lipschitz_source=declared"])),
        "declared_learned_source_with_backtrack_paper_proxy": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "generation.method=backward_euler", "certificate.profile=backtrack_paper_proxy",
            "certificate.learned_lipschitz_source=declared"])),
        "declared_learned_source_with_disturbed_action": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "generation.method=noisy_action", "certificate.profile=disturbed_action", "certificate.true_state_lipschitz=0.1",
            "certificate.true_action_lipschitz=0.1", "certificate.demonstration_residual_bound=0",
            "certificate.learned_lipschitz_source=declared"])),
        "action_neighborhood_with_backtrack_paper_proxy": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "generation.method=backward_euler", "certificate.profile=backtrack_paper_proxy", "certificate.neighborhood_action=0.1"])),
        "produced_population_without_quantile": outcome(lambda: resolve(recipe="analytic_double_integrator", sets=base + [
            "certificate.profile=legacy_label_filtering", "certificate.selection_population=produced"])),
    }
    from ccil.config import ConfigError
    from ccil.pipelines.integrated import run_integrated
    try:
        run_integrated(resolve(recipe="analytic_double_integrator", sets=base + [
            "representation.kind=pca", "representation.dim=2", "certificate.profile=legacy_label_filtering"]))
        integrated = "accepted"
    except ConfigError as e:
        integrated = f"ConfigError: {e.errors[0][:200]}"
    record("configuration_and_migration",
           migrated == {"certificate.profile": "legacy_label_filtering", "certificate.score_threshold": 0.5,
                        "certificate.selection_quantile": 0.3, "certificate.selection_population": "produced",
                        "generation.model_error_threshold": 0.1}
           and all(v.startswith("ConfigError") or v.startswith("ValueError") for v in refused.values())
           and "withdrawn" in refused["withdrawn_max_label_error"] and integrated.startswith("ConfigError"),
           migrated=migrated, refused=refused, integrated=integrated)

    # P7. The CLI with certificate settings, and the pre-#88 candidates reader.
    runs = {}
    work = out / "pipeline-cli"
    work.mkdir()
    write_trajectories(dataset, work / "fixture.npz")
    (work / "true-dynamics.npz").write_bytes(open(truth, "rb").read())
    for name, extra in {"legacy_label_filtering": ["generation.method=noisy_action", f"generation.labels_per_transition={K}",
                                                   f"generation.action_noise_std={SIGMA}", "generation.model_error_threshold=0.5",
                                                   "certificate.profile=legacy_label_filtering", "certificate.selection_quantile=0.3"],
                        "backtrack": ["generation.method=backward_euler", "certificate.profile=backtrack",
                                      "certificate.reference_dynamics=true-dynamics.npz"]}.items():
        args = [sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
                "--set", "data.path=fixture.npz", "--set", "run.output_dir=runs"]
        for e in extra:
            args += ["--set", e]
        r = subprocess.run(args, cwd=work, capture_output=True, text=True)
        lines = [ln.split() for ln in r.stdout.splitlines() if ln.strip()]
        stages = {ln[0]: ln[-1] for ln in lines}
        runs[name] = {"exit": r.returncode, "stages": sorted(stages), "stderr": r.stderr[-300:]}
        if "labels.certify" in stages:
            from ccil.runs import validate_run
            from ccil.schema import read_certificates
            cdir, ldir = work / stages["labels.certify"], work / stages["labels.generate"]
            cert = read_certificates(cdir / "labels/certificates.npz")
            cand = read_candidates(ldir / "labels/candidates.npz")
            runs[name].update(status=validate_run(cdir)[0], classification=sorted(set(cert.classification.tolist())),
                              usable=int((cand.accepted & cert.admissible & cert.selected).sum()), rows=int(len(cand.accepted)),
                              candidates=str(ldir / "labels/candidates.npz"))
    old_root = out / f"reader_{PRE_88}"
    old_root.mkdir()
    tarball = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", PRE_88, "src"], capture_output=True, check=True).stdout
    (old_root / "src.tar").write_bytes(tarball)
    subprocess.run(["tar", "-xf", "src.tar"], cwd=old_root, check=True)
    probe = "import sys; from ccil.schema import read_candidates; c = read_candidates(sys.argv[1]); print(len(c.accepted))"
    for name, v in runs.items():
        if "candidates" in v:
            r = subprocess.run([sys.executable, "-c", probe, v.pop("candidates")], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(old_root / "src")})
            v["read_by_pre_88_reader"] = r.returncode == 0 and r.stdout.strip() == str(v["rows"])
    record("cli_certify_runs_and_pre_88_reader",
           all(v["exit"] == 0 and "labels.certify" in v["stages"] and v.get("status") == "complete"
               and v.get("read_by_pre_88_reader") for v in runs.values())
           and runs["legacy_label_filtering"]["classification"] == ["empirical"]
           and runs["backtrack"]["classification"] == ["certified"],
           runs=runs, pre_88_reader=PRE_88)
