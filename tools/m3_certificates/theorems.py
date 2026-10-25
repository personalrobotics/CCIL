"""Evidence for #88 part 2: the CCIL theorem profiles and model bounds (called by check.py).

On the analytic double integrator, with the true label error computed
independently from the true dynamics, and every bound including the
demonstration residual from the reference dynamics, so coverage is asserted
directly (true error <= total, no tolerance):
- backtrack (Thm 4.2) with K_hat_z from model_lipschitz and K_z from
  reference_lipschitz, both verified exactly for linear maps, is certified and
  covers every row of both backtrack generators; its bound has no action term,
  so a model with 100x the action sensitivity gets the same formula, and rows
  that move the action are unsupported;
- disturbed_action (a direct corollary of the proof of Thm 4.3) uses the true
  constants only: it is certified and covers every row of both
  disturbed-action generators, records solver_error without adding it
  (compared with the double-counted total), and declared zero constants fail
  coverage;
- MLP lipschitz_bound: the bound of relu, tanh, and spectral-normalized MLPs
  (also with a masked output and non-unit scales) covers every sampled exact
  Jacobian and finite difference; for a model whose actions are in
  milli-units, the raw weight product without the normalization constants
  falls short of the sampled action sensitivity; a spectral-normalized layer
  whose stored power-iteration estimate is stale exceeds its constraint, so
  the constraint is not a bound, while the exact-norm bound covers;
  unsupported activations are refused;
- with an MLP model, backtrack is conditional (its MLP bound and model terms
  hold in exact real arithmetic only) and disturbed_action, which uses no
  learned quantity, is certified; both cover every row and pass structural
  verification and the source audit;
- providers of the wrong kind, forged provider kinds, a missing demonstration
  specification, and profiles applied to the other generator family are
  refused; the pipeline writes complete backtrack and disturbed_action
  certify runs.
"""
import copy

import numpy as np

from ccil.core import (BackwardEuler, ExactPreimage, Finite, IdentityRepresentation, InterfaceError,
                       LinearTransitionModel, NoisyAction, NoisyActionPreimage, generate_labels)
from ccil.core.certificates import (PROVIDERS, BackTrackProfile, ContinuityProfile, Context, Declared,
                                    DisturbedActionProfile, Provided, Sources, audit_sources, certify, decide,
                                    resolved_of, verify_structure)
from ccil.schema import (Episode, SchemaError, TrajectoryDataset, read_candidates, read_certificates, write_candidates,
                         write_trajectories)

SEED, K, SIGMA = 20261025, 4, 0.05


def outcome(fn):
    try:
        fn()
        return "accepted"
    except (InterfaceError, SchemaError) as e:
        return f"{type(e).__name__}: {str(e)[:160]}"


def demo_residual(c, dataset, sys_, scale=None):
    """Per candidate row, ||(z_{t+1} - z_t - f(z_t, a_t)) / scale|| of its stored anchor, from the true f."""
    eps = {ep.episode_id: ep for ep in dataset.episodes}
    out = np.empty(len(c.source_frame))
    for i, (e, t) in enumerate(zip(c.source_episode_id, c.source_frame)):
        ep, t = eps[str(e)], int(t)
        z = ep.observations["state"].astype(np.float64)
        r = z[t + 1] - sys_.step_np(z[t:t + 1], ep.actions[t:t + 1].astype(np.float64))[0]
        out[i] = np.linalg.norm(r / (1.0 if scale is None else scale))
    return out


def true_label_error(c, sys_, scale=None):
    z, a = c.observations["state"].astype(np.float64), c.actions.astype(np.float64)
    e = sys_.step_np(z, a) - c.target_observations["state"].astype(np.float64)
    return np.linalg.norm(e / (1.0 if scale is None else scale), axis=1)


def coverage(cert, c, dataset, sys_, scale=None):
    """true_error <= total_error_value on every row, asserted directly (no tolerance)."""
    err = true_label_error(c, sys_, scale)
    r = demo_residual(c, dataset, sys_, scale)
    return {"rows": int(len(err)), "covered": bool((err <= cert.total_error_value).all()),
            "violated_rows": int((err > cert.total_error_value).sum()), "min_slack": float((cert.total_error_value - err).min()),
            "max_demonstration_residual": float(r.max()),
            "max_persisted_demonstration_residual": float(np.nanmax(cert.terms["demonstration_residual"])),
            "max_true_error": float(err.max()), "max_bound": float(cert.total_error_value.max())}


def synthetic(sys_, seed, episodes=30, length=25):
    """Rollouts of the true double integrator from random states and actions, stored as float32."""
    rng = np.random.default_rng(seed)
    eps = []
    for k in range(episodes):
        z = [rng.uniform(-1, 1, 2)]
        a = rng.uniform(-1, 1, (length, 1))
        for t in range(length - 1):
            z.append(sys_.step_np(z[-1][None], a[t:t + 1])[0])
        eps.append(Episode(f"ep{k:02d}", {"state": np.array(z, np.float32)}, a.astype(np.float32)))
    return TrajectoryDataset(eps).validate()


def scaled_jacobians(model, z, a, sz, sa):
    J = model.jacobian_za(z, a)
    d = z.shape[1]
    Jz = J[:, :, :d] * sz[None, None, :] / sz[None, :, None]
    Ja = J[:, :, d:] * sa[None, None, :] / sz[None, :, None]
    return np.linalg.norm(Jz, 2, axis=(1, 2)), np.linalg.norm(Ja, 2, axis=(1, 2))


def run(out, record, dataset, data_path, sys_, truth, bundle):
    import torch
    from ccil.models.mlp import MLPTransitionModel, _effective_weight

    rep = IdentityRepresentation("state")
    A_hat, B_hat = sys_.A + np.array([[0.05, 0.0], [-0.08, 0.04]]), sys_.B * 1.6
    models = {"wrong": LinearTransitionModel(A_hat, B_hat), "wrong_action_x100": LinearTransitionModel(A_hat, B_hat * 100)}
    for name, m in models.items():
        m.save(out / f"model-{name}.npz")
    generators = {"backward_euler": BackwardEuler(), "exact_preimage": ExactPreimage(max_iterations=50),
                  "noisy_action": NoisyAction(K, SIGMA), "noisy_action_preimage": NoisyActionPreimage(K, SIGMA, max_iterations=50)}

    def sources(model, gen, data=dataset, dpath=data_path, model_obj=None, model_path=None):
        m = model_obj if model_obj is not None else models[model]
        c = generate_labels(data, rep, m, generators[gen], [Finite()], run_id=f"{model}-{gen}", seed=SEED)
        path = out / f"candidates-{model}-{gen}.npz"
        write_candidates(c, path)
        return Sources(path, dpath, model_path or out / f"model-{model}.npz", rep, references={"true": truth})

    demo = Provided("reference_residual", "true")
    backtrack = BackTrackProfile(Provided("model_lipschitz"), Provided("reference_lipschitz", "true"), demo)
    disturbed = DisturbedActionProfile(Provided("reference_lipschitz", "true"), Provided("reference_lipschitz", "true"), demo)
    continuity = ContinuityProfile(Provided("linear_discrepancy", "true"), Provided("linear_discrepancy", "true"), demo)

    # T1. BackTrack: certified and covered for both backtrack generators.
    bt = {}
    for g in ("backward_euler", "exact_preimage"):
        s = sources("wrong", g)
        c = read_candidates(s.candidates)
        cert = certify(s, backtrack)
        cont = certify(s, continuity)
        cov = coverage(cert, c, dataset, sys_)
        bt[g] = (s, cert)
        consts = {r: v["value"] for r, v in cert.descriptor["constants"].items()}
        record(f"backtrack_certified_and_covered[{g}]",
               (cert.classification == "certified").all() and cov["covered"]
               and consts["K_hat_z"] >= np.linalg.norm(A_hat, 2) and np.isclose(consts["K_hat_z"], np.linalg.norm(A_hat, 2), rtol=1e-9)
               and consts["K_z"] >= np.linalg.norm(sys_.A, 2) and np.isclose(consts["K_z"], np.linalg.norm(sys_.A, 2), rtol=1e-9)
               and all(v["source"] == "analytic_bound" for v in cert.descriptor["constants"].values())
               and (cert.total_error_value >= cont.total_error_value - 1e-15).all(),
               constants=consts, coverage=cov,
               continuity_comparison={"note": "K_hat_z + K_z >= ||A - A_hat||, so backtrack is never tighter than continuity",
                                      "max_backtrack_over_continuity": float(np.max(cert.total_error_value / cont.total_error_value))})

    # T2. BackTrack has no action term.
    s1, c1 = bt["backward_euler"]
    s2 = sources("wrong_action_x100", "backward_euler")
    c2 = certify(s2, backtrack)
    cand2 = read_candidates(s2.candidates)
    action_bounds = {name: m.lipschitz_bound("action", np.ones(2), np.ones(1))["value"] for name, m in models.items()}
    formula = all(np.allclose(c.model_error_value, c.terms["anchor_model_error"] + c.terms["demonstration_residual"]
                              + sum(v["value"] for v in c.descriptor["constants"].values()) * c.terms["state_displacement"],
                              rtol=1e-14, atol=0) for c in (c1, c2))
    moved = {k: v.copy() for k, v in c1.terms.items()}
    moved["action_displacement"][:3] = 1e-3
    rows, _ = decide(backtrack, resolved_of(c1.descriptor), moved, np.ones(len(c1.selected), bool),
                     {"mode": "upper_bound", "threshold": None}, None)
    record("backtrack_has_no_action_term",
           formula and c1.descriptor["constants"]["K_hat_z"]["value"] == c2.descriptor["constants"]["K_hat_z"]["value"]
           and action_bounds["wrong_action_x100"] > 50 * action_bounds["wrong"]
           and coverage(c2, cand2, dataset, sys_)["covered"] and (c2.classification == "certified").all()
           and (rows["classification"][:3] == "unsupported").all() and (rows["classification"][3:] == "certified").all()
           and set(c1.descriptor["constants"]) == {"K_hat_z", "K_z"},
           K_hat_z={"wrong": c1.descriptor["constants"]["K_hat_z"]["value"], "wrong_action_x100": c2.descriptor["constants"]["K_hat_z"]["value"]},
           model_action_lipschitz=action_bounds, coverage_action_x100=coverage(c2, cand2, dataset, sys_),
           rows_with_moved_action_unsupported=int((rows["classification"] == "unsupported").sum()))

    # T3. DisturbedAction: direct form, true constants only, solver error never added.
    for g in ("noisy_action", "noisy_action_preimage"):
        s = sources("wrong", g)
        c = read_candidates(s.candidates)
        cert = certify(s, disturbed)
        cov = coverage(cert, c, dataset, sys_)
        double = cert.total_error_value + cert.terms["solver_error"]
        kv = {r: v["value"] for r, v in cert.descriptor["constants"].items()}
        direct = (kv["K_a"] * cert.terms["action_displacement"] + (1 + kv["K_z"]) * cert.terms["state_displacement"]
                  + cert.terms["demonstration_residual"] + cert.terms["target_offset"])
        declared = certify(s, DisturbedActionProfile(Declared(0.0, "assumed (wrongly) f is constant"),
                                                      Declared(0.0, "assumed (wrongly) f is constant"), demo))
        dcov = coverage(declared, c, dataset, sys_)
        record(f"disturbed_action_direct_and_covered[{g}]",
               (cert.classification == "certified").all() and cov["covered"] and np.isnan(cert.model_error_value).all()
               and float(cert.terms["target_offset"].max()) == 0.0 and (cert.terms["solver_error"] > 0).any()
               and np.allclose(cert.total_error_value, direct, rtol=1e-14, atol=0) and (cert.total_error_value >= direct).all()
               and (declared.classification == "conditional").all() and not dcov["covered"],
               constants={r: v["value"] for r, v in cert.descriptor["constants"].items()}, coverage=cov,
               solver_error={"recorded_max": float(cert.terms["solver_error"].max()),
                             "rows_positive": int((cert.terms["solver_error"] > 0).sum()),
                             "double_counted_over_direct_max": float(np.max(double / cert.total_error_value)),
                             "double_counted_over_direct_mean": float(np.mean(double / cert.total_error_value))},
               max_target_offset=float(cert.terms["target_offset"].max()),
               declared_zero={"classification": "conditional", "violated_rows": dcov["violated_rows"], "rows": dcov["rows"]})

    # T4-T5. MLP model_bound.
    syn = synthetic(sys_, SEED)
    syn_path = out / "synthetic.npz"
    write_trajectories(syn, syn_path)
    z = np.concatenate([ep.observations["state"][:-1] for ep in syn.episodes]).astype(np.float64)
    a = np.concatenate([ep.actions[:-1] for ep in syn.episodes]).astype(np.float64)
    zn = np.concatenate([ep.observations["state"][1:] for ep in syn.episodes]).astype(np.float64)
    rng = np.random.default_rng(SEED + 1)
    pts = rng.integers(0, len(z), 2000)
    zs = z[pts] + rng.normal(0, 0.2, (len(pts), 2))
    as_ = a[pts] + rng.normal(0, 0.2, (len(pts), 1))
    trained = {}
    for name, kw in {"relu": {"activation": "relu"}, "tanh": {"activation": "tanh"},
                     "spectral_relu": {"activation": "relu", "lipschitz_type": "spectral_normalization", "lipschitz_constraint": 0.5}}.items():
        m = MLPTransitionModel([32, 32], epochs=40, batch_size=64, device="cpu", deterministic=True, **kw)
        m.fit(z, a, zn, seed=SEED)
        trained[name] = m
    # The same system with actions in milli-units: the input scale of a is ~1/1000, so the action sensitivity
    # comes from the normalization, and the raw weights alone cannot bound it.
    milli = MLPTransitionModel([32, 32], epochs=40, batch_size=64, device="cpu", deterministic=True, activation="relu")
    milli.fit(z, a / 1000.0, zn, seed=SEED)
    masked = copy.deepcopy(trained["relu"])
    with torch.no_grad():
        masked.net.out_scale[0] = 0.0
    trained["relu_masked_output"] = masked
    trained["relu_milli_action_units"] = milli
    mlp_cases = {}
    for name, m in trained.items():
        a_in = as_ / 1000.0 if name == "relu_milli_action_units" else as_
        for scale in ((np.ones(2), np.ones(1)), (np.array([2.0, 0.5]), np.array([4.0]))):
            sz, sa = scale
            Kz, Ka = (m.lipschitz_bound(mp, sz, sa) for mp in ("state", "action"))
            jz, ja = scaled_jacobians(m, zs, a_in, sz, sa)
            # Finite differences along state and along action, in scaled coordinates.
            dz = rng.normal(0, 0.05, zs.shape)
            da = rng.normal(0, 0.05, as_.shape) * (a_in.std() / as_.std())
            base = m.delta(zs, a_in)
            fz = np.linalg.norm((m.delta(zs + dz, a_in) - base) / sz, axis=1) / np.linalg.norm(dz / sz, axis=1)
            fa = np.linalg.norm((m.delta(zs, a_in + da) - base) / sz, axis=1) / np.linalg.norm(da / sa, axis=1)
            weights = [_effective_weight(layer).detach().double().numpy() for layer in m.net.fc_layers]
            naive = float(np.prod([np.linalg.norm(w, 2) for w in weights]))  # no normalization or output scales
            tol = 1e-5  # float32 evaluation of the network
            mlp_cases[f"{name}, scale={sz.tolist()}"] = {
                "K_z": Kz["value"], "K_a": Ka["value"], "factors_z": Kz["factors"],
                "max_sampled_jacobian_z": float(jz.max()), "max_sampled_jacobian_a": float(ja.max()),
                "max_finite_difference_z": float(fz.max()), "max_finite_difference_a": float(fa.max()),
                "naive_weight_product": naive,
                "covered": bool(jz.max() <= Kz["value"] * (1 + tol) and ja.max() <= Ka["value"] * (1 + tol)
                                and fz.max() <= Kz["value"] * (1 + tol) and fa.max() <= Ka["value"] * (1 + tol)),
                "naive_violated": bool(max(jz.max(), ja.max()) > naive)}
    gelu = copy.deepcopy(trained["relu"])
    gelu.settings["activation"] = "gelu"
    unsupported = outcome(lambda: PROVIDERS["model_lipschitz"].resolve(
        "state", None, Context(None, None, rep, gelu, np.ones(2), np.ones(1), {}, {"transition_model": "0" * 64})))
    record("mlp_model_bound_covers_sampled_jacobians",
           all(v["covered"] for v in mlp_cases.values())
           and mlp_cases["relu_milli_action_units, scale=[1.0, 1.0]"]["naive_violated"]
           and mlp_cases["relu_masked_output, scale=[1.0, 1.0]"]["K_z"] <= mlp_cases["relu, scale=[1.0, 1.0]"]["K_z"]
           and unsupported.startswith("InterfaceError"),
           cases=mlp_cases, sampled_points=int(len(pts)), unsupported_activation=unsupported)

    # T5. Stale power-iteration estimates: the constraint is not a bound, the exact norm is.
    sn = copy.deepcopy(trained["spectral_relu"])
    gen = torch.Generator().manual_seed(SEED)
    stale = []
    with torch.no_grad():
        for layer in sn.net.fc_layers:
            for buf in ("weight_u", "weight_v"):
                v = getattr(layer, buf)
                v.copy_(torch.nn.functional.normalize(torch.randn(v.shape, generator=gen, dtype=v.dtype), dim=0))
            W = layer.weight_orig.reshape(layer.weight_orig.size(0), -1)
            sigma_hat = float(torch.dot(layer.weight_u, torch.mv(W, layer.weight_v)))
            stale.append({"sigma_hat": sigma_hat, "exact_effective_norm": float(np.linalg.norm(_effective_weight(layer).double().numpy(), 2)),
                          "constraint": 0.5})
    sz, sa = np.ones(2), np.ones(1)
    Kz = sn.lipschitz_bound("state", sz, sa)
    jz, _ = scaled_jacobians(sn, zs, as_, sz, sa)
    with torch.no_grad():
        in_div = (sn.net.s_scale + 1e-8).double().numpy()
        out_mul = ((sn.net.out_scale + 1e-8) * (sn.net.out_scale >= 1e-8)).double().numpy()
    constraint_bound = float(0.5 ** len(sn.net.fc_layers) * out_mul.max() / in_div.min())
    record("spectral_norm_estimates_not_used",
           any(s_["exact_effective_norm"] > s_["constraint"] * 1.01 for s_ in stale) and jz.max() > constraint_bound
           and jz.max() <= Kz["value"] * (1 + 1e-5),
           layers=stale, constraint_times_normalization=constraint_bound, exact_bound=Kz["value"],
           max_sampled_jacobian_z=float(jz.max()))

    # T6. With an MLP model: backtrack uses f_hat's bound and terms, so it is conditional on exact real arithmetic;
    # disturbed_action uses no learned quantity, so it stays certified.
    mlp = trained["relu"]
    mlp_path = out / "model-mlp.npz"
    mlp.save(mlp_path)
    s = sources("mlp", "backward_euler", data=syn, dpath=syn_path, model_obj=mlp, model_path=mlp_path)
    c = read_candidates(s.candidates)
    cert = certify(s, backtrack)
    cov = coverage(cert, c, syn, sys_)
    struct, audit = verify_structure(cert, s.candidates), audit_sources(cert, s)
    sd = sources("mlp", "noisy_action", data=syn, dpath=syn_path, model_obj=mlp, model_path=mlp_path)
    cd = read_candidates(sd.candidates)
    cert_da = certify(sd, disturbed)
    cov_da = coverage(cert_da, cd, syn, sys_)
    struct_da, audit_da = verify_structure(cert_da, sd.candidates), audit_sources(cert_da, sd)
    bundle["backtrack_mlp"] = (cert, s)
    bundle["disturbed_action_mlp"] = (cert_da, sd)
    record("mlp_models_classified_by_what_the_bound_uses",
           (cert.classification == "conditional").all() and cert.is_upper_bound.all() and cov["covered"]
           and cert.descriptor["constants"]["K_hat_z"]["source"] == "model_bound"
           and cert.descriptor["model_evaluation"] == "assumed_exact" and struct == [] and audit == []
           and np.isclose(cert.descriptor["constants"]["K_hat_z"]["value"], mlp.lipschitz_bound("state", np.ones(2), np.ones(1))["value"])
           and (cert_da.classification == "certified").all() and cov_da["covered"] and struct_da == [] and audit_da == [],
           backtrack={"classification": "conditional", "constants": {r: {k: v[k] for k in ("value", "source")}
                                                                     for r, v in cert.descriptor["constants"].items()},
                      "model_evaluation": cert.descriptor["model_evaluation"],
                      "rationale": cert.descriptor["constants"]["K_hat_z"]["rationale"], "coverage": cov,
                      "structure": struct, "audit": audit},
           disturbed_action={"classification": "certified", "coverage": cov_da, "structure": struct_da, "audit": audit_da})

    # T7. Kinds and applicability.
    s_be, cert_be = bt["backward_euler"]
    forged = copy.deepcopy(cert_be)
    forged.descriptor["constants"]["K_z"] = {**forged.descriptor["constants"]["K_hat_z"], "maps": "state"}
    refused = {
        "learned_provider_for_true_constant": outcome(lambda: certify(s_be, BackTrackProfile(Provided("model_lipschitz"), Provided("model_lipschitz"), demo))),
        "discrepancy_provider_for_learned_constant": outcome(lambda: certify(s_be, BackTrackProfile(Provided("linear_discrepancy", "true"), Provided("reference_lipschitz", "true"), demo))),
        "residual_provider_for_a_constant": outcome(lambda: certify(s_be, BackTrackProfile(Provided("reference_residual", "true"), Provided("reference_lipschitz", "true"), demo))),
        "missing_demonstration": outcome(lambda: BackTrackProfile(Provided("model_lipschitz"), Provided("reference_lipschitz", "true"), None)),
        "reference_given_to_model_provider": outcome(lambda: Provided("model_lipschitz", "true")),
        "backtrack_on_disturbed_action_labels": outcome(lambda: certify(sources("wrong", "noisy_action"), backtrack)),
        "disturbed_action_on_backtrack_labels": outcome(lambda: certify(s_be, disturbed)),
    }
    record("provider_kinds_and_applicability_enforced",
           all(v.startswith("InterfaceError") for v in refused.values())
           and any("provider" in p for p in verify_structure(forged, s_be.candidates)),
           refused=refused, forged_kind_structure=verify_structure(forged, s_be.candidates))

    # T8. Pipeline runs for both theorem profiles.
    from ccil.config import resolve
    from ccil.pipelines.offline import certify_stage, generate_offline
    from ccil.runs import validate_run
    work = out / "pipeline-theorems"
    work.mkdir()
    write_trajectories(dataset, work / "fixture.npz")
    results = {}
    for method, profile in (("backward_euler", backtrack), ("noisy_action", disturbed)):
        cfg = resolve(recipe="analytic_double_integrator",
                      sets=[f"data.path={work / 'fixture.npz'}", f"run.output_dir={work / 'runs'}", f"generation.method={method}",
                            f"generation.labels_per_transition={K}", f"generation.action_noise_std={SIGMA}"])
        prepared, fitted, labeled = generate_offline(cfg)
        res = certify_stage(cfg, prepared, fitted, labeled, rep, profile, references={"true": truth})
        pc = read_certificates(res.out("labels/certificates.npz"))
        cand = read_candidates(labeled.out("labels/candidates.npz"))
        results[profile.name] = {"run": res.run_dir.name, "status": validate_run(res.run_dir)[0],
                                 "classification": sorted(set(pc.classification.tolist())),
                                 "coverage": coverage(pc, cand, dataset, sys_)}
    record("pipeline_theorem_profiles",
           all(r["status"] == "complete" and r["classification"] == ["certified"] and r["coverage"]["covered"]
               for r in results.values()),
           runs=results)
