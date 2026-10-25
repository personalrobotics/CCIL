#!/usr/bin/env python3
"""Evidence for #22 part 1: noisy-action measurements and the NoiseBC input perturbation.

    uv run --group evidence python tools/m3_labels/check.py --out DIR

Noisy-action generation, on the analytic double integrator with its exact
linear model:
- noisy_action and noisy_action_fe match their closed forms, and every
  candidate carries action_noise_norm, state_displacement_norm (and, for the
  solved variant, solver_residual) whose values agree with the stored arrays;
- measurement_specs name every measurement, the generator's full params are in
  the file, and both survive a write-read round trip and accepted_labels (2.2);
- generation is batch-invariant, byte for byte;
- failed proposals (a model singular for some actions) are rejected as
  generation_failed and never refilled: every transition keeps exactly k rows;
- `ccil labels generate --set generation.method=noisy_action_fe` writes a
  complete run (the inspectable artifact);
- measurements equal what is computed from the arrays actually written, at
  float32 and a high dynamic range where rounding erases the perturbation;
- a solved candidate's convergence is judged on its stored values;
- a CRC32 collision pair gets distinct random streams;
- a producer with missing or mismatched measurement specs raises.
NoiseBC, on fixed observations (full masks and noise kept in noise_bc_draws.npz):
- legacy mode is biased by scale/2 in [0, scale); symmetric mode is zero-mean in
  [-scale, scale); both perturb about half of each batch;
- a draw is reproducible from (seed, step, sample, draw index), fresh at a new
  step or draw, and independent of batch order and composition;
- perturb never modifies its input and rejects inputs it could not perturb as
  configured; the trace hash is reproducible and changes with the seed.
Filtering and weighting (#22 part 2):
- the label-error estimate equals ||df/d(z, a)|| times the state-action
  distance on the analytic system (the branch's next-state Jacobian shown
  beside it); model_misfit rejects and records the candidates of misfit
  transitions; the quantile and absolute label-error filters are
  reconstructible from the file and batch-invariant; reasons stay
  independent; a weighting never changes validity; the branch's fixed-point
  forward_euler reaches Newton's root; the branch's keys migrate; every
  label-filtering capability of the inventory is classified; and the pipeline
  writes a complete 2.3 run with the filters.
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

from ccil.core import (Finite, GenerationFailed, IdentityRepresentation, InterfaceError,  # noqa: E402
                       LinearTransitionModel, ModelResidual, NoisyAction, NoisyActionPreimage, SupportRadius,
                       generate_labels)
from ccil.runs import validate_run  # noqa: E402
from ccil.schema import (REASON_CODES, Episode, TrajectoryDataset, read_candidates, write_candidates,  # noqa: E402
                         write_trajectories)
from ccil.schema.io import _atomic_savez  # noqa: E402
from ccil.training.noise_bc import NoiseBC, NoiseTrace  # noqa: E402

SEED, K, SIGMA = 20261023, 4, 0.05


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    model = LinearTransitionModel(sys_.A, sys_.B)
    rep = IdentityRepresentation("state")
    validators = [Finite(), GenerationFailed(), SupportRadius(0.01), ModelResidual()]
    s = np.concatenate([ep.observations["state"][:-1] for ep in dataset.episodes]).astype(np.float64)
    sp = np.concatenate([ep.observations["state"][1:] for ep in dataset.episodes]).astype(np.float64)
    a = np.concatenate([ep.actions[:-1] for ep in dataset.episodes]).astype(np.float64)

    # 1-2. Closed forms and measurements, both generators.
    for gen, exact in ((NoisyAction(K, SIGMA), lambda tgt, at: sys_.noisy_action(tgt, at)),
                       (NoisyActionPreimage(K, SIGMA, max_iterations=50), lambda tgt, at: sys_.exact_preimage(tgt, at))):
        c = generate_labels(dataset, rep, model, gen, validators, run_id=f"analytic-{gen.name}", seed=SEED)
        src = np.repeat(np.arange(len(s)), K)
        at = c.actions.astype(np.float64)
        zg = c.observations["state"].astype(np.float64)
        expected = exact(sp[src], at)
        names = set(c.measurements)
        need = {"generator.action_noise_norm", "generator.state_displacement_norm"} | (
            {"generator.solver_residual"} if gen.name == "noisy_action_preimage" else set())
        consistent = (np.allclose(c.measurements["generator.action_noise_norm"], np.linalg.norm(at - a[src], axis=1), atol=1e-6)
                      and np.allclose(c.measurements["generator.state_displacement_norm"], np.linalg.norm(zg - s[src], axis=1), atol=1e-6))
        if gen.name == "noisy_action_preimage":
            consistent = consistent and (c.measurements["generator.solver_residual"] <= 1e-4).all()
        noise = at - a[src]
        # The closed form is exact for noisy_action; the solved variant stops once its residual is within the solver
        # tolerance, which bounds its error by ||(I + A)^-1|| * tolerance on this linear system.
        bound = 1e-5 if gen.name == "noisy_action" else \
            float(np.linalg.norm(np.linalg.inv(np.eye(2) + sys_.A), 2)) * gen.params["tolerance"] + 1e-6
        record(f"closed_form_and_measurements[{gen.name}]",
               len(c.accepted) == K * len(s) and float(np.linalg.norm(zg - expected, axis=1).max()) <= bound
               and need <= names and consistent
               and set(c.measurement_specs) == names and c.generator["params"]["labels_per_transition"] == K,
               candidates=int(len(c.accepted)), accepted=int(c.accepted.sum()),
               max_l2_vs_closed_form=float(np.linalg.norm(zg - expected, axis=1).max()), bound=bound,
               measurements=sorted(names),
               measurement_specs=c.measurement_specs, generator=c.generator,
               action_noise={"mean": float(noise.mean()), "std": float(noise.std()), "configured_std": SIGMA})

        # 3. Round trip and accepted_labels keep specs and params; 4. batch invariance.
        path = out / f"{gen.name}.npz"
        write_candidates(c, path)
        back = read_candidates(path, dataset)
        with np.load(path, allow_pickle=False) as z:
            version = json.loads(str(z["meta"]))["version"]
        labels = back.accepted_labels()
        files = []
        for bs in (1, None):
            cb = generate_labels(dataset, rep, model, gen, validators, run_id=f"analytic-{gen.name}", seed=SEED, batch_size=bs)
            write_candidates(cb, out / f"{gen.name}-bs{bs}.npz")
            files.append(sha(out / f"{gen.name}-bs{bs}.npz"))
        record(f"specs_round_trip_and_batch_invariance[{gen.name}]",
               version == "2.2" and back.measurement_specs == c.measurement_specs and back.generator == c.generator
               and labels.measurement_specs == c.measurement_specs and files[0] == files[1],
               file_version=version, batch_sha256=files)

    # 5. Failed proposals are rejected, never refilled.
    class SingularForLargeActions:
        """Exact dynamics except delta = -z (I + df/dz singular) where the action exceeds 0.1."""

        def __init__(self, exact):
            self.exact = exact

        def delta(self, z, act):
            d = self.exact.delta(z, act)
            bad = act[:, 0] > 0.1
            d[bad] = -z[bad]
            return d

        def jacobian_z(self, z, act):
            J = np.array(self.exact.jacobian_z(z, act))
            J[act[:, 0] > 0.1] = -np.eye(z.shape[1])
            return J

        def save(self, path):
            _atomic_savez(path, {"A": self.exact.A, "B": self.exact.B, "kind": np.array("singular for a > 0.1")})

    c = generate_labels(dataset, rep, SingularForLargeActions(model), NoisyActionPreimage(K, SIGMA), validators,
                        run_id="failed-proposals", seed=SEED)
    gf = c.reasons[:, REASON_CODES.index("generation_failed")]
    per_transition = np.unique(np.stack([c.source_episode_id, c.source_frame.astype(str)], 1), axis=0, return_counts=True)[1]
    singular = c.actions[:, 0] > 0.1
    record("failed_proposals_rejected_never_refilled",
           gf.any() and np.array_equal(gf, singular) and not c.accepted[gf].any() and len(c.accepted) == K * len(s)
           and set(per_transition.tolist()) == {K},
           failed=int(gf.sum()), candidates=int(len(c.accepted)), rows_per_transition=sorted(set(per_transition.tolist())))

    # 6. A complete pipeline run with noisy_action_fe.
    work = out / "pipeline"
    work.mkdir()
    write_trajectories(dataset, work / "fixture.npz")
    r = subprocess.run([sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
                        "--set", "data.path=fixture.npz", "--set", "run.output_dir=runs",
                        "--set", "generation.method=noisy_action_fe", "--set", f"generation.labels_per_transition={K}",
                        "--set", f"generation.action_noise_std={SIGMA}", "--set", "generation.max_iterations=50"],
                       cwd=work, capture_output=True, text=True)
    runs = {p.name: p for p in sorted((work / "runs").iterdir()) if p.is_dir()} if (work / "runs").exists() else {}
    status = {n: validate_run(p)[0] for n, p in runs.items()}
    lab = next((p for n, p in runs.items() if n.startswith("labels-generate")), None)
    pc = read_candidates(lab / "labels/candidates.npz") if lab else None
    record("noisy_action_fe_pipeline_run", r.returncode == 0 and set(status.values()) == {"complete"} and len(status) == 3
           and pc is not None and pc.generator["name"] == "noisy_action_preimage" and pc.measurement_specs is not None,
           stages=status, generator=pc.generator if pc else None,
           artifact=f"<out>/pipeline/runs/{lab.name}/labels/candidates.npz" if lab else None, stderr=r.stderr[-300:])

    # 7. Measurements describe the stored candidate: float32 at a high dynamic range.
    rng_h = np.random.default_rng(9)
    eps_h = []
    for e in range(4):
        st = (1e4 + np.cumsum(rng_h.normal(scale=1.0, size=(6, 2)), axis=0)).astype(np.float32)
        eps_h.append(Episode(f"hdr{e}", {"state": st}, (1e4 + rng_h.normal(size=(6, 1))).astype(np.float32)))
    hdr = TrajectoryDataset(eps_h).validate()
    ch = generate_labels(hdr, rep, model, NoisyAction(K, 1e-4), [Finite(), ModelResidual()], run_id="hdr", seed=SEED)
    write_candidates(ch, out / "hdr.npz")
    with np.load(out / "hdr.npz", allow_pickle=False) as z:
        a_file, z_file = z["actions"].astype(np.float64), z["obs/state"].astype(np.float64)
        noise_file, disp_file = z["meas/generator.action_noise_norm"], z["meas/generator.state_displacement_norm"]
    order = {(ep.episode_id, t): i for i, (ep, t) in enumerate((ep, t) for ep in hdr.episodes for t in range(ep.length - 1))}
    idx = np.array([order[(str(e), int(f))] for e, f in zip(ch.source_episode_id, ch.source_frame)])
    a_src = np.concatenate([ep.actions[:-1] for ep in hdr.episodes]).astype(np.float64)[idx]
    s_src = np.concatenate([ep.observations["state"][:-1] for ep in hdr.episodes]).astype(np.float64)[idx]
    stored = np.linalg.norm(a_file - a_src, axis=1)
    record("measurements_match_the_stored_float32_candidate",
           np.array_equal(noise_file, stored) and np.array_equal(disp_file, np.linalg.norm(z_file - s_src, axis=1))
           and (stored == 0).any(),
           action_magnitude=1e4, action_noise_std=1e-4, float32_spacing_at_1e4=float(np.spacing(np.float32(1e4))),
           rows=int(len(stored)), rows_whose_stored_perturbation_is_zero=int((stored == 0).sum()),
           stored_noise_norm_max=float(stored.max()))

    # 8. A solved candidate's convergence is judged on its stored (float32) values.
    tight = NoisyActionPreimage(K, SIGMA, max_iterations=50, tolerance=1e-12)
    ct = generate_labels(dataset, rep, model, tight, [Finite(), GenerationFailed(), ModelResidual()], run_id="tight",
                         seed=SEED)
    write_candidates(ct, out / "tight.npz")
    with np.load(out / "tight.npz", allow_pickle=False) as z:
        zf, af, tf = (z[k].astype(np.float64) for k in ("obs/state", "actions", "target/state"))
        resid_file = z["meas/generator.solver_residual"]
    independent = sys_.residual(zf, af, tf)  # the true dynamics are the model here, evaluated in float64
    failed_file = ct.reasons[:, REASON_CODES.index("generation_failed")]
    record("solver_convergence_judged_on_the_stored_candidate",
           np.allclose(resid_file, independent, rtol=0, atol=1e-12) and np.array_equal(failed_file, ~(independent <= 1e-12))
           and failed_file.any(), tolerance=1e-12, rows=int(len(failed_file)), failed=int(failed_file.sum()),
           stored_residual_max=float(independent.max()))

    # 9. Sample identities are digested in full: a CRC32 collision pair gets distinct streams.
    import zlib
    from ccil.core.generators import transition_rng
    from ccil.training.noise_bc import counter_uniforms
    pair = ["4300bByhudVrUGx5", "0BZ5Gzkl9ccfsEbx"]
    crc_equal = zlib.crc32(pair[0].encode()) == zlib.crc32(pair[1].encode())
    u = counter_uniforms(SEED, 3, pair, [7, 7], 0, 4)
    g = [transition_rng(SEED, e, 7).normal(size=4) for e in pair]
    forced = NoiseBC("symmetric_uniform_half_batch", 0.1, seed=SEED, mask_probability=1.0)
    _, fr_rec = forced.perturb(np.zeros((2, 3), np.float32), pair, [7, 7], step=3)
    record("crc32_collision_pair_gets_distinct_streams",
           crc_equal and not np.array_equal(u[0], u[1]) and not np.array_equal(g[0], g[1])
           and not np.array_equal(fr_rec["noise"][0], fr_rec["noise"][1]),
           pair=pair, crc32=zlib.crc32(pair[0].encode()), crc32_equal=crc_equal,
           noise_bc_uniforms=u.tolist(), generator_normals=[x.tolist() for x in g])

    # 10. Measurement contracts are enforced, never silently dropped.
    class Unspecified(NoisyAction):
        measurement_specs = None

    class ExtraSpec(ModelResidual):
        measurement_specs = {**ModelResidual.measurement_specs, "unmeasured": {"definition": "x", "norm": None, "units": "-"}}

    contract = {}
    for name, (gen, vals) in (("generator_without_specs", (Unspecified(K, SIGMA), validators)),
                              ("validator_with_extra_spec", (NoisyAction(K, SIGMA), [Finite(), ExtraSpec()]))):
        try:
            generate_labels(dataset, rep, model, gen, vals, run_id="contract", seed=SEED)
            contract[name] = "accepted"
        except InterfaceError as e:
            contract[name] = f"InterfaceError: {e}"
    record("measurement_contracts_enforced", all(v.startswith("InterfaceError") for v in contract.values())
           and "Unspecified" in contract["generator_without_specs"] and "ExtraSpec" in contract["validator_with_extra_spec"],
           results=contract)

    # 11. NoiseBC rejects inputs it could not perturb as configured.
    nb_in = NoiseBC("symmetric_uniform_half_batch", 0.1, seed=SEED)
    ingress = {}
    for name, args in (("integer_observations", (np.zeros((2, 3), np.int64), ["a", "b"], [0, 1], 0)),
                       ("ids_length_mismatch", (np.zeros((2, 3)), ["a"], [0, 1], 0)),
                       ("frames_not_integer", (np.zeros((2, 3)), ["a", "b"], [0.0, 1.0], 0)),
                       ("negative_frame", (np.zeros((2, 3)), ["a", "b"], [0, -1], 0)),
                       ("negative_step", (np.zeros((2, 3)), ["a", "b"], [0, 1], -1)),
                       ("one_dimensional_observations", (np.zeros(2), ["a", "b"], [0, 1], 0))):
        try:
            nb_in.perturb(*args)
            ingress[name] = "accepted"
        except ValueError as e:
            ingress[name] = f"ValueError: {e}"
    record("noise_bc_validates_inputs", all(v.startswith("ValueError") for v in ingress.values()), results=ingress)

    # NoiseBC.
    rng = np.random.default_rng(0)
    big = rng.normal(size=(40000, 3)).astype(np.float32)
    ids = np.array([f"ep{i % 200}" for i in range(len(big))])
    frames = np.arange(len(big)) // 200
    stats = {}
    for mode in ("legacy_uniform_half_batch", "symmetric_uniform_half_batch"):
        nb = NoiseBC(mode, 0.1, seed=SEED)
        before = big.copy()
        new, rec = nb.perturb(big, ids, frames, step=0)
        applied = rec["noise"][rec["mask"]]
        stats[mode] = {"mask_rate": float(rec["mask"].mean()), "noise_mean": float(applied.mean()),
                       "noise_min": float(applied.min()), "noise_max": float(applied.max()),
                       "unmasked_rows_unchanged": bool(np.array_equal(new[~rec["mask"]], big[~rec["mask"]])),
                       "input_unchanged": bool(np.array_equal(big, before)), "descriptor": nb.descriptor()}
    leg, sym = stats["legacy_uniform_half_batch"], stats["symmetric_uniform_half_batch"]
    record("noise_bc_legacy_vs_symmetric",
           abs(leg["noise_mean"] - 0.05) < 2e-3 and leg["noise_min"] >= 0 and leg["noise_max"] < 0.1
           and abs(sym["noise_mean"]) < 2e-3 and sym["noise_min"] >= -0.1 and sym["noise_max"] < 0.1
           and all(abs(x["mask_rate"] - 0.5) < 0.01 and x["unmasked_rows_unchanged"] and x["input_unchanged"]
                   for x in stats.values()), scale=0.1, samples=len(big), modes=stats)

    # Per-draw determinism and independence, on the analytic states (full draws kept).
    obs = s.astype(np.float32)
    eid = np.array([str(e) for ep in dataset.episodes for e in [ep.episode_id] * (ep.length - 1)])
    fr = np.concatenate([np.arange(ep.length - 1) for ep in dataset.episodes])
    nb = NoiseBC("symmetric_uniform_half_batch", 0.1, seed=SEED)
    r0, r0b = nb.perturb(obs, eid, fr, step=0)[1], nb.perturb(obs, eid, fr, step=0)[1]
    r1, r0d1 = nb.perturb(obs, eid, fr, step=1)[1], nb.perturb(obs, eid, fr, step=0, draw=1)[1]
    perm = np.random.default_rng(1).permutation(len(obs))
    rp = nb.perturb(obs[perm], eid[perm], fr[perm], step=0)[1]
    rs = nb.perturb(obs[:3], eid[:3], fr[:3], step=0)[1]
    raw = lambda rec: np.where(rec["mask"][:, None], rec["noise"], np.nan)  # noqa: E731
    _atomic_savez(out / "noise_bc_draws.npz", {"obs": obs, "episode_id": eid, "frame": fr,
                                               "mask_step0": r0["mask"], "noise_step0": r0["noise"],
                                               "mask_step1": r1["mask"], "noise_step1": r1["noise"],
                                               "mask_step0_draw1": r0d1["mask"], "noise_step0_draw1": r0d1["noise"]})
    record("noise_bc_per_draw_reproducible_and_independent",
           np.array_equal(r0["noise"], r0b["noise"]) and np.array_equal(r0["mask"], r0b["mask"])
           and not np.array_equal(raw(r0), raw(r1)) and not np.array_equal(raw(r0), raw(r0d1))
           and np.array_equal(rp["noise"], r0["noise"][perm]) and np.array_equal(rp["mask"], r0["mask"][perm])
           and np.array_equal(rs["noise"], r0["noise"][:3]) and np.array_equal(rs["mask"], r0["mask"][:3]),
           samples=len(obs), artifact="<out>/noise_bc_draws.npz",
           masked_step0=int(r0["mask"].sum()), masked_step1=int(r1["mask"].sum()))

    def trace(seed):
        t, n = NoiseTrace(), NoiseBC("symmetric_uniform_half_batch", 0.1, seed=seed)
        for step in range(5):
            t.update(step, 0, eid, fr, n.perturb(obs, eid, fr, step=step)[1])
        return t.summary()
    t1, t2, t3 = trace(SEED), trace(SEED), trace(SEED + 1)
    record("noise_bc_trace_hash", t1 == t2 and t1["sha256"] != t3["sha256"] and t1["draws"] == 5,
           same_seed=[t1, t2], other_seed=t3)

    # ---------------- PR 2: filtering, weighting, inventory ----------------
    from ccil.core import BackwardEuler, LabelError, ModelMisfit
    from ccil.schema import SchemaError

    def file_meta(path):
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta"]))

    # 12. The label-error estimate is the residual-Jacobian closed form (the branch used the next-state map).
    ce = generate_labels(dataset, rep, model, BackwardEuler(), [Finite(), LabelError()], run_id="le", seed=SEED)
    L_res = float(np.linalg.norm(np.hstack([sys_.A, sys_.B]), 2))
    L_next = float(np.linalg.norm(np.hstack([np.eye(2) + sys_.A, sys_.B]), 2))
    dist = ce.measurements["state_action_distance"]
    est = ce.measurements["label_error_estimate"]
    record("label_error_estimate_uses_the_residual_jacobian",
           np.allclose(est, L_res * dist, rtol=1e-12, atol=1e-15) and np.allclose(ce.measurements["anchor_local_lipschitz"], L_res)
           and "label_error" not in ce.reason_codes and ce.accepted.all(),
           residual_jacobian_norm=L_res, branch_next_state_jacobian_norm=L_next,
           estimate_mean=float(est.mean()), branch_estimate_mean=float((L_next * dist).mean()),
           distance_mean=float(dist.mean()))

    # 13. model_misfit: the misfit transitions' candidates are recorded as rejected, not dropped.
    frames_all = np.concatenate([np.arange(ep.length - 1) for ep in dataset.episodes])
    wrong = copy.deepcopy(dataset)
    for ep in wrong.episodes:  # perturb the stored next state of frame 2, so the exact model misfits it
        if ep.length > 3:
            ep.observations["state"] = ep.observations["state"].copy()
            ep.observations["state"][3] += np.float32(1.0)
    cm = generate_labels(wrong, rep, model, NoisyAction(K, SIGMA), [Finite(), ModelMisfit(0.5)], run_id="misfit",
                         seed=SEED)
    mm = cm.reasons[:, cm.reason_codes.index("model_misfit")]
    expect = cm.measurements["anchor_model_error"] >= 0.5
    record("model_misfit_rejects_and_records",
           mm.any() and np.array_equal(mm, expect) and not cm.accepted[mm].any() and len(cm.accepted) == K * len(frames_all)
           and set(np.unique(cm.source_frame[mm]).tolist()) <= {2, 3},
           rejected=int(mm.sum()), candidates=int(len(cm.accepted)), rejected_source_frames=sorted(set(cm.source_frame[mm].tolist())))

    # 14. Filters: quantile and absolute, threshold recorded and reconstructible, batch-invariant.
    filt = {}
    for name, val in (("quantile", LabelError(quantile=0.5)), ("absolute", LabelError(max_error=0.002))):
        paths = []
        for bs in (1, None):
            cf = generate_labels(dataset, rep, model, NoisyAction(K, SIGMA), [Finite(), GenerationFailed(), val],
                                 run_id=f"filter-{name}", seed=SEED, batch_size=bs)
            path = out / f"filter-{name}-{bs}.npz"
            write_candidates(cf, path)
            paths.append(path)
        with np.load(paths[1], allow_pickle=False) as z:
            e = z["meas/label_error_estimate"]
            thr = z["meas/label_error_threshold"]
            codes = json.loads(str(z["meta"]))["reason_codes"]
            rej = z["reasons"][:, codes.index("label_error")]
        expected_thr = float(np.quantile(e, 0.5)) if name == "quantile" else 0.002
        filt[name] = {"rows": int(len(e)), "rejected": int(rej.sum()), "threshold": float(thr[0]),
                      "recomputed_threshold": expected_thr, "kept_fraction": float((~rej).mean()),
                      "batch_invariant": sha(paths[0]) == sha(paths[1]), "file_version": file_meta(paths[1])["version"]}
        filt[name]["ok"] = (np.allclose(thr, expected_thr) and np.array_equal(rej, ~(e <= expected_thr))
                            and filt[name]["batch_invariant"] and filt[name]["file_version"] == "2.3" and 0 < rej.sum() < len(e))
    record("label_error_filters_reconstructible_from_the_file", all(v["ok"] for v in filt.values()), results=filt)

    # 15. Independent reasons: support and label error together; accounting closes.
    cb = generate_labels(dataset, rep, model, NoisyAction(K, SIGMA),
                         [Finite(), SupportRadius(0.005), LabelError(max_error=0.0005), ModelResidual()], run_id="both",
                         seed=SEED)
    both = cb.reasons[:, cb.reason_codes.index("out_of_support")] & cb.reasons[:, cb.reason_codes.index("label_error")]
    record("filters_record_independent_reasons", both.any() and np.array_equal(cb.accepted, ~cb.reasons.any(axis=1)),
           rows_with_both=int(both.sum()), accepted=int(cb.accepted.sum()), candidates=int(len(cb.accepted)))

    # 16. Weighting never changes validity.
    class ZeroSomeAccepted:
        name, version, params = "zero_even_frames", "1", {}

        def weights(self, cands, accepted):
            return np.where(accepted & (np.arange(len(accepted)) % 2 == 0), 0.0, accepted.astype(float))

    class WeightRejected:
        name, version, params = "weights_rejected", "1", {}

        def weights(self, cands, accepted):
            return np.ones(len(accepted))

    vals = [Finite(), SupportRadius(0.005), ModelResidual()]
    base = generate_labels(dataset, rep, model, NoisyAction(K, SIGMA), vals, run_id="w", seed=SEED)
    zero = generate_labels(dataset, rep, model, NoisyAction(K, SIGMA), vals, run_id="w", seed=SEED, weighting=ZeroSomeAccepted())
    try:
        generate_labels(dataset, rep, model, NoisyAction(K, SIGMA), vals, run_id="w", seed=SEED, weighting=WeightRejected())
        rejected_weight = "accepted"
    except SchemaError as e:
        rejected_weight = f"SchemaError: {sorted(e.counts)}"
    record("weighting_never_changes_validity",
           np.array_equal(zero.accepted, base.accepted) and np.array_equal(zero.reasons, base.reasons)
           and (zero.weight[zero.accepted] == 0).any() and (~base.accepted).any() and rejected_weight.startswith("SchemaError"),
           accepted_with_zero_weight=int((zero.weight[zero.accepted] == 0).sum()), rejected_rows=int((~base.accepted).sum()),
           positive_weight_on_rejected=rejected_weight)

    # 17. The branch's fixed-point forward_euler reaches the same root as Newton on a contraction.
    def fixed_point(z_star, act, iters=50):  # label-filtering gen_forward_euler: s <- s* - f(s, a)
        x = z_star.copy()
        for _ in range(iters):
            x = z_star - model.delta(x, act)
        return x
    newton = sys_.exact_preimage(s, a)
    fp = fixed_point(s, a)
    record("fixed_point_forward_euler_superseded_by_newton",
           float(np.linalg.norm(sys_.A, 2)) < 1 and float(np.abs(fp - newton).max()) < 1e-6,
           contraction_norm=float(np.linalg.norm(sys_.A, 2)), max_abs_difference=float(np.abs(fp - newton).max()))

    # 18. Migration of the branch's keys.
    from ccil.config import resolve
    legacy_yml = out / "legacy_filtering.yml"
    legacy_yml.write_text((REPO / "config/pendulum_cont.yml").read_text().replace(
        "  epsilon:", "  label_err_quantile: 0.3\n  max_label_err: 0.5\n  model_err_thresh: 0.1\n  epsilon:"))
    rc = resolve(config=legacy_yml)
    record("branch_filter_keys_migrate",
           (rc["generation.label_error_quantile"], rc["generation.max_label_error"], rc["generation.model_error_threshold"])
           == (0.3, 0.5, 0.1), values={k: rc[k] for k in ("generation.label_error_quantile", "generation.max_label_error",
                                                            "generation.model_error_threshold")})

    # 19. Every label-filtering capability of the inventory is classified.
    inventory = json.loads((REPO / "docs/inventory/capabilities.json").read_text())
    caps = inventory["capabilities"] if isinstance(inventory, dict) else inventory
    touched = sorted(c["id"] for c in caps if "label-filtering" in json.dumps(c))
    table = (REPO / "docs/migration/label-methods.md").read_text().split("## Label-filtering branch inventory", 1)[1]
    classified = {line.split("|")[1].strip().strip("`") for line in table.splitlines() if line.startswith("| `")}
    missing = [c for c in touched if c not in classified]
    record("label_filtering_inventory_fully_classified", not missing and touched, capabilities=touched, missing=missing)

    # 20. The pipeline with filters writes a complete 2.3 run.
    wf = out / "pipeline_filters"
    wf.mkdir()
    write_trajectories(dataset, wf / "fixture.npz")
    r = subprocess.run([sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
                        "--set", "data.path=fixture.npz", "--set", "run.output_dir=runs",
                        "--set", "generation.method=noisy_action", "--set", f"generation.labels_per_transition={K}",
                        "--set", f"generation.action_noise_std={SIGMA}", "--set", "generation.label_error_quantile=0.5",
                        "--set", "generation.model_error_threshold=0.5"], cwd=wf, capture_output=True, text=True)
    fruns = {p.name: p for p in sorted((wf / "runs").iterdir()) if p.is_dir()} if (wf / "runs").exists() else {}
    fstat = {n: validate_run(p)[0] for n, p in fruns.items()}
    flab = next((p for n, p in fruns.items() if n.startswith("labels-generate")), None)
    fmeta = file_meta(flab / "labels/candidates.npz") if flab else {}
    record("filters_pipeline_run", r.returncode == 0 and set(fstat.values()) == {"complete"} and len(fstat) == 3
           and fmeta.get("version") == "2.3" and {"model_misfit", "label_error"} <= set(fmeta.get("reason_codes", [])),
           stages=fstat, file_version=fmeta.get("version"), reason_codes=fmeta.get("reason_codes"),
           artifact=f"<out>/pipeline_filters/runs/{flab.name}/labels/candidates.npz" if flab else None, stderr=r.stderr[-300:])

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
