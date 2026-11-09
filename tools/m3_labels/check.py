#!/usr/bin/env python3
"""Evidence for #22 part 1: noisy-action measurements (NoiseBC's moved to il-supervision-benchmarks, #168).

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
Filtering and weighting (#22 part 2; label-error estimates and filters moved to
certificates in #88, with their evidence in tools/m3_certificates):
- anchor model error is measured on every row and rejected above the gate,
  batch-invariantly, as a 2.2 extension reason that the 2.2 reader (520f343)
  reads;
- reasons stay independent, and a weighting never changes validity.
Action chunks (#22 part 3):
- position 0 is the label action and positions k >= 1 the source episode's
  actions from the frame the label reaches (t for backtrack, t + 1 for
  disturbed_action); chunks are masked past the episode end, never padded,
  never read another episode, and stay well formed on failed proposals;
- chunk files are 2.3, round-trip through accepted_labels, and the 2.2
  reader (8071952) reads them, ignoring the chunks; the pipeline writes a
  complete run with chunks, and aug.chunk_size migrates;
- target frames of the wrong dtype or shape, wrong or non-finite target
  frames, nonzero or non-finite masked positions, a
  forged 2.2 file with chunks, undeclared chunk arrays, a spec without its
  target-frame offset, and non-integer chunk sizes are all rejected.
Writes DIR/report.json.
"""
import argparse
import copy
import hashlib
import json
import os
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
    record("noisy_action_fe_pipeline_run", r.returncode == 0 and set(status.values()) == {"complete"} and len(status) == 5
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
    pair = ["4300bByhudVrUGx5", "0BZ5Gzkl9ccfsEbx"]
    crc_equal = zlib.crc32(pair[0].encode()) == zlib.crc32(pair[1].encode())
    g = [transition_rng(SEED, e, 7).normal(size=4) for e in pair]
    record("crc32_collision_pair_gets_distinct_streams", crc_equal and not np.array_equal(g[0], g[1]),
           pair=pair, crc32=zlib.crc32(pair[0].encode()), crc32_equal=crc_equal,
           generator_normals=[x.tolist() for x in g])

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

    # ---------------- PR 2: filtering and weighting ----------------
    from ccil.core import BackwardEuler, ModelMisfit
    from ccil.core.validators import MODEL_MISFIT
    from ccil.schema import SchemaError

    def file_meta(path):
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta"]))

    # 12-14. Model misfit is measured on every row and rejected with a threshold. (The label-error estimates and their
    # filters moved to certificates, #88: tools/m3_certificates.)
    wrong = copy.deepcopy(dataset)
    for ep in wrong.episodes:  # perturb the stored state at frame 3, so the exact model misfits transitions 2 and 3
        if ep.length > 3:
            ep.observations["state"] = ep.observations["state"].copy()
            ep.observations["state"][3] += np.float32(1.0)
    gate = 0.5
    paths = []
    for bs in (1, None):
        cm = generate_labels(wrong, rep, model, NoisyAction(K, SIGMA), [Finite(), GenerationFailed(), ModelMisfit(gate)],
                             run_id="misfit", seed=SEED, batch_size=bs)
        paths.append(out / f"misfit-{bs}.npz")
        write_candidates(cm, paths[-1])
    with np.load(paths[1], allow_pickle=False) as z:
        codes = json.loads(str(z["meta"]))["reason_codes"]
        anchor = z["meas/anchor_model_error"]
        mm = z["reasons"][:, codes.index(MODEL_MISFIT)]
    # The schema reader merged before the extension reasons were used (main at 520f343) reads the file unchanged.
    old_root = out / "reader_520f343"
    old_root.mkdir()
    tarball = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "520f343", "src"],
                             capture_output=True, check=True).stdout
    (old_root / "src.tar").write_bytes(tarball)
    subprocess.run(["tar", "-xf", "src.tar"], cwd=old_root, check=True)
    probe = "import sys; from ccil.schema import read_candidates; c = read_candidates(sys.argv[1]); print(len(c.accepted))"
    old_read = subprocess.run([sys.executable, "-c", probe, str(paths[1])], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(old_root / "src")})
    record("model_misfit_measured_and_rejected",
           mm.any() and np.array_equal(mm, ~(anchor < gate)) and len(cm.accepted) == K * len(s)
           and file_meta(paths[1])["version"] == "2.2" and sha(paths[0]) == sha(paths[1])
           and old_read.returncode == 0 and old_read.stdout.strip() == str(len(cm.accepted)),
           rejected_model_misfit=int(mm.sum()), candidates=int(len(mm)), batch_invariant=sha(paths[0]) == sha(paths[1]),
           read_by_520f343=old_read.returncode == 0, rejected_source_frames=sorted(set(cm.source_frame[mm].tolist())))

    # 18. Independent reasons: support and model misfit together; accounting closes.
    cb = generate_labels(wrong, rep, model, NoisyAction(K, SIGMA),
                         [Finite(), SupportRadius(0.005), ModelMisfit(gate), ModelResidual()], run_id="both", seed=SEED)
    both = cb.reasons[:, cb.reason_codes.index("out_of_support")] & cb.reasons[:, cb.reason_codes.index(MODEL_MISFIT)]
    record("filters_record_independent_reasons", both.any() and np.array_equal(cb.accepted, ~cb.reasons.any(axis=1)),
           rows_with_both=int(both.sum()), accepted=int(cb.accepted.sum()), candidates=int(len(cb.accepted)))

    # 19. Weighting never changes validity.
    class ZeroSomeAccepted:
        name, version, params = "zero_even_rows", "1", {}

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

    # ---------------- PR 3: action chunks ----------------
    CK = 4

    def chunk_ok(c, ds_):
        return chunk_ok_k(c, ds_, CK)

    def chunk_ok_k(c, ds_, CK):
        """Position 0 is the label action; k >= 1 the source episode's action at tau + k - 1, masked past the end."""
        eps_ = {ep.episode_id: ep for ep in ds_.episodes}
        tau = c.measurements["chunk_target_frame"].astype(int)
        for i in range(len(c.accepted)):
            ep = eps_[str(c.source_episode_id[i])]
            frames_k = tau[i] + np.arange(CK - 1)
            inside = frames_k <= ep.length - 1
            if not (c.action_chunk_mask[i, 0] and np.array_equal(c.action_chunks[i, 0], c.actions[i])
                    and np.array_equal(c.action_chunk_mask[i, 1:], inside)
                    and np.array_equal(c.action_chunks[i, 1:][inside], ep.actions[frames_k[inside]])
                    and not c.action_chunks[i, 1:][~inside].any()):
                return False
        return True

    # 24-25. Alignment per profile.
    cbt = generate_labels(dataset, rep, model, BackwardEuler(), [Finite()], run_id="chunks-bt", seed=SEED,
                          action_chunk_size=CK)
    cda = generate_labels(dataset, rep, model, NoisyAction(K, SIGMA), [Finite()], run_id="chunks-da", seed=SEED,
                          action_chunk_size=CK)
    record("action_chunks_align_to_the_label_target",
           chunk_ok(cbt, dataset) and chunk_ok(cda, dataset)
           and np.array_equal(cbt.measurements["chunk_target_frame"], cbt.source_frame)
           and np.array_equal(cda.measurements["chunk_target_frame"], cda.source_frame + 1),
           chunk_size=CK, backtrack_target="t", disturbed_action_target="t + 1",
           masked_positions={"backtrack": int((~cbt.action_chunk_mask).sum()),
                             "disturbed_action": int((~cda.action_chunk_mask).sum())})

    # 26. Boundaries: episode-tagged actions, a 2-frame episode, K longer than the episode, failed proposals.
    tagged = []
    for e, T in enumerate((6, 2, 4)):
        st = np.cumsum(np.full((T, 2), 0.01 * (e + 1)), axis=0).astype(np.float32)
        tagged.append(Episode(f"tag{e}", {"state": st}, (100.0 * e + np.arange(T, dtype=np.float32))[:, None]))
    tagged_ds = TrajectoryDataset(tagged).validate()
    big_k = 7
    cb = generate_labels(tagged_ds, rep, model, NoisyAction(2, SIGMA), [Finite()], run_id="tagged", seed=SEED,
                         action_chunk_size=big_k)
    owner = {f"tag{e}": e for e in range(3)}
    vals = cb.action_chunks[:, 1:, 0]
    same_episode = all((np.floor(vals[i][cb.action_chunk_mask[i, 1:]] / 100.0) == owner[str(cb.source_episode_id[i])]).all()
                       for i in range(len(vals)))
    short_rows = cb.source_episode_id == "tag1"
    cfail = generate_labels(dataset, rep, SingularForLargeActions(model), NoisyActionPreimage(K, SIGMA),
                            [Finite(), GenerationFailed()], run_id="failed-chunks", seed=SEED, action_chunk_size=CK)
    gfc = cfail.reasons[:, REASON_CODES.index("generation_failed")]
    record("action_chunks_never_cross_or_pad_episodes",
           # 2-frame episode: the label reaches frame 1, the last, whose stored action fills position 1 (of 7).
           same_episode and set(cb.action_chunk_mask[short_rows].sum(axis=1).tolist()) == {2}
           and chunk_ok_k(cb, tagged_ds, big_k)
           and gfc.any() and not cfail.accepted[gfc].any() and np.isfinite(cfail.action_chunks[cfail.action_chunk_mask]).all(),
           chunk_size=big_k, episode_lengths=[6, 2, 4], two_frame_episode_valid_positions=int(cb.action_chunk_mask[short_rows].sum(axis=1).max()),
           all_values_from_their_own_episode=bool(same_episode), failed_rows_with_chunks=int(gfc.sum()))

    # 28-29. Schema: 2.3 round trip, accepted_labels, the 2.2 reader (8071952) ignores chunks, no chunks stays 2.2.
    write_candidates(cda, out / "chunks.npz")
    back_c = read_candidates(out / "chunks.npz", dataset)
    acc_c = back_c.accepted_labels()
    write_candidates(generate_labels(dataset, rep, model, NoisyAction(K, SIGMA), [Finite()], run_id="no-chunks",
                                     seed=SEED), out / "no_chunks.npz")
    r22 = out / "reader_8071952"
    r22.mkdir()
    (r22 / "src.tar").write_bytes(subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "8071952", "src"],
                                                 capture_output=True, check=True).stdout)
    subprocess.run(["tar", "-xf", "src.tar"], cwd=r22, check=True)
    probe22 = ("import sys; from ccil.schema import read_candidates; c = read_candidates(sys.argv[1]); "
               "print(len(c.accepted), hasattr(c, 'action_chunks'))")
    old22 = subprocess.run([sys.executable, "-c", probe22, str(out / "chunks.npz")], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(r22 / "src")})
    record("action_chunk_schema_round_trip_and_compatibility",
           file_meta(out / "chunks.npz")["version"] == "2.3" and np.array_equal(back_c.action_chunks, cda.action_chunks)
           and np.array_equal(acc_c.action_chunk_mask, cda.action_chunk_mask[cda.accepted])
           and old22.returncode == 0 and old22.stdout.split()[1] == "False"
           and file_meta(out / "no_chunks.npz")["version"] == "2.2"
           and "one-step" in back_c.action_chunk_spec["guarantee"],
           file_version=file_meta(out / "chunks.npz")["version"], optional_fields=file_meta(out / "chunks.npz")["optional_fields"],
           read_by_8071952=old22.stdout.strip() or old22.stderr[-300:], action_chunk_spec=back_c.action_chunk_spec)

    # 30. The pipeline, and 31. migration of aug.chunk_size.
    wc = out / "pipeline_chunks"
    wc.mkdir()
    write_trajectories(dataset, wc / "fixture.npz")
    rcmd = subprocess.run([sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
                           "--set", "data.path=fixture.npz", "--set", "run.output_dir=runs",
                           "--set", f"generation.action_chunk_size={CK}"], cwd=wc, capture_output=True, text=True)
    cruns = {p.name: p for p in sorted((wc / "runs").iterdir()) if p.is_dir()} if (wc / "runs").exists() else {}
    cstat = {n: validate_run(p)[0] for n, p in cruns.items()}
    clab = next((p for n, p in cruns.items() if n.startswith("labels-generate")), None)
    from ccil.config import resolve
    legacy_chunk_yml = out / "legacy_chunks.yml"
    legacy_chunk_yml.write_text((REPO / "config/pendulum_cont.yml").read_text().replace("  epsilon:", "  chunk_size: 5\n  epsilon:"))
    rc5 = resolve(config=legacy_chunk_yml)
    record("action_chunks_pipeline_and_migration",
           rcmd.returncode == 0 and set(cstat.values()) == {"complete"} and len(cstat) == 5
           and file_meta(clab / "labels/candidates.npz")["version"] == "2.3" and rc5["generation.action_chunk_size"] == 5,
           stages=cstat, migrated_chunk_size=rc5["generation.action_chunk_size"],
           artifact=f"<out>/pipeline_chunks/runs/{clab.name}/labels/candidates.npz" if clab else None, stderr=rcmd.stderr[-300:])

    # 32. Malformed chunks and chunk files are rejected (review of #93).
    def schema_rejects(build):
        try:
            build()
            return "accepted"
        except SchemaError as e:
            return f"SchemaError: {sorted(e.counts)}: {e.violations[0]['detail'][:120]}"

    def mutated_set(base, edit):
        c = copy.deepcopy(base)
        edit(c)
        return lambda: c.validate(dataset)

    def bump_frame(c):
        c.measurements["chunk_target_frame"][0] += 1

    rej_row = int(np.flatnonzero(~cfail.accepted)[0])

    def nan_rejected_frame(c):
        c.measurements["chunk_target_frame"][rej_row] = np.nan

    masked = np.argwhere(~cda.action_chunk_mask)

    def nonzero_tail(c):
        c.action_chunks[tuple(masked[0])] = 5.0

    def nan_tail(c):
        c.action_chunks[tuple(masked[0])] = np.nan

    def edited_file(name, edit):
        with np.load(out / "chunks.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays["meta"]))
        edit(meta)
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(out / name, **arrays)
        return lambda: read_candidates(out / name)

    def edited_arrays(name, edit):
        with np.load(out / "chunks.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        edit(arrays)
        np.savez(out / name, **arrays)
        return lambda: read_candidates(out / name)

    malformed = {
        "target_frame_string_dtype": schema_rejects(edited_arrays(
            "tau_str.npz", lambda a: a.__setitem__("meas/chunk_target_frame", a["meas/chunk_target_frame"].astype(str)))),
        "target_frame_wrong_shape": schema_rejects(edited_arrays(
            "tau_2d.npz", lambda a: a.__setitem__("meas/chunk_target_frame",
                                                  np.stack([a["meas/chunk_target_frame"]] * 2, axis=1)))),
        "wrong_integral_target_frame": schema_rejects(mutated_set(cda, bump_frame)),
        "non_finite_target_frame_on_rejected_row": schema_rejects(mutated_set(cfail, nan_rejected_frame)),
        "nonzero_masked_position": schema_rejects(mutated_set(cda, nonzero_tail)),
        "non_finite_masked_position": schema_rejects(mutated_set(cda, nan_tail)),
        "forged_2_2_file_with_chunks": schema_rejects(edited_file("forged22.npz", lambda m: m.update(version="2.2"))),
        "chunk_arrays_not_declared_optional": schema_rejects(edited_file("undeclared.npz", lambda m: m.update(optional_fields=[]))),
        "spec_without_target_frame_offset": schema_rejects(edited_file(
            "no_offset.npz", lambda m: m["action_chunk_spec"].pop("target_frame_offset"))),
    }
    api = {}
    for value in (2.5, True, 0):
        try:
            generate_labels(dataset, rep, model, BackwardEuler(), [Finite()], run_id="k", seed=SEED, action_chunk_size=value)
            api[repr(value)] = "accepted"
        except InterfaceError as e:
            api[repr(value)] = f"InterfaceError: {e}"
    record("malformed_chunks_and_chunk_files_rejected",
           all(v.startswith("SchemaError") for v in malformed.values()) and all(v.startswith("InterfaceError") for v in api.values())
           and cda.action_chunk_spec["target_frame_offset"] == 1 and cbt.action_chunk_spec["target_frame_offset"] == 0,
           malformed=malformed, action_chunk_size=api)

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
