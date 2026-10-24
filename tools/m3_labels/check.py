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
  complete run (the inspectable artifact).
NoiseBC, on fixed observations (full masks and noise kept in noise_bc_draws.npz):
- legacy mode is biased by scale/2 in [0, scale); symmetric mode is zero-mean in
  [-scale, scale); both perturb about half of each batch;
- a draw is reproducible from (seed, step, sample, draw index), fresh at a new
  step or draw, and independent of batch order and composition;
- perturb never modifies its input; the trace hash is reproducible and changes
  with the seed.
Writes DIR/report.json.
"""
import argparse
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

from ccil.core import (IdentityRepresentation, Finite, GenerationFailed,  # noqa: E402
                       LinearTransitionModel, ModelResidual, NoisyAction, NoisyActionPreimage, SupportRadius,
                       generate_labels)
from ccil.runs import validate_run  # noqa: E402
from ccil.schema import REASON_CODES, read_candidates, write_candidates, write_trajectories  # noqa: E402
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
        record(f"closed_form_and_measurements[{gen.name}]",
               len(c.accepted) == K * len(s) and float(np.abs(zg - expected).max()) < 1e-5 and need <= names and consistent
               and set(c.measurement_specs) == names and c.generator["params"]["labels_per_transition"] == K,
               candidates=int(len(c.accepted)), accepted=int(c.accepted.sum()),
               max_abs_vs_closed_form=float(np.abs(zg - expected).max()), measurements=sorted(names),
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
