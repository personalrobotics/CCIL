#!/usr/bin/env python3
"""Evidence for the vector CCIL dynamics port (#21).

    uv run --group evidence python tools/m3_dynamics/check.py --out DIR [--device cpu|cuda] [--only CASE ...]

Cases (all by default):
- legacy: legacy WorldModel weights, exported by legacy_export.py in the pinned
  legacy harness environment, give the same predictions in the new model
  (eval mode bit-identical; training mode within 1e-5 relative, with the same
  power-iteration buffer updates).
- analytic: `ccil labels generate --recipe analytic_double_integrator_mlp` on a
  generated double-integrator dataset; accepted labels are finite and inside
  the support radius, accounting closes, each backward-Euler label differs from
  the exact closed form by exactly the model's one-step error, and the model's
  correction residual is bounded by its local Lipschitz constant times the
  label distance.
- pendulum: `ccil labels generate --recipe pendulum_cont` with backward_euler on
  --device; reports the label and held-out evaluation model errors and the
  label counts.
- lipschitz: every Lipschitz type trains with finite losses; spectral-norm
  layers stay within the constraint.
- persistence, determinism: checkpoints reload exactly and re-save
  byte-identically; malformed checkpoints and unavailable devices fail as
  specified; equal seeds give byte-identical checkpoints.
- extra: on Python 3.10 (no torch) kind=mlp fails before any stage, with
  install guidance.
Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
sys.path.insert(0, str(REPO / "correct_il"))
import torch  # noqa: E402
from system import DoubleIntegrator  # noqa: E402

from ccil.core import jacobian_fd, load_transition_model  # noqa: E402
from ccil.models.mlp import MLPTransitionModel  # noqa: E402
from ccil.runs import layout, validate_run  # noqa: E402
from ccil.schema import Episode, TrajectoryDataset, read_candidates, read_trajectories, write_trajectories  # noqa: E402

CASES = ("legacy", "analytic", "pendulum", "lipschitz", "persistence", "determinism", "extra")
LEGACY_TYPES = ("none", "spectral_normalization", "slack", "spectral_normalization_slack", "soft_sampling",
                "soft_sampling_slack", "soft_jac")
TRAIN_MODE_RTOL = 1e-5  # stated before running: float32 power iteration across torch 2.1 and the locked torch


def ccil(*args, cwd, python=sys.executable):
    return subprocess.run([str(python), "-m", "ccil", *map(str, args)], cwd=cwd, capture_output=True, text=True)


def runs(d: Path) -> dict:
    return {p.name: p for p in sorted(d.iterdir()) if p.is_dir() and not p.name.startswith(".")} if d.exists() else {}


def stage(rs: dict, prefix: str) -> Path:
    return next(p for n, p in rs.items() if n.startswith(prefix))


def double_integrator_dataset(path, episodes=40, frames=26, seed=11):
    """Episodes of the analytic system from uniform initial states under uniform random actions."""
    rng = np.random.default_rng(seed)
    sys_ = DoubleIntegrator(0.1)
    eps = []
    for e in range(episodes):
        s = [rng.uniform(-1, 1, size=2)]
        a = rng.uniform(-1, 1, size=(frames, 1))
        for t in range(frames - 1):
            s.append(sys_.step_np(s[-1][None], a[t][None])[0])
        eps.append(Episode(f"di{e:02d}", {"state": np.array(s, np.float32)}, a.astype(np.float32)))
    write_trajectories(TrajectoryDataset(eps), path)
    return sys_


def legacy_model(d: dict, lt: str):
    st = json.loads(str(d["meta"]))["settings"]
    m = MLPTransitionModel(list(st["hidden"]), epochs=1, batch_size=1, lipschitz_type=lt,
                           lipschitz_constraint=st["lipschitz_constraint"], soft_penalty_weight=0.01)
    m._build(2, 1)
    state = {k[6:]: torch.from_numpy(v) for k, v in d.items() if k.startswith("state/")}
    state.update({k[5:]: torch.from_numpy(v) for k, v in d.items() if k.startswith("norm/")})
    m.net.load_state_dict(state, strict=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--only", nargs="*", choices=CASES)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    only = set(args.only or CASES)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)

    # --- legacy equivalence at fixed weights ------------------------------------
    if "legacy" in only:
        exp = out / "legacy_export"
        r = subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with-requirements",
                            str(REPO / "tools/dynamics_checkpoint/requirements.txt"), "python",
                            str(REPO / "tools/m3_dynamics/legacy_export.py"), "--out", str(exp)],
                           cwd=REPO, capture_output=True, text=True)
        rows = {}
        for lt in LEGACY_TYPES:
            f = exp / f"{lt}.npz"
            if not f.exists():
                rows[lt] = {"error": r.stderr[-300:]}
                continue
            with np.load(f, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            m = legacy_model(d, lt)
            m.net.eval()
            f_eval = m.delta(d["z"], d["a"]).astype(np.float32)
            m.net.train()
            f_train = m.delta(d["z"], d["a"]).astype(np.float32)
            buffers = dict(m.net.named_buffers())
            after = {k[6:]: v for k, v in d.items() if k.startswith("after/")}
            scale = max(float(np.abs(d["f_train"]).max()), 1e-12)
            rows[lt] = {"eval_bit_identical": bool(np.array_equal(f_eval, d["f_eval"])),
                        "train_max_rel_diff": float(np.abs(f_train - d["f_train"]).max() / scale),
                        "power_iteration_buffers_max_diff": max((float(np.abs(buffers[k].numpy() - v).max())
                                                                 for k, v in after.items()), default=0.0),
                        "legacy_eval_differs_from_train": bool(not np.array_equal(d["f_eval"], d["f_train"])),
                        "legacy_torch": json.loads(str(d["meta"]))["torch"]}
        ok = all("error" not in v and v["eval_bit_identical"] and v["train_max_rel_diff"] <= TRAIN_MODE_RTOL
                 and v["power_iteration_buffers_max_diff"] <= TRAIN_MODE_RTOL for v in rows.values())
        # The training-mode defect (#21) shows exactly in the spectral-norm variants.
        ok = ok and {lt for lt, v in rows.items() if v.get("legacy_eval_differs_from_train")} == \
            {"spectral_normalization", "spectral_normalization_slack"}
        record("legacy_worldmodel_equivalence_at_fixed_weights", ok, tolerance_train_mode_rel=TRAIN_MODE_RTOL,
               torch=torch.__version__, types=rows)

    # --- analytic system through the CLI -----------------------------------------
    work = out / "analytic"
    if only & {"analytic", "persistence"}:
        work.mkdir()
        sys_ = double_integrator_dataset(work / "di.npz")
        ar = ccil("labels", "generate", "--recipe", "analytic_double_integrator_mlp", "--set", "data.path=di.npz",
                  "--set", "run.output_dir=runs", cwd=work)
        ars = runs(work / "runs")  # the analytic runs; the persistence case reads their checkpoint
        statuses = {n: validate_run(p)[0] for n, p in ars.items()}
    if "analytic" in only:
        detail = {"stages": statuses, "stderr": ar.stderr[-400:]}
        ok = ar.returncode == 0 and len(ars) == 3 and set(statuses.values()) == {"complete"}
        if ok:
            fit_dir, lab_dir = stage(ars, "transition-fit"), stage(ars, "labels-generate")
            ds = read_trajectories(stage(ars, "data-prepare") / layout.DATASET)
            c = read_candidates(lab_dir / layout.CANDIDATES, ds)
            model = load_transition_model(fit_dir / "checkpoints/transition_model.npz")
            s = np.concatenate([ep.observations["state"][:-1] for ep in ds.episodes]).astype(np.float64)
            a = np.concatenate([ep.actions[:-1] for ep in ds.episodes]).astype(np.float64)
            order = {(ep.episode_id, t): i for i, (ep, t) in enumerate((ep, t) for ep in ds.episodes for t in range(ep.length - 1))}
            idx = np.array([order[(str(e), int(f))] for e, f in zip(c.source_episode_id, c.source_frame)])
            s, a = s[idx], a[idx]
            zg = c.observations["state"].astype(np.float64)
            dist, resid = c.measurements["distance"], c.measurements["residual"]
            r_support = 0.05
            exact = sys_.backward_euler(s, a)
            model_err = sys_.f_np(s, a) - model.delta(s, a)
            closed_form_gap = float(np.abs((zg - exact) - model_err).max())
            # Correction residual |z_G + f(z_G, a) - z| <= sup_segment ||df/dz|| * ||z_G - z||, sampled on 11 points.
            lips = np.max([np.linalg.norm(jacobian_fd(model, s + t * (zg - s), a), ord=2, axis=(1, 2))
                           for t in np.linspace(0, 1, 11)], axis=0)
            bound_slack = float(np.max(resid - (1.05 * lips * dist + 1e-6)))
            acc = c.accepted
            true_accept = np.linalg.norm(sys_.f_np(s, a), axis=1) < r_support
            fit_metrics = json.loads((fit_dir / layout.METRICS).read_text())["values"]
            detail = {
                "stages": statuses, "candidates": int(len(acc)), "accepted": int(acc.sum()),
                "rejections": {code: int(c.reasons[:, i].sum()) for i, code in enumerate(c.reason_codes)},
                "I1_accepted_finite": bool(np.isfinite(zg[acc]).all()),
                "I2_accepted_in_support": bool((dist[acc] < r_support).all()),
                "I3_accounting": bool(np.array_equal(acc, ~c.reasons.any(axis=1))),
                "label_minus_exact_equals_model_error_max_gap": closed_form_gap,
                "label_error_vs_exact_mean": float(np.linalg.norm(zg - exact, axis=1).mean()),
                "correction_residual_max": float(resid.max()),
                "residual_minus_lipschitz_bound_max": bound_slack,
                "local_lipschitz_dz_max": float(lips.max()),
                "acceptance_agrees_with_exact_model": float((acc == true_accept).mean()),
                "label_model_train": {k: fit_metrics["label_model"][k] for k in ("transitions", "mean_l2", "max_l2")},
                "held_out_evaluation_model": fit_metrics.get("held_out"),
                "artifact": f"<out>/analytic/runs/{lab_dir.name}/labels/candidates.npz"}
            ok = (detail["I1_accepted_finite"] and detail["I2_accepted_in_support"] and detail["I3_accounting"]
                  and closed_form_gap <= 1e-5 and bound_slack <= 0 and acc.any() and (~acc).any())
        record("analytic_mlp_backward_euler_invariants", ok, **detail)

    # --- Pendulum held-out metrics ----------------------------------------------------
    if "pendulum" in only:
        from trajectory_io import convert_legacy_pickle  # correct_il/trajectory_io.py
        pw = out / f"pendulum_{args.device}"
        (pw / "data").mkdir(parents=True)
        sha = {x["path"]: x["sha256"] for x in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
               if x.get("path")}
        convert_legacy_pickle(REPO / "data/pendulum_cont_100.pkl", pw / "data/pendulum_cont_100.npz",
                              expect_sha256=sha["data/pendulum_cont_100.pkl"])
        r = ccil("labels", "generate", "--recipe", "pendulum_cont", "--set", "data.path=data/pendulum_cont_100.npz",
                 "--set", "run.output_dir=runs", "--set", "generation.method=backward_euler",
                 "--set", f"compute.device={args.device}", cwd=pw)
        rs = runs(pw / "runs")
        st = {n: validate_run(p)[0] for n, p in rs.items()}
        detail = {"device": args.device, "stages": st, "stderr": r.stderr[-400:]}
        ok = r.returncode == 0 and set(st.values()) == {"complete"} and len(st) == 3
        if ok:
            fm = json.loads((stage(rs, "transition-fit") / layout.METRICS).read_text())["values"]
            lm = json.loads((stage(rs, "labels-generate") / layout.METRICS).read_text())["values"]
            detail.update({"label_model": {k: fm["label_model"][k] for k in ("transitions", "mean_l2", "max_l2", "epochs")},
                           "label_model_final_losses": fm["label_model"]["final_epoch_losses"],
                           "held_out_evaluation_model": fm["held_out"], "labels": lm,
                           "settings": "recipe pendulum_cont (legacy config/pendulum_cont.yml), generation.method=backward_euler"})
            ok = np.isfinite(fm["held_out"]["mean_l2"]) and lm["accepted"] > 0
        record(f"pendulum_held_out_metrics[{args.device}]", ok, **detail)

    # --- every Lipschitz type -----------------------------------------------------------
    if "lipschitz" in only:
        rng = np.random.default_rng(5)
        s = rng.uniform(-1, 1, size=(512, 2))
        a = rng.uniform(-1, 1, size=(512, 1))
        sp = DoubleIntegrator(0.1).step_np(s, a)
        rows = {}
        for lt in LEGACY_TYPES:
            m = MLPTransitionModel([32, 32], epochs=8, batch_size=64, lipschitz_type=lt, lipschitz_constraint=1.5,
                                   soft_penalty_weight=0.01)
            fit = m.fit(s, a, sp, seed=3)
            losses = [x for v in fit["epoch_losses"].values() for x in v]
            row = {"final_losses": fit["final_epoch_losses"], "finite": bool(np.isfinite(losses).all()),
                   "train_mean_l2": fit["mean_l2"]}
            if "spectral" in lt:
                m.net.eval()
                sig = []
                with torch.no_grad():
                    for layer in m.net.fc_layers:
                        layer(torch.zeros(1, layer.in_features))  # the pre-hook recomputes the effective weight
                        sig.append(float(torch.linalg.matrix_norm(layer.weight, ord=2)))
                row["layer_spectral_norms"] = sig
                row["within_constraint"] = all(x <= 1.5 * 1.05 for x in sig)
            rows[lt] = row
        record("every_lipschitz_type_trains", all(v["finite"] and v.get("within_constraint", True) for v in rows.values()),
               constraint=1.5, types=rows)

    # --- persistence ---------------------------------------------------------------------
    if "persistence" in only:
        ck = stage(ars, "transition-fit") / "checkpoints/transition_model.npz"
        m = load_transition_model(ck)
        probe = np.random.default_rng(1).uniform(-1, 1, size=(32, 2)), np.random.default_rng(2).uniform(-1, 1, size=(32, 1))
        m.save(out / "resaved.npz")
        m2 = load_transition_model(out / "resaved.npz")
        with np.load(ck, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays["meta"]))

        def variant(name, edit):
            a2 = dict(arrays)
            edit(a2)
            np.savez(out / name, **a2)
            return out / name

        some_key = next(k for k in arrays if k.startswith("dynamics/fc_layers.0."))
        failures = {}
        for name, call in (
                ("missing_tensor", lambda: load_transition_model(variant("ck_missing.npz", lambda x: x.pop(some_key)))),
                ("wrong_shape", lambda: load_transition_model(variant("ck_shape.npz", lambda x: x.__setitem__(some_key, x[some_key][..., :1])))),
                ("other_version", lambda: load_transition_model(variant("ck_ver.npz", lambda x: x.__setitem__(
                    "meta", np.array(json.dumps({**meta, "version": "2"})))))),
                ("cuda_without_cuda", lambda: load_transition_model(ck, device="cuda"))):
            if name == "cuda_without_cuda" and torch.cuda.is_available():
                failures[name] = "skipped: CUDA is available on this host"
                continue
            try:
                call()
                failures[name] = "loaded"
            except Exception as e:  # noqa: BLE001  (recording which error type each failure surfaces as)
                failures[name] = f"{type(e).__name__}: {str(e)[:160]}"
        expected = {"missing_tensor": "RuntimeError", "wrong_shape": "RuntimeError", "other_version": "CheckpointError",
                    "cuda_without_cuda": "RuntimeError"}
        same_delta = np.array_equal(m.delta(*probe), m2.delta(*probe))
        same_bytes = ck.read_bytes() == (out / "resaved.npz").read_bytes()
        record("checkpoint_persistence", same_delta and same_bytes and meta["derived_from"]["commit"].startswith("477ddb8")
               and all(v.startswith(expected[k]) or v.startswith("skipped") for k, v in failures.items()),
               meta={k: meta[k] for k in ("component", "kind", "version", "state_dim", "act_dim", "training", "derived_from")},
               reload_delta_identical=same_delta, resave_byte_identical=same_bytes,
               failures={k: v.replace(str(out), "<out>") for k, v in failures.items()})

    # --- determinism ------------------------------------------------------------------------
    if "determinism" in only:
        rng = np.random.default_rng(8)
        s = rng.uniform(-1, 1, size=(256, 2))
        a = rng.uniform(-1, 1, size=(256, 1))
        sp = DoubleIntegrator(0.1).step_np(s, a)
        digests = {}
        for tag, seed in (("a", 4), ("b", 4), ("c", 5)):
            m = MLPTransitionModel([16, 16], epochs=4, batch_size=32, lipschitz_type="soft_sampling",
                                   lipschitz_constraint=1.5, soft_penalty_weight=0.01)
            m.fit(s, a, sp, seed=seed)
            m.save(out / f"det_{tag}.npz")
            digests[tag] = hashlib.sha256((out / f"det_{tag}.npz").read_bytes()).hexdigest()
        record("same_seed_byte_identical_checkpoints", digests["a"] == digests["b"] != digests["c"], sha256=digests)

    # --- optional extra on Python 3.10 ----------------------------------------------------------
    if "extra" in only:
        dist = out / "dist"
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=REPO, capture_output=True, text=True)
        wheel = next(dist.glob("ccil-*.whl"))
        venv = out / "venv-3.10"
        subprocess.run(["uv", "venv", "--python", "3.10", str(venv)], capture_output=True, text=True)
        subprocess.run(["uv", "pip", "install", "--python", str(venv / "bin/python"), f"{wheel}[torch]"],
                       capture_output=True, text=True)
        ew = out / "extra"
        ew.mkdir()
        double_integrator_dataset(ew / "di.npz")
        r = ccil("labels", "generate", "--recipe", "analytic_double_integrator_mlp", "--set", "data.path=di.npz",
                 "--set", "run.output_dir=runs", cwd=ew, python=venv / "bin/python")
        has_torch = subprocess.run([str(venv / "bin/python"), "-c", "import torch"], capture_output=True).returncode == 0
        record("mlp_without_torch_fails_before_any_stage", r.returncode == 2 and "ccil[torch]" in r.stderr
               and not (ew / "runs").exists() and not has_torch, exit=r.returncode, error=r.stderr.strip()[-300:],
               torch_installed=has_torch)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                              "numpy": np.__version__, "device": args.device,
                              "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              "host_load": os.getloadavg()}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    shutil.rmtree(out / "dist", ignore_errors=True)
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
