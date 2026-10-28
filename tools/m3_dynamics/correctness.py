#!/usr/bin/env python3
"""Before-and-after evidence for the #21 correctness changes (migration contract section 6).

    uv run --group evidence python tools/m3_dynamics/correctness.py --out DIR [--before REF] [--skip-pendulum]

"Before" is the code at REF (default: the #21 port, 6074e33), extracted with
git archive and run in a subprocess with its own source first on PYTHONPATH;
"after" is the working tree. Both run the same scenarios with the same seeds.

1. Eval-mode inference. A spectral-normalized model, left in training mode
   after fitting: before, repeated delta calls differ, the power-iteration
   buffers change, and labels at batch size 1 and in one batch differ well above
   float32 rounding; after, repeated calls are identical, buffers and the
   module mode are unchanged, and the batch-size difference falls to the
   float32 floor that every torch model has (set by a model without spectral
   normalization). That model gives bit-identical labels before and after.
1b. soft_jac. On a system whose residual Jacobian (<= 0.087) differs from its
   next-state Jacobian (~1), with constraint 0.5: before, the penalty sits at
   the floor ||I + df/ds|| - L (about 0.5) that no residual can remove; after,
   it is zero and the fitted residual Jacobian stays within the constraint.
   Both fits are reported.
1c. soft_sampling_slack. After, per-sample terms permute with the batch and do
   not depend on the other samples in it; before, the legacy expression
   broadcasts a (B, 1) slack against all B * n perturbations.
2. Soft-sampling penalty. On Pendulum (recipe pendulum_cont, backward_euler,
   CPU) before and after: final loss terms, held-out one-step error, labels, and
   the measured finite-difference Lipschitz ratio of the fitted residual model
   against the configured constraint.
Writes DIR/report.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BEFORE = "6074e3347f4db06ff9f25d0aec1f4646ed1feaf8"

SCENARIO = r'''
import hashlib, json, sys
import numpy as np, torch
import ccil
from ccil.core import (BackwardEuler, Finite, IdentityRepresentation, ModelResidual, SupportRadius, generate_labels,
                       load_transition_model)
from ccil.models.mlp import MLPTransitionModel
from ccil.schema import Episode, TrajectoryDataset, write_candidates

out = sys.argv[1]
rng = np.random.default_rng(5)
s, a = rng.uniform(-1, 1, (512, 2)), rng.uniform(-1, 1, (512, 1))
sp = s + np.stack([0.1 * s[:, 1], 0.1 * a[:, 0]], axis=1)
eps = []
r2 = np.random.default_rng(11)
for e in range(12):
    x = [r2.uniform(-1, 1, 2)]
    act = r2.uniform(-1, 1, (20, 1))
    for t in range(19):
        x.append(x[-1] + np.array([0.1 * x[-1][1], 0.1 * act[t, 0]]))
    eps.append(Episode(f"di{e:02d}", {"state": np.array(x, np.float32)}, act.astype(np.float32)))
ds = TrajectoryDataset(eps)
probe = np.random.default_rng(1).uniform(-1, 1, (64, 2)), np.random.default_rng(2).uniform(-1, 1, (64, 1))
result = {"ccil": ccil.__file__, "version": MLPTransitionModel.version}
for lt in ("spectral_normalization", "none"):
    m = MLPTransitionModel([32, 32], epochs=8, batch_size=64, lipschitz_type=lt, lipschitz_constraint=1.5)
    m.fit(s, a, sp, seed=3)
    ck = f"{out}/{lt}.npz"
    m.save(ck)
    bufs = lambda net: {k: v.clone() for k, v in net.named_buffers() if k.endswith(("weight_u", "weight_v"))}
    b0, mode0 = bufs(m.net), m.net.training
    d1, d2 = m.delta(*probe), m.delta(*probe)
    b1 = bufs(m.net)
    labels = {}
    for bs in (1, None):
        model = load_transition_model(ck)  # a fresh copy of the saved model for each labeling
        c = generate_labels(ds, IdentityRepresentation("state"), model, BackwardEuler(),
                            [Finite(), SupportRadius(0.05), ModelResidual()], run_id="x", seed=0, batch_size=bs)
        write_candidates(c, f"{out}/{lt}-labels-{bs}.npz")
        labels[str(bs)] = c.observations["state"]
    np.save(f"{out}/{lt}-labels.npy", labels["None"])
    result[lt] = {
        "mode_before_delta_training": bool(mode0), "mode_after_delta_training": bool(m.net.training),
        "repeated_delta_max_diff": float(np.abs(d1 - d2).max()),
        "buffers_changed_by_delta": max((float((b0[k] - b1[k]).abs().max()) for k in b0), default=0.0),
        "labels_batch1_vs_full_max_diff": float(np.abs(labels["1"] - labels["None"]).max()),
        "labels_batch1_vs_full_bytes_equal": hashlib.sha256(open(f"{out}/{lt}-labels-1.npz", "rb").read()).hexdigest()
        == hashlib.sha256(open(f"{out}/{lt}-labels-None.npz", "rb").read()).hexdigest()}
print(json.dumps(result))
'''

SOFT_JAC = r'''
import json, sys
import numpy as np, torch
from ccil.models.mlp import MLPTransitionModel
# Residual f(s, a) = 0.05 * (sin(s) + a): ||df/d(s, a)|| <= ~0.087, while the next-state map s + f has
# Jacobian I + df/ds with spectral norm ~1. The constraint L = 0.5 separates the two.
rng = np.random.default_rng(21)
s, a = rng.uniform(-1, 1, (1024, 2)), rng.uniform(-1, 1, (1024, 2))
f = lambda s, a: 0.05 * (np.sin(s) + a)
m = MLPTransitionModel([32, 32], epochs=30, batch_size=64, lipschitz_type="soft_jac", lipschitz_constraint=0.5,
                       soft_penalty_weight=1.0,
                       **({"lipschitz_coordinates": "raw"} if "lipschitz_coordinates" in MLPTransitionModel.__init__.__code__.co_varnames else {}))
fit = m.fit(s, a, s + f(s, a), seed=2)
x = torch.as_tensor(s[:256], dtype=torch.float32), torch.as_tensor(a[:256], dtype=torch.float32)
m.net.eval()
jac = torch.func.vmap(torch.func.jacrev(lambda sa: m.net.predict(sa[..., :2], sa[..., 2:])))(torch.cat(x, -1))
norms = torch.linalg.norm(jac, ord=2, dim=(-2, -1)).detach().numpy()
err = np.linalg.norm(m.delta(s, a) - f(s, a), axis=1)
print(json.dumps({"final_epoch_losses": fit["final_epoch_losses"], "residual_error_mean": float(err.mean()),
                  "residual_scale_mean": float(np.linalg.norm(f(s, a), axis=1).mean()),
                  "fitted_residual_jacobian_norm": {"mean": float(norms.mean()), "max": float(norms.max())}}))
'''

SLACK_AFTER = r'''
import json
import numpy as np, torch
from ccil.models.mlp import MLPTransitionModel, slack_sampling_terms
rng = np.random.default_rng(4)
s, a = rng.uniform(-1, 1, (256, 2)), rng.uniform(-1, 1, (256, 1))
m = MLPTransitionModel([16, 16], epochs=2, batch_size=32, lipschitz_type="soft_sampling_slack",
                       lipschitz_constraint=0.01, soft_penalty_weight=0.01,
                       **({"lipschitz_coordinates": "raw"} if "lipschitz_coordinates" in MLPTransitionModel.__init__.__code__.co_varnames else {}))
m.fit(s, a, s + 0.3 * np.tanh(s) + 0.1 * a, seed=1)
n, eps = 5, 0.0003
g = torch.Generator().manual_seed(0)
X = torch.as_tensor(s[:8], dtype=torch.float32), torch.as_tensor(a[:8], dtype=torch.float32)
Z = torch.as_tensor(s[100:104], dtype=torch.float32), torch.as_tensor(a[100:104], dtype=torch.float32)
nx = torch.randn(8 * n, 2, generator=g), torch.randn(8 * n, 1, generator=g)
nz = torch.randn(4 * n, 2, generator=g), torch.randn(4 * n, 1, generator=g)
lam = lambda s, a: m.lam.predict(s, a)
def terms(x, noise):
    with torch.no_grad():
        return slack_sampling_terms(x, m.net.predict, lam, 0.01, eps, n, noise)[0].numpy()
base = terms(X, nx)
perm = torch.as_tensor(np.random.default_rng(3).permutation(8))
rows = (perm[:, None] * n + torch.arange(n)).reshape(-1)
permuted = terms((X[0][perm], X[1][perm]), (nx[0][rows], nx[1][rows]))
mixed = terms((torch.cat([X[0][:4], Z[0]]), torch.cat([X[1][:4], Z[1]])),
              (torch.cat([nx[0][:4 * n], nz[0]]), torch.cat([nx[1][:4 * n], nz[1]])))
scale = float(np.abs(base).max())
print(json.dumps({"terms": base.tolist(), "nonzero_terms": int((base != 0).sum()),
                  "permutation_max_diff": float(np.abs(permuted - base[perm.numpy()]).max()),
                  "composition_max_diff": float(np.abs(mixed[:4] - base[:4]).max()), "scale": scale}))
'''

SLACK_BEFORE = r'''
import json
import torch
from ccil.models import mlp
# The legacy expression: slack (B, 1) times the per-perturbation hinge (B * n,).
B, n = 8, 5
print(json.dumps({"has_paired_terms": hasattr(mlp, "slack_sampling_terms"),
                  "legacy_product_shape": list((torch.zeros(B, 1) * torch.zeros(B * n)).shape)}))
'''

LIPSCHITZ = r'''
import json, sys
import numpy as np
from ccil.core import load_transition_model
from ccil.schema import read_trajectories
ck, dataset = sys.argv[1], sys.argv[2]
model = load_transition_model(ck, device="cpu")
ds = read_trajectories(dataset)
s = np.concatenate([e.observations["state"][:-1] for e in ds.episodes]).astype(np.float64)
a = np.concatenate([e.actions[:-1] for e in ds.episodes]).astype(np.float64)
idx = np.random.default_rng(0).choice(len(s), 2000, replace=False)
s, a = s[idx], a[idx]
rng = np.random.default_rng(1)
ratios = []
for _ in range(5):  # the training penalty's estimator: eps = 3e-4, state and action perturbed together
    ds_, da_ = rng.normal(size=s.shape) * 3e-4, rng.normal(size=a.shape) * 3e-4
    diff = np.linalg.norm(model.delta(s + ds_, a + da_) - model.delta(s, a), axis=1)
    ratios.append(diff / np.linalg.norm(np.hstack([ds_, da_]), axis=1))
r = np.concatenate(ratios)
print(json.dumps({"p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95)), "max": float(r.max()),
                  "fraction_above_constraint": float((r > 2.0).mean())}))
'''


def env_for(root):
    env = dict(os.environ)
    if root is not None:
        env["PYTHONPATH"] = str(root / "src")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--before", default=BEFORE)
    ap.add_argument("--skip-pendulum", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:800]}", flush=True)

    before_root = out / "before_code"
    before_root.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", args.before, "src"],
                             capture_output=True, check=True).stdout
    tar = out / "before.tar"
    tar.write_bytes(archive)
    with tarfile.open(tar) as t:
        t.extractall(before_root, filter="data")
    tar.unlink()
    roots = {"before": before_root, "after": None}

    # 1. Eval-mode inference.
    res = {}
    for tag, root in roots.items():
        d = out / f"scenario_{tag}"
        d.mkdir()
        r = subprocess.run([sys.executable, "-c", SCENARIO, str(d)], env=env_for(root), capture_output=True, text=True)
        res[tag] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-500:]}
    ok = all("error" not in v for v in res.values())
    if ok:
        b, a = res["before"]["spectral_normalization"], res["after"]["spectral_normalization"]
        sn_diff = np.abs(np.load(out / "scenario_before/spectral_normalization-labels.npy")
                         - np.load(out / "scenario_after/spectral_normalization-labels.npy"))
        none_equal = np.array_equal(np.load(out / "scenario_before/none-labels.npy"),
                                    np.load(out / "scenario_after/none-labels.npy"))
        # Batch shape changes float32 matmul rounding on any torch model, so the non-spectral-norm model sets
        # the floor; after the fix a spectral-norm model's batch dependence falls to that floor.
        floor = res["after"]["none"]["labels_batch1_vs_full_max_diff"]
        ok = (str(before_root) in res["before"]["ccil"] and str(before_root) not in res["after"]["ccil"]
              and b["repeated_delta_max_diff"] > 0 and b["buffers_changed_by_delta"] > 0
              and b["labels_batch1_vs_full_max_diff"] > 10 * max(floor, 1e-12)
              and a["repeated_delta_max_diff"] == 0 and a["buffers_changed_by_delta"] == 0
              and a["mode_after_delta_training"] == a["mode_before_delta_training"]
              and a["labels_batch1_vs_full_max_diff"] <= max(floor, 1e-7) and none_equal)
        res["float32_batch_floor"] = floor
        res["spectral_normalization_labels_before_vs_after"] = {"max_abs": float(sn_diff.max()),
                                                                "mean_abs": float(sn_diff.mean())}
        res["no_spectral_norm_labels_bit_identical_before_after"] = bool(none_equal)
    for v in res.values():
        if isinstance(v, dict) and "ccil" in v:
            v["ccil"] = v["ccil"].replace(str(out), "<out>").replace(str(REPO), "<repo>")
    record("eval_mode_inference_before_after", ok, before_ref=args.before, results=res)

    # 1b. soft_jac: the residual Jacobian, not the next-state Jacobian.
    jac = {}
    for tag, root in roots.items():
        r = subprocess.run([sys.executable, "-c", SOFT_JAC], env=env_for(root), capture_output=True, text=True)
        jac[tag] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-400:]}
    ok = all("error" not in v for v in jac.values())
    if ok:
        b, a = jac["before"], jac["after"]
        # Before, the penalty measures ||I + df/ds|| - L, a floor near 1 - L = 0.5 that no residual can remove;
        # after, it measures the residual's own Jacobian, which this system keeps within L.
        ok = (a["final_epoch_losses"]["lipschitz_penalty"] == 0 and a["fitted_residual_jacobian_norm"]["max"] <= 0.5
              and b["final_epoch_losses"]["lipschitz_penalty"] >= 0.4)
    record("soft_jac_constrains_residual_jacobian_before_after", ok, constraint=0.5, penalty_weight=1.0,
           system="f(s, a) = 0.05 (sin s + a): ||df/d(s,a)|| <= 0.087, ||I + df/ds|| ~ 1", results=jac)

    # 1c. soft_sampling_slack: each sample's slack pairs only with its own perturbations.
    r_after = subprocess.run([sys.executable, "-c", SLACK_AFTER], env=env_for(None), capture_output=True, text=True)
    r_before = subprocess.run([sys.executable, "-c", SLACK_BEFORE], env=env_for(roots["before"]), capture_output=True,
                              text=True)
    after = json.loads(r_after.stdout.strip().splitlines()[-1]) if r_after.returncode == 0 else {"error": r_after.stderr[-400:]}
    before = json.loads(r_before.stdout.strip().splitlines()[-1]) if r_before.returncode == 0 else {"error": r_before.stderr[-400:]}
    tol = 1e-6 * max(after.get("scale", 0.0), 1e-12)
    record("soft_sampling_slack_pairs_samples_with_their_perturbations",
           "error" not in after and "error" not in before and after["nonzero_terms"] > 0
           and after["permutation_max_diff"] <= tol and after["composition_max_diff"] <= tol
           and before["legacy_product_shape"] == [8, 40] and not before["has_paired_terms"],
           tolerance=tol, after=after, before=before)

    # 2. Soft-sampling penalty on Pendulum.
    if not args.skip_pendulum:
        sys.path.insert(0, str(REPO / "correct_il"))
        from trajectory_io import convert_legacy_pickle  # correct_il/trajectory_io.py
        sha = {x["path"]: x["sha256"] for x in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
               if x.get("path")}
        pend = {}
        for tag, root in roots.items():
            w = out / f"pendulum_{tag}"
            (w / "data").mkdir(parents=True)
            convert_legacy_pickle(REPO / "data/pendulum_cont_100.pkl", w / "data/pendulum_cont_100.npz",
                                  expect_sha256=sha["data/pendulum_cont_100.pkl"])
            r = subprocess.run([sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "pendulum_cont",
                                "--set", "data.path=data/pendulum_cont_100.npz", "--set", "run.output_dir=runs",
                                "--set", "generation.method=backward_euler", "--set", "compute.device=cpu"],
                               cwd=w, env=env_for(root), capture_output=True, text=True)
            if r.returncode != 0:
                pend[tag] = {"error": r.stderr[-500:]}
                continue
            rs = {p.name: p for p in (w / "runs").iterdir() if p.is_dir()}
            fit = next(p for n, p in rs.items() if n.startswith("transition-fit"))
            lab = next(p for n, p in rs.items() if n.startswith("labels-generate"))
            prep = next(p for n, p in rs.items() if n.startswith("data-prepare"))
            fm = json.loads((fit / "metrics.json").read_text())["values"]
            lm = json.loads((lab / "metrics.json").read_text())["values"]
            lip = subprocess.run([sys.executable, "-c", LIPSCHITZ, str(fit / "checkpoints/transition_model.npz"),
                                  str(prep / "data/dataset.npz")], env=env_for(root), capture_output=True, text=True)
            with np.load(lab / "labels/candidates.npz", allow_pickle=False) as z:
                np.save(out / f"pendulum_{tag}_labels.npy", z["obs/state"])
            pend[tag] = {"final_epoch_losses": fm["label_model"]["final_epoch_losses"],
                         "label_model_train_mean_l2": fm["label_model"]["mean_l2"],
                         "held_out_mean_l2": fm["held_out"]["mean_l2"], "held_out_max_l2": fm["held_out"]["max_l2"],
                         "labels_accepted": lm["accepted"], "labels": lm["candidates"],
                         "residual_lipschitz_ratio": json.loads(lip.stdout) if lip.returncode == 0 else lip.stderr[-300:]}
        ok = all("error" not in v for v in pend.values())
        if ok:
            d = np.linalg.norm(np.load(out / "pendulum_before_labels.npy") - np.load(out / "pendulum_after_labels.npy"), axis=1)
            pend["label_shift_before_to_after"] = {"mean_l2": float(d.mean()), "max_l2": float(d.max())}
            ok = all(np.isfinite(pend[t]["held_out_mean_l2"]) for t in roots)
        record("soft_sampling_penalty_before_after_pendulum", ok, constraint=2.0, penalty_weight=0.001,
               sampling_eps=0.0003, results=pend)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    import torch
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"), "before_commit": args.before,
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
