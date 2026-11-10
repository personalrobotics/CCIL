#!/usr/bin/env python3
"""Evidence for batched transition-model fits (#110).

    uv run --group evidence python tools/m3_fit_v2/check.py --out DIR [--device cuda] [--parts all]

1. correctness (any host): for none and slack, one full-batch step (order-free) equals
   MLPTransitionModel.fit's step from the same initialization (float32 tolerance); for soft_sampling and
   soft_sampling_slack, with the same injected perturbations and the constraint at half the fixture's median
   ratio, so the hinge is active,
   the two engines give the same loss and gradients (float64); a rerun is identical; fit_group refuses the
   regularizers it does not support, and so does preflight for trainer_version 2; compute.deterministic is
   honored, recorded and restored.
2. stage_hooks (any host): fit_transition_stage(prefit=...) accepts a group's fits for their own unit and
   refuses another unit's (a swapped fit), a model whose weights are not its record's, and fits run differently
   from the unit's request (a unit asking for compute.deterministic given non-deterministic fits).
3. lipschitz (any host; #125): the model_normalized sampled Lipschitz ratio
   - equals a direct computation of ||g(x + dx) - g(x)|| / ||dx|| from the state dict (x the normalized input,
     g the normalized residual, dx = eps * N(0, I) in normalized space);
   - is invariant to a change of units (states, actions and residuals rescaled per coordinate, normalization
     refitted, the same weights), while the raw ratio scales with the units;
   - gives the same penalty, and the same gradients, in both engines from the same perturbations, for
     soft_sampling and soft_sampling_slack;
   - is recorded (coordinates, units, normalization hash) in fit records, and a checkpoint saved before #125
     (no lipschitz_coordinates in its settings) loads as raw; soft_jac refuses model_normalized.
4. clean_holdout (any host): on 20 double-integrator episodes (recipe analytic_double_integrator_mlp, 2 dynamics
   epochs), the label pipeline runs on the data and on a copy whose held-out episodes (the data.prepare split
   at seed 501) are perturbed. With transition_model.fit_episodes train, the label model and every label from a
   training episode are identical between the two (held-out episodes influence no label); with all, they
   differ (the check can see the leak it rules out).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))

from ccil.models.mlp import MLPTransitionModel  # noqa: E402
from ccil.models.mlp_batched import FitJob, fit_group  # noqa: E402

DATA_SEED = 501


def _data(n, seed):
    r = np.random.default_rng(seed)
    z, a = r.normal(size=(n, 3)), r.normal(size=(n, 1))
    return z, a, z + 0.05 * np.tanh(z @ r.normal(size=(3, 3)) + a) + 0.001 * r.normal(size=z.shape)


BASE = dict(hidden_layers=[64, 64], epochs=1, batch_size=512, activation="relu", lipschitz_constraint=2.0,
            soft_penalty_weight=0.001, soft_sampling_eps=0.0003, soft_samples=5, slack_weight=0.1,
            learning_rate=1e-3, weight_decay=1e-5)


def activated_references() -> dict:
    """v1's and v2's sampled Lipschitz losses and gradients from the same perturbations, hinge active (float64)."""
    from ccil.models import mlp as mlp_mod
    from ccil.models import mlp_batched as mb
    out, g = {}, torch.Generator().manual_seed(0)
    n, eps = BASE["soft_samples"], BASE["soft_sampling_eps"]
    for lt in ("soft_sampling", "soft_sampling_slack"):
        z, a, zn = _data(2048, 6)
        mm = MLPTransitionModel(**{**BASE, "lipschitz_type": lt, "epochs": 2})
        mm.fit(z, a, zn, seed=6)
        mm.net.double()
        if mm.lam is not None:
            mm.lam.double()
        s_, a_, sp_ = (torch.as_tensor(x[:128], dtype=torch.float64) for x in (z, a, zn))
        nz = torch.randn((128 * n, 4), generator=g, dtype=torch.float64)
        with torch.no_grad():  # the constraint at half the fixture's median ratio, so the hinge is active
            L = 0.5 * float(mlp_mod.lipschitz_ratio(s_, a_, mm.net.predict, n, eps, noise=(nz[:, :3], nz[:, 3:])).median())
        trainable = {"net": {k: v.detach().clone().requires_grad_(True) for k, v in mm.net.named_parameters()}}
        buffers = {"net": {k: v.clone() for k, v in mm.net.named_buffers()}}
        if mm.lam is not None:
            trainable["lam"] = {k: v.detach().clone().requires_grad_(True) for k, v in mm.lam.named_parameters()}
            buffers["lam"] = {k: v.clone() for k, v in mm.lam.named_buffers()}
        k = {"layers": len(BASE["hidden_layers"]) + 1, "relu": True, "lipschitz_type": lt, "lipschitz_constraint": L,
             "soft_penalty_weight": 1.0, "soft_sampling_eps": eps, "soft_samples": n, "slack_weight": BASE["slack_weight"],
             "coordinates": "raw"}
        loss2, terms2 = mb._loss(trainable, buffers, s_, a_, sp_, nz, k)
        g2 = torch.autograd.grad(loss2, [v for d in trainable.values() for v in d.values()])
        y = (sp_ - s_ - mm.net.out_shift) / (mm.net.out_scale + 1e-8)
        if lt == "soft_sampling":
            ratio = mlp_mod.lipschitz_ratio(s_, a_, mm.net.predict, n, eps, noise=(nz[:, :3], nz[:, 3:]))
            pen = torch.relu(ratio - L).mean()
        else:
            terms, slack = mlp_mod.slack_sampling_terms((s_, a_), mm.net.predict, lambda s0, a0: mm.lam.predict(s0, a0),
                                                        L, eps, n, noise=(nz[:, :3], nz[:, 3:]))
            pen = terms.mean() + 1 - torch.exp(-0.1 * torch.abs(slack).mean())
        loss1 = torch.nn.functional.mse_loss(mm.net(s_, a_), y) + pen
        g1 = torch.autograd.grad(loss1, list(mm.net.parameters()) + (list(mm.lam.parameters()) if mm.lam else []))
        dl = abs(float(loss2) - float(loss1)) / abs(float(loss1))
        dg = max(float((x - y_).abs().max() / (y_.abs().max() + 1e-12)) for x, y_ in zip(g2, g1))
        active = float((ratio > L).float().mean()) if lt == "soft_sampling" else None
        out[f"activated_reference_{lt}"] = {"pass": dl < 1e-9 and dg < 1e-8 and float(pen) > 0, "dtype": "float64",
                                            "penalty": float(pen), "loss_relative_difference": dl,
                                            "gradient_max_relative_difference": dg, "active_share": active,
                                            "constraint": L, "note": "constraint at half the median ratio"}
    return out


def determinism_and_preflight() -> dict:
    from ccil.config import ConfigError, resolve
    from ccil.pipelines import offline
    z, a, zn = _data(1024, 2)
    before = torch.are_deterministic_algorithms_enabled()
    [(_, rec)] = fit_group([FitJob(z, a, zn, 1)], {**BASE, "lipschitz_type": "none"}, device="cpu", compile=False,
                           deterministic=True)
    restored = torch.are_deterministic_algorithms_enabled() == before
    try:
        offline.preflight(resolve(recipe="pendulum_cont", sets=["data.path=x.npz", "transition_model.trainer_version=2",
                                                                "transition_model.lipschitz_type=soft_jac"]))
        refused = None
    except ConfigError as e:
        refused = str(e)
    return {"pass": rec["group"]["deterministic"] is True and restored and refused is not None
            and "trainer_version" in refused, "recorded": rec["group"]["deterministic"], "flag_restored": restored,
            "preflight_refusal": refused}


def _double_integrator(path, episodes=20, frames=26, seed=11):
    """A small double-integrator dataset (tools/analytic_fixture/system.py), uniform random states and actions."""
    from system import DoubleIntegrator

    from ccil.schema import Episode, TrajectoryDataset, write_trajectories
    sys_, rng, eps = DoubleIntegrator(0.1), np.random.default_rng(seed), []
    for e in range(episodes):
        s = [rng.uniform(-1, 1, size=2)]
        a = rng.uniform(-1, 1, size=(frames, 1))
        for t in range(frames - 1):
            s.append(sys_.step_np(s[-1][None], a[t][None])[0])
        eps.append(Episode(f"di{e:02d}", {"state": np.array(s, np.float32)}, a.astype(np.float32)))
    write_trajectories(TrajectoryDataset(eps), path)


def _rows(prepared, rep):
    """(z, a, z_next, train mask) of a prepared stage, as fit_transition_stage reads them (forward direction)."""
    from ccil.core import transitions
    from ccil.schema import read_trajectories
    b = transitions(read_trajectories(prepared.out("data/dataset.npz")), rep)
    split = json.loads(prepared.out("data/split.json").read_text())
    return b.z, b.a, b.z_next, np.isin(b.episode_id, list(split["train"]))


def stage_hooks(out: Path) -> dict:
    """A group's fits are accepted for their own unit and refused for another's."""
    from ccil.config import resolve
    from ccil.core import IdentityRepresentation
    from ccil.data.prepare import prepare_stage
    from ccil.pipelines import offline
    root = out / "hooks"
    root.mkdir()
    data = root / "small.npz"
    _double_integrator(data)

    def cfg(seed, deterministic=False):
        return resolve(recipe="analytic_double_integrator_mlp", sets=[
            f"run.seed={seed}", "compute.device=cpu", f"data.path={data}",
            f"compute.deterministic={str(deterministic).lower()}", f"run.output_dir={root / 'stages'}",
            "transition_model.epochs=1", "transition_model.trainer_version=2"])
    c1, c2 = cfg(1), cfg(2)
    rep = IdentityRepresentation("state")
    fits, prepared = {}, {}
    for c in (c1, c2):
        p = prepare_stage(c)
        prepared[c["run.seed"]] = p
        zz, aa, zn, tr = _rows(p, rep)
        settings = dict(offline.build_model(c.values).settings)
        out_ = fit_group([FitJob(zz, aa, zn, c["run.seed"]), FitJob(zz[tr], aa[tr], zn[tr], c["run.seed"])], settings,
                         device="cpu", compile=False)
        fits[c["run.seed"]] = {"label": out_[0], "evaluation": out_[1]}
    res = {}

    def refused(call, needle):
        try:
            call()
            return False
        except Exception as e:  # noqa: BLE001 - the stage wraps the refusal
            return needle in repr(e)
    wrong_model = {role: (fits[2][role][0], fits[1][role][1]) for role in fits[1]}  # seed 2's models, seed 1's records
    c1_det = cfg(1, deterministic=True)  # the same unit, asking for deterministic kernels the fits did not use
    res["swapped_fit_refused"] = refused(lambda: offline.fit_transition_stage(c1, prepared[1], rep, prefit=fits[2]),
                                         "fit identity")
    res["wrong_model_right_record_refused"] = refused(
        lambda: offline.fit_transition_stage(c1, prepared[1], rep, prefit=wrong_model), "not the model its record")
    res["wrong_execution_right_record_refused"] = refused(
        lambda: offline.fit_transition_stage(c1_det, prepare_stage(c1_det), rep, prefit=fits[1]), "deterministic")
    offline.fit_transition_stage(c1, prepared[1], rep, prefit=fits[1])
    res["own_fit_accepted"] = True
    return {"pass": all(res.values()), **res}


def correctness(device):
    out = {}
    z, a, zn = _data(512, 0)
    for lt in ("none", "slack"):
        st = {**BASE, "lipschitz_type": lt}
        ref = MLPTransitionModel(**st, device=device)
        ref.fit(z, a, zn, seed=3)
        [(m, _)] = fit_group([FitJob(z, a, zn, 3)], st, device=device, compile=False)
        nets = [(m.net, ref.net)] + ([(m.lam, ref.lam)] if lt == "slack" else [])
        d = max(float((x - y).abs().max() / (y.abs().max() + 1e-12)) for p, q in nets
                for x, y in zip(p.state_dict().values(), q.state_dict().values()))
        out[f"one_step_matches_fit_{lt}"] = {"pass": d < 1e-5, "max_relative_difference": d}
    z, a, zn = _data(4096, 1)
    st = {**BASE, "lipschitz_type": "soft_sampling", "epochs": 2}
    g1 = fit_group([FitJob(z, a, zn, s) for s in (1, 2, 3)], st, device=device)
    g2 = fit_group([FitJob(z, a, zn, s) for s in (1, 2, 3)], st, device=device)
    same = all(torch.equal(x, y) for (m1, _), (m2, _) in zip(g1, g2)
               for x, y in zip(m1.net.state_dict().values(), m2.net.state_dict().values()))
    out["rerun_identical"] = {"pass": same}
    try:
        fit_group([FitJob(z, a, zn, 1)], {**BASE, "lipschitz_type": "soft_jac"}, device=device)
        refused = False
    except ValueError:
        refused = True
    out["refuses_unsupported_regularizer"] = {"pass": refused}
    return out


def clean_holdout(out: Path, device: str) -> dict:
    import dataclasses

    from ccil.config import resolve
    from ccil.pipelines import offline
    from ccil.schema import read_candidates, read_trajectories, write_trajectories
    from ccil.schema.trajectories import TrajectoryDataset
    root = out / "holdout"
    root.mkdir()
    paths = {"original": root / "original.npz", "perturbed": root / "perturbed.npz"}
    _double_integrator(paths["original"])
    small = read_trajectories(paths["original"])

    def run(which, fit_episodes):
        cfg = resolve(recipe="analytic_double_integrator_mlp", sets=[
            f"run.seed={DATA_SEED}", f"compute.device={device}", f"data.path={paths[which]}",
            f"run.output_dir={root / which / fit_episodes}", "transition_model.epochs=2",
            f"transition_model.fit_episodes={fit_episodes}"])
        prepared, fitted, labeled = offline.generate_offline(cfg)[:3]
        split = json.loads(prepared.out("data/split.json").read_text())
        with np.load(fitted.out(offline.MODEL_CHECKPOINT), allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k.startswith(("dynamics/", "lambda/"))}
        return split, params, read_candidates(labeled.out("labels/candidates.npz"))
    split, _, _ = run("original", "train")
    held = set(split["validation"])
    r = np.random.default_rng(0)
    eps = [dataclasses.replace(ep, observations={k: (x * 1.05 + 0.01 * r.standard_normal(x.shape)).astype(x.dtype)
                                                for k, x in ep.observations.items()})
           if ep.episode_id in held else ep for ep in small.episodes]
    write_trajectories(TrajectoryDataset(eps, small.meta), paths["perturbed"])
    out_ = {}
    for fit_episodes in ("train", "all"):
        s0, p0, c0 = run("original", fit_episodes)
        s1, p1, c1 = run("perturbed", fit_episodes)
        tr0 = ~np.isin(c0.source_episode_id, list(held))
        tr1 = ~np.isin(c1.source_episode_id, list(held))
        same_rows = np.array_equal(c0.source_episode_id[tr0], c1.source_episode_id[tr1]) and \
            np.array_equal(c0.source_frame[tr0], c1.source_frame[tr1])
        labels_equal = same_rows and all(np.array_equal(a_, b_) for a_, b_ in (
            (c0.actions[tr0], c1.actions[tr1]), (c0.accepted[tr0], c1.accepted[tr1]),
            (c0.observations["state"][tr0], c1.observations["state"][tr1]),
            (c0.target_observations["state"][tr0], c1.target_observations["state"][tr1])))
        out_[fit_episodes] = {"split_unchanged": s0 == s1, "held_out_episodes": sorted(held),
                              "label_model_equal": all(np.array_equal(p0[k], p1[k]) for k in p0) and set(p0) == set(p1),
                              "training_episode_labels_equal": bool(labels_equal),
                              "training_episode_labels": int(tr0.sum())}
    t, a_ = out_["train"], out_["all"]
    ok = (t["split_unchanged"] and t["label_model_equal"] and t["training_episode_labels_equal"]
          and not a_["label_model_equal"] and not a_["training_episode_labels_equal"])
    return {"pass": ok, **out_}


def _fitted(lt, coordinates, seed=5):
    z, a, zn = _data(2048, seed)
    m = MLPTransitionModel(**{**BASE, "lipschitz_type": lt, "epochs": 3, "lipschitz_coordinates": coordinates})
    m.fit(z, a, zn, seed=seed)
    return m, z, a, zn


def _reference_ratio(sd: dict, s, a, noise, eps, relu=True):
    """||g(x + dx) - g(x)|| / ||dx|| written out from the state dict, float64."""
    sd = {k: v.double() for k, v in sd.items()}
    x = torch.cat([(s - sd["s_shift"]) / (sd["s_scale"] + 1e-8), (a - sd["a_shift"]) / (sd["a_scale"] + 1e-8)], -1)
    layers = len([k for k in sd if k.startswith("fc_layers.") and k.endswith(".weight")])

    def g(v):
        for i in range(layers):
            v = v @ sd[f"fc_layers.{i}.weight"].T + sd[f"fc_layers.{i}.bias"]
            if i < layers - 1:
                v = torch.relu(v) if relu else torch.tanh(v)
        return v
    n = noise.shape[0] // len(s)
    xr = x.repeat_interleave(n, 0)
    dx = noise * eps
    return torch.linalg.norm(g(xr + dx) - g(xr), dim=-1) / torch.linalg.norm(dx, dim=-1)


def lipschitz_checks() -> dict:
    from ccil.models import mlp as mlp_mod
    from ccil.models import mlp_batched as mb
    out = {}
    eps, n = BASE["soft_sampling_eps"], BASE["soft_samples"]
    # In float64: the ratio divides a difference of O(1) outputs by a 3e-4 step, which amplifies float32 rounding
    # about 3,000-fold; float64 tests the definition rather than the precision.
    m, z, a, zn = _fitted("soft_sampling", "model_normalized")
    m.net.double()
    s, act = torch.as_tensor(z[:256], dtype=torch.float64), torch.as_tensor(a[:256], dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn((256 * n, 4), generator=g, dtype=torch.float64)
    with torch.no_grad():
        got = mlp_mod.lipschitz_ratio(s, act, m.net.predict, n, eps, noise=(noise[:, :3], noise[:, 3:]),
                                      coordinates="model_normalized", net=m.net)
    ref = _reference_ratio(dict(m.net.state_dict()), s.double(), act.double(), noise.double(), eps)
    rel = float(((got - ref).abs() / ref.abs().clamp_min(1e-12)).max())
    out["definition"] = {"pass": rel < 1e-9, "max_relative_difference": rel, "dtype": "float64"}

    # A change of units: per-coordinate scales of state (and therefore residual) and action.
    cs, ca = torch.tensor([10.0, 0.1, 3.0], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64)
    m2, _, _, _ = _fitted("soft_sampling", "model_normalized")
    m2.net.double()
    m2.net.load_state_dict(m.net.state_dict())
    sh = {k: v.clone() for k, v in m.net.named_buffers()}
    m2.net.set_transformations(s_shift=sh["s_shift"] * cs, s_scale=sh["s_scale"] * cs, a_shift=sh["a_shift"] * ca,
                               a_scale=sh["a_scale"] * ca, out_shift=sh["out_shift"] * cs, out_scale=sh["out_scale"] * cs)
    with torch.no_grad():
        r_n = [mlp_mod.lipschitz_ratio(ss, aa, mm.net.predict, n, eps, noise=(noise[:, :3], noise[:, 3:]),
                                       coordinates="model_normalized", net=mm.net)
               for mm, ss, aa in ((m, s, act), (m2, s * cs, act * ca))]
        r_raw = [mlp_mod.lipschitz_ratio(ss, aa, mm.net.predict, n, eps, noise=(noise[:, :3], noise[:, 3:]))
                 for mm, ss, aa in ((m, s, act), (m2, s * cs, act * ca))]
    # Exact up to the normalization's absolute 1e-8 in (scale + 1e-8), which shifts normalized inputs by up to
    # about 1e-8 / scale after rescaling and can tip a sample lying at a ReLU kink: the bulk is tested tightly.
    rel = ((r_n[0] - r_n[1]).abs() / r_n[0].clamp_min(1e-12)).sort().values
    q = {"median": float(rel.median()), "p99": float(rel[int(0.99 * len(rel))]), "max": float(rel[-1])}
    raw_factor = float((r_raw[1] / r_raw[0]).median())
    out["rescaling_invariance"] = {"pass": q["median"] <= 1e-9 and q["p99"] <= 1e-5 and q["max"] <= 1e-3
                                   and abs(raw_factor - 1) > 0.5, "dtype": "float64",
                                   "model_normalized_relative_change": q,
                                   "raw_median_ratio_of_ratios": raw_factor,
                                   "units": {"state": cs.tolist(), "action": ca.tolist()}}

    # The engines: the same penalty and gradients from the same perturbations.
    for lt in ("soft_sampling", "soft_sampling_slack"):
        mm, z_, a_, zn_ = _fitted(lt, "model_normalized", seed=6)
        mm.net.double()
        if mm.lam is not None:
            mm.lam.double()
        s_, aa_, sp_ = (torch.as_tensor(x[:128], dtype=torch.float64) for x in (z_, a_, zn_))
        nz = torch.randn((128 * n, 4), generator=g, dtype=torch.float64)
        trainable = {"net": {k: v.detach().clone().requires_grad_(True) for k, v in mm.net.named_parameters()}}
        buffers = {"net": {k: v.clone() for k, v in mm.net.named_buffers()}}
        if mm.lam is not None:
            trainable["lam"] = {k: v.detach().clone().requires_grad_(True) for k, v in mm.lam.named_parameters()}
            buffers["lam"] = {k: v.clone() for k, v in mm.lam.named_buffers()}
        k = {"layers": len(BASE["hidden_layers"]) + 1, "relu": True, "lipschitz_type": lt,
             "lipschitz_constraint": 0.05, "soft_penalty_weight": 1.0, "soft_sampling_eps": eps, "soft_samples": n,
             "slack_weight": BASE["slack_weight"], "coordinates": "model_normalized"}
        loss2, terms2 = mb._loss(trainable, buffers, s_, aa_, sp_, nz, k)
        g2 = torch.autograd.grad(loss2, [v for d in trainable.values() for v in d.values()])
        mm.net.zero_grad()
        if mm.lam is not None:
            mm.lam.zero_grad()
        y = (sp_ - s_ - mm.net.out_shift) / (mm.net.out_scale + 1e-8)
        lam_fn = (lambda s0, a0: mm.lam.predict(s0, a0)) if mm.lam is not None else None
        if lt == "soft_sampling":
            ratio = mlp_mod.lipschitz_ratio(s_, aa_, mm.net.predict, n, eps, noise=(nz[:, :3], nz[:, 3:]),
                                            coordinates="model_normalized", net=mm.net)
            pen = 1.0 * torch.relu(ratio - 0.05).mean()
        else:
            terms, slack = mlp_mod.slack_sampling_terms((s_, aa_), mm.net.predict, lam_fn, 0.05, eps, n,
                                                        noise=(nz[:, :3], nz[:, 3:]), coordinates="model_normalized",
                                                        net=mm.net)
            pen = 1.0 * (terms.mean() + 1 - torch.exp(-0.1 * torch.abs(slack).mean()))
        loss1 = torch.nn.functional.mse_loss(mm.net(s_, aa_), y) + pen
        params1 = list(mm.net.parameters()) + (list(mm.lam.parameters()) if mm.lam is not None else [])
        g1 = torch.autograd.grad(loss1, params1)
        dpen = abs(float(terms2["lipschitz_penalty"]) - float(pen)) / max(abs(float(pen)), 1e-12)
        dgrad = max(float((x - y_).abs().max() / (y_.abs().max() + 1e-12)) for x, y_ in zip(g2, g1))
        out[f"engines_agree_{lt}"] = {"pass": dpen < 1e-9 and dgrad < 1e-8 and float(pen) > 0, "dtype": "float64",
                                      "penalty": float(pen), "penalty_relative_difference": dpen,
                                      "gradient_max_relative_difference": dgrad,
                                      "note": "constraint lowered to 0.05 so the hinge is active"}

    # Records, checkpoint descriptors, old checkpoints and refusals.
    import tempfile
    m, *_ = _fitted("soft_sampling", "model_normalized")
    rec = m.fit(z, a, zn, seed=5)
    [(_, rec2)] = fit_group([FitJob(z, a, zn, 5)], dict(m.settings), device="cpu", compile=False)

    def rewrite(path, change_meta=None, change_arrays=None):
        with np.load(path, allow_pickle=False) as f:
            arrays = {k_: f[k_] for k_ in f.files}
        meta = json.loads(str(arrays["meta"]))
        if change_meta:
            change_meta(meta)
        if change_arrays:
            change_arrays(arrays)
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez(path, **arrays)

    with tempfile.TemporaryDirectory() as td:
        new_path, old_path, bad_path = (Path(td) / f"{n}.npz" for n in ("new", "old", "bad"))
        m.save(new_path)
        stored = json.loads(str(np.load(new_path, allow_pickle=False)["meta"])).get("lipschitz")
        reloaded = MLPTransitionModel.load(new_path).settings["lipschitz_coordinates"]
        mr, *_ = _fitted("soft_sampling", "raw")
        mr.save(old_path)  # as written before #125: no coordinates in its settings and no descriptor

        def strip(meta):
            del meta["settings"]["lipschitz_coordinates"]
            meta.pop("lipschitz", None)
        rewrite(old_path, change_meta=strip)
        old = MLPTransitionModel.load(old_path)
        m.save(bad_path)  # a descriptor that no longer matches the weights' normalization

        def shift(arrays):
            arrays["dynamics/s_scale"] = arrays["dynamics/s_scale"] * 1.01
        rewrite(bad_path, change_arrays=shift)
        try:
            MLPTransitionModel.load(bad_path)
            tampered_refused = False
        except ValueError:
            tampered_refused = True
    try:
        MLPTransitionModel(**{**BASE, "lipschitz_type": "soft_jac", "lipschitz_coordinates": "model_normalized"})
        jac_refused = False
    except ValueError:
        jac_refused = True
    default = MLPTransitionModel(**{**BASE, "lipschitz_type": "soft_sampling"}).settings["lipschitz_coordinates"]
    recorded = all(r.get("lipschitz", {}).get("coordinates") == "model_normalized"
                   and len(r["lipschitz"]["normalization_sha256"]) == 64 for r in (rec, rec2))
    out["records_and_old_checkpoints"] = {
        "pass": recorded and stored == rec["lipschitz"] and reloaded == "model_normalized"
        and old.settings["lipschitz_coordinates"] == "raw" and tampered_refused and jac_refused
        and default == "model_normalized",
        "recorded": recorded, "checkpoint_descriptor": stored, "checkpoint_reloads_as": reloaded,
        "old_checkpoint_reads_as": old.settings["lipschitz_coordinates"],
        "mismatched_normalization_refused": tampered_refused, "soft_jac_model_normalized_refused": jac_refused,
        "constructor_default": default}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parts", default="correctness")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parts = (["correctness", "stage_hooks", "clean_holdout", "lipschitz"] if a.parts == "all" else a.parts.split(","))
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)
    if "correctness" in parts:
        for name, r in {**correctness(a.device), **activated_references()}.items():
            record(f"correctness_{name}", r.pop("pass"), **r)
        r = determinism_and_preflight()
        record("correctness_determinism_and_preflight", r.pop("pass"), **r)
    if "stage_hooks" in parts:
        r = stage_hooks(out)
        record("stage_hooks", r.pop("pass"), **r)
    if "clean_holdout" in parts:
        r = clean_holdout(out, a.device)
        record("clean_holdout", r.pop("pass"), **r)
    if "lipschitz" in parts:
        for name, r in lipschitz_checks().items():
            record(f"lipschitz_{name}", r.pop("pass"), **r)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "device": a.device,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
