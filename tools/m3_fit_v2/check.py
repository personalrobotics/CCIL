#!/usr/bin/env python3
"""Evidence for batched transition-model fits (#110).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_fit_v2/check.py --out DIR [--device cuda] [--parts all]

1. correctness (any host): for none and slack, one full-batch step (order-free) equals
   MLPTransitionModel.fit's step from the same initialization (float32 tolerance); for soft_sampling and
   soft_sampling_slack, with the same injected perturbations and the constraint at half the fixture's median
   ratio, so the hinge is active,
   the two engines give the same loss and gradients (float64); a rerun is identical; fit_group refuses the
   regularizers it does not support, and so does preflight for trainer_version 2; compute.deterministic is
   honored, recorded and restored.
1b. stage_hooks (any host): fit_transition_stage(prefit=...) and run_policy_harness(trained=...) accept a
   group's result for its own unit and refuse another unit's (a swapped fit, a swapped policy) and a policy
   whose weights are not its record's.
2. pendulum (CUDA host; declared before running): on each task's training transitions (the ordinary
   pipeline's data.prepare at seed 501), with the task's paper settings (continuous soft_sampling,
   discontinuous slack) and with lipschitz_type none, fit seeds 401-405 per-fit (MLPTransitionModel.fit)
   and as one batched group. Report the held-out one-step error of every fit; the 95% union interval of
   mean(error_batched - error_perfit) over the 5 paired seeds is reported (no decision: the fits use
   different shuffles and perturbation draws, so this measures how much the engine changes the fitted
   model, against the seed-to-seed spread).
2b. equivalence (CUDA host; the gate). Declared before any of its fits ran: for each task, with the task's
   paper regularizer and with lipschitz_type none, fit fresh seeds 601-640 per-fit (MLPTransitionModel.fit)
   and as one batched group, on the task's training transitions (data.prepare at seed 501). The batched engine
   is equivalent when the 90% union interval (BCa with paired t; TOST at 5% per side) of
   mean(error_batched - error_perfit) lies within +-25% of the per-fit mean held-out one-step error. The 25%
   bound is a declared practical-equivalence margin, fixed before these fits and not derived from any observed
   spread.
3. timing (CUDA host): wall time of one per-fit fit, and of 20 seeds as one batched group, per task.
5. lipschitz (any host; #125): the model_normalized sampled Lipschitz ratio
   - equals a direct computation of ||g(x + dx) - g(x)|| / ||dx|| from the state dict (x the normalized input,
     g the normalized residual, dx = eps * N(0, I) in normalized space);
   - is invariant to a change of units (states, actions and residuals rescaled per coordinate, normalization
     refitted, the same weights), while the raw ratio scales with the units;
   - gives the same penalty, and the same gradients, in both engines from the same perturbations, for
     soft_sampling and soft_sampling_slack;
   - is recorded (coordinates, units, normalization hash) in fit records, and a checkpoint saved before #125
     (no lipschitz_coordinates in its settings) loads as raw; soft_jac refuses model_normalized.
6. lipschitz_pendulum (CUDA host; reported, no decision; no policy is trained): continuous Pendulum at the paper's
   settings (soft_sampling, constraint 2, eps 3e-4), seeds 401-405, fitted as one batched group under each
   coordinate system: per-epoch penalty, the share of training samples whose ratio exceeds the constraint at the
   final weights, and the held-out one-step error. At the raw fits' weights it also evaluates legacy's term
   (correct_il soft_sampling_lipschitz_loss: relu(||Y_hat - predict(s + eps n_s, a + eps n_a)|| - L ||n||), with
   Y_hat the normalized residual and predict the next state s + f), to show what legacy's ablation removed.
4. clean_holdout (any host): on the first 20 continuous Pendulum episodes (labels_per_transition 2, 2 dynamics
   epochs), the label pipeline runs on the data and on a copy whose held-out episodes (the data.prepare split
   at seed 501) are perturbed. With transition_model.fit_episodes train, the label model and every label from a
   training episode are identical between the two (held-out episodes influence no label); with all, they
   differ (the check can see the leak it rules out).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))

from ccil.models.mlp import MLPTransitionModel  # noqa: E402
from ccil.models.mlp_batched import FitJob, fit_group  # noqa: E402

SEEDS, TIMING_SEEDS, DATA_SEED = tuple(range(401, 406)), tuple(range(401, 421)), 501


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


def _rows(prepared, rep):
    """(z, a, z_next, train mask) of a prepared stage, as fit_transition_stage reads them (forward direction)."""
    from ccil.core import transitions
    from ccil.schema import read_trajectories
    b = transitions(read_trajectories(prepared.out("data/dataset.npz")), rep)
    split = json.loads(prepared.out("data/split.json").read_text())
    return b.z, b.a, b.z_next, np.isin(b.episode_id, list(split["train"]))


def stage_hooks(out: Path) -> dict:
    """A group's fit and policy are accepted for their own unit and refused for another's."""
    import study

    from ccil.config import resolve
    from ccil.core import IdentityRepresentation
    from ccil.data.prepare import prepare_stage
    from ccil.pipelines import offline, policy_harness
    from ccil.schema import read_trajectories, write_trajectories
    from ccil.schema.trajectories import TrajectoryDataset
    from ccil.supervision import build_supervision
    from ccil.training import vector_bc_v2 as v2
    root = out / "hooks"
    study.prepare(root)
    full = read_trajectories(root / "inputs" / "data" / f"{study.DATASETS['continuous']}.npz")
    data = root / "small.npz"
    write_trajectories(TrajectoryDataset(full.episodes[:20], full.meta), data)
    os.chdir(root / "inputs")
    config = str(root / "inputs" / "config" / study.CONFIGS["continuous"])

    def cfg(seed):
        return resolve(config=config, sets=[f"run.seed={seed}", "compute.device=cpu", f"data.path={data}",
                                            f"run.output_dir={root / 'stages'}", "transition_model.epochs=1",
                                            "transition_model.trainer_version=2", "policy.trainer_version=2",
                                            "policy.method=bc", "policy.epochs=1",
                                            "policy.exclude_validation_episodes=false",
                                            "generation.labels_per_transition=2"])
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
    try:
        offline.fit_transition_stage(c1, prepared[1], rep, prefit=fits[2])
        res["swapped_fit_refused"] = False
    except Exception as e:  # noqa: BLE001 - the stage wraps the refusal
        res["swapped_fit_refused"] = "PrefitMismatch" in repr(e) or "fit identity" in str(e)
    offline.fit_transition_stage(c1, prepared[1], rep, prefit=fits[1])
    res["own_fit_accepted"] = True
    offline.generate_offline(c1)
    sup = build_supervision(prepared[1].out("data/dataset.npz"), "bc", representation=rep,
                            demonstration_weight=c1["policy.demonstration_weight"])
    [own, other] = v2.train_group([v2.Job(sup, 1), v2.Job(sup, 2)], policy_harness.settings_of(c1.values),
                                  device="cpu", compile=False)
    for name, trained in (("swapped_policy_refused", other),
                          ("tampered_weights_refused", (other[0], {**own[1]}))):
        try:
            policy_harness.run_policy_harness(c1, trained=trained)
            res[name] = False
        except Exception as e:  # noqa: BLE001 - the stage wraps the refusal
            res[name] = "PolicyMismatch" in repr(e) or "not this unit" in str(e) or "not the ones" in str(e)
    policy_harness.run_policy_harness(c1, trained=own)
    res["own_policy_accepted"] = True
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


def _transitions(out: Path, task: str):
    """(z, a, z_next) of the training episodes and of the held-out episodes, from data.prepare at seed 501."""
    import study

    from ccil.config import resolve
    from ccil.core import IdentityRepresentation, transitions
    from ccil.data.prepare import prepare_stage
    from ccil.schema import read_trajectories
    root = out / "pendulum"
    study.prepare(root)
    data = root / "inputs" / "data" / f"{study.DATASETS[task]}.npz"
    os.chdir(root / "inputs")
    cfg = resolve(config=str(root / "inputs" / "config" / study.CONFIGS[task]),
                  sets=[f"run.seed={DATA_SEED}", "compute.device=cuda", f"data.path={data}",
                        f"run.output_dir={root / 'stages'}"])
    prepared = prepare_stage(cfg)
    split = json.loads(prepared.out("data/split.json").read_text())
    batch = transitions(read_trajectories(prepared.out("data/dataset.npz")), IdentityRepresentation("state"))
    train = np.isin(batch.episode_id, split["train"])
    v = cfg.values
    settings = {k: v[f"transition_model.{k}"] for k in ("hidden_layers", "activation", "lipschitz_type",
                                                         "lipschitz_constraint", "soft_penalty_weight",
                                                         "soft_sampling_eps", "soft_samples", "slack_weight",
                                                         "learning_rate", "weight_decay", "epochs", "batch_size",
                                                         "lipschitz_coordinates")}
    return (batch.z[train], batch.a[train], batch.z_next[train]), (batch.z[~train], batch.a[~train], batch.z_next[~train]), settings


def pendulum(out: Path, device: str):
    sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
    import analyze
    results = {}
    for task in ("continuous", "discontinuous"):
        (z, a, zn), (zh, ah, znh), paper = _transitions(out, task)
        for lt in (paper["lipschitz_type"], "none"):
            st = {**paper, "lipschitz_type": lt}
            err = lambda m: float(np.linalg.norm(m.delta(zh, ah) + zh - znh, axis=1).mean())  # noqa: E731
            per = []
            for s in SEEDS:
                m = MLPTransitionModel(**st, device=device)
                m.fit(z, a, zn, seed=s)
                per.append(err(m))
            batched = [err(m) for m, _ in fit_group([FitJob(z, a, zn, s) for s in SEEDS], st, device=device)]
            d = np.array(batched) - np.array(per)
            iv = analyze.union_interval(d, 0.95)
            results[f"{task}/{lt}"] = {"held_out_error_perfit": per, "held_out_error_batched": batched,
                                       "mean_difference": iv["mean"], "interval": iv["interval"],
                                       "perfit_seed_sd": float(np.std(per, ddof=1))}
    return {"pass": True, "results": results, "seeds": list(SEEDS),
            "criterion": "reported, no decision (declared in the tool's docstring)"}


EQUIVALENCE_SEEDS, EQUIVALENCE_MARGIN, EQUIVALENCE_LEVEL = tuple(range(601, 641)), 0.25, 0.90


def equivalence(out: Path, device: str) -> dict:
    sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
    import analyze
    results = {}
    for task in ("continuous", "discontinuous"):
        (z, a, zn), (zh, ah, znh), paper = _transitions(out, task)
        for lt in (paper["lipschitz_type"], "none"):
            st = {**paper, "lipschitz_type": lt}
            err = lambda m: float(np.linalg.norm(m.delta(zh, ah) + zh - znh, axis=1).mean())  # noqa: E731
            per = []
            for s_ in EQUIVALENCE_SEEDS:
                m = MLPTransitionModel(**st, device=device)
                m.fit(z, a, zn, seed=s_)
                per.append(err(m))
            batched = [err(m) for m, _ in fit_group([FitJob(z, a, zn, s_) for s_ in EQUIVALENCE_SEEDS], st,
                                                    device=device)]
            d = np.array(batched) - np.array(per)
            iv = analyze.union_interval(d, EQUIVALENCE_LEVEL)
            margin = EQUIVALENCE_MARGIN * float(np.mean(per))
            results[f"{task}/{lt}"] = {"equivalent": bool(-margin < iv["interval"][0] and iv["interval"][1] < margin),
                                       "interval": iv["interval"], "mean_difference": iv["mean"], "margin": margin,
                                       "perfit_mean": float(np.mean(per)), "batched_mean": float(np.mean(batched)),
                                       "held_out_error_perfit": per, "held_out_error_batched": batched,
                                       "lipschitz_coordinates": st.get("lipschitz_coordinates", "raw")}
    return {"pass": all(r["equivalent"] for r in results.values()), "seeds": list(EQUIVALENCE_SEEDS),
            "criterion": f"{int(EQUIVALENCE_LEVEL * 100)}% union interval of mean(batched - per-fit) within "
                         f"+-{int(EQUIVALENCE_MARGIN * 100)}% of the per-fit mean", "results": results}


def timing(out: Path, device: str):
    rows = []
    for task in ("continuous", "discontinuous"):
        (z, a, zn), _, st = _transitions(out, task)
        torch.cuda.synchronize()
        t0 = time.time()
        MLPTransitionModel(**st, device=device).fit(z, a, zn, seed=TIMING_SEEDS[0])
        torch.cuda.synchronize()
        one = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        g = fit_group([FitJob(z, a, zn, s) for s in TIMING_SEEDS], st, device=device)
        torch.cuda.synchronize()
        rows.append({"task": task, "lipschitz_type": st["lipschitz_type"], "transitions": len(z),
                     "epochs": st["epochs"], "batch_size": st["batch_size"], "perfit_one_seed_seconds": one,
                     "batched_20_seeds_seconds": time.time() - t0, "engine": g[0][1]["execution"]["engine"],
                     "epoch_seconds": g[0][1]["group"]["epoch_seconds"][:3],
                     "peak_device_gb": torch.cuda.max_memory_allocated() / 2**30})
        print(rows[-1], flush=True)
    return {"pass": all(r["engine"].startswith("compiled") for r in rows), "measurements": rows}


def clean_holdout(out: Path, device: str) -> dict:
    import dataclasses

    import study

    from ccil.config import resolve
    from ccil.pipelines import offline
    from ccil.schema import read_candidates, read_trajectories, write_trajectories
    from ccil.schema.trajectories import TrajectoryDataset
    root = out / "holdout"
    study.prepare(root)
    full = read_trajectories(root / "inputs" / "data" / f"{study.DATASETS['continuous']}.npz")
    small = TrajectoryDataset(full.episodes[:20], full.meta)
    paths = {"original": root / "original.npz", "perturbed": root / "perturbed.npz"}
    write_trajectories(small, paths["original"])
    os.chdir(root / "inputs")
    config = str(root / "inputs" / "config" / study.CONFIGS["continuous"])

    def run(which, fit_episodes):
        cfg = resolve(config=config, sets=[f"run.seed={DATA_SEED}", f"compute.device={device}",
                                           f"data.path={paths[which]}", f"run.output_dir={root / which / fit_episodes}",
                                           "transition_model.epochs=2", "generation.labels_per_transition=2",
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

    # Records, old checkpoints and refusals.
    import tempfile
    m, *_ = _fitted("soft_sampling", "model_normalized")
    rec = m.fit(z, a, zn, seed=5)
    [(_, rec2)] = fit_group([FitJob(z, a, zn, 5)], dict(m.settings), device="cpu", compile=False)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "old.npz"
        mr, *_ = _fitted("soft_sampling", "raw")
        mr.save(path)
        with np.load(path, allow_pickle=False) as f:
            arrays = {k_: f[k_] for k_ in f.files}
        meta = json.loads(str(arrays["meta"]))
        del meta["settings"]["lipschitz_coordinates"]
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez(path, **arrays)
        old = MLPTransitionModel.load(path)
    try:
        MLPTransitionModel(**{**BASE, "lipschitz_type": "soft_jac", "lipschitz_coordinates": "model_normalized"})
        jac_refused = False
    except ValueError:
        jac_refused = True
    recorded = all(r.get("lipschitz", {}).get("coordinates") == "model_normalized"
                   and len(r["lipschitz"]["normalization_sha256"]) == 64 for r in (rec, rec2))
    out["records_and_old_checkpoints"] = {"pass": recorded and old.settings["lipschitz_coordinates"] == "raw"
                                          and jac_refused, "recorded": recorded,
                                          "old_checkpoint_reads_as": old.settings["lipschitz_coordinates"],
                                          "soft_jac_model_normalized_refused": jac_refused,
                                          "record": rec["lipschitz"]}
    return out


def lipschitz_pendulum(out: Path, device: str) -> dict:
    from ccil.models import mlp as mlp_mod
    (z, a, zn), (zh, ah, znh), paper = _transitions(out, "continuous")
    res = {}
    for coords in ("raw", "model_normalized"):
        st = {**paper, "lipschitz_coordinates": coords}
        fits = fit_group([FitJob(z, a, zn, s) for s in SEEDS], st, device=device)
        rows = []
        for (m, rec), seed in zip(fits, SEEDS):
            s_ = torch.as_tensor(z, dtype=torch.float32, device=device)
            a_ = torch.as_tensor(a, dtype=torch.float32, device=device)
            g = torch.Generator(device=device).manual_seed(seed)
            nz = torch.randn((len(z) * st["soft_samples"], z.shape[1] + a.shape[1]), generator=g, device=device)
            with torch.no_grad():
                r = mlp_mod.lipschitz_ratio(s_, a_, m.net.predict, st["soft_samples"], st["soft_sampling_eps"],
                                            noise=(nz[:, :z.shape[1]], nz[:, z.shape[1]:]), coordinates=coords, net=m.net)
            pen = rec["epoch_losses"]["lipschitz_penalty"]
            legacy = None
            if coords == "raw":
                with torch.no_grad():
                    n_, eps_ = st["soft_samples"], st["soft_sampling_eps"]
                    ns_, na_ = nz[:, :z.shape[1]], nz[:, z.shape[1]:]
                    S_, A_ = s_.repeat_interleave(n_, 0), a_.repeat_interleave(n_, 0)
                    y_hat = m.net(s_, a_).repeat_interleave(n_, 0)          # legacy Y_hat: the normalized residual
                    nxt = (S_ + eps_ * ns_) + m.net.predict(S_ + eps_ * ns_, A_ + eps_ * na_)  # legacy predict_fn
                    hinge = torch.relu(torch.linalg.norm(y_hat - nxt, dim=-1)
                                       - st["lipschitz_constraint"] * torch.linalg.norm(torch.cat([ns_, na_], -1), dim=-1))
                legacy = {"active_share": float((hinge > 0).float().mean()), "mean_hinge": float(hinge.mean())}
            rows.append({"legacy_term_at_these_weights": legacy, "seed": seed, "share_above_constraint": float((r > st["lipschitz_constraint"]).float().mean()),
                         "ratio_quantiles": {q: float(r.quantile(q)) for q in (0.5, 0.99)},
                         "ratio_max": float(r.max()), "epochs_with_penalty": int(sum(p > 0 for p in pen)),
                         "penalty_last_epoch": pen[-1],
                         "held_out_error": float(np.linalg.norm(m.delta(zh, ah) + zh - znh, axis=1).mean())})
        res[coords] = rows
    return {"pass": True, "criterion": "reported, no decision (declared in the tool's docstring)",
            "constraint": paper["lipschitz_constraint"], "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parts", default="correctness")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parts = (["correctness", "stage_hooks", "clean_holdout", "lipschitz", "pendulum", "lipschitz_pendulum", "timing"]
             if a.parts == "all" else a.parts.split(","))
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
    if "lipschitz_pendulum" in parts:
        r = lipschitz_pendulum(out, a.device)
        record("lipschitz_pendulum", r.pop("pass"), **r)
    if "equivalence" in parts:
        r = equivalence(out, a.device)
        record("equivalence", r.pop("pass"), **r)
    if "pendulum" in parts:
        r = pendulum(out, a.device)
        record("pendulum_held_out_error", r.pop("pass"), **r)
    if "timing" in parts:
        r = timing(out, a.device)
        record("timing", r.pop("pass"), **r)
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
