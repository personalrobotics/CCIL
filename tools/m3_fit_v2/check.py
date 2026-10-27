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
   group's result for its own unit and refuse another unit's (a swapped fit, a swapped policy), a model or
   policy whose weights are not its record's, and a result run differently from the unit's request (a unit
   asking for compute.deterministic given non-deterministic fits and policy).
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

    def cfg(seed, deterministic=False):
        return resolve(config=config, sets=[f"run.seed={seed}", "compute.device=cpu", f"data.path={data}",
                                            f"compute.deterministic={str(deterministic).lower()}",
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
    offline.generate_offline(c1)
    sup = build_supervision(prepared[1].out("data/dataset.npz"), "bc", representation=rep,
                            demonstration_weight=c1["policy.demonstration_weight"])
    [own, other] = v2.train_group([v2.Job(sup, 1), v2.Job(sup, 2)], policy_harness.settings_of(c1.values),
                                  device="cpu", compile=False)
    for name, trained, needle in (("swapped_policy_refused", other, "not this unit"),
                                  ("tampered_weights_refused", (other[0], {**own[1]}), "not the ones")):
        res[name] = refused(lambda t=trained: policy_harness.run_policy_harness(c1, trained=t), needle)
    res["policy_wrong_execution_refused"] = refused(lambda: policy_harness.run_policy_harness(c1_det, trained=own),
                                                    "deterministic")
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
                                                         "learning_rate", "weight_decay", "epochs", "batch_size")}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parts", default="correctness")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parts = (["correctness", "stage_hooks", "clean_holdout", "pendulum", "timing"] if a.parts == "all"
             else a.parts.split(","))
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
