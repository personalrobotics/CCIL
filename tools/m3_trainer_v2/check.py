#!/usr/bin/env python3
"""Evidence for policy trainer v2 (#108).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_trainer_v2/check.py --out DIR [--device cuda] [--parts all]

Parts, each a case in DIR/report.json:
1. correctness (any host)
   - one step on a batch that is the whole training set (order-free) equals one
     torch.optim.Adam step of v1's model on v1's weighted loss: gradients, Adam,
     scalers and loss together (float32 tolerance);
   - a model whose weights are all zero keeps its initial parameters exactly,
     inside a group whose other models train;
   - device NoiseBC draws equal noise_bc.counter_uniforms bit for bit on random keys;
   - index streams: each epoch visits every training row at most once and all
     but fewer than one minibatch of them, and balancing repeats demonstrations
     as v1's training_rows does.
2. modes and determinism: resident and streaming give bit-identical parameters,
   and a rerun gives byte-identical checkpoints (on the requested device).
2b. robustness (any host; the #117 review): a failing streaming producer
   raises promptly; a synthetic block byte-identical to the demonstration
   block keeps its source; the execution identity differs between groups while
   the training identity does not; a compiler failure on the first step falls
   back to eager, recorded, with unchanged results; TF32 is off during
   training and the caller's setting is restored.
3. equivalence, "no detected difference" (CUDA host; declared before running): one supervision set per
   task from the ordinary pipeline at seed 501; v1 and v2 each train BC and CCIL
   with training seeds 201-210; every policy is evaluated on the canonical
   Pendulum environment (tools/m3_pendulum/env.py) at its seed's initial states;
   per task and arm, the 95% union interval (BCa with t) of mean(R_v2 - R_v1)
   must contain 0. Its width is reported.
4. equivalence_margin (CUDA host; declared before running, after the 10-seed
   result was seen, to tighten it): CCIL only, fresh training seeds 211-250,
   the same seed-501 supervision; the 95% union interval of mean(R_v2 - R_v1)
   must lie within [-250, 250] return units (about a third of the per-seed
   CCIL - BC effects of the v4 run) on both tasks.
4b. realworkload (CUDA host; the #108 timing criterion): labels from the
   ordinary pipeline for seeds 301-320 per task (20 distinct label sets), then
   20-seed groups of BC, NoiseBC (1e-4) and CCIL (balanced on discontinuous),
   and a 1-seed CCIL group, each timed: wall, setup, one-time, steady ms per
   step, peak memory, table rows.
5. scaling (CUDA host): steady-state ms per step (second epoch), one-time cost
   (first-epoch excess) and setup, and peak device memory, as models M, rows N
   (resident to 10^7, streaming to 10^8), observation dimension D and hidden
   width grow; whether a repeated group of the same shapes reuses the compiled
   step; peak memory at D = 512 in both modes.
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(HERE))

from ccil.core.identity import identity_words64  # noqa: E402
from ccil.supervision import Supervision  # noqa: E402
from ccil.training import vector_bc as v1  # noqa: E402
from ccil.training import vector_bc_v2 as v2  # noqa: E402
from ccil.training.noise_bc import NoiseBC, counter_uniforms  # noqa: E402

EQ_SEEDS = tuple(range(201, 211))
LABEL_SEED = 501


def synthetic(n_demo, n_syn, seed, method="ccil", weights=None, d=3):
    r = np.random.default_rng(0)
    od = r.normal(size=(n_demo, d))
    wv = r.normal(size=(d, 1)) / np.sqrt(d)
    ad = np.tanh(od @ wv)
    r2 = np.random.default_rng(seed)
    os_ = od[r2.integers(0, n_demo, n_syn)] + 0.01 * r2.normal(size=(n_syn, d))
    obs = np.concatenate([od, os_]) if method == "ccil" else od
    act = np.concatenate([ad, np.tanh(os_ @ wv)]) if method == "ccil" else ad
    n = len(obs)
    src = np.array(["demonstration"] * n_demo + (["synthetic"] * n_syn if method == "ccil" else []))
    w = np.ones(n) if weights is None else weights
    return Supervision(obs, act, w, src, np.array([str(i // 50) for i in range(n)]), np.arange(n) % 50,
                       np.where(src == "demonstration", -1, np.arange(n)), src, {"method": method, "inputs": {}})


SETTINGS = {"hidden_layers": [64, 64], "learning_rate": 1e-3, "batch_size": 64, "epochs": 3,
            "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None}


def correctness(device) -> dict:
    out = {}
    # (a) one full-batch step against torch.optim.Adam on v1's model and loss.
    sup = synthetic(64, 0, 1, "bc")
    st = {**SETTINGS, "epochs": 1}
    [(p2, _)] = v2.train_group([v2.Job(sup, 7)], st, device=device, compile=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        ref = v1.VectorBCPolicy(3, 1, st["hidden_layers"])
    obs, act, w = sup.arrays()
    ref.fit_scalers(obs, act)
    opt = torch.optim.Adam(ref.net.parameters(), lr=st["learning_rate"], **v1.ADAM)
    x = ref.scale_obs(torch.as_tensor(obs, dtype=torch.float32))
    y = ref.scale_act(torch.as_tensor(act, dtype=torch.float32))
    loss = v1.weighted_loss(ref.net(x), y, torch.as_tensor(w, dtype=torch.float32))
    opt.zero_grad()
    loss.backward()
    opt.step()
    diffs = [float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in
             zip(p2.net.state_dict().values(), ref.net.state_dict().values())]
    out["one_step_matches_adam"] = {"pass": max(diffs) < 1e-5, "max_relative_difference": max(diffs)}
    # (b) an all-zero-weight model keeps its initial parameters exactly inside a training group.
    zero = synthetic(256, 0, 2, "bc", weights=np.zeros(256))
    group = v2.train_group([v2.Job(synthetic(256, 0, 3, "bc"), 3), v2.Job(zero, 4)], SETTINGS, device=device, compile=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        init = v1.VectorBCPolicy(3, 1, SETTINGS["hidden_layers"])
    unchanged = all(torch.equal(a, b) for a, b in zip(group[1][0].net.state_dict().values(), init.net.state_dict().values()))
    out["zero_weight_model_unchanged"] = {"pass": unchanged and group[1][1]["epochs"][0]["zero_weight_steps"] == 4,
                                          "zero_weight_steps": group[1][1]["epochs"][0]["zero_weight_steps"]}
    # (c) device NoiseBC draws equal the host draws.
    rng = np.random.default_rng(3)
    ok = True
    for _ in range(20):
        eps, fr = [str(e) for e in rng.integers(0, 10**6, 64)], rng.integers(0, 10**6, 64)
        seed, step = int(rng.integers(0, 2**63)), int(rng.integers(0, 2**40))
        words = torch.tensor(np.array([identity_words64(e, f) for e, f in zip(eps, fr)], np.uint64).view(np.int64))
        dev = v2.counter_uniforms_device(torch.tensor([v2._s64(seed)]), torch.tensor([[step]]), words[None], 0, 4)[0]
        ok &= np.array_equal(dev.numpy(), counter_uniforms(seed, step, eps, fr, 0, 4))
    out["noise_draws_bit_identical"] = {"pass": bool(ok), "trials": 20}
    # (d) index streams: coverage per epoch, and balancing as v1.
    jobs = [v2.Job(synthetic(500, 900 + 37 * s, s), s) for s in (1, 2, 3)]
    table, blocks = v2._table(jobs)
    cov = {}
    for bal in (None, 4):
        segs = [v2._segments(b, bal) for b in blocks]
        s = v2._Streams(segs, [1, 2, 3], 64, torch.device("cpu"))
        for e in (0, 1):
            idx = s.chunk(e, 0, int(s.steps.max()))
            for i in range(3):
                got = idx[i, :int(s.steps[i])].reshape(-1).numpy()
                want = np.concatenate([np.arange(a, a + n) for a, n in segs[i]])
                vals, counts = np.unique(want, return_counts=True)
                gv, gc = np.unique(got, return_counts=True)
                allowed = dict(zip(vals, counts))
                v1_rows = len(v1.training_rows(jobs[i].supervision, bal))
                cov[f"balance{bal}-epoch{e}-model{i}"] = bool(
                    all(c <= allowed.get(v, 0) for v, c in zip(gv, gc)) and 0 <= len(want) - len(got) < 64
                    and len(want) == v1_rows)
    out["index_streams"] = {"pass": all(cov.values()), "cases": cov}
    return out


def robustness(device) -> dict:
    """The #117 review's failure modes, each as observable behavior."""
    import threading
    out = {}
    # A producer failure surfaces promptly as an error, never as a hang.
    jobs = [v2.Job(synthetic(800, 1500, 1), 1)]
    original = v2._Streams.chunk

    def failing(self, *a, **k):
        if threading.current_thread().name == "ccil-v2-producer":
            raise MemoryError("injected producer failure")
        return original(self, *a, **k)
    v2._Streams.chunk = failing
    t0, err = time.time(), None
    try:
        v2.train_group(jobs, SETTINGS, device=device, data_mode="streaming", chunk_steps=4)
    except v2.TrainingError as e:
        err = str(e)
    finally:
        v2._Streams.chunk = original
    out["producer_failure_raises_promptly"] = {"pass": err is not None and "injected" in err and time.time() - t0 < 60,
                                               "seconds": round(time.time() - t0, 2), "error": err}
    # A synthetic block byte-identical to the demonstration block keeps its source.
    base = synthetic(512, 0, 1, "bc")
    obs, act = np.concatenate([base.observations] * 2), np.concatenate([base.actions] * 2)
    src = np.array(["demonstration"] * 512 + ["synthetic"] * 512)
    twin = Supervision(obs, act, np.ones(1024), src, np.concatenate([base.episode_id] * 2),
                       np.concatenate([base.frame] * 2), np.where(src == "demonstration", -1, np.arange(1024)), src,
                       {"method": "ccil", "inputs": {}})
    [(_, r)] = v2.train_group([v2.Job(twin, 1)], SETTINGS, device=device, compile=False)
    seen = r["epochs"][0]["rows_seen"]
    out["identical_blocks_keep_their_source"] = {
        "pass": r["row_sources"] == {"demonstration": 512, "synthetic": 512} and len(r["group"]["table_blocks"]) == 2
        and seen["synthetic"] > 0 and seen["demonstration"] > 0, "row_sources": r["row_sources"], "rows_seen": seen}
    # Execution identity: the same job in groups of different buckets has the same training identity and a
    # different execution identity; members and position are recorded.
    j = [v2.Job(synthetic(512, 512, s), s) for s in (1, 2, 3)]
    alone = v2.train_group(j[:1], SETTINGS, device=device, compile=False)[0][1]
    grouped = v2.train_group(j, SETTINGS, device=device, compile=False)[0][1]
    out["execution_identity"] = {
        "pass": alone["training_identity_sha256"] == grouped["training_identity_sha256"]
        and alone["execution"]["execution_sha256"] != grouped["execution"]["execution_sha256"]
        and (alone["execution"]["bucket"], grouped["execution"]["bucket"]) == (1, 4)
        and [m["seed"] for m in grouped["group"]["members"]] == [1, 2, 3] and grouped["group"]["padding"] == 1,
        "buckets": [alone["execution"]["bucket"], grouped["execution"]["bucket"]]}
    # A compiler failure on the first step falls back to eager, is recorded, and changes nothing.
    original_compiled = v2._compiled

    def broken(fn):
        def raising(*a, **k):
            raise RuntimeError("injected compiler failure")
        return raising
    v2._compiled = broken
    try:
        fb = v2.train_group(j, SETTINGS, device=device, compile=True)
    finally:
        v2._compiled = original_compiled
    eager = v2.train_group(j, SETTINGS, device=device, compile=False)
    same = all(torch.equal(a.net.state_dict()[k], b.net.state_dict()[k]) for (a, _), (b, _) in zip(fb, eager)
               for k in a.net.state_dict())
    out["compile_failure_falls_back"] = {"pass": same and fb[0][1]["execution"]["engine"].startswith("eager (compilation failed"),
                                         "engine": fb[0][1]["execution"]["engine"]}
    # TF32 is off during training and the caller's setting is restored afterwards.
    before = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    try:
        r = v2.train_group(j[:1], SETTINGS, device=device, compile=False)[0][1]
        restored = torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before[0]
        torch.set_float32_matmul_precision(before[1])
    out["precision_enforced_and_restored"] = {"pass": r["group"]["precision"]["tf32_matmul"] is False and restored,
                                              "precision": r["group"]["precision"]}
    return out


def recipe(device) -> dict:
    """The #115 recipe against independent references, on full batches (the shuffle cannot matter)."""
    import math
    out = {}
    sup = synthetic(64, 0, 1, "bc")
    obs, act, w = sup.arrays()
    K, warm, lr = 12, 3, 1e-2

    def lam(step):  # #115: linear warm-up to lr over `warm` steps, then cosine decay to 0 at step K (1-based t)
        t_ = step + 1
        return t_ / warm if t_ <= warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (t_ - warm) / (K - warm))))

    def reference(clip=None, ema_decay=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            ref = v1.VectorBCPolicy(3, 1, SETTINGS["hidden_layers"])
        ref.fit_scalers(obs, act)
        opt = torch.optim.AdamW(ref.net.parameters(), lr=lr, betas=v1.ADAM["betas"], eps=v1.ADAM["eps"], weight_decay=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        x = ref.scale_obs(torch.as_tensor(obs, dtype=torch.float32))
        y = ref.scale_act(torch.as_tensor(act, dtype=torch.float32))
        ema = [p.detach().clone() for p in ref.net.parameters()]
        for _ in range(K):
            loss = v1.weighted_loss(ref.net(x), y, torch.as_tensor(w, dtype=torch.float32))
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(ref.net.parameters(), clip)
            opt.step()
            sched.step()
            if ema_decay:
                ema = [e * ema_decay + p.detach() * (1 - ema_decay) for e, p in zip(ema, ref.net.parameters())]
        return [p.detach() for p in ref.net.parameters()] if not ema_decay else ema

    base = {**SETTINGS, "epochs": K, "learning_rate": lr, "optimizer": "adamw", "weight_decay": 0.1,
            "lr_schedule": "cosine", "warmup_steps": warm}
    for name, extra, ref_args in (("adamw_cosine", {}, {}), ("adamw_cosine_clip", {"grad_clip_norm": 0.05}, {"clip": 0.05}),
                                  ("adamw_cosine_ema", {"ema_decay": 0.9}, {"ema_decay": 0.9})):
        [(p2, _)] = v2.train_group([v2.Job(sup, 7)], {**base, **extra}, device=device, compile=False)
        ref = reference(**ref_args)
        d = max(float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in zip(p2.net.parameters(), ref))
        out[name] = {"pass": d < 1e-4, "steps": K, "max_relative_difference": d}
    # Validation selection: the returned weights are the best epoch's, judged on held-out rows only. A small,
    # noisily labelled training set overfits, so the best held-out epoch is interior, not the first or the last.
    def labelled(n, seed, noise):
        r = np.random.default_rng(seed)
        o = r.normal(size=(n, 3))
        a_ = np.tanh(o @ np.array([[0.5], [-0.3], [0.2]]) + noise * r.normal(size=(n, 1)))
        src = np.full(n, "demonstration")
        return Supervision(o, a_, np.ones(n), src, np.array([str(i // 50) for i in range(n)]), np.arange(n) % 50,
                           np.full(n, -1), src, {"method": "bc", "inputs": {}})
    train, held = labelled(128, 3, 1.0), labelled(2048, 4, 0.0)
    [(p, r)] = v2.train_group([v2.Job(train, 5, validation=held)],
                              {**SETTINGS, "batch_size": 32, "epochs": 150, "learning_rate": 3e-3,
                               "checkpoint_selection": "validation"}, device=device, compile=False)
    o, a_, ww = held.arrays()
    x = p.scale_obs(torch.as_tensor(o, dtype=torch.float32))
    y = p.scale_act(torch.as_tensor(a_, dtype=torch.float32))
    with torch.no_grad():
        returned = float(v1.weighted_loss(p.net(x), y, torch.as_tensor(ww, dtype=torch.float32)))
    hist = [e["loss"] for e in r["validation"]["evaluations"]]
    interior = 0 < r["validation"]["selected"] < len(hist) - 1
    out["validation_selection"] = {"pass": abs(returned - min(hist)) <= 1e-6 * max(1.0, abs(min(hist)))
                                   and r["validation"]["selected"] == int(np.argmin(hist)) and interior,
                                   "epochs": len(hist), "interior": interior,
                                   "last_epoch_loss": hist[-1],
                                   "returned_loss": returned, "best_recorded": min(hist),
                                   "selected": r["validation"]["selected"]}
    # A step cadence shared by models of very different dataset sizes: under one step budget, a 128-row and a
    # 2,048-row training set are validated at the same steps (every 40 and the last), the selection is the best
    # of those, and with EMA on, the EMA weights are both judged and returned.
    big = labelled(2048, 6, 1.0)
    budget, every = 300, 40
    outs = v2.train_group([v2.Job(train, 5, validation=held), v2.Job(big, 6, validation=held)],
                          {**SETTINGS, "batch_size": 32, "epochs": 1, "learning_rate": 3e-3, "step_budget": budget,
                           "ema_decay": 0.9, "checkpoint_selection": "validation", "validation_interval_steps": every},
                          device=device, compile=False)
    want = [s_ for s_ in range(every, budget + 1, every)] + ([budget] if budget % every else [])
    cadence = []
    for p_, r_ in outs:
        steps_seen = [e["step"] for e in r_["validation"]["evaluations"]]
        losses = [e["loss"] for e in r_["validation"]["evaluations"]]
        with torch.no_grad():
            got = float(v1.weighted_loss(p_.net(p_.scale_obs(torch.as_tensor(o, dtype=torch.float32))),
                                         p_.scale_act(torch.as_tensor(a_, dtype=torch.float32)),
                                         torch.as_tensor(ww, dtype=torch.float32)))
        cadence.append({"steps": steps_seen, "selected_step": r_["validation"]["selected_step"],
                        "returned_loss": got, "best_recorded": min(losses),
                        "ok": steps_seen == want and abs(got - min(losses)) <= 1e-6 * max(1.0, abs(min(losses)))
                        and r_["validation"]["selected"] == int(np.argmin(losses))})
    out["validation_step_cadence"] = {"pass": all(c_["ok"] for c_ in cadence), "expected_steps": want,
                                      "models": cadence}
    try:
        v2.train_group([v2.Job(train, 5)], {**SETTINGS, "checkpoint_selection": "validation"}, device=device)
        refused = False
    except v2.TrainingError:
        refused = True
    out["validation_requires_held_out_data"] = {"pass": refused}

    # A step budget bounds the work: with 64 minibatches per epoch, a budget of 100 runs 64 + 36 steps, and a
    # budget of 10 runs 10, whatever the epoch length; each epoch's record is exact.
    calls = {"n": 0}
    real_call = v2._Engine.__call__

    def counting(self, *a, **k):
        calls["n"] += 1
        return real_call(self, *a, **k)
    big_train = labelled(2048, 7, 0.5)
    budget_rows = []
    v2._Engine.__call__ = counting
    try:
        for budget in (100, 10):
            calls["n"] = 0
            [(_, rb)] = v2.train_group([v2.Job(big_train, 5)], {**SETTINGS, "batch_size": 32, "epochs": 1,
                                                                 "step_budget": budget}, device=device, compile=False)
            budget_rows.append({"budget": budget, "steps_per_epoch": rb["steps_per_epoch"], "engine_calls": calls["n"],
                                "epoch_steps": [e["steps"] for e in rb["epochs"]],
                                "optimizer_steps": sum(e["optimizer_steps"] for e in rb["epochs"]),
                                "rows_seen": sum(sum(e["rows_seen"].values()) for e in rb["epochs"])})
    finally:
        v2._Engine.__call__ = real_call
    out["step_budget_bounds_work"] = {
        "pass": all(r["engine_calls"] == r["budget"] == sum(r["epoch_steps"]) == r["optimizer_steps"]
                    and r["rows_seen"] == r["budget"] * 32 for r in budget_rows)
        and budget_rows[0]["epoch_steps"] == [64, 36] and budget_rows[1]["epoch_steps"] == [10],
        "cases": budget_rows}

    # Validation is deduplicated and reduced in bounded chunks: two models sharing one held-out set store it once,
    # and a tiny chunk (7 rows per model) selects the same checkpoints with the same losses as one large chunk.
    vs = {"batch_size": 32, "epochs": 6, "learning_rate": 3e-3, "checkpoint_selection": "validation"}
    jobs_v = [v2.Job(train, 5, validation=held), v2.Job(train, 6, validation=held)]
    tiny = v2.train_group(jobs_v, {**SETTINGS, **vs}, device=device, compile=False,
                          stream_chunk_bytes=2 * 7 * 4 * (3 + 1 + 1))
    large = v2.train_group(jobs_v, {**SETTINGS, **vs}, device=device, compile=False)
    same = all(a_[1]["validation"]["selected"] == b_[1]["validation"]["selected"]
               and max(abs(x["loss"] - y["loss"]) / max(abs(y["loss"]), 1e-12)
                       for x, y in zip(a_[1]["validation"]["evaluations"], b_[1]["validation"]["evaluations"])) < 1e-5
               for a_, b_ in zip(tiny, large))
    g_tiny, g_large = tiny[0][1]["group"]["validation"], large[0][1]["group"]["validation"]
    out["validation_bounded_and_deduplicated"] = {
        "pass": same and g_tiny["distinct_sets"] == 1 and g_tiny["rows_per_chunk"] == 7 and g_large["rows_per_chunk"] > 1000,
        "same_selection_and_losses": same, "tiny": g_tiny, "large": g_large}

    # The training identity includes the validation set: a different held-out set is a different training run.
    other_held = labelled(2048, 8, 0.0)
    ids = [v2.training_identity_v2(train, {**SETTINGS, **vs}, 5, None, h)["sha256"] for h in (held, held, other_held, None)]
    out["validation_in_training_identity"] = {"pass": ids[0] == ids[1] and len({ids[0], ids[2], ids[3]}) == 3,
                                              "identities": ids}

    # Ingress: a cosine warm-up that is not shorter than the budget, and validation with zero total weight.
    refusals = {}
    cases = {"warmup_not_shorter_than_budget": ([v2.Job(train, 5)], {**SETTINGS, "lr_schedule": "cosine",
                                                                      "warmup_steps": 50, "step_budget": 50,
                                                                      "batch_size": 32}),
             "zero_weight_validation": ([v2.Job(train, 5, validation=_zero_weight(held))],
                                        {**SETTINGS, "batch_size": 32, "checkpoint_selection": "validation"}),
             "warmup_with_constant_schedule": ([v2.Job(train, 5)], {**SETTINGS, "lr_schedule": "constant",
                                                                    "warmup_steps": 10, "batch_size": 32})}
    for name, (jobs_, st_) in cases.items():
        try:
            v2.train_group(jobs_, st_, device=device, compile=False)
            refusals[name] = False
        except v2.TrainingError:
            refusals[name] = True
    out["ingress_refusals"] = {"pass": all(refusals.values()), **refusals}
    return out


def _zero_weight(sup):
    """sup with every weight 0 (the same rows)."""
    import dataclasses
    return dataclasses.replace(sup, weights=np.zeros_like(np.asarray(sup.weights)))


def contract(device) -> dict:
    """The shared-harness training contract (il-supervision-benchmarks#11): patience termination on the validation
    history, a fixed synthetic sampling fraction with realized exposure, and nested-K candidate views."""
    from ccil.core import (ActionBounds, Finite, IdentityRepresentation, InterfaceError, LinearTransitionModel,
                           NoisyAction, SupportRadius, generate_labels, nested_view)
    from ccil.schema import Episode, TrajectoryDataset
    out = {}

    def labelled(n, seed, noise):  # as recipe(): a small, noisily labelled set overfits, so validation turns
        r = np.random.default_rng(seed)
        o = r.normal(size=(n, 3))
        a_ = np.tanh(o @ np.array([[0.5], [-0.3], [0.2]]) + noise * r.normal(size=(n, 1)))
        src = np.full(n, "demonstration")
        return Supervision(o, a_, np.ones(n), src, np.array([str(i // 50) for i in range(n)]), np.arange(n) % 50,
                           np.full(n, -1), src, {"method": "bc", "inputs": {}})
    train, held = labelled(128, 3, 1.0), labelled(2048, 4, 0.0)

    # (a) Patience: the stop event and the selected checkpoint recompute from the saved validation history, and the
    # returned weights are the weights a plain run cut at the selected step returns (same seed, same shuffle).
    st = {**SETTINGS, "batch_size": 32, "epochs": 1, "learning_rate": 3e-3, "step_budget": 4000,
          "checkpoint_selection": "validation", "validation_interval_steps": 10, "patience": 8, "min_delta": 1e-4}
    [(p, r)] = v2.train_group([v2.Job(train, 5, validation=held)], st, device=device, compile=False)
    hist = r["validation"]["evaluations"]
    best, sel, stale, stop = float("inf"), None, 0, None
    for i, e in enumerate(hist):
        if e["loss"] < best - st["min_delta"]:
            best, sel, stale = e["loss"], i, 0
        else:
            stale += 1
        if stale >= st["patience"]:
            stop = e["step"]
            break
    [(p_cut, _)] = v2.train_group([v2.Job(train, 5)], {**SETTINGS, "batch_size": 32, "epochs": 1, "learning_rate": 3e-3,
                                                      "step_budget": hist[sel]["step"]}, device=device, compile=False)
    s_ = r["stopping"]
    out["patience_recomputes"] = {
        "pass": s_["outcome"] == "patience" and s_["stopped_at_step"] == stop == hist[-1]["step"]
        and r["validation"]["selected"] == sel and s_["steps"] == stop < st["step_budget"]
        and v2.parameters_sha256(p) == v2.parameters_sha256(p_cut),
        "stopped_at_step": s_["stopped_at_step"], "recomputed_stop": stop, "selected": r["validation"]["selected"],
        "recomputed_selected": sel, "evaluations": len(hist), "ceiling": st["step_budget"],
        "returned_equals_cut_run": v2.parameters_sha256(p) == v2.parameters_sha256(p_cut)}
    [(_, r2)] = v2.train_group([v2.Job(train, 5, validation=held)], {**st, "step_budget": 60, "patience": 1000},
                               device=device, compile=False)
    out["ceiling_is_incomplete_fit"] = {"pass": r2["stopping"]["outcome"] == "incomplete_fit"
                                        and r2["stopping"]["stopped_at_step"] is None and r2["stopping"]["steps"] == 60,
                                        "stopping": r2["stopping"]}

    # (a') The plateau schedule: the learning-rate reductions, the stop at the floor and the selected checkpoint
    # recompute from the saved history; a run extended past its stop returns the same weights and records the gain.
    sp = {**st, "patience": 4, "min_delta": 0.0, "lr_schedule": "plateau", "lr_factor": 0.5, "lr_patience": 3,
          "min_lr": 3e-4}
    [(pp, rp)] = v2.train_group([v2.Job(train, 5, validation=held)], sp, device=device, compile=False)
    hist = rp["validation"]["evaluations"]
    lr_, best, sel, stale, stop, cuts, lrs = sp["learning_rate"], float("inf"), None, 0, None, [], []
    for i, e in enumerate(hist):
        lrs.append(lr_)
        if e["loss"] < best:
            best, sel, stale = e["loss"], i, 0
        else:
            stale += 1
        if stale >= sp["lr_patience"] and lr_ > sp["min_lr"]:
            lr_, stale = max(lr_ * sp["lr_factor"], sp["min_lr"]), 0
            cuts.append({"step": e["step"], "lr": lr_})
        if stale >= sp["patience"] and lr_ <= sp["min_lr"]:
            stop = e["step"]
            break
    [(pe, re_)] = v2.train_group([v2.Job(train, 5, validation=held)], {**sp, "extend_after_stop": 3}, device=device,
                                 compile=False)
    s_, x_ = rp["stopping"], re_["stopping"]["extension"]
    out["plateau_recomputes"] = {
        "pass": s_["outcome"] == "patience" and s_["stopped_at_step"] == stop == hist[-1]["step"]
        and s_["lr_reductions"] == cuts and len(cuts) >= 2 and [e["lr"] for e in hist] == lrs
        and rp["validation"]["selected"] == sel and re_["validation"]["selected"] == sel
        and v2.parameters_sha256(pp) == v2.parameters_sha256(pe) and x_["trained_to_step"] >= 3 * stop
        and x_["best_loss_extended"] == min(e["loss"] for e in re_["validation"]["evaluations"]),
        "stopped_at_step": s_["stopped_at_step"], "recomputed_stop": stop, "lr_reductions": s_["lr_reductions"],
        "recomputed_reductions": cuts, "selected": rp["validation"]["selected"], "recomputed_selected": sel,
        "extension": x_, "extended_returns_stop_checkpoint": v2.parameters_sha256(pp) == v2.parameters_sha256(pe)}

    # (b) Synthetic fraction: every minibatch carries exactly b_s synthetic rows; the record's rows and weight mass
    # equal a host replay of the index stream; demonstrations are each drawn at most once per epoch and the
    # synthetic stream draws every row once before any repeats.
    wts = np.random.default_rng(11).uniform(0.2, 2.0, size=500 + 1700)
    sup = synthetic(500, 1700, 12, weights=wts)
    frac_cases = {}
    for frac in (0.5, 0.3):
        stf = {**SETTINGS, "epochs": 3, "synthetic_fraction": frac}
        [(_, rf)] = v2.train_group([v2.Job(sup, 21)], stf, device=device, compile=False)
        table, blocks = v2._table([v2.Job(sup, 21)])
        ms = v2._MixedStreams(blocks, [21], SETTINGS["batch_size"], frac, torch.device("cpu"))
        steps = int(ms.steps[0])
        idx = torch.cat([ms.chunk(e, 0, steps)[0] for e in range(3)]).numpy()  # (3 * steps, B)
        syn_flag, w_tab = table["synthetic"], table["w"].astype(np.float64)
        demo_ok = all(len(np.unique(idx[e * steps:(e + 1) * steps, :ms.b_d])) == steps * ms.b_d for e in range(3))
        syn_draws = idx[:, ms.b_d:].reshape(-1)
        n_s = int(ms.n_s[0])
        first_pass_ok = len(np.unique(syn_draws[:n_s])) == n_s if len(syn_draws) >= n_s else \
            len(np.unique(syn_draws)) == len(syn_draws)
        ex = rf["exposure"]
        replay_rows = {"demonstration": int((~syn_flag[idx]).sum()), "synthetic": int(syn_flag[idx].sum())}
        replay_mass = {"demonstration": float(w_tab[idx][~syn_flag[idx]].sum()),
                       "synthetic": float(w_tab[idx][syn_flag[idx]].sum())}
        frac_cases[str(frac)] = {
            "ok": ex["rows"] == replay_rows and ex["per_minibatch"]["synthetic"] == ms.b_s == round(frac * 64)
            and all(abs(ex["weight_mass"][k] - replay_mass[k]) <= 1e-6 * replay_mass[k] for k in replay_mass)
            and ex["within_tolerance"] and demo_ok and first_pass_ok and syn_flag[idx[:, ms.b_d:]].all()
            and not syn_flag[idx[:, :ms.b_d]].any(),
            "realized_row_fraction": ex["realized_synthetic_row_fraction"],
            "realized_mass_fraction": ex["realized_synthetic_mass_fraction"], "tolerance": ex["tolerance"],
            "rows": ex["rows"], "replayed_rows": replay_rows}
    out["synthetic_fraction_exposure"] = {"pass": all(c_["ok"] for c_ in frac_cases.values()), "cases": frac_cases}

    # (c) Nested K: a view at k of a k_max file equals a direct generation at k, array for array.
    rng = np.random.default_rng(31)
    A_, B_ = np.array([[1.0, 0.02], [-0.2, 0.98]]), np.array([[0.0], [0.02]])
    eps = []
    for e in range(4):
        s = [rng.normal(size=2)]
        acts = rng.normal(size=(30, 1))
        for t in range(30):
            s.append(A_ @ s[-1] + B_ @ acts[t])
        eps.append(Episode(f"e{e}", {"state": np.array(s, np.float32)},
                           np.vstack([acts, np.zeros((1, 1))]).astype(np.float32)))
    ds = TrajectoryDataset(eps).validate()
    model, rep = LinearTransitionModel(A_, B_), IdentityRepresentation("state")
    vals = [Finite(), SupportRadius(1.4)]  # about half the candidates are in support (median displacement 1.35)
    full = generate_labels(ds, rep, model, NoisyAction(20, 0.5), vals, run_id="kmax", seed=7)
    eq = {}
    for k in (1, 4, 20):
        direct = generate_labels(ds, rep, model, NoisyAction(k, 0.5), vals, run_id="direct", seed=7)
        view = nested_view(full, k, "0" * 64)
        eq[str(k)] = bool(
            all(np.array_equal(getattr(view, f), getattr(direct, f)) for f in
                ("source_episode_id", "source_frame", "actions", "accepted", "reasons", "weight"))
            and all(np.array_equal(view.observations[m], direct.observations[m]) for m in direct.observations)
            and all(np.array_equal(view.target_observations[m], direct.target_observations[m])
                    for m in direct.target_observations)
            and all(np.array_equal(view.measurements[m], direct.measurements[m]) for m in direct.measurements)
            and view.generator["params"] == direct.generator["params"] and 0 < view.accepted.sum() < len(view.accepted))
    out["nested_view_equals_direct_generation"] = {"pass": all(eq.values()), "k_max": 20, "cases": eq,
                                                   "accepted_fraction_k_max": float(full.accepted.mean())}

    # (c2) Bounded proposals: on demonstrations saturated at the bound, the first k feasible proposals of each stream.
    bnd = ([-0.5], [0.5])
    sat = TrajectoryDataset([Episode(e.episode_id, e.observations, np.clip(e.actions, -0.5, 0.5)) for e in eps]
                            ).validate()
    bvals = [Finite(), ActionBounds(*bnd)]
    bfull = generate_labels(sat, rep, model, NoisyAction(20, 0.3, action_bounds=bnd), bvals, run_id="kmax", seed=7)
    beq = {}
    for k in (1, 4, 20):
        direct = generate_labels(sat, rep, model, NoisyAction(k, 0.3, action_bounds=bnd), bvals, run_id="d", seed=7)
        view = nested_view(bfull, k, "0" * 64)
        beq[str(k)] = bool(np.array_equal(view.actions, direct.actions)
                           and all(np.array_equal(view.measurements[m], direct.measurements[m])
                                   for m in direct.measurements))
    att = bfull.measurements["generator.attempt_index"]
    out["bounded_proposals"] = {
        "pass": all(beq.values()) and bool(bfull.accepted.all()) and float(np.abs(bfull.actions).max()) <= 0.5
        and bool((att > bfull.measurements["generator.draw_index"]).any()),
        "views_equal_direct_generation": beq, "saturated_demonstration_fraction":
        float(np.mean([np.isclose(np.abs(e.actions[:-1]), 0.5).mean() for e in sat.episodes])),
        "accepted_fraction": float(bfull.accepted.mean()), "skipped_proposals": int(att.max() + 1 - 20)}

    # (d) Refusals.
    import dataclasses
    refusals = {}
    train_cases = {
        "fraction_with_balance": ([v2.Job(sup, 1)], {**SETTINGS, "synthetic_fraction": 0.5, "balance_labels": 2}),
        "fraction_without_synthetic_rows": ([v2.Job(synthetic(300, 0, 1, "bc"), 1)],
                                            {**SETTINGS, "synthetic_fraction": 0.5}),
        "fraction_out_of_range": ([v2.Job(sup, 1)], {**SETTINGS, "synthetic_fraction": 1.0}),
        "patience_without_validation": ([v2.Job(train, 5)], {**SETTINGS, "patience": 5}),
        "min_delta_without_patience": ([v2.Job(train, 5, validation=held)],
                                       {**SETTINGS, "checkpoint_selection": "validation", "min_delta": 0.1}),
        "plateau_without_patience": ([v2.Job(train, 5, validation=held)],
                                     {**sp, "patience": None}),
        "plateau_keys_without_plateau": ([v2.Job(train, 5, validation=held)], {**sp, "lr_schedule": "constant"}),
        "min_lr_not_below_learning_rate": ([v2.Job(train, 5, validation=held)], {**sp, "min_lr": sp["learning_rate"]}),
        "extension_without_patience": ([v2.Job(train, 5, validation=held)],
                                       {**SETTINGS, "checkpoint_selection": "validation", "extend_after_stop": 2}),
    }
    for name, (jobs_, st_) in train_cases.items():
        try:
            v2.train_group(jobs_, st_, device=device, compile=False)
            refusals[name] = False
        except v2.TrainingError:
            refusals[name] = True
    perm = np.random.default_rng(0).permutation(len(full.actions))
    shuffled = dataclasses.replace(full, source_episode_id=full.source_episode_id[perm],
                                   source_frame=full.source_frame[perm])
    # Every column of each transition's rows reversed together: a file whose draws are reordered within their
    # transitions, which a position-based view would silently turn into a different smaller-K view.
    within = (np.arange(len(full.actions)).reshape(-1, 20)[:, ::-1]).reshape(-1)
    reordered = dataclasses.replace(
        full, observations={m: v[within] for m, v in full.observations.items()}, actions=full.actions[within],
        target_observations={m: v[within] for m, v in full.target_observations.items()},
        accepted=full.accepted[within], reasons=full.reasons[within], weight=full.weight[within],
        measurements={m: v[within] for m, v in full.measurements.items()})
    no_ordinal = dataclasses.replace(full, measurements={m: v for m, v in full.measurements.items()
                                                         if m != "generator.draw_index"},
                                     measurement_specs={m: v for m, v in full.measurement_specs.items()
                                                        if m != "generator.draw_index"})
    wrong_order = dataclasses.replace(bfull, measurements={**bfull.measurements, "generator.attempt_index":
                                                           bfull.measurements["generator.attempt_index"][::-1].copy()})
    view_cases = {"k_above_k_max": (full, 21), "k_zero": (full, 0),
                  "attempts_not_increasing": (wrong_order, 4),
                  "rows_not_grouped_by_transition": (shuffled, 4),
                  "draws_reordered_within_transition": (reordered, 4),
                  "no_draw_ordinal": (no_ordinal, 4),
                  "population_weighting": (dataclasses.replace(full, weighting={"name": "inverse_density",
                                                                                "version": "1", "params": {}}), 4),
                  "generator_without_nested_draws": (dataclasses.replace(full, generator={**full.generator,
                                                                                          "name": "backward_euler"}), 1)}
    for name, (c_, k) in view_cases.items():
        try:
            nested_view(c_, k, "0" * 64)
            refusals[name] = False
        except InterfaceError:
            refusals[name] = True
    try:
        generate_labels(ds, rep, model, NoisyAction(4, 0.3, action_bounds=bnd), bvals, run_id="x", seed=7)
        refusals["demonstration_action_outside_bounds"] = False
    except ValueError:
        refusals["demonstration_action_outside_bounds"] = True
    out["refusals"] = {"pass": all(refusals.values()), **refusals}
    return out


def modes_and_determinism(device) -> dict:
    jobs = [v2.Job(synthetic(800, 1500 + 50 * s, s), s) for s in (1, 2, 3)]
    nb = [v2.Job(synthetic(800, 0, s, "noise_bc"), s, NoiseBC("symmetric_uniform_half_batch", 0.1, seed=s)) for s in (1, 2)]

    def params(out):
        return [p.net.state_dict() for p, _ in out]

    def same(a, b):
        return all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)
    held = synthetic(600, 0, 9, "bc")
    fj = [v2.Job(synthetic(800, 1500 + 50 * s, s), s, validation=held) for s in (1, 2, 3)]
    fst = {**SETTINGS, "epochs": 1, "step_budget": 300, "synthetic_fraction": 0.5, "checkpoint_selection": "validation",
           "validation_interval_steps": 10, "patience": 3}  # the shared-harness contract, on the default engine
    pst = {**fst, "step_budget": 600, "lr_schedule": "plateau", "lr_factor": 0.5, "lr_patience": 2, "min_lr": 2.5e-4,
           "extend_after_stop": 1.5}  # host-side reductions reach the compiled step identically in every mode
    res = {}
    for name, js, st in (("ccil", jobs, SETTINGS), ("noise_bc", nb, SETTINGS), ("ccil_fraction_patience", fj, fst),
                         ("ccil_fraction_plateau", fj, pst)):
        a = params(v2.train_group(js, st, device=device, data_mode="resident"))
        b = params(v2.train_group(js, st, device=device, data_mode="streaming", chunk_steps=5))
        c = params(v2.train_group(js, st, device=device, data_mode="resident"))
        res[name] = {"resident_equals_streaming": same(a, b), "rerun_identical": same(a, c)}
    return {"pass": all(v for r in res.values() for v in r.values()), "device": str(device), "cases": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parts", default="correctness,modes")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parts = a.parts.split(",") if a.parts != "all" else ["correctness", "robustness", "recipe", "contract", "modes",
                                                          "equivalence", "realworkload", "scaling"]
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:700]}", flush=True)
    if "correctness" in parts:
        for name, r in correctness(a.device).items():
            record(f"correctness_{name}", r.pop("pass"), **r)
    if "robustness" in parts:
        for name, r in robustness(a.device).items():
            record(f"robustness_{name}", r.pop("pass"), **r)
    if "recipe" in parts:
        for name, r in recipe(a.device).items():
            record(f"recipe_{name}", r.pop("pass"), **r)
    if "contract" in parts:
        for name, r in contract(a.device).items():
            record(f"contract_{name}", r.pop("pass"), **r)
    if "modes" in parts:
        r = modes_and_determinism(a.device)
        record("modes_and_determinism", r.pop("pass"), **r)
    if "equivalence" in parts:
        import equivalence
        r = equivalence.run(out, a.device, EQ_SEEDS, LABEL_SEED)
        record("no_detected_difference_v1_v2", r.pop("pass"), **r)
    if "equivalence_margin" in parts:
        import equivalence
        r = equivalence.run(out, a.device, equivalence.MARGIN_SEEDS, LABEL_SEED, arms=("ccil",),
                            margin=equivalence.MARGIN, tag="equivalence_margin")
        record("equivalence_within_margin_v1_v2_ccil", r.pop("pass"), **r)
    if "realworkload" in parts:
        import realworkload
        r = realworkload.run(out, a.device)
        record("real_workload_timing", r.pop("pass"), **r)
    if "scaling" in parts:
        import scaling
        r = scaling.run(a.device)
        record("scaling", r.pop("pass"), **r)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "numpy": np.__version__, "device": a.device,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
