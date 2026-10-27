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


def modes_and_determinism(device) -> dict:
    jobs = [v2.Job(synthetic(800, 1500 + 50 * s, s), s) for s in (1, 2, 3)]
    nb = [v2.Job(synthetic(800, 0, s, "noise_bc"), s, NoiseBC("symmetric_uniform_half_batch", 0.1, seed=s)) for s in (1, 2)]

    def params(out):
        return [p.net.state_dict() for p, _ in out]

    def same(a, b):
        return all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)
    res = {}
    for name, js in (("ccil", jobs), ("noise_bc", nb)):
        a = params(v2.train_group(js, SETTINGS, device=device, data_mode="resident"))
        b = params(v2.train_group(js, SETTINGS, device=device, data_mode="streaming", chunk_steps=5))
        c = params(v2.train_group(js, SETTINGS, device=device, data_mode="resident"))
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
    parts = a.parts.split(",") if a.parts != "all" else ["correctness", "robustness", "modes", "equivalence",
                                                          "realworkload", "scaling"]
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
