#!/usr/bin/env python3
"""Evidence for #89 part 2: the experimental vector-policy harness and NoiseBC adapter.

    uv run --group evidence python tools/m3_policy/check.py --out DIR [--device cuda]

On rollouts of the analytic double integrator (8 episodes, a 25% episode
split), with noisy-action labels and disturbed_action certificates:
- ccil policy train writes complete bc, noise_bc, and ccil runs from the same
  prepared data, reusing the label stages; each run has its checkpoint,
  supervision manifest, training record, and offline metrics;
- the legacy protocol (train_bc_policy.py, CustomBC.py, d3rlpy@e89a7da) holds:
  layers, ReLU and tanh, scaler formulas transcribed from d3rlpy, Adam
  settings, floor(N / batch_size) steps, balance_data arithmetic, and the
  unit-weight loss equal to F.mse_loss;
- the weighted loss is the declared formula, and zero-weight minibatches take
  no step;
- the rows seen per epoch reconcile with the supervision manifest;
- the same seed gives byte-identical CPU checkpoints for every arm, and a
  different seed does not;
- NoiseBC draws replay from their key, are fresh per step, never mutate the
  batch or the supervision, and are recorded; legacy and corrected noise are
  compared at the same scale;
- resuming at an epoch boundary is byte-identical to uninterrupted CPU
  training (with --device cuda, the CUDA difference is reported as measured);
- checkpoints load without pickle and record their supervision; with torch
  blocked, ccil labels generate works and ccil policy train refuses with
  install guidance.
Writes DIR/report.json and DIR/artifacts/<arm>/ (manifest, training record, offline metrics).
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
sys.path.insert(0, str(REPO / "tools" / "m3_supervision"))
from system import DoubleIntegrator  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ccil.core import IdentityRepresentation  # noqa: E402
from ccil.core.certificates import write_reference_dynamics  # noqa: E402
from ccil.runs import validate_run  # noqa: E402
from ccil.schema import write_trajectories  # noqa: E402
from ccil.supervision import build_supervision  # noqa: E402
from ccil.training import vector_bc  # noqa: E402
from ccil.training.noise_bc import NoiseBC  # noqa: E402

SEED, K, SIGMA = 20261026, 4, 0.05
SETTINGS = {"hidden_layers": [256, 256], "learning_rate": 1e-3, "batch_size": 32, "epochs": 4,
            "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr_sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:900]}", flush=True)

    from check import rollouts  # tools/m3_supervision/check.py
    sys_ = DoubleIntegrator(0.1)
    rep = IdentityRepresentation("state")
    work = out / "work"
    work.mkdir()
    write_trajectories(rollouts(sys_, SEED), work / "rollouts.npz")
    write_reference_dynamics(work / "true-dynamics.npz", sys_.A, sys_.B, "analytic double integrator")
    base = ["data.path=rollouts.npz", "run.output_dir=runs", "data.validation_fraction=0.25", "compute.device=cpu",
            "generation.method=noisy_action", f"generation.labels_per_transition={K}", f"generation.action_noise_std={SIGMA}",
            "policy.batch_size=32", "policy.epochs=4"]
    certified = ["certificate.profile=disturbed_action", "certificate.reference_dynamics=true-dynamics.npz",
                 "certificate.max_total_error=0.015", "certificate.selection_quantile=0.6"]
    arms = {"bc": ["policy.method=bc"], "noise_bc": ["policy.method=noise_bc", "policy.noise_scale=0.1"],
            "ccil": ["policy.method=ccil", *certified]}

    def cli(sets, cwd=work):
        a = [sys.executable, "-m", "ccil", "policy", "train", "--recipe", "analytic_double_integrator"]
        for s in sets:
            a += ["--set", s]
        return subprocess.run(a, cwd=cwd, capture_output=True, text=True)

    # 1. Three arms from the same prepared data.
    runs, art = {}, out / "artifacts"
    for arm, extra in arms.items():
        r = cli(base + extra)
        stages = {ln.split()[0]: ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip()}
        reused = [ln.split()[0] for ln in r.stdout.splitlines() if " reused " in ln]
        runs[arm] = {"exit": r.returncode, "stages": stages, "reused": reused, "stderr": r.stderr[-300:]}
        if r.returncode == 0:
            tr, ev = work / stages["policy.train"], work / stages["policy.evaluate"]
            (art / arm).mkdir(parents=True)
            shutil.copy(tr / "supervision.json", art / arm / "supervision.json")
            (art / arm / "training.json").write_text(json.dumps(json.loads((tr / "metrics.json").read_text())["values"]["training"],
                                                                indent=1, sort_keys=True) + "\n")
            shutil.copy(ev / "metrics.json", art / arm / "offline-metrics.json")
            runs[arm].update(status={s: validate_run(work / d)[0] for s, d in stages.items()},
                             checkpoint=f"<out>/work/{stages['policy.train']}/checkpoints/policy.npz",
                             offline=json.loads((ev / "metrics.json").read_text())["values"])
    prepared = {a: v["stages"].get("data.prepare") for a, v in runs.items()}
    labels = {a: v["stages"].get("labels.generate") for a, v in runs.items()}
    record("three_arms_from_the_same_prepared_data",
           all(v["exit"] == 0 and set(v["status"].values()) == {"complete"} for v in runs.values())
           and len(set(prepared.values())) == 1 and len(set(labels.values())) == 1
           and "labels.certify" in runs["ccil"]["stages"] and "data.prepare" in runs["noise_bc"]["reused"],
           runs={a: {k: v[k] for k in v if k != "stderr"} for a, v in runs.items()}, artifacts="<out>/artifacts/<arm>/")

    # Supervision objects for the API-level cases.
    d = work / "runs"
    P = {k: next(d.glob(f"{k}-*")) for k in ("data-prepare", "labels-generate", "labels-certify")}
    sup = {"bc": build_supervision(P["data-prepare"] / "data/dataset.npz", "bc", representation=rep,
                                   split=P["data-prepare"] / "data/split.json"),
           "ccil": build_supervision(P["data-prepare"] / "data/dataset.npz", "ccil", representation=rep,
                                     candidates=P["labels-generate"] / "labels/candidates.npz",
                                     certificates=P["labels-certify"] / "labels/certificates.npz",
                                     split=P["data-prepare"] / "data/split.json")}
    sup["noise_bc"] = build_supervision(P["data-prepare"] / "data/dataset.npz", "noise_bc", representation=rep,
                                        split=P["data-prepare"] / "data/split.json")

    def noise(mode="symmetric_uniform_half_batch", scale=0.1, seed=SEED):
        return NoiseBC(mode, scale, seed=seed, units="standardized observation units")

    # 2. The legacy protocol.
    pol, rec = vector_bc.train_vector_bc(sup["ccil"], {**SETTINGS, "epochs": 1}, seed=SEED)
    layers = [type(m).__name__ + (f"({m.in_features},{m.out_features})" if hasattr(m, "in_features") else "") for m in pol.net]
    obs, act, _ = sup["ccil"].arrays()
    o, a = np.asarray(obs, np.float32), np.asarray(act, np.float32)
    # d3rlpy StandardScaler.fit: float64 sums over transitions, population std; transform in float32 with eps 1e-3.
    total = np.zeros(o.shape[1])
    for x in o:
        total += np.asarray(x)
    mean = total / len(o)
    sq = np.zeros(o.shape[1])
    for x in o:
        sq += (np.asarray(x) - mean) ** 2
    std = np.sqrt(sq / len(o))
    d3_obs = (torch.as_tensor(o) - torch.tensor(mean, dtype=torch.float32)) / (torch.tensor(std, dtype=torch.float32) + 1e-3)
    lo, hi = a.min(axis=0), a.max(axis=0)
    d3_act = ((torch.as_tensor(a) - torch.tensor(lo)) / (torch.tensor(hi) - torch.tensor(lo) + 1e-8)) * 2.0 - 1.0
    opt = vector_bc._optimizer(pol, 1e-3)
    g = opt.param_groups[0]
    pred, target = torch.randn(32, 1), torch.randn(32, 1)
    L = 4
    pol_b, rec_b = vector_bc.train_vector_bc(sup["ccil"], {**SETTINGS, "epochs": 8, "balance_labels": L}, seed=SEED)
    man = sup["ccil"].manifest()["counts"]
    record("legacy_protocol",
           layers == ["Linear(2,256)", "ReLU", "Linear(256,256)", "ReLU", "Linear(256,1)", "Tanh"]
           and torch.equal(pol.scale_obs(torch.as_tensor(o)), d3_obs) and torch.equal(pol.scale_act(torch.as_tensor(a)), d3_act)
           and np.allclose(pol.unscale_act(pol.scale_act(torch.as_tensor(a))).numpy(), a, atol=1e-6)
           and (g["betas"], g["eps"], g["weight_decay"], g["amsgrad"]) == ((0.9, 0.999), 1e-8, 0.0, False)
           and rec["steps_per_epoch"] == len(o) // 32 and rec["rows_dropped_per_epoch"] == len(o) % 32
           and rec_b["rows"] == man["synthetic_rows"] + man["demonstration_rows"] * (L + 1) and rec_b["epochs_run"] == 8 // L
           and torch.equal(vector_bc.weighted_loss(pred, target, torch.ones(32)), F.mse_loss(pred, target)),
           layers=layers, steps_per_epoch=rec["steps_per_epoch"], rows=rec["rows"],
           balance={"labels_per_transition": L, "rows": rec_b["rows"], "demonstration_rows": man["demonstration_rows"],
                    "synthetic_rows": man["synthetic_rows"], "epochs": f"8 // {L} = {rec_b['epochs_run']}"},
           sources="archive/upstream-main-2026-10-22 correct_il/{train_bc_policy,CustomBC,utils}.py; personalrobotics/d3rlpy@e89a7da")

    # 3. Weights: the declared formula, and zero-weight batches take no step.
    w = torch.rand(32)
    manual = float((w.numpy() * ((pred - target).numpy() ** 2).mean(axis=1)).sum() / w.numpy().sum())
    zero = build_supervision(P["data-prepare"] / "data/dataset.npz", "bc", representation=rep,
                             split=P["data-prepare"] / "data/split.json", demonstration_weight=0.0)
    torch.manual_seed(SEED)
    init = vector_bc.VectorBCPolicy(2, 1, [256, 256])
    pol_z, rec_z = vector_bc.train_vector_bc(zero, {**SETTINGS, "epochs": 2}, seed=SEED)
    record("weights_follow_the_declared_loss",
           np.isclose(float(vector_bc.weighted_loss(pred, target, w)), manual, rtol=1e-6)
           and all(torch.equal(p, q) for p, q in zip(init.net.state_dict().values(), pol_z.net.state_dict().values()))
           and all(e["zero_weight_steps"] == e["steps"] for e in rec_z["epochs"]),
           weighted_loss=float(vector_bc.weighted_loss(pred, target, w)), manual=manual,
           zero_weight_steps=[e["zero_weight_steps"] for e in rec_z["epochs"]])

    # 4. Data-count reconciliation.
    _, rec4 = vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED)
    rows = vector_bc.training_rows(sup["ccil"], None)
    seen = set()
    for e in range(SETTINGS["epochs"]):
        perm = rows[np.random.default_rng(np.random.SeedSequence([SEED, e])).permutation(len(rows))]
        seen |= set(perm[: rec4["steps_per_epoch"] * 32].tolist())
    record("rows_reconcile_with_the_supervision_manifest",
           rec4["row_sources"] == {"demonstration": man["demonstration_rows"], "synthetic": man["synthetic_rows"]}
           and all(sum(e["rows_seen"].values()) == e["steps"] * 32 for e in rec4["epochs"])
           and seen <= set(range(len(sup["ccil"].weights))),
           manifest_counts=man, row_sources=rec4["row_sources"], rows_seen_per_epoch=[e["rows_seen"] for e in rec4["epochs"]],
           distinct_rows_seen=len(seen), rows=len(rows))

    # 5. Determinism.
    det = {}
    for arm in ("bc", "noise_bc", "ccil"):
        paths = []
        for i, s in enumerate((SEED, SEED, SEED + 1)):
            p_, _ = vector_bc.train_vector_bc(sup[arm], SETTINGS, seed=s, noise=noise(seed=s) if arm == "noise_bc" else None)
            paths.append(out / f"det-{arm}-{i}.npz")
            p_.save(paths[-1])
        det[arm] = {"same_seed_identical": sha(paths[0]) == sha(paths[1]), "other_seed_differs": sha(paths[0]) != sha(paths[2])}
    record("cpu_training_is_deterministic", all(all(v.values()) for v in det.values()), arms=det)

    # 6. NoiseBC: replay, freshness, no mutation, recorded.
    captured = []
    nz = noise()
    original = nz.perturb

    def spy(batch, ids, frames, step, draw=0):
        before = arr_sha(batch)
        res = original(batch, ids, frames, step, draw)
        captured.append({"step": step, "ids": np.array(ids), "frames": np.array(frames), "batch": batch.copy(),
                         "noise": res[1]["noise"], "mask": res[1]["mask"], "mutated": arr_sha(batch) != before})
        return res
    nz.perturb = spy
    sup_before = arr_sha(*sup["noise_bc"].arrays())
    _, rec6 = vector_bc.train_vector_bc(sup["noise_bc"], {**SETTINGS, "epochs": 2}, seed=SEED, noise=nz)
    replay = all(np.array_equal(noise().perturb(c["batch"], c["ids"], c["frames"], c["step"], 0)[1]["noise"], c["noise"])
                 for c in captured[:5])
    first, second = captured[0], captured[1]
    common = {(e, int(f)) for e, f in zip(first["ids"], first["frames"])} & {(e, int(f)) for e, f in zip(second["ids"], second["frames"])}
    # Freshness: the same sample's draw at step 0 and step 1 differs (the key includes the step).
    from ccil.training.noise_bc import counter_uniforms
    u = [counter_uniforms(SEED, step, first["ids"], first["frames"], 0, 3) for step in (0, 1)]
    fresh = bool(not np.isclose(u[0], u[1]).any())
    del second, common
    trace = [{"step": c["step"], "rows": len(c["ids"]), "masked": int(c["mask"].sum()),
              "noise_sha256": arr_sha(c["noise"])[:16]} for c in captured[:4]]
    desc = rec6["noise_bc"]
    record("noise_bc_replays_fresh_and_never_mutates",
           replay and fresh is not False and not any(c["mutated"] for c in captured)
           and arr_sha(*sup["noise_bc"].arrays()) == sup_before
           and {"mode", "scale", "units", "truncation", "mask_probability", "seed", "rng", "draw_schedule", "trace_sha256",
                "version", "distribution"} <= set(desc)
           and len(desc["trace_epochs"]) == 2,
           trace=trace, replayed_draws=5, fresh_across_steps=fresh, descriptor={k: desc[k] for k in desc if k != "trace_epochs"})

    # 7. Legacy versus corrected NoiseBC at the same scale.
    from ccil.core import transitions
    from ccil.schema import read_trajectories
    batch = transitions(read_trajectories(P["data-prepare"] / "data/dataset.npz"), rep)
    split = json.loads((P["data-prepare"] / "data/split.json").read_text())
    val = np.isin(batch.episode_id, split["validation"])
    cmp = {}
    for mode in ("legacy_uniform_half_batch", "symmetric_uniform_half_batch"):
        cap = []
        nzm = noise(mode)
        orig = nzm.perturb

        def spy2(batch, ids, frames, step, draw=0, orig=orig, cap=cap):
            res = orig(batch, ids, frames, step, draw)
            cap.append((res[1]["mask"], res[1]["noise"]))
            return res
        nzm.perturb = spy2
        pm, _ = vector_bc.train_vector_bc(sup["noise_bc"], SETTINGS, seed=SEED, noise=nzm)
        masked = np.concatenate([n[m] for m, n in cap])
        err = ((pm.predict(batch.z) - batch.a) ** 2).sum(axis=1)
        cmp[mode] = {"mask_fraction": float(np.mean(np.concatenate([m for m, _ in cap]))),
                     "noise_mean": float(masked.mean()), "noise_min": float(masked.min()), "noise_max": float(masked.max()),
                     "expert_data_error": float(err.mean()), "validation_action_error": float(err[val].mean())}
    record("legacy_versus_corrected_noise_bc",
           0.04 < cmp["legacy_uniform_half_batch"]["noise_mean"] < 0.06 and abs(cmp["symmetric_uniform_half_batch"]["noise_mean"]) < 0.01
           and cmp["legacy_uniform_half_batch"]["noise_min"] >= 0 and cmp["symmetric_uniform_half_batch"]["noise_min"] < 0,
           scale=0.1, comparison=cmp, note="reported, not tuned; noise in standardized observation units")

    # 8. Resume at an epoch boundary.
    res = {}
    for arm in ("bc", "noise_bc", "ccil"):
        nz_a = (lambda: noise()) if arm == "noise_bc" else (lambda: None)
        ck = {}

        def keep(policy, opt, epoch, state, ck=ck):
            path = out / f"resume-{arm}-epoch{epoch}.npz"
            policy.save(path, opt, epoch, state)
            ck[epoch] = (path, state)
        straight, _ = vector_bc.train_vector_bc(sup[arm], SETTINGS, seed=SEED, noise=nz_a(), on_epoch=keep)
        resumed, _ = vector_bc.train_vector_bc(sup[arm], SETTINGS, seed=SEED, noise=nz_a(), resume=ck[2][0])
        a_, b_ = out / f"straight-{arm}.npz", out / f"resumed-{arm}.npz"
        straight.save(a_)
        resumed.save(b_)
        res[arm] = {"identical": sha(a_) == sha(b_)}
    cuda = None
    if args.device == "cuda" and torch.cuda.is_available():
        ck = {}

        def keep_cuda(policy, opt, epoch, state):
            path = out / f"resume-cuda-epoch{epoch}.npz"
            policy.save(path, opt, epoch, state)
            ck[epoch] = path
        s1, _ = vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, device="cuda", on_epoch=keep_cuda)
        s2, _ = vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, device="cuda")
        r1, _ = vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, device="cuda", resume=ck[2])

        def maxdiff(x, y):
            return max(float((p.cpu() - q.cpu()).abs().max()) for p, q in zip(x.net.state_dict().values(), y.net.state_dict().values()))
        cuda = {"straight_vs_straight_max_abs": maxdiff(s1, s2), "straight_vs_resumed_max_abs": maxdiff(s1, r1),
                "device": torch.cuda.get_device_name(0)}
    # Resume is refused for any checkpoint that does not belong to the requested run.
    ck = {}

    def keep_ccil(policy, opt, epoch, state):
        path = out / f"identity-ccil-epoch{epoch}.npz"
        policy.save(path, opt, epoch, state)
        ck[epoch] = path
    vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, on_epoch=keep_ccil)
    bc_ck = {}

    def keep_bc(policy, opt, epoch, state):
        path = out / f"identity-bc-epoch{epoch}.npz"
        policy.save(path, opt, epoch, state)
        bc_ck[epoch] = path
    vector_bc.train_vector_bc(sup["bc"], SETTINGS, seed=SEED, on_epoch=keep_bc)
    nb_ck = {}

    def keep_nb(policy, opt, epoch, state):
        path = out / f"identity-noise-epoch{epoch}.npz"
        policy.save(path, opt, epoch, state)
        nb_ck[epoch] = path
    vector_bc.train_vector_bc(sup["noise_bc"], SETTINGS, seed=SEED, noise=noise(), on_epoch=keep_nb)
    from ccil.schema.io import _atomic_savez
    with np.load(ck[2], allow_pickle=False) as z:
        partial = {k: z[k] for k in z.files if k != "adam/0/exp_avg_sq"}
    _atomic_savez(out / "identity-incomplete-adam.npz", partial)

    def refused(fn):
        try:
            fn()
            return "accepted"
        except vector_bc.TrainingError as e:
            return f"TrainingError: {str(e)[:220]}"
    refusals = {
        "bc_checkpoint_resumed_as_ccil": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, resume=bc_ck[2])),
        "other_seed": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED + 1, resume=ck[2])),
        "other_learning_rate": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], {**SETTINGS, "learning_rate": 3e-4}, seed=SEED, resume=ck[2])),
        "other_batch_size": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], {**SETTINGS, "batch_size": 16}, seed=SEED, resume=ck[2])),
        "other_balancing": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], {**SETTINGS, "epochs": 8, "balance_labels": 4}, seed=SEED, resume=ck[2])),
        "other_supervision": refused(lambda: vector_bc.train_vector_bc(
            build_supervision(P["data-prepare"] / "data/dataset.npz", "ccil", representation=rep,
                              candidates=P["labels-generate"] / "labels/candidates.npz",
                              split=P["data-prepare"] / "data/split.json"), SETTINGS, seed=SEED, resume=ck[2])),
        "other_noise_scale": refused(lambda: vector_bc.train_vector_bc(sup["noise_bc"], SETTINGS, seed=SEED, noise=noise(scale=0.2), resume=nb_ck[2])),
        "inference_checkpoint": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, resume=out / "straight-ccil.npz")),
        "incomplete_adam_state": refused(lambda: vector_bc.train_vector_bc(sup["ccil"], SETTINGS, seed=SEED, resume=out / "identity-incomplete-adam.npz")),
    }
    reason = {"bc_checkpoint_resumed_as_ccil": "method", "other_seed": "seed", "other_learning_rate": "settings",
              "other_batch_size": "settings", "other_balancing": "settings", "other_supervision": "supervision",
              "other_noise_scale": "noise_bc", "inference_checkpoint": "only a resumable checkpoint",
              "incomplete_adam_state": "incomplete"}
    # The command level restarts an interrupted policy.train; determinism makes the rerun identical.
    reruns = []
    for i in range(2):
        wd = out / f"rerun-{i}"
        wd.mkdir()
        shutil.copy(work / "rollouts.npz", wd / "rollouts.npz")
        shutil.copy(work / "true-dynamics.npz", wd / "true-dynamics.npz")
        r = cli(base + arms["ccil"], cwd=wd)
        reruns.append(sha(next((wd / "runs").glob("policy-train-*")) / "checkpoints/policy.npz") if r.returncode == 0 else None)
    record("resume_matches_uninterrupted_training",
           all(v["identical"] for v in res.values())
           and all(v.startswith("TrainingError") and reason[k] in v for k, v in refusals.items())
           and reruns[0] is not None and reruns[0] == reruns[1],
           cpu=res, cuda_measured=cuda, refused=refusals, command_level_rerun_identical=reruns[0] == reruns[1],
           contract="library resume: CPU byte-identical, identity-checked; CUDA reported as measured, not claimed. "
                    "Command level: an interrupted policy.train restarts, and determinism reproduces it.")

    # 9. Checkpoints and dependency isolation.
    ckpt = work / runs["ccil"]["stages"]["policy.train"] / "checkpoints/policy.npz"
    with np.load(ckpt, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        no_pickle = all(z[k].dtype != object for k in z.files)
    blocked = ("import sys, runpy; sys.modules['torch'] = None; sys.argv = ['ccil'] + sys.argv[1:]; "
               "runpy.run_module('ccil', run_name='__main__')")
    iso = out / "isolation"
    iso.mkdir()
    shutil.copy(work / "rollouts.npz", iso / "rollouts.npz")
    lab = subprocess.run([sys.executable, "-c", blocked, "labels", "generate", "--recipe", "analytic_double_integrator",
                          "--set", "data.path=rollouts.npz", "--set", "run.output_dir=runs"], cwd=iso, capture_output=True, text=True)
    pt = subprocess.run([sys.executable, "-c", blocked, "policy", "train", "--recipe", "analytic_double_integrator",
                         *sum([["--set", s] for s in base + arms["bc"]], [])], cwd=iso, capture_output=True, text=True)
    record("checkpoints_and_dependency_isolation",
           no_pickle and meta["trainer"] == "vector_mlp" and meta["supervision"].get("certificates")
           and lab.returncode == 0 and pt.returncode == 2 and "ccil[torch]" in pt.stderr,
           checkpoint_meta={k: meta[k] for k in ("trainer", "version", "epoch", "method", "seed")},
           records_supervision_inputs=sorted(meta["supervision"]), labels_generate_without_torch=lab.returncode,
           policy_train_without_torch=pt.returncode, refusal=pt.stderr.strip()[-200:])

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                              "torch": torch.__version__, "device_requested": args.device}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    for p in art.rglob("*.json"):
        p.write_text(p.read_text().replace(str(out), "<out>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
