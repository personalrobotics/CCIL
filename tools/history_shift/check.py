#!/usr/bin/env python3
"""A history state's known shift is not learned (#30): transition_model.structure history_shift, end to end.

    uv run --extra torch python tools/history_shift/check.py --out DIR

The source is a two-part history state, each part [x_{t-1}, x_t]: a PD-controlled agent (state, 2 + 2) and a block it
pushes (state.block, 1 + 1), 10 Hz. The CLI runs generate with representation.modality state+state.block, a spectrally
constrained MLP under transition_model.structure history_shift, history_steps 2, and noisy_action_fe. Then:
1. the network is trained on the newest slots alone: its output, normalization and fit target are the 3 newest
   coordinates, and the checkpoint records its slot layout (steps 2, parts [4, 2]);
2. the shift is exact: on random states and actions, delta's older-slot rows equal z[newer slot] - z[older slot]
   bit for bit, and the Jacobian's are exactly +1 and -1 there; its learned rows match central finite differences;
3. every accepted preimage label's newest slots equal its target's older slots (the source state x_t) within
   eps_struct = max(10 root_tolerance, 10 eps32 max(1, ||x_t||_inf)), the solver's and float32's error alone; the
   same configuration without the structure is reported for contrast, its labels off by the learned shift's error;
4. the labels hold under the reloaded checkpoint, independently of the generator: for every accepted label,
   Phi_hat(h_G, a~) assembled from first principles (the newer slots copied from h_G, the newest from the reloaded
   network) is within root_tolerance of its target, equal to the solver residual the generator recorded; and a Newton
   solve from the reloaded checkpoint, with the finite-difference Jacobian disabled, uses only the structured
   jacobian_z (exact shift rows on every call), converges for every accepted label within root_tolerance, and agrees
   with it within eps_struct (float32 products depend on the batch's composition, so not bit for bit);
5. the post-fit Lipschitz measurement is the learned head's, equal to a numpy recomputation from the saved arrays,
   and the known shift's is recorded apart (sqrt 2 for two slots in unit scales);
6. an unstructured configuration is unchanged: its checkpoint settings, meta and binding carry no history entry;
7. the structure is refused (exit 2, no labels written) with noisy_action, a certificate profile, trainer_version 2,
   history_steps 3 on parts of width 4 and 2, and a checkpoint whose structure is not the run's, either way.
Writes DIR/report.json, with no paths.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ccil.schema import Episode, TrajectoryDataset, read_candidates, write_trajectories  # noqa: E402

JOINED = "state+state.block"
EPS32 = float(np.finfo(np.float32).eps)
SETS = ["data.path=source.npz", "data.validation_fraction=0.25", "compute.device=cpu", "transition_model.kind=mlp",
        "transition_model.trainer_version=1", "transition_model.fit_episodes=train",
        "transition_model.hidden_layers=[64,64]", "transition_model.epochs=40", "transition_model.batch_size=64",
        "transition_model.lipschitz_type=spectral_normalization", "transition_model.lipschitz_constraint=3.0",
        f"representation.modality={JOINED}", "generation.labels_per_transition=2",
        "generation.action_noise_scale=training_std", "generation.action_noise_std=0.3", "generation.recovery_steps=1"]
STRUCTURED = ["transition_model.structure=history_shift", "transition_model.history_steps=2",
              "generation.method=noisy_action_fe"]


def source(path: Path) -> None:
    """24 episodes of 49 frames: the agent's velocity decays and is pulled toward the action; the block moves when
    the agent is near. Each part is stored as [x_{t-1}, x_t]; row k is [x_k, x_{k+1}] with the action taken from
    x_{k+1}, so the stored state is Markov and each transition shifts its history exactly."""
    rng = np.random.default_rng(0)
    eps = []
    for e in range(24):
        p, v, b = rng.uniform(100, 400, 2), np.zeros(2), rng.uniform(150, 350, 1)
        goal = rng.uniform(150, 350, 2)
        P, B, A = [p.copy()], [b.copy()], []
        for _ in range(50):
            a = goal + 15 * rng.normal(size=2)
            v = 0.6 * v + 0.3 * (a - p)
            b = b + 2.0 * np.tanh((p[0] - b) / 40.0)
            p = p + v
            P.append(p.copy()); B.append(b.copy()); A.append(a)
        P, B = np.array(P), np.array(B)
        state = np.concatenate([P[:-1], P[1:]], 1)[:-1]  # frames 1..49 as [x_{t-1}, x_t]
        block = np.concatenate([B[:-1], B[1:]], 1)[:-1]
        eps.append(Episode(str(e), {"state": state.astype(np.float32), "state.block": block.astype(np.float32)},
                           np.array(A[1:], np.float32), timestamps=np.arange(49) / 10.0))
    write_trajectories(TrajectoryDataset(eps), path)


def ccil(*args, cwd):
    return subprocess.run([sys.executable, "-m", "ccil", *args], cwd=cwd, capture_output=True, text=True)


def sets(*extra):
    return [x for s in (*SETS, *extra) for x in ("--set", s)]


def latest(runs: Path, prefix: str) -> Path:
    found = sorted(runs.glob(f"{prefix}-*"), key=lambda p: p.stat().st_mtime)
    return found[-1]


def head_bound(ckpt: Path, maps: str) -> float:
    """The learned head's weight-based bound in raw units, from the checkpoint's arrays with numpy alone."""
    with np.load(ckpt) as z:
        meta = json.loads(str(z["meta"]))
        c = meta["settings"]["lipschitz_constraint"]
        layers = sorted({k.split(".")[1] for k in z.files if k.startswith("dynamics/fc_layers.")}, key=int)
        weights = []
        for i in layers:
            w, u, v = (z[f"dynamics/fc_layers.{i}.weight_{x}"] for x in ("orig", "u", "v"))
            sigma = np.float32(u @ (w @ v))
            weights.append((w / max(sigma / np.float32(c), np.float32(1.0))).astype(np.float64))
        d = z["dynamics/s_scale"].shape[0]
        scale = (z["dynamics/s_scale"] if maps == "state" else z["dynamics/a_scale"]).astype(np.float64) + 1e-8
        out = (z["dynamics/out_scale"].astype(np.float64) + 1e-8) * (z["dynamics/out_scale"] >= 1e-8)
    cols = slice(0, d) if maps == "state" else slice(d, None)
    norms = [np.linalg.norm(weights[0][:, cols] / scale[None, :], 2)] + \
            [np.linalg.norm(w, 2) for w in weights[1:-1]] + [np.linalg.norm(out[:, None] * weights[-1], 2)]
    return float(np.prod(norms))


def history_errors(cand_path: Path, dataset_path: Path):
    """Per accepted label: max over parts of |label's newest slot - target's older slot|, and that row's eps_struct."""
    from ccil.schema import read_trajectories
    c = read_candidates(cand_path, read_trajectories(dataset_path))
    keep = np.asarray(c.accepted, bool)
    err, size = np.zeros(int(keep.sum())), np.zeros(int(keep.sum()))
    for name in ("state", "state.block"):
        z, t = np.asarray(c.observations[name], np.float64)[keep], np.asarray(c.target_observations[name], np.float64)[keep]
        w = z.shape[1] // 2
        err = np.maximum(err, np.abs(z[:, w:] - t[:, :w]).max(1))
        size = np.maximum(size, np.abs(t[:, :w]).max(1))
    return err, size, int(keep.sum()), int(len(keep))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)

    w = out / "work"
    w.mkdir()
    source(w / "source.npz")
    g = ccil("generate", *sets(*STRUCTURED, "run.output_dir=runs"), cwd=w)
    u = ccil("generate", *sets("generation.method=noisy_action_fe", "run.output_dir=runs_plain"), cwd=w)
    if g.returncode or u.returncode:
        record("pipeline_runs", False, structured=g.stderr[-800:], unstructured=u.stderr[-800:])
        return 1
    import torch  # the mlp extra
    from ccil.models.mlp import MLPTransitionModel
    fit = latest(w / "runs", "transition-fit")
    ckpt = fit / "checkpoints" / "transition_model.npz"
    model = MLPTransitionModel.load(ckpt)
    with np.load(ckpt) as z:
        meta = json.loads(str(z["meta"]))
    learned = [2, 3, 5]  # state's newest slot (2, 3) and state.block's (5)
    net_out = model.net.fc_layers[-1].out_features
    record("only_the_newest_slots_are_trained",
           net_out == 3 and list(model.net.out_shift.shape) == [3] and list(model.history["learned"]) == learned
           and meta["history"] == {"steps": 2, "parts": [4, 2]} and meta["settings"]["structure"] == "history_shift"
           and meta["binding"]["representation"]["history"] == {"structure": "history_shift", "steps": 2, "parts": [4, 2]},
           network_outputs=net_out, learned=list(map(int, model.history["learned"])), history=meta["history"],
           binding_history=meta["binding"]["representation"]["history"])

    rng = np.random.default_rng(1)
    zs = rng.uniform(100, 400, (64, 6)).astype(np.float32).astype(np.float64)
    acts = rng.uniform(100, 400, (64, 2))
    d = model.delta(zs, acts)
    dst, src = model.history["shift_dst"], model.history["shift_src"]
    J = model.jacobian_za(zs, acts)
    eps = 0.1  # float32 rounding of delta over 2 eps is about 1.5e-4 here
    fd = np.stack([(model.delta(zs + eps * np.eye(6)[j], acts) - model.delta(zs - eps * np.eye(6)[j], acts)) / (2 * eps)
                   for j in range(6)], -1)
    expect = np.zeros((len(dst), 6))
    expect[np.arange(len(dst)), src], expect[np.arange(len(dst)), dst] = 1.0, -1.0
    record("shift_is_exact",
           np.array_equal(d[:, dst], zs[:, src] - zs[:, dst]) and np.array_equal(J[:, dst, :6], np.broadcast_to(expect, (64, *expect.shape)))
           and not J[:, dst, 6:].any() and np.allclose(J[:, learned, :6], fd[:, learned], atol=5e-3),
           shift_rows=list(map(int, dst)), learned_rows_fd_max_diff=float(np.abs(J[:, learned, :6] - fd[:, learned]).max()))

    tol = 1e-4  # generation.root_tolerance, its default
    labels = latest(w / "runs", "labels-generate") / "labels" / "candidates.npz"
    dataset = latest(w / "runs", "data-prepare") / "data" / "dataset.npz"
    err, size, kept, total = history_errors(labels, dataset)
    bound = np.maximum(10 * tol, 10 * EPS32 * np.maximum(1.0, size))
    plain = latest(w / "runs_plain", "labels-generate") / "labels" / "candidates.npz"
    perr, _, pkept, ptotal = history_errors(plain, latest(w / "runs_plain", "data-prepare") / "data" / "dataset.npz")
    q = lambda x: {p: float(np.percentile(x, p)) for p in (50, 90, 100)} if len(x) else None  # noqa: E731
    record("preimage_labels_keep_the_history",
           kept > 0 and bool((err <= bound).all()) and float(np.median(perr)) > float(np.median(bound)),
           accepted=kept, candidates=total, error=q(err), eps_struct=q(bound),
           unstructured_contrast={"accepted": pkept, "candidates": ptotal, "error": q(perr)})

    import ccil.core.generators as gen
    from ccil.schema import read_trajectories
    reloaded = MLPTransitionModel.load(ckpt)  # a fresh load: the generator's in-process model is not used
    c = read_candidates(labels, read_trajectories(dataset))
    keep = np.asarray(c.accepted, bool)
    names = JOINED.split("+")
    zg = np.concatenate([np.asarray(c.observations[n]) for n in names], 1)[keep]
    tgt = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in names], 1)[keep]
    act = np.asarray(c.actions)[keep]
    # Phi_hat from first principles: newer slots copied, newest slots from the network, in eval mode (a training-mode
    # forward would run the spectral-norm power iteration and change the weights it applies)
    was = reloaded.net.training
    reloaded.net.eval()
    with torch.no_grad():
        head = reloaded.net.predict(torch.as_tensor(zg, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32))
    reloaded.net.train(was)
    head = head.numpy().astype(np.float64)
    nxt = np.empty_like(tgt)
    h = reloaded.history
    nxt[:, h["shift_dst"]] = zg[:, h["shift_src"]]
    nxt[:, h["learned"]] = zg[:, h["learned"]].astype(np.float64) + head
    resid = np.linalg.norm(nxt - tgt, axis=1)
    recorded = np.asarray(c.measurements["generator.solver_residual"], np.float64)[keep]
    ds = read_trajectories(dataset)
    by_id = {ep.episode_id: np.concatenate([ep.observations[n] for n in names], 1) for ep in ds.episodes}
    init = np.stack([by_id[e][f] for e, f in zip(np.asarray(c.source_episode_id)[keep], np.asarray(c.source_frame)[keep])])
    calls = {"n": 0, "shift_rows_exact": True}
    jz = reloaded.jacobian_z

    def checked_jacobian_z(z, a):
        J = jz(z, a)
        calls["n"] += 1
        calls["shift_rows_exact"] &= bool(np.array_equal(J[:, h["shift_dst"]],
                                                         np.broadcast_to(expect, (len(J), *expect.shape))))
        return J

    def no_fd(*_):
        raise AssertionError("Newton fell back to finite differences")

    reloaded.jacobian_z, saved = checked_jacobian_z, gen.jacobian_fd
    gen.jacobian_fd = no_fd
    try:
        solved, failed, _ = gen.newton_preimage(reloaded, tgt, act, init, 50, tol, out_dtype=zg.dtype)
    finally:
        gen.jacobian_fd = saved
    record("labels_hold_under_the_reloaded_checkpoint",
           bool((resid <= tol).all()) and np.allclose(resid, recorded, rtol=0, atol=1e-9) and calls["n"] > 0
           and calls["shift_rows_exact"] and not failed.any()
           and bool((np.abs(solved.astype(np.float64) - zg).max(1) <= bound).all())
           and bool((np.linalg.norm(solved + reloaded.delta(solved, act) - tgt, axis=1) <= tol).all()),
           labels=int(keep.sum()), residual_max=float(resid.max()), root_tolerance=tol,
           residual_vs_recorded_max_diff=float(np.abs(resid - recorded).max()), newton_jacobian_calls=calls["n"],
           newton_shift_rows_exact=calls["shift_rows_exact"], resolved_bitwise=bool(np.array_equal(solved, zg)),
           resolved_max_abs_diff=float(np.abs(solved.astype(np.float64) - zg).max()),
           resolved_residual_max=float(np.linalg.norm(solved + reloaded.delta(solved, act) - tgt, axis=1).max()))

    metrics = json.loads((fit / "metrics.json").read_text())["values"]["lipschitz_measured"]
    rec = {m: head_bound(ckpt, m) for m in ("state", "action")}
    record("lipschitz_is_the_learned_heads_with_the_shift_apart",
           all(abs(metrics[m]["value"] - rec[m]) <= 1e-6 * max(1.0, rec[m]) for m in rec)
           and abs(metrics["known_shift"]["state"] - np.sqrt(2)) < 1e-12 and metrics["known_shift"]["action"] == 0.0
           and "learned head" in metrics["scope"],
           measured={m: metrics[m]["value"] for m in rec}, recomputed=rec, known_shift=metrics["known_shift"])

    plain_ckpt = latest(w / "runs_plain", "transition-fit") / "checkpoints" / "transition_model.npz"
    with np.load(plain_ckpt) as z:
        pmeta = json.loads(str(z["meta"]))
    record("unstructured_configuration_unchanged",
           "history" not in pmeta and not {"structure", "history_steps"} & set(pmeta["settings"])
           and "history" not in pmeta["binding"]["representation"],
           settings=sorted(pmeta["settings"]), binding_representation=sorted(pmeta["binding"]["representation"]))

    refusals = {
        "noisy_action": ("transition_model.structure=history_shift", "transition_model.history_steps=2",
                         "generation.method=noisy_action"),
        "certificate_profile": (*STRUCTURED, "certificate.profile=disturbed_action_paper_proxy"),
        "trainer_version_2": (*STRUCTURED, "transition_model.trainer_version=2"),
        "uneven_slots": ("transition_model.structure=history_shift", "transition_model.history_steps=3",
                         "generation.method=noisy_action_fe"),
        "structured_checkpoint_into_plain_run": ("generation.method=noisy_action_fe", f"transition_model.checkpoint={ckpt}"),
        "plain_checkpoint_into_structured_run": (*STRUCTURED, f"transition_model.checkpoint={plain_ckpt}"),
    }
    reason = {"noisy_action": "noisy_action_fe", "certificate_profile": "no certificate.profile",
              "trainer_version_2": "trainer_version 1", "uneven_slots": "is not 3 equal slots",
              "structured_checkpoint_into_plain_run": "history", "plain_checkpoint_into_structured_run": "history"}
    results = {}
    for i, (name, extra) in enumerate(refusals.items()):
        r = ccil("generate", *sets(*extra, f"run.output_dir=refused{i}"), cwd=w)
        results[name] = {"exit": r.returncode, "wrote_labels": any((w / f"refused{i}").glob("labels-generate-*")),
                         "error": r.stderr.strip().splitlines()[-1][:800] if r.stderr.strip() else ""}
    record("refusals", all(v["exit"] == 2 and not v["wrote_labels"] and reason[k] in v["error"]
                           for k, v in results.items()), **results)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
