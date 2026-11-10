#!/usr/bin/env python3
"""A curated selection handed to an unchanged LeRobot trainer (#30), on a two-feature LeRobot fixture.

    uv run --extra lerobot --extra torch python tools/m4_export/check.py --out DIR [--lerobot-python PY]

A LeRobot v3.0 source with observation.state (2) and observation.environment_state (4) is converted, and the
CLI runs generate (representation.modality state+state.environment_state, a spectrally constrained MLP, noisy_action
with noise in training-action standard deviations, recovery length 1), curate, and export lerobot. Then:
1. labels keep each modality under its own name, and the supervision rows concatenate them in the declared order;
2. a checkpoint fitted on one modality order loads under that order and is refused under the other (exit 2);
3. the post-fit Lipschitz measurement equals a numpy recomputation from the saved checkpoint's arrays;
4. training_std noise resolves to 0.5 x each action dimension's standard deviation over the training transitions;
5. the export holds the training episodes' frames exactly as the source holds them, one one-frame episode per usable
   label with exactly its observation and action, no held-out episode, and a source-index hash that recomputes;
6. in LeRobot 0.6.1 (its own environment, built under DIR unless --lerobot-python names one), stock LeRobotDataset
   and ACT (n_obs_steps 1, chunk_size 1) return each requested frame as exactly its row with an unpadded action,
   read only the two observation features, and train one step with a finite loss and gradients;
7. an existing output, another dataset's conversion, a non-unit label weight, and a changed export file are refused
   (exit 2), and nothing is written;
8. a recorded split is honored whatever supervision.exclude_validation_episodes says: with it false, the export
   still holds no held-out episode and no label of a held-out transition;
9. a joined modality concatenates vectors only: an image part is refused by the configuration (exit 2) and by the
   representation itself when built directly.
Writes DIR/report.json, with no paths.
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ccil import InputError  # noqa: E402
from ccil.core import IdentityRepresentation  # noqa: E402
from ccil.core.representation import transitions  # noqa: E402
from ccil.lerobot import convert, write_dataset  # noqa: E402
from ccil.schema import Episode, TrajectoryDataset, read_candidates, read_trajectories  # noqa: E402
from ccil.supervision import build_supervision  # noqa: E402

LEROBOT = "0.6.1"
JOINED, REVERSED = "state+state.environment_state", "state.environment_state+state"
SETS = ["data.path=source.npz", "data.validation_fraction=0.25", "compute.device=cpu", "transition_model.kind=mlp",
        "transition_model.trainer_version=1", "transition_model.fit_episodes=train", "transition_model.hidden_layers=[32,32]", "transition_model.epochs=20",
        "transition_model.batch_size=64", "transition_model.lipschitz_type=spectral_normalization",
        "transition_model.lipschitz_constraint=2.0", "generation.method=noisy_action",
        "generation.labels_per_transition=2", "generation.action_noise_scale=training_std",
        "generation.action_noise_std=0.5", "generation.recovery_steps=1"]


def source(path: Path) -> None:
    """16 episodes of 40 frames at 10 Hz: an agent tracking its action target, pushing a block it is near."""
    rng = np.random.default_rng(0)
    eps = []
    for e in range(16):
        s, b = rng.uniform(100, 400, 2), rng.uniform(150, 350, 4)
        goal = rng.uniform(150, 350, 2)
        S, B, A = [], [], []
        for _ in range(40):
            a = goal + 20 * rng.normal(size=2)
            S.append(s.copy()); B.append(b.copy()); A.append(a)
            push = 0.2 * np.tanh((s - b[:2]) / 50.0)
            s = s + 0.3 * (a - s)
            b = b + np.concatenate([push, push])
        eps.append(Episode(str(e), {"state": np.array(S, np.float32), "state.environment_state": np.array(B, np.float32)},
                           np.array(A, np.float32), timestamps=np.arange(40) / 10.0, task="push the block"))
    write_dataset(TrajectoryDataset(eps), path, fps=10)


def ccil(*args, cwd, prelude=None):
    cmd = [sys.executable, "-m", "ccil", *args] if prelude is None else \
        [sys.executable, "-c", prelude + "import sys\nfrom ccil.__main__ import main\nsys.exit(main(sys.argv[1:]))\n", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def sets(*extra):
    return [x for s in (*SETS, *extra) for x in ("--set", s)]


def recomputed_lipschitz(ckpt: Path, maps: str) -> float:
    """The weight-based bound of f_hat in raw units, from the checkpoint's arrays with numpy alone."""
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
    first = weights[0][:, cols] / scale[None, :]
    norms = [np.linalg.norm(first, 2)] + [np.linalg.norm(w, 2) for w in weights[1:-1]] + \
            [np.linalg.norm(out[:, None] * weights[-1], 2)]
    return float(np.prod(norms))


def lerobot_python(out: Path, given) -> Path:
    if given:
        return Path(given)
    env = out / "envs" / LEROBOT
    subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(env)], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(env / "bin" / "python"), f"lerobot[dataset]=={LEROBOT}"],
                   check=True)
    return env / "bin" / "python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lerobot-python", help="a Python with lerobot 0.6.1 installed (default: build one under --out)")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)

    w = out / "work"
    w.mkdir()
    source(w / "lerobot")
    convert(w / "lerobot", w / "source.npz", w / "conversion.json")
    g = ccil("generate", *sets(f"representation.modality={JOINED}", "run.output_dir=runs"), cwd=w)
    c = ccil("curate", "runs/generate-result.json", cwd=w)
    if g.returncode or c.returncode:
        record("pipeline_runs", False, generate=g.stderr[-600:], curate=c.stderr[-600:])
        return 1
    runs = w / "runs"
    gen = json.loads((runs / "generate-result.json").read_text())
    cur_path = sorted(runs.glob("curate-*.json"))[0]
    cur = json.loads(cur_path.read_text())
    art = {k: runs / v["path"] for k, v in {**gen["artifacts"], **cur["artifacts"]}.items()}
    cands = read_candidates(art["candidates"])
    split = json.loads(art["split"].read_text())

    # 1. Each modality under its own name; supervision concatenates them in order.
    sup = build_supervision(art["dataset"], "demonstrations_plus_selection", representation=IdentityRepresentation(JOINED),
                            split=art["split"], candidates=art["candidates"], selection=art["selection"])
    syn = sup.source == "synthetic"
    rows = sup.candidate_row[syn]
    expect = np.hstack([cands.observations["state"][rows], cands.observations["state.environment_state"][rows]])
    record("labels_keep_modalities_and_supervision_concatenates_in_order",
           sorted(cands.observations) == ["state", "state.environment_state"] and syn.any()
           and np.array_equal(sup.observations[syn], expect), modalities=sorted(cands.observations),
           synthetic_rows=int(syn.sum()), width=int(sup.observations.shape[1]))

    # 2. The checkpoint is bound to the modality order.
    ckpt = runs / gen["artifacts"]["transition_model"]["path"]
    same = ccil("generate", *sets(f"representation.modality={JOINED}", "run.output_dir=runs_same",
                                  f"transition_model.checkpoint={ckpt}"), cwd=w)
    other = ccil("generate", *sets(f"representation.modality={REVERSED}", "run.output_dir=runs_reversed",
                                   f"transition_model.checkpoint={ckpt}"), cwd=w)
    labels_written = any((w / "runs_reversed").glob("labels-generate-*")) if (w / "runs_reversed").exists() else False
    record("checkpoint_bound_to_modality_order", same.returncode == 0 and other.returncode == 2
           and "representation" in other.stderr and "Traceback" not in other.stderr and not labels_written,
           same_order_exit=same.returncode, reversed_exit=other.returncode, error=other.stderr.strip()[-240:])

    # 3. The post-fit Lipschitz measurement, recomputed from the saved arrays.
    fit = next(runs / s["run_dir"] for s in gen["stages"] if s["stage"] == "transition.fit")
    measured = json.loads((fit / "metrics.json").read_text())["values"]["lipschitz_measured"]
    again = {m: recomputed_lipschitz(ckpt, m) for m in ("state", "action")}
    record("lipschitz_measured_from_the_saved_weights",
           all(np.isclose(measured[m]["value"], again[m], rtol=1e-5) for m in again),
           measured={m: measured[m]["value"] for m in again}, recomputed=again, scope=measured["scope"])

    # 4. Noise in training-action standard deviations.
    batch = transitions(read_trajectories(art["dataset"]), IdentityRepresentation(JOINED))
    train = np.isin(batch.episode_id.astype(str), split["train"])
    std = np.asarray(batch.a, np.float64)[train].std(axis=0)
    sigma = cands.generator["params"]["action_noise_std"]
    record("noise_resolved_from_training_action_std", np.allclose(sigma, 0.5 * std, rtol=1e-12),
           sigma=sigma, training_std=std.tolist())

    # 5. The export against the source and the candidates.
    e = ccil("export", "lerobot", str(cur_path.relative_to(w)), "--conversion", "conversion.json", "--out", "export",
             "--revision", "fixture", cwd=w)
    exp = w / "export"
    import pyarrow.parquet as pq
    table = pq.read_table(sorted((exp / "data").rglob("*.parquet"))).to_pydict() if exp.exists() else {}
    manifest = json.loads((exp / "ccil" / "export.json").read_text()) if exp.exists() else {}
    src = {int(ep.episode_id): ep for ep in read_trajectories(w / "source.npz").episodes}
    synthetic = np.array(table.get("ccil.synthetic", []), bool)
    ep_col, fr_col = np.array(table.get("ccil.source_episode", [])), np.array(table.get("ccil.source_frame", []))
    state, env, act = (np.array(table.get(k, []), np.float32) for k in
                       ("observation.state", "observation.environment_state", "action"))
    demo_ok = bool(len(synthetic)) and all(
        np.array_equal(state[i], src[ep_col[i]].observations["state"][fr_col[i]])
        and np.array_equal(env[i], src[ep_col[i]].observations["state.environment_state"][fr_col[i]])
        and np.array_equal(act[i], src[ep_col[i]].actions[fr_col[i]]) for i in np.flatnonzero(~synthetic))
    crow = np.array(table.get("ccil.candidate_row", []))[synthetic]
    label_ok = bool(synthetic.any()) and np.array_equal(crow, rows) and \
        np.array_equal(state[synthetic], cands.observations["state"][crow].astype(np.float32)) and \
        np.array_equal(env[synthetic], cands.observations["state.environment_state"][crow].astype(np.float32)) and \
        np.array_equal(act[synthetic], cands.actions[crow].astype(np.float32))
    held = {int(x) for x in split["validation"]}
    starts = np.cumsum([0] + [src[i].length for i in range(len(src))])
    index = np.sort(np.array([starts[ep_col[i]] + fr_col[i] for i in np.flatnonzero(synthetic)], np.int64))
    record("export_holds_the_source_frames_and_the_labels", e.returncode == 0 and demo_ok and label_ok
           and not (set(ep_col[~synthetic].tolist()) & held)
           and set(ep_col[~synthetic].tolist()) == {int(x) for x in split["train"]}
           and manifest.get("synthetic_source_index_sha256") == hashlib.sha256(index.tobytes()).hexdigest()
           and manifest["counts"]["synthetic_rows"] == int(synthetic.sum()) == int(syn.sum()),
           counts=manifest.get("counts"), held_out_episodes=sorted(held), error=e.stderr[-300:])

    # 6. Stock LeRobot 0.6.1 and ACT.
    py = lerobot_python(out, a.lerobot_python)
    first_syn = int(np.flatnonzero(synthetic)[0])
    picks = [0, 5, first_syn, first_syn + 1, len(synthetic) - 1]
    (w / "picks.json").write_text(json.dumps(picks))
    r = subprocess.run([str(py), str(HERE / "lerobot_side.py"), str(exp), str(w / "picks.json"), str(w / "lerobot.json")],
                       capture_output=True, text=True, cwd=w)
    side = json.loads((w / "lerobot.json").read_text()) if r.returncode == 0 else {}
    items_ok = bool(side) and all(
        np.allclose(it["observation.state"], state[int(i)]) and np.allclose(it["observation.environment_state"], env[int(i)])
        and np.allclose(it["action"], [act[int(i)]]) and it["action_is_pad"] == [False]
        and it["ccil.synthetic"] == int(synthetic[int(i)]) for i, it in side.get("items", {}).items())
    version = subprocess.run([str(py), "-c", "import lerobot; print(lerobot.__version__)"], capture_output=True,
                             text=True).stdout.strip()
    record("stock_lerobot_act_consumes_each_row", r.returncode == 0 and items_ok and version == LEROBOT
           and side["input_features"] == ["observation.environment_state", "observation.state"]
           and side["delta_timestamps"] == {"action": [0.0]} and np.isfinite(side["act"]["loss"])
           and side["act"]["finite_gradients"], lerobot=version, picks=picks,
           delta_timestamps=side.get("delta_timestamps"), input_features=side.get("input_features"),
           act=side.get("act"), error=r.stderr[-400:] if r.returncode else "")

    # 7. Refusals: nothing is written.
    def leaves(path):
        return sorted(p.name for p in path.iterdir())
    before = leaves(w)
    again_out = ccil("export", "lerobot", str(cur_path.relative_to(w)), "--conversion", "conversion.json", "--out", "export",
                     cwd=w)
    other_conv = json.loads((w / "conversion.json").read_text())
    other_conv["output"]["sha256"] = "0" * 64
    (w / "other-conversion.json").write_text(json.dumps(other_conv))
    before = leaves(w)
    wrong = ccil("export", "lerobot", str(cur_path.relative_to(w)), "--conversion", "other-conversion.json", "--out",
                 "export_wrong", cwd=w)
    halved = ccil("export", "lerobot", str(cur_path.relative_to(w)), "--conversion", "conversion.json", "--out",
                  "export_halved", cwd=w,
                  prelude="import ccil.supervision as s\n_b = s.build_supervision\n"
                          "def half(*a, **k):\n    x = _b(*a, **k)\n    x.weights[x.source == 'synthetic'] *= 0.5\n    return x\n"
                          "s.build_supervision = half\n")
    after = leaves(w)
    victim = sorted((exp / "data").rglob("*.parquet"))[0]
    original = victim.read_bytes()
    victim.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    ins = ccil("inspect", "export", cwd=w)
    victim.write_bytes(original)
    refusals = {"existing_output": again_out.returncode, "other_conversion": wrong.returncode,
                "non_unit_weight": halved.returncode, "changed_file": ins.returncode}
    record("refusals_write_nothing", set(refusals.values()) == {2} and before == after
           and "weights other than 1" in halved.stderr and "not this conversion's output" in wrong.stderr
           and "changed since export" in ins.stderr
           and not any("Traceback" in x.stderr for x in (again_out, wrong, halved, ins)), exits=refusals)

    # 8. The split is honored even when supervision.exclude_validation_episodes is false.
    gf = ccil("generate", *sets(f"representation.modality={JOINED}", "run.output_dir=runs_flag",
                                "supervision.exclude_validation_episodes=false"), cwd=w)
    cf = ccil("curate", "runs_flag/generate-result.json", cwd=w)
    flag_cur = sorted((w / "runs_flag").glob("curate-*.json"))
    ef = ccil("export", "lerobot", str(flag_cur[0].relative_to(w)) if flag_cur else "missing", "--conversion",
              "conversion.json", "--out", "export_flag", cwd=w)
    ft = pq.read_table(sorted((w / "export_flag" / "data").rglob("*.parquet"))).to_pydict() \
        if (w / "export_flag").exists() else {}
    flag_eps = set(ft.get("ccil.source_episode", []))
    fm = json.loads((w / "export_flag" / "ccil" / "export.json").read_text()) if (w / "export_flag").exists() else {}
    record("recorded_split_is_honored_whatever_the_supervision_flag",
           gf.returncode == 0 and cf.returncode == 0 and ef.returncode == 0 and bool(flag_eps) and not (flag_eps & held)
           and fm["counts"] == manifest["counts"] and sorted(fm["episodes"]["held_out"]) == sorted(split["validation"]),
           episodes_written=len(flag_eps), held_out_written=sorted(flag_eps & held), counts=fm.get("counts"),
           error=(gf.stderr + cf.stderr + ef.stderr)[-300:])

    # 9. Joined modalities are vectors: an image part is refused at the configuration and by the representation.
    cfg_refusal = ccil("generate", *sets("representation.modality=state+image.top", "run.output_dir=runs_image"), cwd=w)
    img = TrajectoryDataset([Episode("0", {"state": np.zeros((3, 2), np.float32),
                                           "image.top": np.zeros((3, 4, 4, 3), np.uint8)}, np.zeros((3, 1), np.float32))])
    try:
        transitions(img, IdentityRepresentation("state+image.top"))
        api_refusal = "accepted"
    except InputError as err:
        api_refusal = f"{type(err).__name__}: {err}"
    record("joined_image_modalities_refused", cfg_refusal.returncode == 2 and "vector state modalities only"
           in cfg_refusal.stderr and not (w / "runs_image").exists() and "concatenates vectors" in api_refusal,
           configuration_exit=cfg_refusal.returncode, configuration_error=cfg_refusal.stderr.strip()[-200:],
           representation=api_refusal[:200])

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "numpy": np.__version__, "lerobot": LEROBOT}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    shutil.rmtree(out / "envs", ignore_errors=True) if not a.lerobot_python else None
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
