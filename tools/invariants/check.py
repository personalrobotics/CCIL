#!/usr/bin/env python3
"""Representative failures for every protected data invariant (#14).

    python tools/invariants/check.py --out DIR

A: ingress validation (correct_il/invariants.py) on the analytic double
   integrator, one malformed dataset per invariant.
B: candidate-level checks through the repository's collect_labels.
C: the real train_dynamics_model.py and gen_aug_label.py on Pendulum (CPU,
   legacy-harness stubs), clean and with one corrupted frame.
Writes DIR/report.json with every accepted and rejected example and its reason.
"""
import argparse
import copy
import json
import os
import pickle
import platform
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "correct_il", REPO / "tools" / "analytic_fixture"):
    sys.path.insert(0, str(p))
import invariants as inv  # noqa: E402
from legacy import load_definitions  # noqa: E402
from run import trajectories as fixture_trajectories  # noqa: E402
from system import DoubleIntegrator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(section, name, ok, **detail):
        cases.append({"section": section, "case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {section}.{name} {'' if ok else detail}")

    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    clean = [t for t in fixture_trajectories(config) if np.all(np.isfinite(t["actions"]))]  # slow, fast

    # --- A: ingress -------------------------------------------------------
    def expect(name, data, invariant, episode, frame, **bounds):
        snapshot = copy.deepcopy(data)
        try:
            inv.validate_trajectories(data, **bounds)
            got = None
        except inv.DataValidationError as e:
            got = e.violations
        hit = got is not None and any(v["invariant"] == invariant and v["episode"] == episode and v["frame"] == frame
                                      for v in got)
        unchanged = all(np.array_equal(np.asarray(a["observations"]), np.asarray(b["observations"]), equal_nan=True)
                        and np.array_equal(np.asarray(a["actions"]), np.asarray(b["actions"]), equal_nan=True)
                        for a, b in zip(data, snapshot) if isinstance(a, dict) and isinstance(b, dict) and "actions" in a)
        record("A", name, hit and unchanged, expected={"invariant": invariant, "episode": episode, "frame": frame},
               violations=got, input_unchanged=unchanged)

    def mutate(fn):
        data = copy.deepcopy(clean)
        fn(data)
        return data

    record("A", "clean_dataset_passes", inv.validate_trajectories(clean) == (2, 1))
    record("A", "nan_in_unused_last_action_passes",
           inv.validate_trajectories(mutate(lambda d: d[0]["actions"].__setitem__((-1, 0), np.nan))) == (2, 1))
    expect("nan_action", mutate(lambda d: d[0]["actions"].__setitem__((2, 0), np.nan)), "non_finite", 0, 2)
    expect("inf_observation", mutate(lambda d: d[1]["observations"].__setitem__((3, 1), np.inf)), "non_finite", 1, 3)
    expect("float32_overflow", mutate(lambda d: d[1].__setitem__("observations", d[1]["observations"].astype(np.float64))
                                      or d[1]["observations"].__setitem__((1, 0), 1e39)), "float32_overflow", 1, 1)
    expect("length_mismatch", mutate(lambda d: d[0].__setitem__("actions", d[0]["actions"][:-1])), "length_mismatch", 0, None)
    expect("integer_dtype", mutate(lambda d: d[1].__setitem__("actions", d[1]["actions"].astype(np.int64))), "dtype", 1, None)
    expect("observation_rank", mutate(lambda d: d[0].__setitem__("observations", d[0]["observations"][:, :, None])), "shape", 0, None)
    expect("dimension_mismatch", mutate(lambda d: d[1].__setitem__("observations", np.hstack(
        [d[1]["observations"], d[1]["observations"][:, :1]]))), "dimension_mismatch", 1, None)
    expect("too_short", mutate(lambda d: d[0].update(observations=d[0]["observations"][:1], actions=d[0]["actions"][:1])),
           "too_short", 0, None)
    expect("action_out_of_bounds", copy.deepcopy(clean), "action_out_of_bounds", 0, 2, action_low=-0.1, action_high=0.1)
    expect("missing_key", mutate(lambda d: d[1].pop("actions")), "episode_structure", 1, None)
    try:
        inv.validate_trajectories({"observations": []})
        ok = False
    except inv.DataValidationError as e:
        ok = e.violations[0]["invariant"] == "dataset_structure"
    record("A", "not_a_list", ok)

    # --- B: candidate level, through collect_labels -------------------------
    impl = load_definitions({
        "correct_il/gen_aug_label.py": ["_candidates", "_root_find_one", "gen_noisy_action_be", "gen_backward_euler_fast",
                                        "collect_labels"],
        "correct_il/utils.py": ["root_finder_s", "build_dataset"]}, extra={"tqdm": lambda x: x})
    model = DoubleIntegrator(config["h"]).model()
    np_s, np_a, np_sp = impl["build_dataset"](clean)
    data = tuple(torch.as_tensor(x).float() for x in (np_s, np_a, np_sp))
    ids = inv.transition_index(clean)
    torch.manual_seed(0)
    labels = impl["collect_labels"](partial(impl["gen_noisy_action_be"], 4, 0.2, 1.0), model, data, np_s, len(np_s),
                                    transition_ids=ids, action_bounds=(-0.3, 0.3))
    acc, oob = labels["accepted"], labels["reason"] == "action_out_of_bounds"
    within = bool(np.all(np.abs(labels["a"][acc]) <= 0.3))
    ids_ok = all(labels["episode"][k] == ids[0][labels["datapoint"][k]] and labels["frame"][k] == ids[1][labels["datapoint"][k]]
                 for k in range(len(acc)))
    record("B", "action_bounds_reject_never_clamp", oob.any() and acc.any() and within and ids_ok
           and bool(np.all(np.abs(labels["a"][oob]) > 0.3)), counts=labels["counts"],
           rejected_examples=[{"episode": int(labels["episode"][k]), "frame": int(labels["frame"][k]),
                               "action": labels["a"][k].tolist(), "reason": labels["reason"][k]} for k in np.flatnonzero(oob)[:5]])

    class NanModel:
        def f(self, s, a):
            out = model.f(s, a)
            return out + torch.where(s[..., :1] > 0.5, torch.nan, 0.0)  # NaN for x > 0.5 (the 'fast' episode)

    labels = impl["collect_labels"](partial(impl["gen_backward_euler_fast"], 50, None), NanModel(), data, np_s, len(np_s),
                                    transition_ids=ids)
    nf = labels["reason"] == "non_finite"
    record("B", "non_finite_model_output_rejected_with_source", nf.any() and not labels["accepted"][nf].any()
           and set(labels["episode"][nf].tolist()) == {1}, counts=labels["counts"],
           rejected_frames=[[int(e), int(f)] for e, f in zip(labels["episode"][nf], labels["frame"][nf])])

    def bad_generator(model_, datapoint):
        rec = impl["gen_backward_euler_fast"](50, None, model_, datapoint)
        rec["s"] = rec["s"][:, :1]  # drop a state dimension
        return rec

    try:
        impl["collect_labels"](bad_generator, model, data, np_s, len(np_s))
        ok, msg = False, "no error"
    except ValueError as e:
        ok, msg = "shape" in str(e), str(e)
    record("B", "malformed_generator_output_raises", ok, error=msg)

    # --- C: real entry points on Pendulum -----------------------------------
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'tools/legacy_harness/stubs'}"}

    def run(script, cfg, outdir, *extra):
        return subprocess.run([sys.executable, f"correct_il/{script}", str(cfg), "no_gpu", "true", "output.location",
                               str(outdir), "dynamics.train_epochs", "2", *extra],
                              cwd=REPO, env=env, capture_output=True, text=True)

    cfg = REPO / "config/pendulum_cont.yml"
    r1 = run("train_dynamics_model.py", cfg, out / "e2e")
    r2 = run("gen_aug_label.py", cfg, out / "e2e", "debug", "true")
    dec = list((out / "e2e").glob("seed*/*/data/*/label_decisions.npz"))
    ok = r1.returncode == 0 and r2.returncode == 0 and len(dec) == 1
    detail = {"train_stderr": r1.stderr[-600:], "gen_stderr": r2.stderr[-600:]}
    if ok:
        d = np.load(dec[0], allow_pickle=False)
        ok = {"episode", "frame", "reason", "accepted"} <= set(d.files) and len(d["accepted"]) > 0
        detail = {"files": sorted(d.files), "candidates": int(len(d["accepted"])), "accepted": int(d["accepted"].sum()),
                  "episodes_frames": sorted({(int(e), int(f)) for e, f in zip(d["episode"], d["frame"])}),
                  "statistics": (dec[0].parent / "statistics.txt").read_text()}
    record("C", "pendulum_train_and_generate_end_to_end", ok, **detail)

    with open(REPO / "data/pendulum_cont_100.pkl", "rb") as f:  # trusted repository data
        pend = pickle.load(f)
    pend[7]["actions"] = np.array(pend[7]["actions"], dtype=float)
    pend[7]["actions"][123] = np.nan
    bad_pkl = out / "pendulum_cont_corrupt.pkl"
    with open(bad_pkl, "wb") as f:
        pickle.dump(pend, f)
    bad_cfg = out / "pendulum_cont_corrupt.yml"
    c = yaml.safe_load(cfg.read_text())
    c["data"]["pkl"] = str(bad_pkl)
    bad_cfg.write_text(yaml.safe_dump(c))
    r3 = run("train_dynamics_model.py", bad_cfg, out / "corrupt")
    r4 = run("gen_aug_label.py", bad_cfg, out / "e2e_gen_on_corrupt", "debug", "true")
    named = "non_finite: episode 7, frame 123" in r3.stderr and "non_finite: episode 7, frame 123" in r4.stderr
    nothing_written = not list((out / "corrupt").glob("seed*/*/dynamics_*/dynamics.pt")) and \
        not list((out / "e2e_gen_on_corrupt").glob("seed*/*/data/*/aug_data.pkl"))
    record("C", "corrupt_frame_fails_at_ingress_before_any_output", r3.returncode != 0 and r4.returncode != 0 and named
           and nothing_written, train_error=r3.stderr.strip().splitlines()[-3:], gen_error=r4.stderr.strip().splitlines()[-3:])

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "correct_il", "config", "tools")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "numpy": np.__version__}}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
