#!/usr/bin/env python3
"""End-to-end check of trajectory-level validation in dynamics training (#13).

    python tools/validation_split/check.py --out DIR

Runs the real correct_il/train_dynamics_model.py on the Pendulum data (CPU, a
few epochs) through the legacy harness stubs, then checks the persisted split,
held-out metrics, split reuse, small-dataset diagnostics, and that held-out
evaluation leaves the label model unchanged. Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "correct_il"))
import correct_il.models.nn_dynamics as nd  # noqa: E402
import validation  # noqa: E402  (correct_il/validation.py)
import trajectory_io as tio  # noqa: E402

CONFIG = "config/pendulum_cont.yml"
EPOCHS = "2"
DATA = None  # converted Pendulum .npz, set in main()


def train(out, *overrides):
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'tools/legacy_harness/stubs'}"}
    cmd = [sys.executable, "correct_il/train_dynamics_model.py", CONFIG, "no_gpu", "true",
           "dynamics.train_epochs", EPOCHS, "output.location", str(out), "data.path", str(DATA), *overrides]
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)


def dynamics_dir(out):
    hits = list(Path(out).glob("seed*/*/dynamics_*"))
    return hits[0] if len(hits) == 1 else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def portable(value):
        """Remove machine-local roots from the committed evidence artifact."""
        if isinstance(value, str):
            for root, marker in ((str(REPO), "<repo>"), (str(out), "<artifact>"),
                                 (str(Path.home()), "<home>")):
                value = value.replace(root, marker)
            return value
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    global DATA
    DATA = out / "pendulum_cont_100.npz"
    pend_sha = {a["path"]: a["sha256"] for a in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
                if a.get("path")}["data/pendulum_cont_100.pkl"]
    tio.convert_legacy_pickle(REPO / "data/pendulum_cont_100.pkl", DATA, expect_sha256=pend_sha)
    trajectories = tio.read_trajectories(DATA)[0]

    # A: validation on (config default 0.2)
    r = train(out / "val")
    d = dynamics_dir(out / "val")
    record("train_with_validation_exits_0", r.returncode == 0 and d is not None, stderr=r.stderr[-1500:])
    if d is None:
        return 1
    split = json.loads((d / "split.json").read_text())
    report = json.loads((d / "validation.json").read_text())
    train_ids, val_ids = set(split["train"]), set(split["validation"])
    record("split_is_trajectory_partition", not (train_ids & val_ids) and len(train_ids | val_ids) == len(trajectories),
           train=len(train_ids), validation=len(val_ids), overlap=sorted(train_ids & val_ids))
    held = report["held_out"]["per_trajectory"]
    expected_counts = {i: len(trajectories[i]["observations"]) - 1 for i in val_ids}
    record("held_out_metrics_only_from_validation_trajectories",
           sorted(r_["trajectory"] for r_ in held) == sorted(val_ids)
           and all(r_["transitions"] == expected_counts[r_["trajectory"]] for r_ in held),
           held_out_trajectories=sorted(r_["trajectory"] for r_ in held))
    agg = report["held_out"]["aggregate"]
    pooled = sum(r_["mean_l2"] * r_["transitions"] for r_ in held) / sum(r_["transitions"] for r_ in held)
    record("aggregate_matches_per_trajectory_rows", abs(pooled - agg["pooled_mean_l2"]) < 1e-12, aggregate=agg)
    stats = (d / "statistics.txt").read_text()
    record("statistics_label_training_error_honestly",
           "not a held-out metric" in stats and "Held-out one-step L2 error, evaluation model" in stats, statistics=stats)

    # B: rerun with the same configuration reuses the persisted split
    before = sha(d / "split.json")
    r2 = train(out / "val")
    record("rerun_reuses_split", r2.returncode == 0 and "Reusing split" in r2.stdout and sha(d / "split.json") == before)

    # A persisted split is untrusted input. Duplicates, coercible values and
    # other malformed partitions must fail instead of changing the training set.
    valid_small = validation.make_split(4, 0.25, 0)
    malformed = {
        "duplicate": {**valid_small, "train": [0, 0, 1, 2], "validation": [3]},
        "boolean_id": {**valid_small, "train": [0, 1, True], "validation": [3]},
        "out_of_range": {**valid_small, "train": [0, 1, 4], "validation": [3]},
        "unsorted": {**valid_small, "train": list(reversed(valid_small["train"]))},
        "wrong_validation_count": {**valid_small, "validation_fraction": 0.5},
        "missing_field": {k: v for k, v in valid_small.items() if k != "seed"},
    }
    rejected = {}
    for name, bad in malformed.items():
        try:
            validation.check_disjoint(bad)
            rejected[name] = None
        except validation.SplitError as e:
            rejected[name] = str(e)
    broken_json = out / "broken-split.json"
    broken_json.write_text("{not json\n")
    try:
        validation.load_or_create_split(broken_json, 4, 0.25, 0)
        rejected["invalid_json"] = None
    except validation.SplitError as e:
        rejected["invalid_json"] = str(e)
    record("malformed_persisted_splits_fail_before_reuse",
           all(rejected.values()) and "duplicate" in rejected["duplicate"]
           and "cannot read persisted split" in rejected["invalid_json"], errors=rejected)

    # C: an explicit configuration change produces a new split
    r3 = train(out / "val", "dynamics.validation_fraction", "0.3")
    split3 = json.loads((d / "split.json").read_text())
    record("config_change_creates_new_split", r3.returncode == 0 and "Created split" in r3.stdout
           and len(split3["validation"]) == 30, validation=len(split3["validation"]))

    # D: held-out evaluation leaves the label model unchanged
    r4 = train(out / "noval", "dynamics.validation_fraction", "0")
    d4 = dynamics_dir(out / "noval")
    a = torch.load(d / "dynamics.pt", weights_only=True)["dynamics_net"]["state_dict"]
    b = torch.load(d4 / "dynamics.pt", weights_only=True)["dynamics_net"]["state_dict"]
    same = r4.returncode == 0 and a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    record("label_model_identical_with_and_without_validation", same,
           no_validation_files=not (d4 / "validation.json").exists() and not (d4 / "split.json").exists())

    # E: a dataset too small to split fails before training, with an actionable message
    one = out / "one_trajectory.npz"
    tio.write_trajectories(trajectories[:1], one, {"note": "first Pendulum episode"})
    r5 = train(out / "small", "data.path", str(one))
    d5 = dynamics_dir(out / "small")
    record("small_dataset_fails_fast_with_diagnostic",
           r5.returncode != 0 and "each split needs at least one" in r5.stderr
           and (d5 is None or not (d5 / "dynamics.pt").exists()), stderr_tail=r5.stderr[-400:])

    # F: training-set error includes the final partial batch
    model = nd.load_world_model(d / "dynamics.pt", "cpu")
    s, a_, sp = validation.transitions(trajectories[:1])[0]
    s, a_, sp = s[:37], a_[:37], sp[:37]
    manual = float(np.mean(np.linalg.norm(model.predict(s, a_).detach().numpy() - sp, axis=1)))
    got = float(model.eval_prediction_error(s, a_, sp, batch_size=512))
    record("eval_prediction_error_uses_partial_batch", np.isfinite(got) and abs(got - manual) < 1e-6,
           transitions=37, batch_size=512, got=got, manual=manual)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    result = {
        "all_pass": all(c["pass"] for c in cases), "cases": cases,
        "split": split, "validation_report": report,
        "environment": {"code_commit": git("rev-parse", "HEAD"),
                        "code_dirty": bool(git("status", "--porcelain", "--", "correct_il", "config", "tools")),
                        "python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__, "config": CONFIG, "epochs": int(EPOCHS)},
    }
    (out / "report.json").write_text(json.dumps(portable(result), indent=1, sort_keys=True) + "\n")
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
