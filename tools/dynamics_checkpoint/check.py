#!/usr/bin/env python3
"""Behavior check for dynamics checkpoints (#12).

    python tools/dynamics_checkpoint/check.py --out DIR

Exercises save/load round trips, the canonical backward name, explicit failure
modes, CPU label generation with a loaded model, and (when CUDA is available)
cross-device loading. Writes DIR/report.json and exits non-zero on any failure.
"""
import argparse
import hashlib
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))

import correct_il.models.nn_dynamics as nd  # noqa: E402
from legacy import load_definitions  # noqa: E402

SAME_DEVICE_TOL = 1e-6   # same device, same weights: expected exactly equal
CROSS_DEVICE_TOL = 1e-5  # CPU vs CUDA kernels: not bitwise identical
LIPSCHITZ_TYPES = ("none", "spectral_normalization", "slack", "soft_sampling")


def d_config(lipschitz_type, backward=False):
    return nd._Config(lipschitz_type=lipschitz_type, lipschitz_constraint=2.0,
                      soft_lipschitz_penalty_weight=0.001, soft_lipschitz_sampling_eps=0.0003,
                      soft_lipschitz_n_samples=5, slack_weight=1.0, lr=1e-3, weight_decay=1e-5,
                      layers=[32, 32], activation="relu", train_epochs=3, batch_size=64, backward=backward)


def data(seed=0, n=256, h=0.1):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    a = rng.uniform(-1, 1, (n, 1)).astype(np.float32)
    sp = s + h * np.stack([s[:, 1], a[:, 0]], axis=1).astype(np.float32)
    return s, a, sp


def fit(cfg, device):
    torch.manual_seed(0)
    np.random.seed(0)
    s, a, sp = data()
    m = nd.WorldModel(2, 1, d_config=cfg, hidden_size=tuple(cfg.layers), fit_lr=cfg.lr,
                      fit_wd=cfg.weight_decay, device=device, activation=cfg.activation)
    m.fit_dynamics(s, a, sp, batch_size=cfg.batch_size, train_epochs=cfg.train_epochs)
    return m


def predict(m):
    s, a, _ = data(seed=1, n=64)
    with torch.no_grad():
        return m.predict(s, a).cpu().numpy()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expect_error(fn, needle):
    try:
        fn()
    except nd.CheckpointError as e:
        return needle in str(e), str(e)[:300]
    except Exception as e:  # anything else is an unexplained failure
        return False, f"{type(e).__name__}: {e}"[:300]
    return False, "no error raised"


class _Exploit:
    """A pickle that would create a marker file if it were ever unpickled."""
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work"
    work.mkdir()
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail if not ok else ''}")

    # Round trips on CPU, for every architecture variant.
    for lt in LIPSCHITZ_TYPES:
        cfg = d_config(lt)
        m = fit(cfg, "cpu")
        path = work / f"{lt}.pt"
        nd.save_world_model(m, path, cfg)
        loaded = nd.load_world_model(path, "cpu")
        diff = float(np.max(np.abs(predict(m) - predict(loaded))))
        record(f"roundtrip_cpu[{lt}]", diff <= SAME_DEVICE_TOL, max_abs_diff=diff, tolerance=SAME_DEVICE_TOL,
               training_mode_restored=loaded.dynamics_net.training == m.dynamics_net.training,
               has_lambda=hasattr(loaded, "learnable_lambda"))

    # Canonical backward name, through gen_aug_label's own lookup.
    gen = load_definitions({"correct_il/gen_aug_label.py": ["choose_model_basename"]},
                           extra={"checkpoint_name": nd.checkpoint_name})
    bdir = work / "backward"
    bdir.mkdir()
    cfg = d_config("none", backward=True)
    nd.save_world_model(fit(cfg, "cpu"), bdir / nd.checkpoint_name(cfg.backward), cfg)
    name = gen["choose_model_basename"](nd._Config(type="backward_model"))
    ok = name == "dynamics_backward.pt" and (bdir / name).exists()
    record("backward_name_lookup", ok and nd.load_world_model(bdir / name, "cpu") is not None, lookup=name)

    # CPU label generation with a loaded model (CUDA not required).
    impl = load_definitions({
        "correct_il/gen_aug_label.py": ["_candidates", "_root_find_one", "gen_backward_euler_fast", "collect_labels"],
        "correct_il/utils.py": ["root_finder_s"]}, extra={"tqdm": lambda x: x})
    loaded = nd.load_world_model(work / "none.pt", "cpu")
    s, a, sp = (torch.as_tensor(x) for x in data(seed=2, n=32))
    labels = impl["collect_labels"](partial(impl["gen_backward_euler_fast"], 50, 1.0), loaded, (s, a, sp), s.numpy(), 32)
    acc = labels["accepted"]
    record("cpu_label_generation", acc.any() and np.all(np.isfinite(labels["s"][acc])), counts=labels["counts"])

    # Explicit failures.
    bad = work / "malformed.pt"
    bad.write_bytes(b"not a checkpoint")
    ok, msg = expect_error(lambda: nd.load_world_model(bad, "cpu"), "cannot read")
    record("malformed_file", ok, error=msg)

    marker = work / "pickle-executed"
    legacy = work / "legacy.pkl"
    with open(legacy, "wb") as f:
        pickle.dump(_Exploit(str(marker)), f)
    ok, msg = expect_error(lambda: nd.load_world_model(legacy, "cpu"), "cannot read")
    record("legacy_pickle_not_executed", ok and not marker.exists(), error=msg, marker_created=marker.exists())

    def tampered(src, dst, mutate):
        payload = torch.load(src, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, dst)
        return dst

    p = tampered(work / "none.pt", work / "arch.pt", lambda pl: pl["meta"].update(hidden_size=[16, 16]))
    ok, msg = expect_error(lambda: nd.load_world_model(p, "cpu"), "incompatible architecture")
    record("incompatible_architecture", ok, error=msg)

    p = tampered(work / "slack.pt", work / "nets.pt", lambda pl: pl["meta"]["d_config"].update(lipschitz_type="none"))
    ok, msg = expect_error(lambda: nd.load_world_model(p, "cpu"), "do not match")
    record("network_set_mismatch", ok, error=msg)

    p = tampered(work / "none.pt", work / "fmt.pt", lambda pl: pl["meta"].update(format="ccil-dynamics/0"))
    ok, msg = expect_error(lambda: nd.load_world_model(p, "cpu"), "unsupported checkpoint format")
    record("unsupported_format", ok, error=msg)

    missing_meta = work / "missing-meta.pt"
    torch.save({"meta": {"format": nd.CHECKPOINT_FORMAT}}, missing_meta)
    ok, msg = expect_error(lambda: nd.load_world_model(missing_meta, "cpu"), "metadata missing")
    record("missing_metadata", ok and "d_config" in msg and "state_dim" in msg, error=msg)

    p = tampered(work / "none.pt", work / "transforms.pt",
                 lambda pl: pl["dynamics_net"]["transformations"].pop("s_scale"))
    ok, msg = expect_error(lambda: nd.load_world_model(p, "cpu"), "transformations must be")
    record("malformed_network_payload", ok and "s_scale" in msg, error=msg)

    if not torch.cuda.is_available():
        ok, msg = expect_error(lambda: nd.load_world_model(work / "none.pt", "cuda"), "CUDA is unavailable")
        record("cuda_requested_without_cuda", ok, error=msg)

    target = work / "keep.pt"
    nd.save_world_model(nd.load_world_model(work / "none.pt", "cpu"), target, d_config("none"))
    before = sha(target)
    real_save = torch.save

    def failing_save(obj, f, *args, **kwargs):
        Path(f).write_bytes(b"partial")
        raise OSError("simulated disk failure")

    nd.torch.save = failing_save
    try:
        nd.save_world_model(nd.load_world_model(work / "slack.pt", "cpu"), target, d_config("slack"))
        raised = False
    except OSError:
        raised = True
    finally:
        nd.torch.save = real_save
    leftovers = sorted(x.name for x in work.iterdir() if ".tmp-" in x.name)
    record("failed_save_preserves_existing", raised and sha(target) == before and not leftovers,
           raised=raised, unchanged=sha(target) == before, leftover_temp_files=leftovers)

    # Cross-device cases (only where CUDA exists).
    if torch.cuda.is_available():
        for lt in LIPSCHITZ_TYPES:
            cfg = d_config(lt)
            gpu = fit(cfg, "cuda")
            path = work / f"{lt}-cuda.pt"
            nd.save_world_model(gpu, path, cfg)
            ref = predict(gpu)
            for dev in ("cpu", "cuda"):
                diff = float(np.max(np.abs(ref - predict(nd.load_world_model(path, dev)))))
                tol = SAME_DEVICE_TOL if dev == "cuda" else CROSS_DEVICE_TOL
                record(f"saved_cuda_loaded_{dev}[{lt}]", diff <= tol, max_abs_diff=diff, tolerance=tol)
            cpu_ref = predict(nd.load_world_model(work / f"{lt}.pt", "cpu"))
            diff = float(np.max(np.abs(cpu_ref - predict(nd.load_world_model(work / f"{lt}.pt", "cuda")))))
            record(f"saved_cpu_loaded_cuda[{lt}]", diff <= CROSS_DEVICE_TOL, max_abs_diff=diff, tolerance=CROSS_DEVICE_TOL)

    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {
        "all_pass": all(c["pass"] for c in cases),
        "cases": cases,
        "environment": {
            "code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--porcelain", "--", "correct_il", "tools")),
            "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
        "tolerances": {"same_device": SAME_DEVICE_TOL, "cross_device": CROSS_DEVICE_TOL},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
