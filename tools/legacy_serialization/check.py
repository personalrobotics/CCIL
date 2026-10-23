#!/usr/bin/env python3
"""Trust-boundary check for legacy pickle data (#15).

    python tools/legacy_serialization/check.py --out DIR

Shows that the shipped datasets convert only after sha256 verification and
round-trip exactly; that tampered, unacknowledged, or malformed pickles are
refused before they are unpickled or before anything is written; that the
normal entry points never unpickle; and that converted data trains and labels
end to end. Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "correct_il"))
import trajectory_io as tio  # noqa: E402
from invariants import DataValidationError  # noqa: E402


class Exploit:
    """Unpickling this creates a marker file: proof that a pickle was executed."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'tools/legacy_harness/stubs'}"}
    py = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=REPO, env=env, capture_output=True, text=True)  # noqa: E731

    # 1. Shipped datasets: sha256-verified conversion, exact round trip, schema metadata.
    conv = out / "converted"
    r = py("correct_il/convert_legacy_dataset.py", "--shipped", "--out-dir", conv)
    manifest = {a["path"]: a["sha256"] for a in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
                if (a.get("path") or "").endswith(".pkl") and a.get("sha256")}
    rows = []
    for rel, digest in sorted(manifest.items()):
        npz = conv / (Path(rel).stem + ".npz")
        eps, meta = tio.read_trajectories(npz)
        with open(REPO / rel, "rb") as f:  # trusted: its sha256 was verified above
            legacy = pickle.load(f)
        exact = len(eps) == len(legacy) and all(
            np.array_equal(np.asarray(l["observations"]), e["observations"])
            and np.asarray(l["observations"]).dtype == e["observations"].dtype
            and np.array_equal(np.asarray(l["actions"]).reshape(len(l["actions"]), -1), e["actions"])
            for e, l in zip(eps, legacy))
        rows.append({"dataset": rel, "episodes": len(eps), "exact": exact, "meta": meta,
                     "meta_ok": meta["source"]["sha256"] == digest and meta["format"] == tio.FORMAT
                     and meta["converter_version"] == tio.CONVERTER_VERSION and meta["trust"] == "sha256"})
    record("shipped_datasets_convert_exactly_with_provenance", r.returncode == 0 and len(rows) == 15
           and all(x["exact"] and x["meta_ok"] for x in rows), datasets=rows)

    # 2. A tampered file is refused before unpickling: its hash no longer matches.
    marker = out / "EXECUTED"
    evil = out / "pendulum_cont_100.pkl"
    with open(evil, "wb") as f:
        pickle.dump(Exploit(str(marker)), f)
    r = py("correct_il/convert_legacy_dataset.py", evil, out / "evil.npz", "--expect-sha256",
           manifest["data/pendulum_cont_100.pkl"])
    record("tampered_pickle_refused_before_unpickling", r.returncode != 0 and "does not match" in r.stderr
           and not marker.exists() and not (out / "evil.npz").exists(), stderr=r.stderr.strip())

    # 3. Without a hash or an acknowledgement, nothing is unpickled.
    r = py("correct_il/convert_legacy_dataset.py", evil, out / "evil.npz")
    record("unacknowledged_pickle_refused", r.returncode != 0 and "refusing to unpickle" in r.stderr
           and not marker.exists(), stderr=r.stderr.strip())

    # 4. An acknowledged but malformed pickle fails validation and writes nothing;
    #    an existing destination is left untouched.
    bad = out / "malformed.pkl"
    with open(bad, "wb") as f:
        pickle.dump([{"observations": np.zeros((5, 2)), "actions": np.zeros((4, 1))}], f)
    keep = out / "keep.npz"
    tio.write_trajectories([{"observations": np.zeros((3, 2)), "actions": np.zeros((3, 1))}], keep, {"note": "existing"})
    before = sha(keep)
    r = py("correct_il/convert_legacy_dataset.py", bad, keep, "--trust-legacy-pickle")
    leftovers = sorted(p.name for p in out.iterdir() if ".tmp-" in p.name)
    record("malformed_pickle_fails_validation_without_writing", r.returncode != 0 and "length_mismatch" in r.stderr
           and sha(keep) == before and not leftovers, stderr=r.stderr.strip()[-300:])

    # 5. Malformed or foreign .npz files are rejected by the loader.
    np.savez(out / "foreign.npz", x=np.zeros(3))
    np.savez(out / "lengths.npz", observations=np.zeros((5, 2)), actions=np.zeros((5, 1)),
             episode_lengths=np.array([4]), meta=np.array(json.dumps({"format": tio.FORMAT})))
    errs = {}
    for name in ("foreign.npz", "lengths.npz"):
        try:
            tio.read_trajectories(out / name)
        except tio.LegacyInputError as e:
            errs[name] = str(e)
    record("malformed_npz_rejected", len(errs) == 2, errors=errs)

    # 6. The normal entry point refuses a .pkl path and never executes it.
    r = py("correct_il/train_dynamics_model.py", "config/pendulum_cont.yml", "no_gpu", "true", "dynamics.train_epochs", "2",
           "output.location", out / "refuse", "data.path", evil)
    record("training_refuses_pickle_path", r.returncode != 0 and "legacy pickle" in r.stderr and not marker.exists()
           and not list((out / "refuse").glob("seed*/*/dynamics_*/dynamics.pt")), stderr=r.stderr.strip().splitlines()[-1:])

    # 7. Converted data trains and labels end to end; no pickle is written.
    pend = conv / "pendulum_cont_100.npz"
    args = ("config/pendulum_cont.yml", "no_gpu", "true", "dynamics.train_epochs", "2", "output.location", out / "e2e",
            "data.path", pend)
    r1 = py("correct_il/train_dynamics_model.py", *args)
    r2 = py("correct_il/gen_aug_label.py", *args, "debug", "true")
    aug = list((out / "e2e").glob("seed*/*/data/*/aug_data.npz"))
    pickles = sorted(str(p.relative_to(out)) for p in (out / "e2e").rglob("*.pkl"))
    ok = r1.returncode == 0 and r2.returncode == 0 and len(aug) == 1 and not pickles
    if ok:
        with np.load(aug[0], allow_pickle=False) as z:
            ok = {"observations", "actions", "next_obs", "original_states"} <= set(z.files) and len(z["observations"]) > 0
    record("converted_data_trains_and_labels_without_pickle", ok, pickle_files_written=pickles,
           stderr=(r1.stderr + r2.stderr)[-600:] if not ok else "")

    # 8. Static check: the only pickle.load on input data is the converter boundary.
    loads = []
    for p in sorted((REPO / "correct_il").rglob("*.py")):
        rel = p.relative_to(REPO)
        if "envs/f1" in str(rel):
            continue  # retired benchmark tooling (migration contract, section 3)
        loads += [f"{rel}:{i}" for i, line in enumerate(p.read_text().splitlines(), 1) if re.search(r"pickle\.loads?\(", line)]
    record("only_converter_unpickles", loads == ["correct_il/trajectory_io.py:" + str(
        next(i for i, line in enumerate((REPO / "correct_il/trajectory_io.py").read_text().splitlines(), 1)
             if "pickle.load(" in line))], pickle_load_sites=loads)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "correct_il", "config", "tools")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
