#!/usr/bin/env python3
"""Convert the retained legacy Pendulum inputs into canonical artifacts (#23).

    uv run python tools/legacy_conversion/convert_pendulum.py --out DIR

The retained inputs (migration contract, sections 3 and 7) are the two
Pendulum datasets and their configurations. Each pickle crosses the trust
boundary through correct_il's converter, verified against the SHA-256 in
docs/artifacts/manifest.json before it is unpickled, into an M1 file; ccil data
convert turns that into ccil.trajectories; ccil config convert turns each
legacy YAML into a canonical configuration in both fidelities. Writes:

    DIR/data/<name>.npz              canonical datasets
    DIR/config/<name>.<fidelity>.yaml canonical configurations
    DIR/manifests/<name>.dataset.json, <name>.<fidelity>.config.json
    DIR/conversion.json              the summary, with every SHA-256
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "correct_il"))
from trajectory_io import convert_legacy_pickle  # noqa: E402  (the trusted-input boundary, outside ccil)

from ccil.config.convert import FIDELITIES, convert_config  # noqa: E402
from ccil.data.convert import convert_m1  # noqa: E402

RETAINED = {"pendulum_cont": ("data/pendulum_cont_100.pkl", "config/pendulum_cont.yml"),
            "pendulum_disc": ("data/pendulum_disc_500.pkl", "config/pendulum_disc.yml")}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_pendulum(out: Path) -> dict:
    artifacts = {a["path"]: a["sha256"] for a in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
                 if a.get("path")}
    for d in ("m1", "data", "config", "manifests"):
        (out / d).mkdir(parents=True, exist_ok=True)
    summary = {"retained": {}}
    for name, (pkl, yml) in RETAINED.items():
        stem = Path(pkl).stem
        m1 = out / "m1" / f"{stem}.npz"
        convert_legacy_pickle(REPO / pkl, m1, expect_sha256=artifacts[pkl])  # refuses before unpickling on a mismatch
        canonical = out / "data" / f"{stem}.npz"
        dm = convert_m1(m1, canonical, out / "manifests" / f"{stem}.dataset.json")
        configs = {}
        for fidelity in FIDELITIES:
            cfg = out / "config" / f"{name}.{fidelity}.yaml"
            cm = convert_config(REPO / yml, cfg, out / "manifests" / f"{name}.{fidelity}.config.json", fidelity=fidelity,
                                dataset_map={pkl: f"data/{stem}.npz"})
            configs[fidelity] = {"path": f"config/{cfg.name}", "sha256": sha(cfg), "counts": cm["counts"],
                                 "implicit_written": {i["key"]: i["written"] for i in cm["implicit_legacy_behavior"]}}
        summary["retained"][name] = {
            "legacy_pickle": {"path": pkl, "sha256": artifacts[pkl]}, "legacy_config": {"path": yml, "sha256": sha(REPO / yml)},
            "m1": {"sha256": sha(m1)}, "dataset": {"path": f"data/{canonical.name}", "sha256": dm["output"]["sha256"],
                                                   "counts": dm["counts"]},
            "configs": configs}
    (out / "conversion.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    s = convert_pendulum(out)
    for name, v in s["retained"].items():
        print(f"{name}: {v['dataset']['counts']['episodes']} episodes -> {v['dataset']['path']}; "
              f"configs {sorted(v['configs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
