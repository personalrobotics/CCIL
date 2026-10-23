#!/usr/bin/env python3
"""Run an implementation on the analytic fixture and write the artifact.

    python tools/analytic_fixture/run.py --impl {legacy,reference} --out DIR
    python tools/analytic_fixture/run.py --check DIR   # re-evaluate an existing artifact

Exits 0 only if every invariant passes. See docs/fixtures/analytic-dynamics.md.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import oracle  # noqa: E402
from system import DoubleIntegrator  # noqa: E402

ARTIFACT_FILES = ("config.resolved.json", "transitions.json", "candidates.json", "metrics.json", "plot.png")


def trajectories(config):
    """Legacy dataset format: per-episode observations and actions of equal length.

    Legacy build_dataset drops the last action, so a zero action pads each episode.
    """
    by_ep = {}
    for tr in oracle.episodes_to_transitions(config):
        by_ep.setdefault(tr["episode"], []).append(tr)
    out = []
    for trs in by_ep.values():
        obs = np.stack([t["s"] for t in trs] + [trs[-1]["s_next"]]).astype(np.float32)
        act = np.concatenate([t["a"] for t in trs] + [np.zeros(1, np.float32)])[:, None].astype(np.float32)
        out.append({"observations": obs, "actions": act})
    return out


def dumps(obj):
    return json.dumps(oracle.jsonable(obj), indent=1, sort_keys=True) + "\n"


def loads_floats(x):
    if isinstance(x, dict):
        return {k: loads_floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [loads_floats(v) for v in x]
    return float(x) if x in ("nan", "inf", "-inf") else x


def plot(config, transitions, results, path):
    """Per method: each candidate's distance from its expert state, coloured by the
    implementation's decision and ringed when it contradicts the oracle."""
    methods = list(config["methods"])
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.4))
    for ax, m in zip(axes, methods):
        eps = config["methods"][m].get("epsilon")
        cands = [c for dp in results[m] for c in dp["candidates"]]
        finite = [c for c in cands if "oracle_distance" in c]
        for k, c in enumerate(finite):
            ax.scatter(c["oracle_distance"], k, s=16, marker="o" if c["accepted"] else "x",
                       color="tab:green" if c["accepted"] else "tab:red")
            if c["expected_class"] != "boundary" and c["accepted"] != (c["expected_class"] == "accepted"):
                ax.scatter(c["oracle_distance"], k, s=80, facecolors="none", edgecolors="black")
        if eps is not None:
            ax.axvline(eps, color="0.3", linestyle="--", linewidth=1)
        bad_nan = sum(c["accepted"] and c["expected_class"] == "non_finite" for c in cands)
        ax.set_title(f"{m}\nepsilon={eps}  accepted non-finite={bad_nan}", fontsize=8)
        ax.set_xlabel("||s_G - s*||")
        ax.set_yticks([])
    axes[0].scatter([], [], marker="o", color="tab:green", label="accepted")
    axes[0].scatter([], [], marker="x", color="tab:red", label="rejected")
    axes[0].scatter([], [], s=80, facecolors="none", edgecolors="black", label="contradicts oracle")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100, metadata={"Software": None})
    plt.close(fig)


def summarize(verdicts):
    lines = []
    for name, v in verdicts.items():
        if "pass" in v:
            lines.append(f"{'PASS' if v['pass'] else 'FAIL'} {name}")
        else:
            for m, r in v.items():
                lines.append(f"{'PASS' if r['pass'] else 'FAIL'} {name} [{m}] violations={r['violations']}")
    return "\n".join(lines)


def check(out):
    config = json.loads((out / "config.resolved.json").read_text())
    transitions = loads_floats(json.loads((out / "transitions.json").read_text()))
    results = loads_floats(json.loads((out / "candidates.json").read_text()))
    _, verdicts = oracle.evaluate(config, transitions, results)
    print(summarize(verdicts))
    return 0 if oracle.all_pass(verdicts) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["legacy", "reference"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", type=Path, help="re-evaluate an existing artifact directory")
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    args = ap.parse_args()
    if args.check:
        return check(args.check)
    if not (args.impl and args.out):
        ap.error("--impl and --out are required unless --check is given")

    config = json.loads(args.config.read_text())
    model = DoubleIntegrator(config["h"]).model()
    if args.impl == "legacy":
        import legacy as impl
    else:
        import reference as impl

    runs = [impl.run(config, model, trajectories(config)) for _ in range(2)]
    serial = [(dumps(t), dumps(r)) for t, r, _ in runs]
    transitions, results, meta = runs[0]
    _, verdicts = oracle.evaluate(config, transitions, results)
    verdicts["I7_replay_determinism"] = {"pass": serial[0] == serial[1]}

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.resolved.json").write_text(dumps(config))
    (out / "transitions.json").write_text(dumps(transitions))
    (out / "candidates.json").write_text(dumps(results))
    (out / "metrics.json").write_text(dumps({"implementation": args.impl, "all_pass": oracle.all_pass(verdicts), "verdicts": verdicts}))
    plot(config, transitions, results, out / "plot.png")

    git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    manifest = {
        "implementation": args.impl,
        "code_commit": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain", "--", str(HERE))),
        **meta,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "matplotlib": matplotlib.__version__},
        "sha256": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ARTIFACT_FILES},
    }
    (out / "manifest.json").write_text(dumps(manifest))
    print(summarize(verdicts))
    print(f"artifact: {out}")
    return 0 if oracle.all_pass(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
