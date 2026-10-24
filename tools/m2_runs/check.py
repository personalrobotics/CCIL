#!/usr/bin/env python3
"""Evidence for atomic run directories and provenance (#18).

    uv run --group evidence python tools/m2_runs/check.py --out DIR

Writes an analytic run (fixture dataset, closed-form candidates with accepted
and rejected rows, metrics, plot, log) twice and checks byte-identical replay;
checks data prepare under the new layout; interrupts writers by exception and
by SIGKILL; and classifies damaged copies as corrupt, incomplete, or
incompatible. Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "m2_schemas"))
sys.path.insert(0, str(REPO / "correct_il"))
from check import fixture_dataset  # noqa: E402  (tools/m2_schemas/check.py)
from system import DoubleIntegrator  # noqa: E402

from ccil.config import resolve  # noqa: E402
from ccil.runs import RunWriter, layout, validate_run  # noqa: E402
from ccil.runs.provenance import sha256  # noqa: E402
from ccil.schema import REASON_CODES, CandidateSet, read_candidates, read_trajectories, write_candidates, write_trajectories  # noqa: E402

RUN_ID = "analytic-0001"


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def analytic_run(workdir: Path, overwrite=False, fail_after_partial=False) -> Path:
    """Write the analytic run under workdir/runs with relative paths, so replays are identical."""
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        ds, config = fixture_dataset()
        write_trajectories(ds, "fixture.npz")
        cfg = resolve(overrides={"run.seed": 0, "run.output_dir": "runs", "data.path": "fixture.npz",
                                 "data.validation_fraction": 0.0, "run.overwrite": overwrite})
        sys_ = DoubleIntegrator(config["h"])
        s, a, sp, ids, frames = ds.transitions("state")
        s64, a64 = s.astype(np.float64), a.astype(np.float64)
        s_g = sys_.backward_euler(s64, a64)
        dist = np.linalg.norm(s_g - s64, axis=1)
        eps = config["methods"]["backward_euler_fast"]["epsilon"]
        reasons = np.zeros((len(s), len(REASON_CODES)), bool)
        reasons[:, REASON_CODES.index("out_of_support")] = ~(dist < eps)
        cands = CandidateSet(run_id=RUN_ID, generator={"name": "backward_euler_closed_form", "version": "1",
                                                       "params_hash": hashlib.sha256(f"eps={eps}".encode()).hexdigest()[:16]},
                             source_episode_id=ids, source_frame=frames.astype(np.int64),
                             observations={"state": s_g.astype(np.float32)}, actions=a, target_observations={"state": s},
                             accepted=~reasons.any(axis=1), reasons=reasons,
                             measurements={"distance": dist, "residual": sys_.residual(s_g, a64, s64)})
        with RunWriter(cfg["run.output_dir"], RUN_ID, "analytic.backward_euler", cfg, overwrite=overwrite) as run:
            run.add_input("dataset", "fixture.npz")
            write_trajectories(ds, run.path(layout.DATASET))
            write_candidates(cands, run.path(layout.CANDIDATES))
            if fail_after_partial:
                raise RuntimeError("simulated failure after partial writes")
            run.metrics({"candidates": int(len(dist)), "accepted": int(cands.accepted.sum()),
                         "rejected": int((~cands.accepted).sum()), "support_radius": eps})
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(dist, np.arange(len(dist)), c=np.where(cands.accepted, "tab:green", "tab:red"), s=12)
            ax.axvline(eps, ls="--", c="0.3")
            ax.set_xlabel("||s_G - s*||")
            fig.savefig(run.path(f"{layout.PLOTS}/distance.png"), dpi=80, metadata={"Software": None})
            plt.close(fig)
            run.log(f"generated {len(dist)} candidates; accepted {int(cands.accepted.sum())}")
        return workdir / "runs" / RUN_ID
    finally:
        os.chdir(cwd)


KILLED = """
import sys, time
sys.path.insert(0, {tools!r}); sys.path.insert(0, {correct_il!r})
from ccil.config import resolve
from ccil.runs import RunWriter
cfg = resolve(overrides={{"run.seed": 0, "run.output_dir": "runs", "data.path": "fixture.npz",
                          "data.validation_fraction": 0.0, "run.overwrite": True}})
with RunWriter("runs", {run_id!r}, "analytic.backward_euler", cfg, overwrite=True) as run:
    run.path("labels/candidates.npz").write_bytes(b"partial")
    print("ready", flush=True)
    time.sleep(60)
"""


def ccil_validate(path) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "ccil", "run", "validate", str(path)], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout)["status"]


def refresh_entry(run: Path, rel: str, version=None):
    """Update one manifest entry to match a deliberately changed file, so only the intended defect remains."""
    m = json.loads((run / layout.MANIFEST).read_text())
    m["files"][rel].update(sha256=sha256(run / rel), size=(run / rel).stat().st_size)
    if version is not None:
        m["files"][rel]["version"] = version
    (run / layout.MANIFEST).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    # 1. Analytic run, replayed byte-identically.
    wa, wb = out / "work_a", out / "work_b"
    wa.mkdir()
    wb.mkdir()
    run_a, run_b = analytic_run(wa), analytic_run(wb)
    status = ccil_validate(run_a)
    manifest = json.loads((run_a / layout.MANIFEST).read_text())
    record("analytic_run_complete_and_replayed_byte_identically", status == (0, "complete") and tree(run_a) == tree(run_b),
           status=status[1], files=sorted(manifest["files"]), manifest=manifest)

    # 5. Accepted and rejected rows share the run ID and trace to the stored dataset.
    dataset = read_trajectories(run_a / layout.DATASET)
    cands = read_candidates(run_a / layout.CANDIDATES, dataset)
    record("accepted_and_rejected_share_run_and_source_ids",
           cands.run_id == manifest["run_id"] and cands.accepted.any() and (~cands.accepted).any(),
           run_id=cands.run_id, accepted=int(cands.accepted.sum()), rejected=int((~cands.accepted).sum()),
           sources=[[str(e), int(f), bool(acc)] for e, f, acc in zip(cands.source_episode_id, cands.source_frame, cands.accepted)])

    # 2. data prepare under the new layout; the legacy 1.0 layout is incompatible.
    pdir = out / "prepare"
    (pdir / "data").mkdir(parents=True)
    sha = {a["path"]: a["sha256"] for a in json.loads((REPO / "docs/artifacts/manifest.json").read_text())["artifacts"]
           if a.get("path")}
    from trajectory_io import convert_legacy_pickle  # correct_il/trajectory_io.py
    convert_legacy_pickle(REPO / "data/pendulum_cont_100.pkl", pdir / "data/pendulum_cont_100.npz",
                          expect_sha256=sha["data/pendulum_cont_100.pkl"])
    r = subprocess.run([sys.executable, "-m", "ccil", "data", "prepare", "--recipe", "pendulum_cont",
                        "--set", "run.output_dir=runs"], cwd=pdir, capture_output=True, text=True)
    prep = next((pdir / "runs").iterdir(), None) if r.returncode == 0 else None
    legacy = out / "legacy_v1_run"
    if prep is not None:
        shutil.copytree(prep, legacy)
        m = json.loads((legacy / layout.MANIFEST).read_text())
        m.update(version="1.0")
        (legacy / layout.MANIFEST).write_text(json.dumps(m))
    record("data_prepare_complete_and_v1_manifest_incompatible",
           prep is not None and ccil_validate(prep) == (0, "complete") and ccil_validate(legacy) == (1, "incompatible"),
           prepare_run=prep.name if prep else None, stderr=r.stderr[-300:])

    # 3a. An exception after partial writes leaves the previous run intact and no staging directory.
    before = tree(run_a)
    try:
        analytic_run(wa, overwrite=True, fail_after_partial=True)
        raised = False
    except RuntimeError:
        raised = True
    staging = [p.name for p in (wa / "runs").iterdir() if p.name.startswith(".")]
    record("exception_leaves_previous_run_intact", raised and tree(run_a) == before and not staging, staging_left=staging)

    # 3b. SIGKILL mid-write: destination untouched, staging is incomplete, the next writer cleans it up.
    script = KILLED.format(tools=str(REPO / "tools" / "m2_schemas"), correct_il=str(REPO / "correct_il"), run_id=RUN_ID)
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=wa, stdout=subprocess.PIPE, text=True)
    ready = proc.stdout.readline().strip() == "ready"
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    stale = [p for p in (wa / "runs").iterdir() if p.name.startswith(".")]
    staged_status = ccil_validate(stale[0]) if stale else None
    intact = tree(run_a) == before
    time.sleep(0.1)
    analytic_run(wa, overwrite=True)
    left = [p.name for p in (wa / "runs").iterdir() if p.name.startswith(".")]
    record("sigkill_leaves_destination_intact_and_staging_is_reclaimed",
           ready and intact and len(stale) == 1 and staged_status == (1, "incomplete") and not left
           and ccil_validate(run_a) == (0, "complete"),
           staging_name=stale[0].name if stale else None, staging_status=staged_status, left_after_rerun=left)

    # 4. Validation transcript on damaged copies.
    def damaged(name, damage, expected):
        d = out / f"damaged_{name}"
        shutil.copytree(run_a, d)
        damage(d)
        got = ccil_validate(d)
        record(f"validate[{name}]", got == (1, expected), expected=expected, got=got[1])

    def flip(d):
        p = d / layout.DATASET
        b = bytearray(p.read_bytes())
        b[len(b) // 2] ^= 0xFF
        p.write_bytes(bytes(b))

    def mismatched_run_id(d):
        c = read_candidates(d / layout.CANDIDATES)
        c.run_id = "some-other-run"
        write_candidates(c, d / layout.CANDIDATES)
        refresh_entry(d, layout.CANDIDATES)

    def future_component(d):
        m = json.loads((d / layout.METRICS).read_text())
        m["version"] = "9.0"
        (d / layout.METRICS).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        refresh_entry(d, layout.METRICS, version="9.0")

    def future_manifest(d):
        m = json.loads((d / layout.MANIFEST).read_text())
        m["version"] = "9.0"
        (d / layout.MANIFEST).write_text(json.dumps(m))

    damaged("flipped_byte", flip, "corrupt")
    damaged("missing_file", lambda d: (d / layout.METRICS).unlink(), "incomplete")
    damaged("missing_manifest", lambda d: (d / layout.MANIFEST).unlink(), "incomplete")
    damaged("unlisted_file", lambda d: (d / "notes.txt").write_text("stray"), "corrupt")
    damaged("candidates_run_id_mismatch", mismatched_run_id, "corrupt")
    damaged("future_component_schema", future_component, "incompatible")
    damaged("future_manifest_version", future_manifest, "incompatible")

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
