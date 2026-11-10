#!/usr/bin/env python3
"""A new user can install the built wheel and follow docs/quickstart.md, as written, to curated rows (#170).

    uv run python tools/golden_path/check.py --out NEW_DIR [--evidence docs/fixtures/evidence/golden-path/report.json]

Builds the wheel, installs it in a clean Python 3.12 environment with only the core dependencies (no torch, no
LeRobot, no evidence group), and outside the repository runs every block of docs/quickstart.md marked
`quickstart`, in order. Then:
- the quickstart completes, and its pilot (an external numpy learner on build_supervision's rows) runs;
- every artifact's SHA-256 is recorded, and a second run reuses every stage and reproduces them;
- a flipped byte in the candidates makes curate and inspect exit 2 and write nothing;
- an unknown key, a NaN action and a file that is not a ccil dataset each exit 2 before any stage completes.
The report holds no paths or times, so a rerun on the same host reproduces it byte for byte.
"""
import argparse
import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
QUICKSTART = REPO / "docs" / "quickstart.md"


def sh(*cmd, cwd=None, env_bin=None, check=True):
    env = None
    if env_bin is not None:
        import os
        env = {**os.environ, "PATH": f"{env_bin}:{os.environ['PATH']}"}
        env.pop("VIRTUAL_ENV", None)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quickstart(work: Path) -> list[str]:
    """Write the quickstart's named files into work; return its shell blocks, in order."""
    blocks = re.findall(r"```(\w+) quickstart ?(\S*)\n(.*?)```", QUICKSTART.read_text(), re.S)
    shell = []
    for _, name, body in blocks:
        if name:
            (work / name).write_text(body)
        else:
            shell.append(body)
    return shell


def artifacts(work: Path) -> dict:
    """Every artifact the generate and curate results record, by result kind and name: its SHA-256, as recorded."""
    out = {}
    for f in sorted(glob.glob(str(work / "runs" / "*.json"))):
        doc = json.loads(Path(f).read_text())
        kind = "generate" if doc["schema"] == "ccil.generate-result" else f"curate[{doc['counts']['policy']['tolerance']}]"
        out[kind] = {k: a["sha256"] for k, a in sorted(doc["artifacts"].items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--evidence", type=Path)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)

    sh("uv", "build", "--wheel", "--out-dir", out / "dist", cwd=REPO)
    [wheel] = glob.glob(str(out / "dist" / "ccil-*.whl"))
    sh("uv", "venv", "--python", "3.12", out / "venv")
    sh("uv", "pip", "install", "--python", out / "venv" / "bin" / "python", wheel)
    installed = sorted(p["name"].lower() for p in json.loads(
        sh("uv", "pip", "list", "--python", out / "venv" / "bin" / "python", "--format", "json").stdout))
    bin_ = out / "venv" / "bin"

    work = out / "work"
    work.mkdir()
    runs = [sh("bash", "-ec", block, cwd=work, env_bin=bin_, check=False) for block in quickstart(work)]
    pilot = [ln for ln in runs[-1].stdout.splitlines() if ": BC " in ln]
    first = artifacts(work)
    record("quickstart_runs_as_written_from_the_wheel", all(r.returncode == 0 for r in runs)
           and installed == ["ccil", "numpy", "pyyaml"] and len(pilot) == 2 and "generate" in first,
           installed=installed, exits=[r.returncode for r in runs], pilot=pilot,
           errors=[r.stderr[-300:] for r in runs if r.returncode])

    again = [sh("bash", "-ec", block, cwd=work, env_bin=bin_, check=False) for block in quickstart(work)[1:3]]
    stages = [ln.split()[1] for r in again for ln in r.stdout.splitlines()
              if len(ln.split()) > 2 and ln.split()[1] in ("wrote", "reused")]
    record("rerun_reuses_every_stage_and_reproduces_every_artifact", all(r.returncode == 0 for r in again)
           and stages and set(stages) == {"reused"} and artifacts(work) == first, stages=stages, artifacts=first)

    gen = json.loads((work / "runs" / "generate-result.json").read_text())
    cands = work / "runs" / gen["artifacts"]["candidates"]["path"]
    original = cands.read_bytes()
    flipped = bytearray(original)
    flipped[len(flipped) // 2] ^= 1
    cands.write_bytes(bytes(flipped))
    before = sorted(p.name for p in (work / "runs").iterdir())
    cur = sh(bin_ / "ccil", "curate", "runs/generate-result.json", "--set", "curate.tolerance=0.2", cwd=work, check=False)
    ins = sh(bin_ / "ccil", "inspect", "runs/generate-result.json", cwd=work, check=False)
    after = sorted(p.name for p in (work / "runs").iterdir())
    cands.write_bytes(original)
    record("a_changed_artifact_is_refused_before_anything_is_written", cur.returncode == 2 and ins.returncode == 2
           and "changed since Generate wrote it" in cur.stderr and before == after and "Traceback" not in cur.stderr,
           curate_exit=cur.returncode, inspect_exit=ins.returncode)

    bad = out / "bad"
    bad.mkdir()
    for f in ("ccil.yaml", "demos.py"):
        shutil.copy(work / f, bad / f)
    sh(bin_ / "python", "demos.py", cwd=bad, env_bin=bin_)
    with np.load(bad / "demos.npz") as z:
        arrays = {k: z[k] for k in z.files}
    nan = dict(arrays)
    nan["actions"] = arrays["actions"].copy()
    nan["actions"][3] = np.nan
    np.savez(bad / "nan.npz", **nan)
    np.savez(bad / "not_ccil.npz", x=np.zeros(3))
    refusals = {}
    for name, sets in (("unknown_key", ["--set", "data.pth=demos.npz"]), ("nan_action", ["--set", "data.path=nan.npz"]),
                       ("not_a_ccil_dataset", ["--set", "data.path=not_ccil.npz"])):
        r = sh(bin_ / "ccil", "generate", "--config", "ccil.yaml", "--set", f"run.output_dir=runs_{name}", *sets,
               cwd=bad, check=False)
        complete = [p.name for p in (bad / f"runs_{name}").glob("*") if (p / "manifest.json").exists()] \
            if (bad / f"runs_{name}").exists() else []
        refusals[name] = {"exit": r.returncode, "complete_runs": len(complete), "traceback": "Traceback" in r.stderr,
                          "error": r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else ""}
    record("invalid_inputs_exit_2_before_any_stage_completes",
           all(v["exit"] == 2 and not v["complete_runs"] and not v["traceback"] for v in refusals.values()),
           refusals=refusals)

    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases, "wheel_sha256": sha(wheel),
              "quickstart_sha256": sha(QUICKSTART), "python": "3.12"}
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (out / "report.json").write_text(text)
    if a.evidence:
        a.evidence.parent.mkdir(parents=True, exist_ok=True)
        a.evidence.write_text(text)
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
