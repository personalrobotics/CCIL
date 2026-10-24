#!/usr/bin/env python3
"""Dependency gate and torch boundary evidence (#84).

    uv run python tools/dependency_gate/check.py --out DIR

Needs network access (PyPI, OSV). Checks that uv.lock is current and has no
known vulnerability outside the accepted list in docs/security.md; that the
evidence group is pinned exactly and limited to Python 3.11+; that the built
wheel's ccil[torch] installs and runs at the minimum and at the locked torch on
3.11, and installs no torch on 3.10; that ccil never unpickles; and that every
legacy manifest pinning a pre-2.6 torch is documented and outside the lock.
Writes DIR/report.json.
"""
import argparse
import json
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TORCH_MINIMUM, TORCH_LOCKED = "2.9.1", "2.14.1"
ACCEPTED_IN_LOCK: set = set()  # advisory IDs accepted for uv.lock; docs/security.md lists none

SMOKE = """
import json, sys
import numpy as np
import torch
import ccil
from ccil.schema.io import _atomic_savez
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).eval()
x = torch.randn(5, 3)
# The #21 checkpoint path: state_dict -> numpy -> .npz without pickle -> state_dict.
_atomic_savez(sys.argv[1], {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
with np.load(sys.argv[1], allow_pickle=False) as z:
    state = {k: torch.from_numpy(z[k]) for k in z.files}
twin = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).eval()
twin.load_state_dict(state)
with torch.no_grad():
    same = bool(torch.equal(net(x), twin(x)))
print(json.dumps({"torch": torch.__version__, "ccil": ccil.__file__, "npz_round_trip_identical": same}))
"""


def run(*cmd, cwd=None):
    return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    # 1. The lock is current and has no known vulnerability outside the accepted list.
    current = run("uv", "lock", "--check", cwd=REPO)
    audit = run("uv", "audit", "--locked", "--output-format", "json", "--preview-features", "audit-command,json-output",
                cwd=REPO)
    try:
        report = json.loads(audit.stdout)
        found = [{"id": v["id"], "package": v["dependency"]["name"], "version": v["dependency"]["version"],
                  "fix_versions": v.get("fix_versions")} for v in report["vulnerabilities"]]
        audited = report["summary"]["audited_packages"]
    except (ValueError, KeyError) as e:
        found, audited = None, f"unreadable audit output: {e}: {audit.stderr[-300:]}"
    blocking = None if found is None else [v for v in found if v["id"] not in ACCEPTED_IN_LOCK]
    record("lock_current_and_free_of_known_vulnerabilities", current.returncode == 0 and blocking == [],
           lock_check=current.stderr.strip()[-200:], audited_packages=audited, vulnerabilities=found,
           accepted=sorted(ACCEPTED_IN_LOCK))

    # 2. The evidence environment is exact and 3.11+; the core stays numpy + pyyaml on 3.10+.
    py = tomllib.loads((REPO / "pyproject.toml").read_text())
    evidence = py["dependency-groups"]["evidence"]
    group_python = py["tool"]["uv"]["dependency-groups"]["evidence"]["requires-python"]
    record("evidence_pinned_exactly_on_python_3_11_plus",
           all(re.fullmatch(r"[a-z0-9_.-]+==[0-9][0-9a-z.]*", d) for d in evidence) and group_python == ">=3.11"
           and f"torch=={TORCH_LOCKED}" in evidence and py["project"]["requires-python"] == ">=3.10"
           and py["project"]["dependencies"] == ["numpy>=1.24", "pyyaml>=6.0"]
           and py["project"]["optional-dependencies"]["torch"] == [f"torch>={TORCH_MINIMUM}; python_version >= '3.11'"],
           evidence=evidence, evidence_requires_python=group_python, core=py["project"]["dependencies"],
           torch_extra=py["project"]["optional-dependencies"]["torch"])

    # 3. ccil[torch] from the built wheel, at the minimum and the locked torch, and on 3.10.
    dist = out / "dist"
    built = run("uv", "build", "--wheel", "--out-dir", dist, cwd=REPO)
    wheel = next(dist.glob("ccil-*.whl"), None)
    for label, python, pin in (("minimum", "3.11", TORCH_MINIMUM), ("locked", "3.11", TORCH_LOCKED), ("core", "3.10", None)):
        venv = out / f"venv-{label}"
        run("uv", "venv", "--python", python, venv)
        pyexe = venv / "bin" / "python"
        reqs = [f"{wheel}[torch]"] + ([f"torch=={pin}"] if pin else [])
        inst = run("uv", "pip", "install", "--python", pyexe, *reqs)
        listed = run("uv", "pip", "list", "--python", pyexe, "--format", "json")
        installed = {p["name"].lower(): p["version"] for p in json.loads(listed.stdout)} if listed.returncode == 0 else {}
        if pin:
            smoke = run(pyexe, "-I", "-c", SMOKE, out / f"smoke-{label}.npz", cwd=out)
            info = json.loads(smoke.stdout) if smoke.returncode == 0 else {"error": smoke.stderr[-400:]}
            filelock = tuple(int(x) for x in installed.get("filelock", "0").split(".")[:3])
            ok = (inst.returncode == 0 and info.get("torch", "").split("+")[0] == pin and info.get("npz_round_trip_identical")
                  and str(venv) in info.get("ccil", "") and filelock >= (4, 2, 1) and "pillow" not in installed)
            record(f"torch_extra_installs_and_runs[{label} {pin}, python {python}]", ok, smoke=info,
                   filelock=installed.get("filelock"), install_stderr=inst.stderr[-300:] if inst.returncode else "")
        else:
            imp = run(pyexe, "-I", "-c", "import ccil, sys; print(sys.version.split()[0])", cwd=out)
            record("python_3_10_core_installs_without_torch", inst.returncode == 0 and imp.returncode == 0
                   and set(installed) == {"ccil", "numpy", "pyyaml"}, installed=installed, python=imp.stdout.strip())
    if wheel is None:
        record("wheel_built", False, stderr=built.stderr[-300:])

    # 4. ccil never unpickles: no torch.load/torch.save/pickle and no allow_pickle=True in the package.
    pattern = re.compile(r"torch\.(load|save|jit\.load)\b|\bimport pickle\b|\bpickle\.|allow_pickle\s*=\s*True")
    hits = [f"{p.relative_to(REPO)}:{i}" for p in sorted((REPO / "src" / "ccil").rglob("*.py"))
            for i, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    record("ccil_package_never_unpickles", not hits, hits=hits)

    # 5. Every legacy manifest pinning torch < 2.6 is documented and outside the project lock.
    doc = (REPO / "docs" / "security.md").read_text()
    lock = (REPO / "uv.lock").read_text()
    legacy = {}
    for req in sorted(REPO.glob("**/requirements*.txt")):
        if ".venv" in req.parts:
            continue
        pin = next((l.strip() for l in req.read_text().splitlines() if l.lower().startswith("torch==")), None)
        if pin and tuple(int(x) for x in pin.split("==")[1].split(".")[:2]) < (2, 6):
            rel = req.relative_to(REPO).as_posix()
            legacy[rel] = {"pin": pin, "documented": f"`{rel}` | `{pin}`" in doc,
                           "torch_in_lock_at_this_version": f'name = "torch"\nversion = "{pin.split("==")[1]}"' in lock}
    record("legacy_pins_documented_and_isolated", len(legacy) == 6 and all(v["documented"] and not v["torch_in_lock_at_this_version"]
                                                                           for v in legacy.values()), manifests=legacy)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    result = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "uv": run("uv", "--version").stdout.strip(), "python": platform.python_version(),
                              "platform": platform.platform()}}
    text = json.dumps(result, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
