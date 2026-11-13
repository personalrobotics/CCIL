#!/usr/bin/env python3
"""Verify reproducible distributions and clean installs of the exact wheel."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHONS = ("3.10", "3.14")
CORE = {"ccil", "numpy", "pyyaml"}
FORBIDDEN = re.compile(r"(__pycache__|\.pyc$|\.npz$|\.pkl$|\.log$|manifest\.json$|^data/|\.venv)")  # caches, data, run outputs
LOCAL_PATHS = re.compile(r"(/Users/|/home/|/private/|/tmp/)")


def run(*cmd, cwd=None, check=True):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{r.stdout}\n{r.stderr}")
    return r


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SCRIPT = """
import json, sys
import numpy as np
from ccil.schema import Episode, TrajectoryDataset, write_trajectories
rng = np.random.default_rng(0)
eps = [Episode(f"ep{i}", {"state": rng.normal(size=(8, 3)).astype(np.float32)}, rng.normal(size=(8, 2)).astype(np.float32))
       for i in range(5)]
write_trajectories(TrajectoryDataset(eps), sys.argv[1])
print(json.dumps({"python": sys.version.split()[0], "file": __import__("ccil").__file__}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--evidence", type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    builds = []
    for n in (1, 2):
        d = out / f"dist{n}"
        run("uv", "build", "--out-dir", d, cwd=REPO)
        # Select by name: newer uv also writes a .gitignore into the output directory.
        builds.append({kind: p for p in d.iterdir() for kind in (".whl", ".tar.gz") if p.name.endswith(kind)})
    wheel, sdist = builds[0][".whl"], builds[0][".tar.gz"]
    record("builds_are_reproducible", all(sha(builds[0][k]) == sha(builds[1][k]) for k in builds[0]),
           wheel_sha256=sha(wheel), sdist_sha256=sha(sdist))

    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
        entry = z.read(next(n for n in names if n.endswith(".dist-info/entry_points.txt"))).decode()
        texts = [z.read(n).decode(errors="replace") for n in names if not n.endswith("/")]
    with tarfile.open(sdist) as t:
        snames = t.getnames()
    bad = [n for n in names if not (n.startswith("ccil/") or ".dist-info/" in n) or FORBIDDEN.search(n)]
    sbad = [n for n in snames if FORBIDDEN.search(n.split("/", 1)[-1])]
    leaks = sorted({m.group(0) for t in texts for m in LOCAL_PATHS.finditer(t)})
    needed = {"ccil/__init__.py", "ccil/__main__.py", "ccil/recipes/pendulum_cont.yaml", "ccil/recipes/pendulum_disc.yaml",
              "ccil/schema/io.py", "ccil/config/params.py", "ccil/data/prepare.py",
              "ccil/models/LICENSE-mjrl.txt", "ccil/models/LICENSE-pytorch.txt", "ccil/models/NOTICE"}
    record("wheel_contains_package_only", not bad and not sbad and not leaks and needed <= set(names),
           wheel_files=names, unexpected=bad + sbad, local_paths=leaks, missing=sorted(needed - set(names)))
    license_expression = [line.split(":", 1)[1].strip() for line in meta.splitlines()
                          if line.startswith("License-Expression:")]
    license_files = sorted(line.split(":", 1)[1].strip() for line in meta.splitlines()
                           if line.startswith("License-File:"))
    expected_licenses = ["LICENSE", "src/ccil/models/LICENSE-mjrl.txt",
                         "src/ccil/models/LICENSE-pytorch.txt", "src/ccil/models/NOTICE"]
    wheel_licenses = sorted(n.split(".dist-info/licenses/", 1)[1] for n in names
                            if ".dist-info/licenses/" in n and not n.endswith("/"))
    sdist_files = {n.split("/", 1)[1] for n in snames if "/" in n}
    record("distributions_carry_project_and_third_party_licenses",
           license_expression == ["MIT"] and license_files == expected_licenses
           and wheel_licenses == expected_licenses and set(expected_licenses) <= sdist_files,
           license_expression=license_expression, license_files=license_files,
           wheel_license_files=wheel_licenses,
           sdist_missing=sorted(set(expected_licenses) - sdist_files))
    requires = sorted(line.split(":", 1)[1].strip() for line in meta.splitlines()
                      if line.startswith("Requires-Dist:"))
    pyreq = [line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Requires-Python:")]
    extras = sorted(line.split(":", 1)[1].strip() for line in meta.splitlines()
                    if line.startswith("Provides-Extra:"))
    urls = sorted(line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Project-URL:"))
    relative_links = sorted(target for target in re.findall(r"\]\(([^)]+)\)", meta)
                            if not target.startswith(("https://", "http://", "#", "mailto:")))
    core = [r for r in requires if "extra ==" not in r]
    optional = [r for r in requires if "extra ==" in r]
    by_extra = {e: sorted(r for r in optional if f"extra == '{e}'" in r) for e in extras}
    record("metadata_declares_core_dependencies_only", core == ["numpy>=1.24", "pyyaml>=6.0"] and pyreq == [">=3.10"]
           and extras == ["lerobot", "torch"] and len(optional) == 3
           and len(by_extra["torch"]) == 1 and by_extra["torch"][0].startswith("torch>=2.9.1")
           and re.search(r"python(_full)?_version >= ['\"]3\.11['\"]", by_extra["torch"][0]) is not None
           and [r.split(" ;")[0] for r in by_extra["lerobot"]] == ["huggingface-hub>=1.0.0", "pyarrow>=21.0.0"]
           and re.search(r"python(_full)?_version >= ['\"]3\.11['\"]", by_extra["lerobot"][0]) is not None
           and len(urls) == 5 and not relative_links
           and "ccil = ccil.__main__:main" in entry,
           requires_dist=requires, requires_python=pyreq, extras=extras, project_urls=urls,
           relative_readme_links=relative_links, entry_points=entry.strip())

    for py in PYTHONS:
        venv = out / f"venv-{py}"
        run("uv", "venv", "--python", py, venv)
        pyexe = venv / "bin" / "python"
        run("uv", "pip", "install", "--python", pyexe, wheel)
        listed = json.loads(run("uv", "pip", "list", "--python", pyexe, "--format", "json").stdout)
        installed = {p["name"].lower(): p["version"] for p in listed}
        work = out / f"work-{py}"
        work.mkdir()
        info = json.loads(run(pyexe, "-I", "-c", SCRIPT, work / "tiny.npz", cwd=work).stdout)
        help_ok = run(venv / "bin" / "ccil", "--help", cwd=work).returncode == 0
        shown = json.loads(run(venv / "bin" / "ccil", "config", "show", "--recipe", "pendulum_cont", cwd=work).stdout)
        prep = run(venv / "bin" / "ccil", "data", "prepare", "--set", "data.path=tiny.npz", "--set", "run.output_dir=runs",
                   cwd=work, check=False)
        runs = list((work / "runs").iterdir()) if (work / "runs").exists() else []
        metrics = json.loads((runs[0] / "metrics.json").read_text())["values"] if len(runs) == 1 else {}
        validated = json.loads(run(venv / "bin" / "ccil", "run", "validate", runs[0], cwd=work, check=False).stdout)["status"] \
            if len(runs) == 1 else None
        labels = run(venv / "bin" / "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
                     "--set", "data.path=tiny.npz", "--set", "run.output_dir=label_runs", cwd=work, check=False)
        label_runs = sorted((work / "label_runs").iterdir()) if (work / "label_runs").exists() else []
        label_status = sorted(json.loads(run(venv / "bin" / "ccil", "run", "validate", d, cwd=work, check=False).stdout)["status"]
                              for d in label_runs)
        # Imported from the venv's site-packages, not the source tree (the output dir may sit inside the repo).
        from_wheel = Path(info["file"]).resolve().is_relative_to(venv.resolve()) and "site-packages" in Path(info["file"]).parts
        record(f"clean_install_runs[python {py}]",
               set(installed) == CORE and from_wheel and help_ok and shown["schema"] == "ccil.config"
               and prep.returncode == 0 and validated == "complete"
               and (metrics.get("train_episodes"), metrics.get("validation_episodes")) == (4, 1)
               and labels.returncode == 0 and label_status == ["complete"] * 5,
               python=info["python"], installed=installed, imported_from_venv=from_wheel,
               prepare_stderr=prep.stderr[-300:], run_status=validated, metrics=metrics,
               labels_generate_stages=label_status, labels_stderr=labels.stderr[-300:])

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock", "tools")),
                              "uv": run("uv", "--version").stdout.strip(), "host": platform.platform(),
                              "optional_dependencies": extras}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    text = text.replace(str(out), "<out>").replace(str(REPO), "<repo>")
    (out / "report.json").write_text(text)
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(text)
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
