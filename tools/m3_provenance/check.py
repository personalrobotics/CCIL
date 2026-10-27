#!/usr/bin/env python3
"""Evidence for once-per-process code state and the writer-owned hash cache (#127).

    uv run --group evidence python tools/m3_provenance/check.py --out DIR

Run it from a clean commit: the git cases clone HEAD. Each case runs the offline analytic pipeline
(`ccil labels generate`) in a fresh child process:
1. a ccil source edit between two stages fails the unit (exit 3) and commits no later stage, in a git clone and in
   a non-git copy (only the source hash can see it there); a tracked non-source edit fails the command at its end;
   the unedited clone exits 0; a direct RunWriter, with no stage identity, captures the code state on entry and
   refuses to commit after a source edit in its body;
2. an unchanged rerun in one process reuses every stage, hashes pipeline artifacts from the cache and the user's
   dataset in full;
3. an atomically replaced artifact (same size, same nanosecond mtime, new inode) is rehashed;
4. an in-place edit with a changed mtime is rehashed; with its metadata restored it is not, which is the documented
   limit, and strict mode (CCIL_PROVENANCE_STRICT=1) rehashes it;
5. a same-size in-place rewrite of the user's dataset with its mtime restored is detected: data.prepare writes anew;
6. cached and strict runs write byte-identical runs, and strict mode never uses the cache.
Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "m2_interfaces"))
from check import fixture  # noqa: E402  (tools/m2_interfaces/check.py)

from ccil.schema import write_trajectories  # noqa: E402

ARGS = ["labels", "generate", "--recipe", "analytic_double_integrator", "--set", "data.path=fixture.npz",
        "--set", "run.output_dir=runs"]

# Runs ccil's CLI after PRELUDE; on the second RunWriter (data.prepare has committed) it appends to EDIT.
EDIT_MID_UNIT = """
import sys
sys.path.insert(0, {src!r})
import ccil
assert ccil.__file__.startswith({src!r}), ccil.__file__
from ccil.runs import writer
enter, n = writer.RunWriter.__enter__, [0]
def patched(self):
    n[0] += 1
    if n[0] == 2 and {edit!r}:
        with open({edit!r}, "a") as f:
            f.write("# edited while a unit ran\\n")
    return enter(self)
writer.RunWriter.__enter__ = patched
from ccil.__main__ import main
sys.exit(main(sys.argv[1:]))
"""

# A direct RunWriter workflow, with no stage identity: the writer captures the code state on entry, and the
# ccil source is edited inside its body.
DIRECT_WRITER = """
import json, sys
sys.path.insert(0, {src!r})
import ccil
assert ccil.__file__.startswith({src!r}), ccil.__file__
from ccil.config import resolve
from ccil.runs import RunWriter, provenance
cfg = resolve(overrides={{"run.seed": 0, "run.output_dir": "runs", "data.path": "fixture.npz"}})
before = provenance._IMPLEMENTATION is None
try:
    with RunWriter("runs", "direct-0001", "direct.writer", cfg) as run:
        captured = provenance._IMPLEMENTATION is not None
        run.path("data/note.txt").write_text("written by a direct writer\\n")
        if {edit!r}:
            with open({edit!r}, "a") as f:
                f.write("# edited inside a writer body\\n")
    outcome = "committed"
except provenance.CodeStateChanged as e:
    outcome = "CodeStateChanged"
print(json.dumps({{"uncaptured_before_entry": before, "captured_in_body": captured, "outcome": outcome}}))
"""

# Cases 2-5 in one process: the cache is per process.
PROBE = """
import contextlib, hashlib, io, json, os, sys
from pathlib import Path
from ccil.__main__ import main
from ccil.runs import provenance as p
full = []
orig = p._full_sha256
def counted(path):
    full.append(os.path.relpath(os.path.realpath(path)))
    return orig(path)
p._full_sha256 = counted
def run():
    del full[:]
    p.HASH_STATS.update(full=0, cached=0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(sys.argv[2:])
    lines = [l.split() for l in buf.getvalue().splitlines()]
    return {"rc": rc, "status": {l[0]: l[1] for l in lines}, "full": sorted(set(full)), "stats": dict(p.HASH_STATS)}
def truth(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def meta(path):
    st = os.stat(path)
    return [st.st_ino, st.st_size, st.st_mtime_ns]
def flip_in_place(path, keep_mtime):
    st = os.stat(path)
    with open(path, "r+b") as f:
        f.seek(-1, 2); b = f.read(1); f.seek(-1, 2); f.write(bytes([b[0] ^ 1]))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns if keep_mtime else st.st_mtime_ns + 10**9))
out = {"first": run(), "second": run()}
runs = {d.name.split("-")[0] + "." + d.name.split("-")[1]: d for d in Path("runs").iterdir() if not d.name.startswith(".")}
# 5. user input: a same-size in-place rewrite to another valid dataset, mtime restored
fx, alt = Path("fixture.npz"), Path(sys.argv[1])
before = meta(fx)
st = os.stat(fx)
with open(fx, "r+b") as f:
    f.write(alt.read_bytes()); f.truncate()
os.utime(fx, ns=(st.st_atime_ns, st.st_mtime_ns))
out["user_input"] = {"meta_before": before, "meta_after": meta(fx), "rerun": run(),
                     "hash_is_content": p.sha256(fx) == truth(fx)}
# 3. atomic replacement of a registered artifact: same size and mtime, new inode
a = runs["data.prepare"] / "data" / "dataset.npz"
cached = p.sha256(a)
data = bytearray(a.read_bytes()); data[-1] ^= 1
tmp = a.with_name(".replacement")
tmp.write_bytes(bytes(data))
st = os.stat(a)
os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
m0 = meta(a)
os.replace(tmp, a)
out["atomic_replacement"] = {"meta_before": m0, "meta_after": meta(a), "cached_before": cached,
                             "after": p.sha256(a), "truth": truth(a)}
# 4. in-place edits of a registered artifact
b = runs["labels.generate"] / "labels" / "candidates.npz"
c = runs["transition.fit"] / "checkpoints" / "transition_model.npz"
cached_b, cached_c = p.sha256(b), p.sha256(c)
flip_in_place(b, keep_mtime=False)
flip_in_place(c, keep_mtime=True)
out["in_place"] = {"mtime_changed": {"cached_before": cached_b, "after": p.sha256(b), "truth": truth(b)},
                   "metadata_preserved": {"cached_before": cached_c, "after": p.sha256(c), "truth": truth(c)}}
os.environ["CCIL_PROVENANCE_STRICT"] = "1"
out["in_place"]["metadata_preserved"]["strict"] = p.sha256(c)
print(json.dumps(out))
"""

# Case 6: write then reuse, reporting the cache counters.
TWICE = """
import contextlib, io, json, sys
from ccil.__main__ import main
from ccil.runs import provenance as p
outs = []
for _ in range(2):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(sys.argv[1:])
    outs.append([rc, buf.getvalue()])
print(json.dumps({"runs": outs, "stats": p.HASH_STATS}))
"""


def tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def runs(d: Path) -> list:
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def child(code, *args, cwd, env=None):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    git = lambda *x, at=REPO: subprocess.run(["git", "-C", str(at), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    config["episodes"] = [e for e in config["episodes"] if not e["corrupt_actions"]]
    dataset = fixture(config)

    def workdir(name):
        w = out / name
        w.mkdir()
        write_trajectories(dataset, w / "fixture.npz")
        return w

    # 1. Code state: captured once, rechecked before every commit and at the command's end.
    clone = out / "clone"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(REPO), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", git("rev-parse", "HEAD")], check=True)
    copy = out / "copy"
    shutil.copytree(REPO / "src" / "ccil", copy / "ccil", ignore=shutil.ignore_patterns("__pycache__"))
    variants = {
        "git_source_edit": (clone / "src", clone / "src/ccil/core/generators.py"),
        "non_git_source_edit": (copy, copy / "ccil/core/generators.py"),
        "git_non_source_edit": (clone / "src", clone / "tools/m2_runs/check.py"),
        "unedited": (clone / "src", ""),
    }
    results = {}
    for name, (src, edit) in variants.items():
        subprocess.run(["git", "-C", str(clone), "checkout", "-q", "--", "."], check=True)  # clean at each start
        w = workdir(f"code_{name}")
        r = child(EDIT_MID_UNIT.format(src=str(src), edit=str(edit)), *ARGS, cwd=w)
        results[name] = {"rc": r.returncode, "runs": runs(w / "runs"), "stdout": r.stdout.strip().splitlines(),
                         "stderr": r.stderr.strip()[-1500:]}
    se, ng, ns, un = (results[k] for k in variants)
    record("source_edit_mid_unit_fails_the_unit",
           all(x["rc"] == 3 and len(x["runs"]) == 1 and x["runs"][0].startswith("data-prepare")
               and "the ccil source changed" in x["stderr"] for x in (se, ng)),
           git_clone=se, non_git_copy=ng)
    record("non_source_edit_fails_the_command_at_its_end",
           ns["rc"] == 3 and len(ns["runs"]) == 3 and "git state" in ns["stderr"]
           and "the ccil source changed" not in ns["stderr"], result=ns)
    record("unedited_clone_finalizes", un["rc"] == 0 and len(un["runs"]) == 3, result=un)

    direct = {}
    for name, edit in (("edited", copy / "ccil/core/generators.py"), ("unedited", "")):
        shutil.rmtree(copy / "ccil", ignore_errors=True)  # a fresh, unedited copy for each run
        shutil.copytree(REPO / "src" / "ccil", copy / "ccil", ignore=shutil.ignore_patterns("__pycache__"))
        w = workdir(f"direct_writer_{name}")
        r = child(DIRECT_WRITER.format(src=str(copy), edit=str(edit)), cwd=w)
        direct[name] = {**(json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else
                           {"stderr": r.stderr[-600:]}), "runs": sorted(p.name for p in (w / "runs").iterdir())}
    de, du = direct["edited"], direct["unedited"]
    record("direct_writer_captures_on_entry_and_refuses_a_source_edit_in_its_body",
           de.get("uncaptured_before_entry") is True and de.get("captured_in_body") is True
           and de.get("outcome") == "CodeStateChanged" and de["runs"] == []
           and du.get("outcome") == "committed" and du["runs"] == ["direct-0001"], edited=de, unedited=du)

    # 2-5. One process: reuse, replacement, in-place edits, the user's dataset.
    w = workdir("probe")
    alt = out / "fixture_alt.npz"
    perturbed = fixture(config)
    perturbed.episodes[0].observations["state"][0, 0] += 1.0
    write_trajectories(perturbed, alt)
    same_size = alt.stat().st_size == (w / "fixture.npz").stat().st_size
    r = child(PROBE, alt, *ARGS, cwd=w)
    probe = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"stderr": r.stderr[-800:]}
    if r.returncode != 0:
        record("probe_ran", False, **probe)
    else:
        first, second = probe["first"], probe["second"]
        record("unchanged_rerun_uses_the_cache",
               first["rc"] == 0 and set(first["status"].values()) == {"wrote"}
               and second["rc"] == 0 and set(second["status"].values()) == {"reused"}
               and second["full"] == ["fixture.npz"] and second["stats"]["cached"] > 0
               and "fixture.npz" in first["full"],
               first=first, second=second)
        ui = probe["user_input"]
        record("same_size_user_input_rewrite_with_mtime_restored_is_detected",
               same_size and ui["meta_before"] == ui["meta_after"] and ui["hash_is_content"]
               and ui["rerun"]["rc"] == 0 and ui["rerun"]["status"].get("data.prepare") == "wrote"
               and "fixture.npz" in ui["rerun"]["full"], **ui)
        ar = probe["atomic_replacement"]
        record("atomic_replacement_is_rehashed",
               ar["meta_before"][1:] == ar["meta_after"][1:] and ar["meta_before"][0] != ar["meta_after"][0]
               and ar["after"] == ar["truth"] != ar["cached_before"], **ar)
        ip = probe["in_place"]
        mc, mp = ip["mtime_changed"], ip["metadata_preserved"]
        record("in_place_edit_with_changed_mtime_is_rehashed", mc["after"] == mc["truth"] != mc["cached_before"], **mc)
        record("metadata_preserving_in_place_edit_is_outside_the_model_and_strict_mode_sees_it",
               mp["after"] == mp["cached_before"] != mp["truth"] and mp["strict"] == mp["truth"], **mp)

    # 6. Strict and cached runs agree byte for byte.
    both = {}
    for mode, env in (("cached", {}), ("strict", {"CCIL_PROVENANCE_STRICT": "1"})):
        w = workdir(f"mode_{mode}")
        r = child(TWICE, *ARGS, cwd=w, env=env)
        both[mode] = {"result": json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None,
                      "trees": {n: tree(w / "runs" / n) for n in runs(w / "runs")}, "stderr": r.stderr[-400:]}
    c, s = both["cached"], both["strict"]
    record("strict_and_cached_records_are_identical",
           c["result"] is not None and s["result"] is not None and len(c["trees"]) == 3 and c["trees"] == s["trees"]
           and c["result"]["runs"] == s["result"]["runs"] and all(rc == 0 for rc, _ in c["result"]["runs"])
           and s["result"]["stats"]["cached"] == 0 and c["result"]["stats"]["cached"] > 0,
           cached_stats=(c["result"] or {}).get("stats"), strict_stats=(s["result"] or {}).get("stats"),
           runs=sorted(c["trees"]), stdout=(c["result"] or {}).get("runs"), stderr=[c["stderr"], s["stderr"]])

    report = {"all_pass": all(x["pass"] for x in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(x['pass'] for x in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
