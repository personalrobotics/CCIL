#!/usr/bin/env python3
"""Verify docs/artifacts/manifest.json.

Offline: recompute each Git-hosted dataset's sha256, size, and blob id at its
archive tag; check status rules; and check that every dataset the ledger names
has a manifest entry. With --network: fetch each pinned code dependency commit,
download and hash each URL artifact, and re-resolve the environment lock with
uv to confirm it reproduces. Needs the archive tags locally:
    git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs/artifacts/manifest.json"
LEDGER = ROOT / "docs/ledger/ledger.json"
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)(==\S+| @ \S+)")


def run(*args, **kw):
    return subprocess.run(args, capture_output=True, **kw)


def pins(text):
    return sorted(m.group(0) for line in text.splitlines() if (m := PIN.match(line)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true", help="also fetch dependencies and re-resolve the lock")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    ledger = json.loads(LEDGER.read_text())
    errors = []

    def check(ok, msg):
        print(("OK    " if ok else "FAIL  ") + msg)
        if not ok:
            errors.append(msg)

    by_path = {a["path"]: a for a in manifest["artifacts"] if a.get("path")}
    entry_ids = {e["id"] for e in ledger["entries"]}

    for a in manifest["artifacts"]:
        aid = a["id"]
        for eid in a.get("ledger_entries", []):
            check(eid in entry_ids, f"{aid}: ledger entry {eid} exists")
        if a["status"] == "unresolved":
            r = a.get("recovery") or {}
            check(all(r.get(k) for k in ("missing", "owner", "action")), f"{aid}: unresolved with recovery action")
            continue
        check(a["status"] == "resolved", f"{aid}: status is resolved or unresolved")
        locator = a.get("locator") or ""
        if locator.startswith("git:"):
            _, ref, path = locator.split(":", 2)
            raw = run("git", "-C", str(ROOT), "show", f"{ref}:{path}").stdout
            blob = run("git", "-C", str(ROOT), "rev-parse", f"{ref}:{path}", text=True).stdout.strip()
            check(hashlib.sha256(raw).hexdigest() == a["sha256"] and len(raw) == a["bytes"] and blob == a["git_blob"],
                  f"{aid}: sha256, size, and blob match at {ref}")
        elif locator.startswith("git+"):
            check(bool(a.get("commit")), f"{aid}: pinned commit recorded")
        elif locator.startswith("https://"):
            check(bool(a.get("sha256")), f"{aid}: sha256 recorded")

    for e in ledger["entries"]:
        for d in e["datasets"]:
            path = d.replace(" (missing)", "")
            check(path in by_path, f"ledger {e['id']}: dataset {path} is in the manifest")

    env = manifest["environment"]
    lock = (ROOT / env["lock"]).read_text()
    locked = {p.split("==")[0].split(" @ ")[0].lower().replace("_", "-") for p in pins(lock)}
    for line in (ROOT / env["inputs"][0]).read_text().splitlines():
        if line.strip():
            name = re.split(r"[=<> @]", line.strip(), 1)[0].lower().replace("_", "-")
            check(name in locked, f"lock pins requirement {name}")

    if args.network:
        with tempfile.TemporaryDirectory() as tmp:
            for a in manifest["artifacts"]:
                loc = a.get("locator") or ""
                if loc.startswith("git+"):
                    url, sha = loc[4:].rsplit("@", 1)
                    repo = Path(tmp) / a["id"].replace(":", "_")
                    run("git", "init", "-q", "--bare", str(repo))
                    ok = run("git", "-C", str(repo), "fetch", "-q", "--depth", "1", "--filter=blob:none", url, sha).returncode == 0
                    check(ok, f"{a['id']}: commit {sha[:12]} fetchable from {url}")
                elif loc.startswith("https://") and a.get("sha256"):
                    with urllib.request.urlopen(loc) as resp:
                        body = resp.read()
                    check(hashlib.sha256(body).hexdigest() == a["sha256"], f"{a['id']}: download sha256 matches")
            out = Path(tmp) / "lock.txt"
            r = run("uv", "pip", "compile", env["inputs"][0], "--python-version", "3.8",
                    "--python-platform", env["platform"], "--config-file", env["inputs"][1],
                    "--exclude-newer", env["resolved_as_of"], "-q", "-o", str(out), cwd=ROOT, text=True)
            check(r.returncode == 0 and pins(out.read_text()) == pins(lock),
                  f"re-resolved lock matches {env['lock']}" + ("" if r.returncode == 0 else f" ({r.stderr.strip()[-200:]})"))

    unresolved = [a["id"] for a in manifest["artifacts"] if a["status"] == "unresolved"]
    print(f"{len(manifest['artifacts'])} artifacts, {len(unresolved)} unresolved {unresolved}, {len(errors)} failures")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
