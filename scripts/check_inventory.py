#!/usr/bin/env python3
"""Check docs/inventory/capabilities.json against the preserved Git history.

Every capability must name its paths, a preserved ref, an owner (or
"unknown"), and a known disposition. Every path must exist at the recorded
commit, every commit must match docs/history/refs.tsv, and every capability
must appear in the human-readable matrix. Needs the archive tags locally:
    git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "docs/inventory/capabilities.json"
MATRIX = ROOT / "docs/inventory/README.md"
MANIFEST = ROOT / "docs/history/refs.tsv"
REQUIRED = ("id", "name", "category", "implementations", "provenance",
            "authors_of_record", "owner", "disposition", "blockers")


def git_has(commit, path):
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}:{path}"],
                          capture_output=True).returncode == 0


def manifest_shas():
    shas = {}
    for line in MANIFEST.read_text().splitlines():
        if line and not line.startswith("#"):
            _, ref, sha = line.split("\t")
            shas[ref.removesuffix("^{}")] = sha  # a peeled tag line overrides its tag object
    return shas


def main():
    inv = json.loads(INVENTORY.read_text())
    matrix = MATRIX.read_text()
    recorded = manifest_shas()
    errors = []

    for name, ref in inv["refs"].items():
        tag = ref["preserved_by"]
        full = tag if tag.startswith("refs/") else f"refs/tags/{tag}"
        if recorded.get(full) != ref["commit"]:
            errors.append(f"ref {name}: {tag} is not recorded at {ref['commit']} in {MANIFEST.name}")

    seen = set()
    for cap in inv["capabilities"]:
        cid = cap.get("id", "?")
        missing = [k for k in REQUIRED if k not in cap]
        if missing:
            errors.append(f"{cid}: missing fields {missing}")
            continue
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        if cap["disposition"] not in inv["dispositions"]:
            errors.append(f"{cid}: unknown disposition {cap['disposition']!r}")
        if not cap["owner"]:
            errors.append(f"{cid}: empty owner (use 'unknown')")
        if not cap["implementations"]:
            errors.append(f"{cid}: no implementations")
        for impl in cap["implementations"]:
            ref = inv["refs"].get(impl["ref"])
            if ref is None:
                errors.append(f"{cid}: unknown ref {impl['ref']!r}")
                continue
            if not impl["paths"]:
                errors.append(f"{cid}: no paths on {impl['ref']}")
            for path in impl["paths"]:
                if not git_has(ref["commit"], path):
                    errors.append(f"{cid}: {path} not found at {impl['ref']} ({ref['commit'][:12]})")
        if f"`{cid}`" not in matrix:
            errors.append(f"{cid}: not listed in {MATRIX.name}")

    for item in inv["do_not_migrate"]:
        if not git_has(inv["refs"][item["ref"]]["commit"], item["path"]):
            errors.append(f"do_not_migrate: {item['path']} not found at {item['ref']}")

    for e in errors:
        print("ERROR", e)
    print(f"{len(inv['capabilities'])} capabilities, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
