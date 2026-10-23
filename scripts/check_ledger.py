#!/usr/bin/env python3
"""Validate docs/ledger/ledger.json against docs/ledger/schema.json and history.

Checks required fields and enums, status rules (evidence for reproduced
entries, blockers for blocked ones, a rationale for out-of-scope ones), and
coverage: every README command, script, and config on the preserved refs, every
config key, and every inventory capability must map to the ledger. A file is
covered if it, or a byte-identical copy on another preserved ref, is covered.
Needs the archive tags locally:
    git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs/ledger/ledger.json"
SCHEMA = ROOT / "docs/ledger/schema.json"
INVENTORY = ROOT / "docs/inventory/capabilities.json"
COVERED_REFS = ("main", "label-filtering", "abhaybd-image")
COMMAND = re.compile(r"^(?:> )?((?:git|cd|pip|python|find|\./)\S*.*)$")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def files_at(commit):
    """Map path -> blob id for scripts and configs at a commit."""
    out = {}
    for line in git("ls-tree", "-r", commit).stdout.splitlines():
        meta, path = line.split("\t", 1)
        if path.endswith((".sh", ".slurm")) or (path.startswith("config/") and path.endswith(".yml")):
            out[path] = meta.split()[2]
    return out


def leaf_keys(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaf_keys(v, f"{prefix}.{k}" if prefix else k)
    else:
        yield prefix


def main():
    schema = json.loads(SCHEMA.read_text())
    ledger = json.loads(LEDGER.read_text())
    inv = json.loads(INVENTORY.read_text())
    enums = schema["enums"]
    refs = inv["refs"]
    errors = []

    if ledger["schema_version"] != schema["schema_version"]:
        errors.append("schema_version mismatch between ledger and schema")

    entries = {}
    for e in ledger["entries"]:
        eid = e.get("id", "?")
        if eid in entries:
            errors.append(f"{eid}: duplicate id")
        entries[eid] = e
        missing = [k for k in schema["entry_required"] if k not in e]
        if missing:
            errors.append(f"{eid}: missing {missing}")
            continue
        for section, req in (("origin", "origin_required"), ("evaluation", "evaluation_required"), ("target", "target_required")):
            gap = [k for k in schema[req] if k not in e[section]]
            if gap:
                errors.append(f"{eid}: {section} missing {gap}")
        if e["kind"] not in enums["kind"]:
            errors.append(f"{eid}: bad kind {e['kind']!r}")
        if e["status"] not in enums["status"]:
            errors.append(f"{eid}: bad status {e['status']!r}")
        if e["target"].get("parity") not in enums["parity"]:
            errors.append(f"{eid}: bad parity {e['target'].get('parity')!r}")
        if e["target"].get("command_status") not in enums["target_command_status"]:
            errors.append(f"{eid}: bad target command_status")
        ref = refs.get(e["origin"].get("ref"))
        if ref is None or ref["commit"] != e["origin"].get("commit"):
            errors.append(f"{eid}: origin ref/commit does not match the inventory refs")
            continue

        status = e["status"]
        if status.startswith("reproduced"):
            if not e["evidence"]:
                errors.append(f"{eid}: {status} requires evidence links")
            if status == "reproduced-with-correctness-change" and not e.get("correctness_change"):
                errors.append(f"{eid}: requires correctness_change")
        for link in e["evidence"]:
            if not (link.startswith("https://") or (ROOT / link).exists()):
                errors.append(f"{eid}: evidence {link!r} is neither an https URL nor a repository path")
        if status == "blocked":
            if not e["blockers"]:
                errors.append(f"{eid}: blocked requires blockers")
            for b in e["blockers"]:
                gap = [k for k in schema["blocker_required"] if not b.get(k)]
                if gap or b.get("missing") not in enums["missing"]:
                    errors.append(f"{eid}: incomplete blocker {b}")
        if status == "out-of-scope" and not e.get("exclusion_rationale"):
            errors.append(f"{eid}: out-of-scope requires exclusion_rationale")
        if status in ("pending", "blocked") and not e["target_milestone"]:
            errors.append(f"{eid}: non-terminal entry needs a target_milestone")
        if not e["terminal_rule"]:
            errors.append(f"{eid}: empty terminal_rule")

        for path in e["configs"] + [d for d in e["datasets"] if "(missing)" not in d]:
            if "*" not in path and git("cat-file", "-e", f"{ref['commit']}:{path}").returncode:
                errors.append(f"{eid}: {path} not found at {e['origin']['ref']}")
        for cap in e["inventory_capabilities"]:
            if cap not in {c["id"] for c in inv["capabilities"]}:
                errors.append(f"{eid}: unknown capability {cap!r}")

    params = {}
    for p in ledger["parameters"]:
        gap = [k for k in schema["parameter_required"] if k not in p]
        if gap or p.get("class") not in enums["value_class"]:
            errors.append(f"parameter {p.get('key')}: incomplete or bad class")
        params[p["key"]] = p

    # README commands on main.
    readme = git("show", f"{refs['main']['commit']}:README.md").stdout
    documented = ledger["documented_examples"]["README.md"]
    in_block = False
    for line in readme.splitlines():
        if line.startswith("```"):
            in_block = not in_block
            continue
        m = COMMAND.match(line.strip()) if in_block else None
        if m:
            target = documented.get(m.group(1))
            if target not in entries:
                errors.append(f"README command not mapped to an entry: {m.group(1)}")

    # Scripts and configs on each preserved ref.
    covered = {name: set() for name in COVERED_REFS}
    for e in entries.values():
        name = e["origin"]["ref"]
        if name in covered:
            covered[name].update(c for c in e["configs"] if "*" not in c)
            if e["origin"].get("script"):
                covered[name].add(e["origin"]["script"])
    for name, mapping in ledger["file_coverage"].items():
        for path, eid in mapping.items():
            if eid not in entries:
                errors.append(f"file_coverage {name}:{path} -> unknown entry {eid}")
            covered[name].add(path)
    snapshot = {name: files_at(refs[name]["commit"]) for name in COVERED_REFS}
    covered_blobs = {snapshot[n][p] for n in COVERED_REFS for p in covered[n] if p in snapshot[n]}
    for name in COVERED_REFS:
        for path, blob in snapshot[name].items():
            if path not in covered[name] and blob not in covered_blobs:
                errors.append(f"{name}:{path} is not covered by any ledger entry")
            if path.endswith(".yml"):
                cfg = yaml.safe_load(git("show", f"{refs[name]['commit']}:{path}").stdout)
                for key in leaf_keys(cfg):
                    if key not in params:
                        errors.append(f"{name}:{path}: config key {key} missing from parameter registry")

    # Inventory capabilities.
    used = {c for e in entries.values() for c in e["inventory_capabilities"]}
    for cap in inv["capabilities"]:
        if cap["id"] not in used and cap["id"] not in ledger["capability_exclusions"]:
            errors.append(f"capability {cap['id']} has no ledger entry or exclusion")

    for err in errors:
        print("ERROR", err)
    counts = {}
    for e in entries.values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print(f"{len(entries)} entries {dict(sorted(counts.items()))}, {len(params)} parameters, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
