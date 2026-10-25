#!/usr/bin/env python3
"""Audit a certificate bundle on this host (#88).

    uv run --group evidence python tools/m3_certificates/audit_bundle.py --bundle DIR --out DIR

check.py writes DIR/bundle: per case, a certificate and copies of every source
it records (candidates, dataset, transition model, references), with a
manifest. This tool runs verify_structure and audit_sources on each case, so a
certificate made on one host is audited on another: the audit recomputes every
term to its row-local tolerance, every provided constant to 1e-9 relative,
and re-verifies every analytic_bound exactly. A case passes when both levels
report no problems. The report lists the SHA-256 of every bundle file, to
match against bundle_sha256 in the generating host's report.
Writes DIR/report.json.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from ccil.core import IdentityRepresentation
from ccil.core.certificates import Sources, audit_sources, verify_structure
from ccil.schema import read_certificates

REPO = Path(__file__).resolve().parents[2]
STANDARD = ("candidates", "dataset", "transition_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bundle, out = args.bundle.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / "manifest.json").read_text())
    cases = []
    for name, entry in sorted(manifest.items()):
        d = bundle / name
        files = {k: d / f for k, f in entry["files"].items()}
        sources = Sources(files["candidates"], files["dataset"], files["transition_model"],
                          IdentityRepresentation(entry["modality"]),
                          extra={k: p for k, p in files.items() if k not in STANDARD and not k.startswith("reference.")},
                          references={k[len("reference."):]: p for k, p in files.items() if k.startswith("reference.")})
        cert = read_certificates(d / entry["certificates"])
        structure, audit = verify_structure(cert, files["candidates"]), audit_sources(cert, sources)
        ok = structure == [] and audit == []
        cases.append({"case": name, "pass": ok, "profile": cert.descriptor["profile"],
                      "classification": sorted(set(cert.classification.tolist())), "rows": int(len(cert.selected)),
                      "constants": {r: {"value": c["value"], "source": c["source"]} for r, c in cert.descriptor["constants"].items()},
                      "structure": structure, "audit": audit[:5]})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps((structure + audit)[:3])[:600]}", flush=True)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              # Compare with bundle_sha256 in the generating host's check.py report: the same bytes were audited.
              "bundle_sha256": {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(bundle.rglob("*")) if p.is_file()},
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
