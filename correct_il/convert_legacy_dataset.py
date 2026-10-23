"""Convert legacy pickle datasets to the non-executable ccil-trajectories format.

Unpickling can execute arbitrary code, so every conversion needs a trust
decision:

    # The shipped datasets, each verified against the sha256 recorded in
    # docs/artifacts/manifest.json before it is unpickled:
    python correct_il/convert_legacy_dataset.py --shipped

    # Any other file, with a known hash or an explicit acknowledgement:
    python correct_il/convert_legacy_dataset.py SRC.pkl DST.npz --expect-sha256 HEX
    python correct_il/convert_legacy_dataset.py SRC.pkl DST.npz --trust-legacy-pickle
"""
import argparse
import json
import os
import sys

from trajectory_io import LegacyInputError, convert_legacy_pickle
from invariants import DataValidationError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "docs", "artifacts", "manifest.json")


def shipped():
    with open(MANIFEST) as f:
        artifacts = json.load(f)["artifacts"]
    return [(a["path"], a["sha256"]) for a in artifacts
            if a["kind"] == "dataset" and a["status"] == "resolved" and a["path"].endswith(".pkl")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--shipped", action="store_true", help="convert every dataset listed in docs/artifacts/manifest.json")
    ap.add_argument("--out-dir", default=None, help="with --shipped: where to write the .npz files (default: next to each source)")
    ap.add_argument("--expect-sha256")
    ap.add_argument("--trust-legacy-pickle", action="store_true",
                    help="acknowledge that SRC is trusted; unpickling it can execute arbitrary code")
    args = ap.parse_args()

    if args.shipped:
        jobs = []
        for rel, digest in shipped():
            src = os.path.join(REPO, rel)
            name = os.path.splitext(os.path.basename(rel))[0] + ".npz"
            jobs.append((src, os.path.join(args.out_dir or os.path.dirname(src), name), digest, False))
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
    elif args.src and args.dst:
        jobs = [(args.src, args.dst, args.expect_sha256, args.trust_legacy_pickle)]
    else:
        ap.error("give SRC and DST, or --shipped")

    failed = 0
    for src, dst, digest, trust in jobs:
        try:
            got = convert_legacy_pickle(src, dst, expect_sha256=digest, trust=trust)
            print(f"converted {src} -> {dst} (sha256 {got[:12]})")
        except (LegacyInputError, DataValidationError, FileNotFoundError) as e:
            failed += 1
            print(f"FAILED {src}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
