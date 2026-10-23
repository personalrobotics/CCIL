"""Command-line entry point: python -m ccil <group> <command>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _schema_docs(args) -> int:
    from .schema.docs import render
    text = render()
    if args.check:
        current = Path(args.out).read_text() if Path(args.out).exists() else ""
        if current != text:
            print(f"{args.out} is out of date; run: python -m ccil schema docs", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")
    return 0


def _schema_validate(args) -> int:
    from .schema import SchemaError, read_candidates, read_trajectories
    try:
        dataset = read_trajectories(args.trajectories)
        print(f"{args.trajectories}: valid ({len(dataset.episodes)} episodes)")
        if args.candidates:
            cands = read_candidates(args.candidates, dataset)
            print(f"{args.candidates}: valid ({len(cands.accepted)} candidates, {int(cands.accepted.sum())} accepted)")
    except SchemaError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccil")
    groups = ap.add_subparsers(dest="group", required=True)
    schema = groups.add_parser("schema", help="data schemas").add_subparsers(dest="command", required=True)
    docs = schema.add_parser("docs", help="generate the schema reference")
    docs.add_argument("--out", default="docs/schemas/reference.md")
    docs.add_argument("--check", action="store_true", help="fail if the committed reference is out of date")
    docs.set_defaults(fn=_schema_docs)
    val = schema.add_parser("validate", help="validate a trajectories file and, optionally, its candidates")
    val.add_argument("trajectories")
    val.add_argument("--candidates")
    val.set_defaults(fn=_schema_validate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
