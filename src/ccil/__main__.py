"""Command-line entry point: ccil <group> <command> (also python -m ccil)."""
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
            print(f"{args.out} is out of date; run: uv run ccil schema docs", file=sys.stderr)
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


def _write_or_check(text: str, out: str, check: bool, regenerate: str) -> int:
    if check:
        current = Path(out).read_text() if Path(out).exists() else ""
        if current != text:
            print(f"{out} is out of date; run: {regenerate}", file=sys.stderr)
            return 1
        print(f"{out} is up to date")
        return 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    print(f"wrote {out}")
    return 0


def _resolve(args):
    from .config import resolve
    return resolve(recipe=args.recipe, config=args.config, sets=args.set)


def _config_show(args) -> int:
    import json
    from .config import ConfigError
    try:
        cfg = _resolve(args)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(cfg.to_dict(), indent=1, sort_keys=True))
    return 0


def _config_docs(args) -> int:
    from .config.docs import render_parameters
    return _write_or_check(render_parameters(), args.out, args.check, "uv run ccil config docs")


def _config_audit(args) -> int:
    from .config.docs import render_audit
    return _write_or_check(render_audit(args.ledger), args.out, args.check, "uv run ccil config audit")


def _data_prepare(args) -> int:
    from .config import ConfigError
    from .data.prepare import prepare
    from .data.split import SplitError
    from .runs import RunError
    from .schema import SchemaError
    try:
        run_dir = prepare(_resolve(args))
    except (ConfigError, SchemaError, SplitError, FileExistsError, RunError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"wrote {run_dir}")
    return 0


def _run_validate(args) -> int:
    import json
    from .runs import validate_run
    status, problems = validate_run(args.run_dir)
    print(json.dumps({"run": args.run_dir, "status": status, "problems": problems}, indent=1))
    return 0 if status == "complete" else 1


def _run_pipeline(args, which) -> int:
    from .config import ConfigError
    from .core import InterfaceError
    from .pipelines.stages import StageError
    from .schema import SchemaError
    from .supervision import SupervisionError
    try:
        cfg = _resolve(args)
        if which == "offline":
            from .pipelines.offline import generate_offline
            results = generate_offline(cfg)
        elif which == "policy":
            from .pipelines.policy_harness import run_policy_harness
            results = run_policy_harness(cfg)
        else:
            from .pipelines.integrated import run_integrated
            results = run_integrated(cfg)
    except (ConfigError, SchemaError, InterfaceError, StageError, SupervisionError) as e:
        print(str(e), file=sys.stderr)
        return 2
    for r in results:
        print(f"{r.stage:18s} {'reused' if r.reused else 'wrote '} {r.run_dir}")
    return 0


def _add_config_args(p) -> None:
    p.add_argument("--recipe", help="named recipe in src/ccil/recipes/")
    p.add_argument("--config", help="user configuration file (YAML)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one parameter; repeatable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ccil")
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
    config = groups.add_parser("config", help="configuration contract").add_subparsers(dest="command", required=True)
    show = config.add_parser("show", help="resolve and print the configuration with each value's source")
    _add_config_args(show)
    show.set_defaults(fn=_config_show)
    cdocs = config.add_parser("docs", help="generate the parameter reference")
    cdocs.add_argument("--out", default="docs/config/parameters.md")
    cdocs.add_argument("--check", action="store_true")
    cdocs.set_defaults(fn=_config_docs)
    audit = config.add_parser("audit", help="generate the repository-wide parameter audit")
    audit.add_argument("--ledger", default="docs/ledger/ledger.json")
    audit.add_argument("--out", default="docs/config/parameter-audit.md")
    audit.add_argument("--check", action="store_true")
    audit.set_defaults(fn=_config_audit)
    data = groups.add_parser("data", help="dataset workflows").add_subparsers(dest="command", required=True)
    prep = data.add_parser("prepare", help="validate a dataset, split it by episode, and write a run directory")
    _add_config_args(prep)
    prep.set_defaults(fn=_data_prepare)
    labels = groups.add_parser("labels", help="label generation").add_subparsers(dest="command", required=True)
    gen = labels.add_parser("generate", help="offline label generation", description='Run offline label generation: data.prepare -> transition.fit -> labels.generate.\n\nrequired:  data.path, generation.method (via --recipe, --config, or --set)\noutputs:   one run directory per stage under run.output_dir; labels in\n           labels-generate-*/labels/candidates.npz (accepted and rejected rows)\n           and a report in its metrics.json\nresume:    a stage whose run directory is already complete is reused\noptional:  no extra dependencies (core install)\nexit:      0 success; 2 invalid configuration, data, or stage failure',
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(gen)
    gen.set_defaults(fn=lambda a: _run_pipeline(a, "offline"))
    pipeline = groups.add_parser("pipeline", help="integrated pipelines").add_subparsers(dest="command", required=True)
    integ = pipeline.add_parser("integrated", help="representation, labels, and downstream policy",
                                description='Run the integrated pipeline: data.prepare -> representation.fit ->\ntransition.fit -> labels.generate -> policy.train -> policy.evaluate.\n\nThe reference components (PCA representation, linear or mean-action policy,\noffline action-error evaluation) stand in for M5 image encoders and real\npolicies; they claim no image-CCIL parity.\n\nrequired:  data.path, generation.method, representation.kind=pca, representation.dim\noutputs:   one run directory per stage under run.output_dir; the policy in\n           policy-train-*/checkpoints/policy.npz and its evaluation in\n           policy-evaluate-*/metrics.json\nresume:    complete stages are reused; changing only policy.* reuses stages 1-4\noptional:  no extra dependencies (core install)\nexit:      0 success; 2 invalid configuration, data, or stage failure', formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(integ)
    integ.set_defaults(fn=lambda a: _run_pipeline(a, "integrated"))
    policy = groups.add_parser("policy", help="experimental vector-policy harness (reference, not required by CCIL)"
                               ).add_subparsers(dest="command", required=True)
    ptrain = policy.add_parser("train", help="train the experimental vector policy on CCIL supervision",
                               description='Run the experimental vector-policy harness: the offline label stages\n(reused when complete), then policy.train and policy.evaluate.\n\nExperimental reference infrastructure for reproducing the BC, NoiseBC, and\nCCIL arms (#24). CCIL ends at the supervision artifacts; any trainer may\nconsume them (docs/supervision.md, docs/policy-harness.md).\n\nrequired:  data.path, generation.method, policy.method, policy.batch_size, policy.epochs\noutputs:   policy-train-*/checkpoints/policy.npz, supervision.json, and the\n           training record in metrics.json; offline metrics in\n           policy-evaluate-*/metrics.json\nresume:    complete stages are reused\noptional:  the ccil[torch] extra (Python 3.11+)\nexit:      0 success; 2 invalid configuration, data, supervision, or stage failure',
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(ptrain)
    ptrain.set_defaults(fn=lambda a: _run_pipeline(a, "policy"))
    runs = groups.add_parser("run", help="run directories").add_subparsers(dest="command", required=True)
    rval = runs.add_parser("validate", help="classify a run directory: complete, corrupt, incompatible, or incomplete")
    rval.add_argument("run_dir")
    rval.set_defaults(fn=_run_validate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
