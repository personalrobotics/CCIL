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
        return 2
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
    from .data.prepare import prepare
    run_dir = prepare(_resolve(args))
    print(f"wrote {run_dir}")
    return 0


def _run_validate(args) -> int:
    import json
    from .runs import validate_run
    status, problems = validate_run(args.run_dir)
    print(json.dumps({"run": args.run_dir, "status": status, "problems": problems}, indent=1))
    return 0 if status == "complete" else 2


def _diagnose(args) -> int:
    from . import api
    print(api.summary(api.diagnose(_resolve(args))))
    return 0


def _generate(args) -> int:
    from . import api
    print(api.summary(api.generate(_resolve(args))))
    return 0


def _curate(args) -> int:
    from . import api
    print(api.summary(api.curate(args.result, sets=args.set)))
    return 0


def _inspect(args) -> int:
    from . import api
    print(api.inspect(args.target).text)
    return 0


GOLDEN_HELP = """required:  data.path, generation.method (via --recipe, --config, or --set)
resume:    a stage whose run directory is already complete is reused
optional:  no extra dependencies (core install)
exit:      0 success, including an empty result; 1 an unexpected failure (with its traceback);
           2 an invalid configuration, dataset, checkpoint or artifact; 3 the code changed while it ran"""


def _run_pipeline(args, which) -> int:
    if which == "offline":  # the deprecated alias keeps its outputs exactly, for one minor release
        print("ccil labels generate is deprecated: use ccil generate, then ccil curate", file=sys.stderr)
    cfg = _resolve(args)
    if which == "offline":
        from .pipelines.offline import generate_offline
        results = generate_offline(cfg)
    else:
        from .pipelines.integrated import run_integrated
        results = run_integrated(cfg)
    for r in results:
        print(f"{r.stage:18s} {'reused' if r.reused else 'wrote '} {r.run_dir}")
    if getattr(args, "result", None):
        from .pipelines.offline import labels_result, write_labels_result
        try:
            write_labels_result(labels_result(cfg, results), args.result)
        except OSError as e:
            print(f"could not write the result {args.result}: {e}", file=sys.stderr)
            return 2
    return 0


def _data_convert(args) -> int:
    from .data.convert import convert_m1
    from .schema import SchemaError
    try:
        m = convert_m1(args.src, args.dst, args.manifest)
    except (SchemaError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"wrote {args.dst}: {m['counts']['episodes']} episodes, {m['counts']['frames']} frames (sha256 {m['output']['sha256']})")
    return 0


def _config_convert(args) -> int:
    from .config import ConfigError
    from .config.convert import convert_config
    mapping = {}
    for item in args.dataset_map:
        old, sep, new = item.partition("=")
        if not sep:
            print(f"--dataset-map expects OLD=NEW, got {item!r}", file=sys.stderr)
            return 2
        mapping[old] = new
    try:
        m = convert_config(args.src, args.out, args.manifest, fidelity=args.fidelity, dataset_map=mapping)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"wrote {args.out} and {args.manifest}: {m['counts']}")
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
    conv = config.add_parser("convert", help="convert a legacy correct_il YAML into a canonical configuration (#23)",
                             description="One-way conversion with a manifest of every legacy key's disposition.\n"
                                         "exit: 0 written; 2 unknown, duplicate, ambiguous, or incompatible settings "
                                         "(nothing written)", formatter_class=argparse.RawDescriptionHelpFormatter)
    conv.add_argument("src")
    conv.add_argument("--out", required=True)
    conv.add_argument("--manifest", required=True)
    conv.add_argument("--fidelity", choices=("corrected", "historical"), default="corrected",
                      help="historical writes legacy behavior that legacy configs left implicit")
    conv.add_argument("--dataset-map", action="append", default=[], metavar="OLD=NEW",
                      help="canonical dataset path for a legacy data.pkl value; repeatable")
    conv.set_defaults(fn=_config_convert)
    data = groups.add_parser("data", help="dataset workflows").add_subparsers(dest="command", required=True)
    dconv = data.add_parser("convert", help="convert an M1 ccil-trajectories/1 file into ccil.trajectories (#23)",
                            description="Legacy pickles first cross the trust boundary through "
                                        "tools/legacy_conversion/convert_legacy_dataset.py; ccil never unpickles.\n"
                                        "exit: 0 written; 2 not an M1 file or unreadable",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    dconv.add_argument("src")
    dconv.add_argument("dst")
    dconv.add_argument("--manifest")
    dconv.set_defaults(fn=_data_convert)
    prep = data.add_parser("prepare", help="validate a dataset, split it by episode, and write a run directory")
    _add_config_args(prep)
    prep.set_defaults(fn=_data_prepare)
    diag = groups.add_parser("diagnose", help="prepare, fit and diagnose: whether labels are worth generating",
                             description="Run data.prepare -> transition.fit -> labels.diagnose. No candidates.\n\n"
                                         "outputs:   one run directory per stage under run.output_dir; the diagnosis in\n"
                                         "           labels-diagnose-*/diagnosis/diagnosis.json\n" + GOLDEN_HELP,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(diag)
    diag.set_defaults(fn=_diagnose)
    gen_ = groups.add_parser("generate", help="diagnose, then generate candidates and their quality evidence",
                             description="Run Diagnose, then labels.generate and, with a certificate.profile,\n"
                                         "labels.certify. No selection: ccil curate writes those.\n\n"
                                         "outputs:   one run directory per stage under run.output_dir, and in it the\n"
                                         "           immutable generate-<id>.json (ccil.generate-result 1.0) that\n"
                                         "           ccil curate reads, copied to generate-result.json (the latest)\n"
                                         + GOLDEN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(gen_)
    gen_.set_defaults(fn=_generate)
    cur = groups.add_parser("curate", help="write one selection of a generate result's candidates",
                            description="Verify every artifact a generate result records, then run labels.curate\n"
                                        "(and labels.calibrate, labels.audit when configured). Any number of\n"
                                        "selections per generate result; nothing is regenerated.\n\n"
                                        "options:   --set may change curate.*, calibration.*, audit.*,\n"
                                        "           generation.evidence_mode and run.overwrite\n"
                                        "outputs:   a labels-curate-* run beside the generated ones, and curate-<id>.json\n"
                                        "           (ccil.curate-result 1.0) beside the generate result\n"
                                        "exit:      0 success; 1 an unexpected failure; 2 a changed or corrupted artifact,\n"
                                        "           or an invalid setting; 3 the code changed while it ran",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    cur.add_argument("result", help="a generate result (generate-result.json)")
    cur.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a curation setting; repeatable")
    cur.set_defaults(fn=_curate)
    ins = groups.add_parser("inspect", help="configuration, provenance, quality, rejections and usable rows",
                            description="Print a generate or curate result's resolved configuration (non-default\n"
                                        "values with their sources), provenance and hashes, quality, rejections,\n"
                                        "usable rows and next step; or a run directory's configuration,\n"
                                        "provenance and metrics.", formatter_class=argparse.RawDescriptionHelpFormatter)
    ins.add_argument("target", help="a generate or curate result, or a run directory")
    ins.set_defaults(fn=_inspect)
    labels = groups.add_parser("labels", help="label generation").add_subparsers(dest="command", required=True)
    gen = labels.add_parser("generate", help="deprecated: ccil generate, then ccil curate", description='Deprecated: use ccil generate, then ccil curate. Run offline label generation: data.prepare -> transition.fit -> labels.diagnose ->\nlabels.generate -> labels.certify (with a certificate.profile) -> labels.curate.\n\nrequired:  data.path, generation.method (via --recipe, --config, or --set)\noutputs:   one run directory per stage under run.output_dir; labels in\n           labels-generate-*/labels/candidates.npz (accepted and rejected rows)\n           and a report in its metrics.json; with --result PATH, the stages and\n           artifact paths as JSON (ccil.labels-result 1.0)\nresume:    a stage whose run directory is already complete is reused\noptional:  no extra dependencies (core install)\nexit:      0 success; 1 an unexpected failure (with its traceback); 2 an invalid\n           configuration, dataset, checkpoint or artifact; 3 the code changed while it ran',
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(gen)
    gen.add_argument("--result", metavar="PATH",
                     help="also write the machine-readable result (ccil.labels-result 1.0) to PATH")
    gen.set_defaults(fn=lambda a: _run_pipeline(a, "offline"))
    pipeline = groups.add_parser("pipeline", help="integrated pipelines").add_subparsers(dest="command", required=True)
    integ = pipeline.add_parser("integrated", help="representation, labels, and downstream policy",
                                description='Run the integrated pipeline: data.prepare -> representation.fit ->\ntransition.fit -> labels.generate -> policy.train -> policy.evaluate.\n\nThe reference components (PCA representation, linear or mean-action policy,\noffline action-error evaluation) stand in for M5 image encoders and real\npolicies; they claim no image-CCIL parity.\n\nrequired:  data.path, generation.method, representation.kind=pca, representation.dim\noutputs:   one run directory per stage under run.output_dir; the policy in\n           policy-train-*/checkpoints/policy.npz and its evaluation in\n           policy-evaluate-*/metrics.json\nresume:    complete stages are reused; changing only supervision.* or integrated.*\n           reuses stages 1-4\noptional:  no extra dependencies (core install)\nexit:      0 success; 1 an unexpected failure (with its traceback); 2 an invalid\n           configuration, dataset, checkpoint or artifact; 3 the code changed while it ran', formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_args(integ)
    integ.set_defaults(fn=lambda a: _run_pipeline(a, "integrated"))
    runs = groups.add_parser("run", help="run directories").add_subparsers(dest="command", required=True)
    rval = runs.add_parser("validate", help="classify a run directory: complete, corrupt, incompatible, or incomplete")
    rval.add_argument("run_dir")
    rval.set_defaults(fn=_run_validate)
    args = ap.parse_args(argv)
    from .errors import InputError
    from .pipelines.stages import StageError
    from .runs.provenance import CodeStateChanged, code, verify_code_state
    code()  # the git state, captured before the command does any work (#127)
    try:
        rc = args.fn(args)
        if rc == 0:  # a command is a unit: never finalize one that saw its code change (#127)
            verify_code_state()
    except CodeStateChanged as e:
        print(f"{e}; this command's outputs are not a consistent unit", file=sys.stderr)
        return 3
    except StageError as e:
        if not isinstance(e.error, (InputError, FileExistsError)):
            raise  # a defect inside a stage: exit 1 with its traceback, never as an input error
        print(str(e), file=sys.stderr)
        return 2
    except (InputError, FileExistsError) as e:  # every refused input exits 2 (docs/contract.md)
        print(str(e), file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
