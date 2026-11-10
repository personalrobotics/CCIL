"""The Python entry points of the golden path; the CLI is a thin wrapper over them (docs/contract.md).

    cfg = ccil.resolve_config(recipe="analytic_double_integrator", overrides={"data.path": "demos.npz"})
    gen = ccil.generate(cfg)                       # candidates and quality evidence; no selection
    sel = ccil.curate(gen)                         # one selection per policy, any number per generate result
    print(ccil.inspect(sel.path).text)

Every refused input raises a ccil.InputError; any other exception is a defect.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import resolve
from .config.resolve import Resolved
from .core.curate import CurationPolicy
from .errors import InputError
from .pipelines import offline
from .pipelines.stages import StageResult
from .runs import layout
from .runs.provenance import sha256
from .schema.results import CURATE, GENERATE, VERSION, immutable_name, read_result, write_once, write_result

Config = Resolved
# The keys a Curate may change on the configuration its generate result recorded.
CURATE_KEYS = ("curate.", "calibration.", "audit.", "generation.evidence_mode", "run.overwrite")
GENERATE_RESULT = "generate-result.json"


@dataclass(frozen=True)
class Stage:
    stage: str
    run_dir: Path
    reused: bool


@dataclass(frozen=True)
class DiagnoseResult:
    """The stages Diagnose ran or reused, the diagnosis, and the admitted recovery length (or None, with why)."""
    stages: tuple[Stage, ...]
    diagnosis: Path
    admitted: int | None
    reason: str | None
    next_step: str | None
    timing: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerateResult:
    """What Generate wrote: its stages, the candidates and quality evidence, their counts, and its result file.

    path is the immutable generate-<id>.json; latest is generate-result.json, a copy of the latest Generate there."""
    stages: tuple[Stage, ...]
    diagnosis: Path
    admitted: int | None
    candidates: Path
    certificates: Path | None
    counts: dict
    path: Path
    latest: Path
    next_step: str


@dataclass(frozen=True)
class CurateResult:
    """What Curate wrote: its stages, the selection (and calibration, audit), their counts, and its result file."""
    stages: tuple[Stage, ...]
    selection: Path
    calibration: Path | None
    audit: Path | None
    counts: dict
    path: Path
    next_step: str


@dataclass(frozen=True)
class Inspection:
    """What `ccil inspect` prints, as data: one section per heading, and the printed text."""
    sections: dict
    text: str


def resolve_config(recipe: str | None = None, config: str | Path | None = None,
                   overrides: dict | None = None) -> Config:
    """Resolve and validate a configuration (ccil.config.resolve); overrides map dotted keys to typed values."""
    return resolve(recipe=recipe, config=config, overrides=overrides)


def _metrics(r: StageResult) -> dict:
    return json.loads(r.out(layout.METRICS).read_text())["values"]


def _stages(results) -> tuple[Stage, ...]:
    return tuple(Stage(r.stage, r.run_dir, r.reused) for r in results)


def _shown(path: Path) -> str:
    """A path as the user should type it: relative to the working directory when it is below it."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def diagnose(config: Config) -> DiagnoseResult:
    """Prepare, fit and diagnose: the data-only decision whether labels are worth generating. No candidates."""
    results = offline.diagnose_offline(config)
    diagnosed = offline.stage(results, "labels.diagnose")
    doc = json.loads(diagnosed.out(layout.DIAGNOSIS).read_text())
    return DiagnoseResult(_stages(results), diagnosed.out(layout.DIAGNOSIS), doc["admitted"], doc["reason"],
                          doc["next_step"], doc["timing"])


def _artifact(path: Path, base: Path) -> dict:
    return {"path": os.path.relpath(path.resolve(), base.resolve()), "sha256": sha256(path)}


def generate(config: Config) -> GenerateResult:
    """Diagnose, then generate the candidates and, with a certificate profile, their quality evidence.

    Writes the generate result that `curate` reads: an immutable generate-<id>.json in run.output_dir, named by its
    content, and generate-result.json beside it, a copy of the latest Generate there.
    """
    results = offline.generate_candidates(config)
    path = Path(config["run.output_dir"]) / GENERATE_RESULT
    base = path.parent
    prepared, fitted, diagnosed, labeled = (offline.stage(results, s) for s in
                                            ("data.prepare", "transition.fit", "labels.diagnose", "labels.generate"))
    certified = offline.stage(results, "labels.certify")
    files = {"dataset": prepared.out(layout.DATASET), "split": prepared.out(layout.SPLIT),
             "transition_model": fitted.out(offline.MODEL_CHECKPOINT), "diagnosis": diagnosed.out(layout.DIAGNOSIS),
             "candidates": labeled.out(layout.CANDIDATES)}
    if certified is not None:
        files["certificates"] = certified.out(layout.CERTIFICATES)
    artifacts = {k: _artifact(p, base) for k, p in files.items() if p.exists()}
    lm, dm = _metrics(labeled), _metrics(diagnosed)
    counts = {"candidates": lm["candidates"], "accepted": lm["accepted"], "rejected_by_reason": lm["rejections_by_reason"],
              "by_classification": _metrics(certified)["by_classification"] if certified is not None else None,
              "admitted": dm["admitted"]}
    shown = _shown(path)
    if dm["admitted"] is None:
        doc = json.loads(diagnosed.out(layout.DIAGNOSIS).read_text())
        next_step = f"nothing admitted ({doc['reason']}): {doc['next_step']}"
    elif certified is not None:
        next_step = f"ccil curate {shown} --set curate.tolerance=<largest acceptable label error>"
    else:
        next_step = f"ccil curate {shown}"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": GENERATE, "version": f"{VERSION[0]}.{VERSION[1]}", "config": config.to_dict(),
           "stages": [{"stage": r.stage, "run_dir": os.path.relpath(r.run_dir.resolve(), base.resolve()),
                       "reused": r.reused} for r in results],
           "artifacts": artifacts, "counts": counts, "next_step": next_step}
    immutable = write_once(doc, base, "generate")  # what a curate result refers to
    write_result(doc, path)
    return GenerateResult(_stages(results), files["diagnosis"], dm["admitted"], files["candidates"],
                          files.get("certificates"), counts, immutable, path, next_step)


def _verified(path: Path, schema: str | None = GENERATE) -> dict:
    """A result whose every recorded artifact still has its recorded hash, or an InputError naming each."""
    if not Path(path).exists():
        raise InputError(f"{path} does not exist: run ccil generate first (it writes run.output_dir/{GENERATE_RESULT})")
    doc = read_result(path)
    if schema is not None and doc["schema"] != schema:
        raise InputError(f"{path} is a {doc['schema']}; curate reads a {schema}")
    problems = []
    for name, a in sorted(doc["artifacts"].items()):
        p = path.parent / a["path"]
        if not p.exists():
            problems.append(f"{name}: {p} is missing")
        elif sha256(p) != a["sha256"]:
            problems.append(f"{name}: {p} changed since Generate wrote it (sha256 {sha256(p)[:12]}, recorded "
                            f"{a['sha256'][:12]})")
    if problems:
        raise InputError(f"{path} cannot be used:\n  " + "\n  ".join(problems))
    return doc


def _immutable(path: Path) -> Path:
    """The immutable generate-<id>.json whose bytes a generate result has (itself, or the one generate-result.json
    copies), or an InputError."""
    text = path.read_bytes()
    immutable = path.parent / immutable_name(text, "generate")
    if not (immutable.is_file() and not immutable.is_symlink() and immutable.read_bytes() == text):
        raise InputError(f"{path} has no immutable copy {immutable.name} beside it with the same content; "
                         "run ccil generate again")
    return immutable


def _generated(doc: dict, base: Path) -> dict:
    """The generate result a curate result records, verified: its recorded hash, then its own artifacts."""
    p = base / doc["generated"]["path"]
    if not p.is_file() or sha256(p) != doc["generated"]["sha256"]:
        raise InputError(f"{p} is missing or changed since Curate recorded it (recorded sha256 "
                         f"{doc['generated']['sha256'][:12]})")
    return _verified(p)


def curate(generated: GenerateResult | str | Path, policy: CurationPolicy | None = None, *,
           overrides: dict | None = None, sets=()) -> CurateResult:
    """Write one selection of a generate result's candidates, after verifying every artifact it records.

    The policy defaults to the recorded configuration's curate.*; overrides and sets may change only curate.*,
    calibration.*, audit.*, generation.evidence_mode and run.overwrite. Writes curate-<id>.json beside the
    generate result.
    """
    path = Path(generated.path if isinstance(generated, GenerateResult) else generated).resolve()
    doc = _verified(path)
    generated_file = _immutable(path)
    keys = list(overrides or {}) + [s.partition("=")[0].strip() for s in sets]
    refused = sorted(k for k in keys if not k.startswith(CURATE_KEYS))
    if refused:
        raise InputError(f"curate changes only {list(CURATE_KEYS)}; {refused} belong to Generate")
    base = path.parent
    generated_stages = [StageResult(s["stage"], (base / s["run_dir"]).resolve(), True) for s in doc["stages"]]
    labeled = offline.stage(generated_stages, "labels.generate")
    recorded = Path(doc["config"]["values"]["run.output_dir"]["value"])
    here = {} if recorded.resolve() == labeled.run_dir.parent else {"run.output_dir": str(labeled.run_dir.parent)}
    cfg = resolve(base=doc["config"], sets=sets, overrides={**(overrides or {}), **here})
    results = offline.curate_candidates(cfg, generated_stages, policy)
    curated = offline.stage(results, "labels.curate")
    calibrated, audited = offline.stage(results, "labels.calibrate"), offline.stage(results, "labels.audit")
    files = {"selection": curated.out(layout.SELECTION)}
    if calibrated is not None:
        files["calibration"] = calibrated.out(layout.CALIBRATION)
    if audited is not None:
        files["audit"] = audited.out(layout.AUDIT)
    m = _metrics(curated)
    counts = {"rows": m["rows"], "selected": m["selected"], "policy": m["policy"], "threshold": m["threshold"],
              "not_selected_by_reason": m["reasons"]}
    next_step = "ccil inspect <this curate result>: the usable rows, and how to load them"
    out = write_once({"schema": CURATE, "version": f"{VERSION[0]}.{VERSION[1]}",
                      "generated": {"path": generated_file.name, "sha256": sha256(generated_file)},
                      "config": cfg.to_dict(),
                      "stages": [{"stage": r.stage, "run_dir": os.path.relpath(r.run_dir.resolve(), base.resolve()),
                                  "reused": r.reused} for r in results],
                      "artifacts": {k: _artifact(p, base) for k, p in files.items()}, "counts": counts,
                      "next_step": next_step}, base, "curate")
    next_step = f"ccil inspect {_shown(out)}: the usable rows, and how to load them"
    return CurateResult(_stages(results), files["selection"], files.get("calibration"), files.get("audit"), counts,
                        out, next_step)


def summary(result: DiagnoseResult | GenerateResult | CurateResult) -> str:
    """The completion summary the CLI prints: artifacts, row counts, rejection and evidence classes, next step."""
    lines = [f"{s.stage:18s} {'reused' if s.reused else 'wrote '} {_shown(s.run_dir)}" for s in result.stages]
    if isinstance(result, DiagnoseResult):
        dt = result.timing.get("dt_ctrl")
        lines += ["", f"diagnosis      {_shown(result.diagnosis)}",
                  f"admitted       {result.admitted if result.admitted is not None else 'nothing'}"
                  + (f" ({result.reason})" if result.reason else ""),
                  f"action period  {f'{dt:g} s' if dt is not None else 'unknown: H is unknown'}"
                  + (f" ({result.timing['note']})" if dt is None and result.timing.get("note") else "")]
        if result.next_step:
            lines.append(f"next           {result.next_step}")
        return "\n".join(lines)
    c = result.counts
    if isinstance(result, GenerateResult):
        rejected = {k: v for k, v in c["rejected_by_reason"].items() if v}
        lines += ["", f"candidates     {_shown(result.candidates)}: {c['candidates']} rows, {c['accepted']} accepted",
                  f"rejected       {', '.join(f'{k} {v}' for k, v in rejected.items()) or 'none'}" if c["candidates"]
                  else "rejected       nothing was generated (see admitted)",
                  "evidence       " + (", ".join(f"{k} {v}" for k, v in c["by_classification"].items() if v) or "none")
                  if c["by_classification"] is not None else "evidence       no certificate.profile: no quality evidence",
                  f"admitted       recovery length {c['admitted']}" if c["admitted"] is not None
                  else "admitted       nothing: an empty synthetic view",
                  f"result         {_shown(result.path)} (latest: {_shown(result.latest)})",
                  f"next           {result.next_step}"]
    else:
        not_sel = [f"{k} {v}" for k, v in c["not_selected_by_reason"].items() if v]
        p = c["policy"]
        if not c["rows"]:
            not_sel = ["(the generate result has no candidates)"]
        elif p["admit"] in ("bound", "score") and p.get("tolerance") is None and p.get("quantile") is None:
            not_sel.append("(no curate.tolerance or curate.quantile: every admissible row is selected)")
        lines += ["", f"selection      {_shown(result.selection)}: {c['selected']} of {c['rows']} rows selected "
                      f"({c['policy']['admit']}" + "".join(f", {k} {c['policy'][k]:g}" for k in ("tolerance", "quantile")
                                                          if c["policy"].get(k) is not None)
                  + (f", threshold {c['threshold']:g}" if c["threshold"] is not None else "") + ")", f"not selected   {', '.join(not_sel) or 'none'}",
                  f"result         {_shown(result.path)}", f"next           {result.next_step}"]
    return "\n".join(lines)


def _usable_rows(doc: dict, gen: dict, base: Path, selection: Path | None) -> dict:
    from .core import IdentityRepresentation
    from .supervision import build_supervision
    art = {k: base / a["path"] for k, a in gen["artifacts"].items()}
    cfg = gen["config"]["values"]
    rep = IdentityRepresentation(cfg["representation.modality"]["value"])
    kwargs = {"split": art["split"]} if "split" in art and cfg["supervision.exclude_validation_episodes"]["value"] else {}
    if selection is not None:
        cur = {k: base / a["path"] for k, a in doc["artifacts"].items()}
        sup = build_supervision(art["dataset"], "demonstrations_plus_selection", representation=rep,
                                candidates=art["candidates"], certificates=art.get("certificates"),
                                selection=selection, calibration=cur.get("calibration"), **kwargs)
    else:
        sup = build_supervision(art["dataset"], "demonstrations_plus_selection", representation=rep,
                                candidates=art["candidates"], **kwargs)
    counts = sup.manifest()["counts"]
    view = "this selection" if selection is not None else "every accepted candidate, uncurated (curate selects)"
    return {"view": view, **{k: counts[k] for k in ("rows", "demonstration_rows", "synthetic_rows") if k in counts}}


def _load_step(gen: dict, base: Path, selection: Path) -> str:
    """The build_supervision call that loads a selection's usable rows, with this run's paths."""
    art = {k: _shown(base / a["path"]) for k, a in gen["artifacts"].items()}
    modality = gen["config"]["values"]["representation.modality"]["value"]
    extra = "".join(f", {k}={art[k]!r}" for k in ("certificates",) if k in art)
    if "split" in art and gen["config"]["values"]["supervision.exclude_validation_episodes"]["value"]:
        extra += f", split={art['split']!r}"
    return ("load the usable rows in Python: from ccil.core import IdentityRepresentation; "
            "from ccil.supervision import build_supervision; "
            f"obs, actions, weights = build_supervision({art['dataset']!r}, 'demonstrations_plus_selection', "
            f"representation=IdentityRepresentation({modality!r}), candidates={art['candidates']!r}{extra}, "
            f"selection={_shown(selection)!r}).arrays()")


def inspect(target: str | Path) -> Inspection:
    """The resolved configuration and its sources, provenance and hashes, quality, rejections, usable rows, and the
    next step of a generate or curate result; of a run directory, its configuration, provenance and metrics."""
    t = Path(target).resolve()
    sections: dict = {}
    if t.is_dir():
        if not (t / layout.MANIFEST).exists():
            raise InputError(f"{t} is not a run directory (no {layout.MANIFEST})")
        manifest = json.loads((t / layout.MANIFEST).read_text())
        config = json.loads((t / layout.CONFIG).read_text())
        sections["configuration"] = {k: v for k, v in config["values"].items() if v["source"] != "default"}
        stage = json.loads((t / layout.STAGE).read_text()).get("stage") if (t / layout.STAGE).exists() else None
        sections["provenance"] = {"run_id": manifest.get("run_id"), "stage": stage,
                                  "inputs": manifest.get("inputs"), "files": manifest.get("files")}
        sections["metrics"] = json.loads((t / layout.METRICS).read_text())["values"] if (t / layout.METRICS).exists() else {}
    else:
        doc = _verified(t, schema=None)
        base = t.parent
        sections["configuration"] = {k: v for k, v in doc["config"]["values"].items() if v["source"] != "default"}
        sections["provenance"] = {"stages": doc["stages"], "artifacts": doc["artifacts"],
                                  **({"generated": doc["generated"]} if doc["schema"] == CURATE else {})}
        gen = doc if doc["schema"] == GENERATE else _generated(doc, base)
        diagnosis = json.loads((base / gen["artifacts"]["diagnosis"]["path"]).read_text())
        quality = {"diagnosis": {"admitted": diagnosis["admitted"], "reason": diagnosis["reason"],
                                 "evidence_class": diagnosis["evidence_class"], "dt_ctrl": diagnosis["timing"]["dt_ctrl"],
                                 "gates": {g: diagnosis["gates"][0][g].get("pass") for g in
                                           ("yield", "displacement", "novelty", "fit", "trust")}}}
        if "certificates" in gen["artifacts"]:
            from .schema import read_certificates
            cert = read_certificates(base / gen["artifacts"]["certificates"]["path"])
            values = np.asarray(cert.total_error_value, float)
            finite = values[np.isfinite(values)]
            quality["certificates"] = {
                "by_classification": {c: int((cert.classification == c).sum()) for c in sorted(set(cert.classification))},
                "total_error_quantiles": {q: float(np.quantile(finite, q)) for q in (0.1, 0.5, 0.9)} if len(finite) else {}}
        sections["quality"] = quality
        sections["rejections"] = {"candidates": gen["counts"]["candidates"], "accepted": gen["counts"]["accepted"],
                                  "by_reason": gen["counts"]["rejected_by_reason"]}
        if doc["schema"] == CURATE:
            sections["selection"] = doc["counts"]
        selection = base / doc["artifacts"]["selection"]["path"] if doc["schema"] == CURATE else None
        sections["usable_rows"] = _usable_rows(doc, gen, base, selection)
        sections["next_step"] = _load_step(gen, base, selection) if selection is not None else doc["next_step"]
    return Inspection(sections, _text(sections))


def _text(sections: dict) -> str:
    out = []
    for name, body in sections.items():
        out.append(name.replace("_", " "))
        if name == "configuration":
            out += [f"  {k} = {json.dumps(v['value'])} ({v['source']})" for k, v in body.items()]
        elif isinstance(body, dict):
            for k, v in body.items():
                shown = json.dumps(v, sort_keys=True, default=str) if not isinstance(v, str) else v
                out.append(f"  {k}: {shown}")
        else:
            out.append(f"  {body}")
    return "\n".join(out)


__all__ = ["Config", "CurateResult", "DiagnoseResult", "GenerateResult", "Inspection", "Stage", "curate", "diagnose",
           "generate", "inspect", "resolve_config", "summary"]
