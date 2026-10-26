"""Explicit, one-way conversion of a legacy correct_il YAML into a canonical configuration (#23).

    ccil config convert LEGACY.yml --out CANON.yml --manifest MANIFEST.json [--fidelity corrected|historical]

resolve() already reads a legacy file by migrating it on the fly
(ccil.config.migrate); this module writes the result as an artifact that can
be inspected and versioned before any model is trained, with a manifest that
records what happened to every legacy key:

- renamed: the value moves unchanged to its canonical key;
- derived: the canonical value is computed from one or more legacy keys
  (policy.naive + policy.noise_bc -> policy.method; no_gpu -> compute.device);
- dropped: the key never affected behavior (with the reason);
- unsupported or unknown: conversion fails.

Every explicitly supplied value that affected behavior must appear unchanged
in the resolved canonical configuration, or conversion fails: a historical
value is never replaced by a new default.

Legacy behavior that no legacy config spelled out, and whose canonical default
differs, is listed in LEGACY_IMPLICIT. With fidelity "historical" the legacy
behavior is written explicitly; with "corrected" (the default) the canonical
default stands. Either way the manifest records both values. Corrections that
are not configuration (#21, #22) are listed as such: they belong to the
faithful-port arm of #24, not to a flag.

Unknown, duplicate, ambiguous, or incompatible settings fail with every
diagnostic, and nothing is written. The canonical file and the manifest are
written atomically, after the canonical file resolves.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from .migrate import CONFIG_VERSION, DROPPED, RENAMED, UNSUPPORTED, flatten, migrate_v0
from .params import REGISTRY
from .resolve import _load_yaml, resolve
from .spec import ConfigError, coerce

CONVERTER = "ccil.config.convert/1"
FIDELITIES = ("corrected", "historical")

# Canonical keys absent from every legacy config whose default differs from what the legacy code did.
LEGACY_IMPLICIT = {
    "policy.exclude_validation_episodes": {
        "legacy": False,
        "why": "legacy trained the BC policy on every expert transition (utils.load_demo_for_policy); its validation "
               "split (dynamics.validation_fraction, #13) only evaluated the dynamics model. The canonical harness "
               "holds the split's validation episodes out of policy training.",
    },
    "policy.noise_model": {
        "legacy": "legacy_uniform_half_batch",
        "why": "legacy CustomBC added noise_bc * U[0, 1), always positive; the canonical default is the zero-centered "
               "symmetric_uniform_half_batch (#22).",
    },
}
# Behavior changes that are not configuration; the #24 faithful-port arm (commit 50cfc535) isolates them.
NON_CONFIGURABLE = [
    "#21: the dynamics model predicts in eval mode (legacy left spectral-norm power iterations running)",
    "#21: soft_sampling uses the residual model's finite-difference ratio in raw units",
    "#21: soft_jac constrains the residual Jacobian df/d(s, a), not d(s + f)",
    "#21: soft_sampling_slack pairs each sample's slack with its own perturbations",
    "#22: NoiseBC draws are counter-keyed per (seed, step, sample, draw) and never mutate the batch",
    "#22: action chunks align to the frame each label reaches (legacy's chunk_size condition was inverted)",
]
# Legacy entry points: each took the configuration path as its only positional argument.
LEGACY_SCRIPTS = {
    "correct_il/train_dynamics_model.py CONFIG": "ccil labels generate --config CANON (stage transition.fit)",
    "correct_il/gen_aug_label.py CONFIG": "ccil labels generate --config CANON (stages labels.generate, labels.certify)",
    "correct_il/train_bc_policy.py CONFIG": "ccil policy train --config CANON (experimental harness, #89)",
    "correct_il/eval_bc_policy.py CONFIG": "simulator evaluation: #24 and M7",
}
# Legacy keys handled by migrate_v0 outside RENAMED, with the canonical keys they produce.
DERIVED = {
    "no_gpu": ["compute.device"], "dynamics.backward": ["transition_model.direction"], "data.pkl": ["data.path"],
    "eval.noise": ["evaluation.noise_levels"], "policy.naive": ["policy.method", "policy.noise_scale"],
    "policy.noise_bc": ["policy.method", "policy.noise_scale"],
}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nest(flat: dict) -> dict:
    out: dict = {}
    for key, value in sorted(flat.items()):
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _dispositions(flat: dict, values: dict, dataset_map: dict) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    for key, value in sorted(flat.items()):
        top = key.split(".")[0]
        if key in DROPPED:
            rows.append({"legacy": key, "value": value, "disposition": "dropped", "reason": DROPPED[key]})
        elif key in UNSUPPORTED or top in UNSUPPORTED:
            rows.append({"legacy": key, "value": value, "disposition": "unsupported",
                         "reason": UNSUPPORTED.get(key) or UNSUPPORTED[top]})
        elif key == "data.pkl":
            if not (isinstance(value, str) and (value.endswith(".pkl") or value in dataset_map)):
                errors.append(f"data.pkl {value!r} is ambiguous: it is not a .pkl path and --dataset-map gives no "
                              "converted dataset for it")
            rows.append({"legacy": key, "value": value, "disposition": "derived", "canonical": ["data.path"],
                         "result": {"data.path": values.get("data.path")},
                         "reason": "the pickle is converted to a ccil.trajectories file first (ccil data convert); "
                                   "data.path names that file"})
        elif key in DERIVED:
            produced = {k: values[k] for k in DERIVED[key] if k in values}
            rows.append({"legacy": key, "value": value, "disposition": "derived", "canonical": DERIVED[key],
                         "result": produced})
        elif key in RENAMED:
            rows.append({"legacy": key, "value": value, "disposition": "renamed", "canonical": RENAMED[key]})
        else:
            rows.append({"legacy": key, "value": value, "disposition": "unknown"})
    return rows, errors


def convert_config(src, out, manifest, fidelity: str = "corrected", dataset_map: dict | None = None) -> dict:
    """Convert src; write out and manifest atomically; return the manifest. ConfigError lists every problem."""
    if fidelity not in FIDELITIES:
        raise ConfigError([f"fidelity must be one of {list(FIDELITIES)}, got {fidelity!r}"])
    src, out, manifest = Path(src), Path(out), Path(manifest)
    dataset_map = dict(dataset_map or {})
    errors: list[str] = []
    raw = _load_yaml(src, errors)
    if errors:
        raise ConfigError(errors)
    if not isinstance(raw, dict):
        raise ConfigError([f"{src} must be a mapping"])
    if raw.get("config_version", 0) != 0:
        raise ConfigError([f"{src} declares config_version {raw.get('config_version')!r}; only legacy (version 0) "
                           "configurations are converted"])
    try:
        flat = flatten(raw)
    except ValueError as e:
        raise ConfigError([str(e)]) from None
    values, notes, migrate_errors = migrate_v0(raw)
    errors += migrate_errors
    if "data.pkl" in flat and flat["data.pkl"] in dataset_map:
        values["data.path"] = dataset_map[flat["data.pkl"]]
    rows, row_errors = _dispositions(flat, values, dataset_map)
    errors += row_errors
    errors += [f"{r['legacy']} has no supported equivalent: {r['reason']}" for r in rows if r["disposition"] == "unsupported"]
    errors += [f"unknown legacy key {r['legacy']!r}" for r in rows if r["disposition"] == "unknown"]
    for r in rows:  # name the legacy key, not the canonical one, when a supplied value has the wrong type or domain
        if r["disposition"] == "renamed":
            try:
                coerce(REGISTRY[r["canonical"]], r["value"])
            except ValueError as e:
                errors.append(f"{r['legacy']} (-> {r['canonical']}): {e}")
    if errors:
        raise ConfigError(sorted(set(errors)))

    implicit = []
    for key, spec in LEGACY_IMPLICIT.items():
        if key in values:
            continue  # the legacy file named it explicitly
        written = spec["legacy"] if fidelity == "historical" else REGISTRY[key].default
        if fidelity == "historical":
            values[key] = spec["legacy"]
        implicit.append({"key": key, "legacy_behavior": spec["legacy"], "canonical_default": REGISTRY[key].default,
                         "written": written, "why": spec["why"]})

    header = (f"# Converted by {CONVERTER} from {src.name} (sha256 {_sha(src)}), fidelity {fidelity}.\n"
              f"# Do not edit by hand; regenerate with: ccil config convert {src.name} --fidelity {fidelity}\n")
    text = header + yaml.safe_dump({"config_version": CONFIG_VERSION, **_nest(values)}, sort_keys=True)

    # Validate before writing anything: the canonical text must resolve, and keep every supplied value.
    out.parent.mkdir(parents=True, exist_ok=True)
    probe = out.with_name(f".{out.name}.probe-{os.getpid()}.yaml")
    try:
        probe.write_text(text)
        resolved = resolve(config=probe)
    finally:
        if probe.exists():
            probe.unlink()
    changed = []
    for r in rows:
        if r["disposition"] != "renamed":
            continue
        p = REGISTRY[r["canonical"]]
        try:
            supplied = coerce(p, r["value"])
        except ValueError as e:
            changed.append(f"{r['legacy']}: {e}")
            continue
        if resolved[p.key] != supplied:
            changed.append(f"{r['legacy']}={r['value']!r} became {p.key}={resolved[p.key]!r}")
        r["resolved"] = resolved[p.key]
    for r in rows:
        if r["disposition"] == "derived":
            r["resolved"] = {k: resolved[k] for k in r["canonical"] if k in resolved.values}
    if changed:
        raise ConfigError([f"a supplied historical value would change: {c}" for c in changed])

    record = {
        "converter": CONVERTER, "fidelity": fidelity,
        "source": {"path": src.name, "sha256": _sha(src)},
        "output": {"path": out.name, "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "dispositions": rows,
        "counts": {d: sum(r["disposition"] == d for r in rows) for d in ("renamed", "derived", "dropped")},
        "implicit_legacy_behavior": implicit,
        "non_configurable_corrections": NON_CONFIGURABLE,
        "dataset": {"legacy_key": "data.pkl" if "data.pkl" in flat else "data.path",
                    "legacy": flat.get("data.pkl", flat.get("data.path")), "canonical": resolved["data.path"],
                    "conversion": "correct_il/convert_legacy_dataset.py (trusted pickle -> M1), then ccil data convert"},
        "invocation": LEGACY_SCRIPTS,
        "resolved": dict(sorted(resolved.values.items())), "config_hash": resolved.config_hash(),
        "notes": notes,
    }
    _atomic_write(out, text)
    _atomic_write(manifest, json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record
