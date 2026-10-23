"""Configuration resolution with explicit precedence and per-value provenance.

Precedence, lowest first: package defaults < recipe < user file < command-line
--set < Python overrides. Every input is validated before anything is built.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .migrate import read_layer
from .params import REGISTRY, RULES
from .spec import ConfigError, coerce, parse_text

RECIPES = Path(__file__).resolve().parent.parent / "recipes"
SCHEMA, VERSION = "ccil.config", "1"


@dataclass
class Resolved:
    values: dict
    sources: dict
    notes: list = field(default_factory=list)

    def __getitem__(self, key):
        return self.values[key]

    def config_hash(self) -> str:
        blob = json.dumps(self.values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "version": VERSION, "config_hash": self.config_hash(),
            "values": {k: {"value": self.values[k], "source": self.sources[k], "units": REGISTRY[k].units,
                           "status": REGISTRY[k].status} for k in sorted(self.values)},
            "notes": self.notes,
        }


def _load_yaml(path: Path, errors: list):
    """Parsed YAML, or None after recording why it could not be used."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        errors.append(f"cannot read {path}: {e}")
        return None
    if raw is None:
        errors.append(f"{path} is empty")
    return raw


def resolve(recipe: str | None = None, config: str | Path | None = None, sets=(), overrides: dict | None = None) -> Resolved:
    """Resolve and validate a configuration; raise ConfigError listing every problem."""
    values = {k: p.default for k, p in REGISTRY.items()}
    sources = {k: "default" for k in REGISTRY}
    errors, notes = [], []

    def apply(flat: dict, source: str, from_text: bool = False):
        for key, raw in flat.items():
            p = REGISTRY.get(key)
            if p is None:
                close = difflib.get_close_matches(key, REGISTRY, n=1)
                errors.append(f"unknown key {key!r} ({source})" + (f"; did you mean {close[0]!r}?" if close else ""))
                continue
            try:
                value = parse_text(p, raw) if from_text else raw
                values[key], sources[key] = coerce(p, value), source
            except ValueError as e:
                errors.append(f"{e} ({source})")

    layers = []
    if recipe is not None:
        path = RECIPES / f"{recipe}.yaml"
        if not path.exists():
            known = sorted(p.stem for p in RECIPES.glob("*.yaml"))
            errors.append(f"unknown recipe {recipe!r}; available: {known}")
        else:
            layers.append((path, f"recipe:{recipe}"))
    if config is not None:
        layers.append((Path(config), f"file:{config}"))
    for path, source in layers:
        raw = _load_yaml(path, errors)
        if raw is None:
            continue
        flat, layer_notes, layer_errors = read_layer(raw)
        notes += [f"{source}: {n}" for n in layer_notes]
        errors += [f"{e} ({source})" for e in layer_errors]
        apply(flat, source)

    cli = {}
    for item in sets:
        key, sep, text = item.partition("=")
        if not sep:
            errors.append(f"--set expects key=value, got {item!r}")
        else:
            cli[key.strip()] = text
    apply(cli, "cli", from_text=True)
    apply(overrides or {}, "python")

    if not errors:
        for name, keys, check in RULES:
            problem = check(values)
            if problem:
                errors.append(f"{problem} [{name}; sources: " + ", ".join(f"{k}={sources[k]}" for k in keys) + "]")
    if errors:
        raise ConfigError(errors)
    return Resolved(values, sources, notes)
