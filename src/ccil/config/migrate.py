"""Versioned configuration migration.

Version 1 files declare `config_version: 1`. A file without it is read as
version 0, the legacy correct_il YAML, and migrated: renamed keys move to their
canonical key; keys that never affected behavior are dropped with a note; keys
that did affect behavior but have no supported home yet fail with guidance.
"""
from __future__ import annotations

from .params import REGISTRY

CONFIG_VERSION = 1

# Legacy keys dropped because they never affected behavior.
DROPPED = {
    "output.folder_suffix": "run directories are named by run ID",
    "data.expert": "never read by the legacy code",
    "data.num_traj": "derived placeholder ('...') filled in at load time",
    "data.horizon": "derived placeholder ('...') filled in at load time",
    "data.num_datapoints": "derived placeholder ('...') filled in at load time",
    "dynamics.val_percentage": "never read by the legacy code (#13)",
    "aug.use_expert_label": "never read by the legacy code",
    "debug": "development aid that truncated label generation to five transitions",
}
# Legacy keys that affected behavior but have no supported home yet.
UNSUPPORTED = {
    "aug.chunk_size": "action-chunk labels from label-filtering; ported in #22",
    "data.data_frac": "trajectory subsampling; superseded by M7 low-data subsets (#51-#55)",
    "state_encoder": "image-observation encoder from the fork image branch; deferred to M5",
    "policy.layers": "policy encoder width from label-filtering; ported with the policy stage in M3",
}
RENAMED = {legacy: p.key for p in REGISTRY.values() for legacy in p.legacy
           if legacy not in ("no_gpu", "dynamics.backward", "policy.naive", "policy.noise_bc", "data.pkl", "eval.noise")}


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested mapping -> dotted keys; ValueError if two spellings name one key (run: {seed} and run.seed)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        for name, value in (flatten(v, key + ".") if isinstance(v, dict) else {key: v}).items():
            if name in out:
                raise ValueError(f"key {name!r} is given more than once")
            out[name] = value
    return out


def migrate_v0(raw: dict) -> tuple[dict, list[str], list[str]]:
    """Legacy correct_il YAML -> (canonical flat values, notes, errors)."""
    flat = flatten(raw)
    values, notes, errors = {}, [], []
    for key, v in flat.items():
        top = key.split(".")[0]
        if key in DROPPED:
            notes.append(f"dropped {key}: {DROPPED[key]}")
        elif key in UNSUPPORTED or top in UNSUPPORTED:
            errors.append(f"{key} has no supported equivalent: {UNSUPPORTED.get(key) or UNSUPPORTED[top]}")
        elif key == "no_gpu":
            values["compute.device"] = "cpu" if v is True else "cuda" if v is False else v
            notes.append("no_gpu -> compute.device")
        elif key == "dynamics.backward":
            values["transition_model.direction"] = "backward" if v is True else "forward" if v is False else v
            notes.append("dynamics.backward -> transition_model.direction")
        elif key == "data.pkl":
            values["data.path"] = v[:-4] + ".npz" if isinstance(v, str) and v.endswith(".pkl") else v
            notes.append("data.pkl -> data.path (.pkl replaced by the converted .npz; see convert_legacy_dataset.py)")
        elif key == "eval.noise":
            values["evaluation.noise_levels"] = v if v else [0.0]
            notes.append("eval.noise -> evaluation.noise_levels")
        elif key in ("policy.naive", "policy.noise_bc"):
            continue  # combined below
        elif key in RENAMED:
            values[RENAMED[key]] = v
            notes.append(f"{key} -> {RENAMED[key]}")
        else:
            errors.append(f"unknown legacy key {key!r}")
    if "policy.naive" in flat or "policy.noise_bc" in flat:
        naive, noise = flat.get("policy.naive", False), flat.get("policy.noise_bc", 0.0)
        if naive is True and noise:
            values["policy.method"], values["policy.noise_scale"] = "noise_bc", noise
        elif naive is True:
            values["policy.method"] = "bc"
        elif naive is False and not noise:
            values["policy.method"] = "ccil"
        else:
            errors.append(f"policy.naive={naive!r} with policy.noise_bc={noise!r} has no supported equivalent")
        notes.append("policy.naive + policy.noise_bc -> policy.method (+ policy.noise_scale)")
    return values, notes, errors


def read_layer(raw: dict) -> tuple[dict, list[str], list[str]]:
    """A parsed YAML layer -> (canonical flat values, notes, errors), migrating when needed."""
    if not isinstance(raw, dict):
        return {}, [], [f"configuration must be a mapping, got {type(raw).__name__}"]
    try:
        return _read_layer(raw)
    except ValueError as e:
        return {}, [], [str(e)]


def _read_layer(raw: dict) -> tuple[dict, list[str], list[str]]:
    version = raw.get("config_version", 0)
    if version == 0:
        values, notes, errors = migrate_v0(raw)
        return values, ["read as legacy correct_il configuration (config_version 0)"] + notes, errors
    if version != CONFIG_VERSION:
        return {}, [], [f"config_version {version!r} is not supported (this version reads 0 and {CONFIG_VERSION})"]
    flat = flatten({k: v for k, v in raw.items() if k != "config_version"})
    return flat, [], []
