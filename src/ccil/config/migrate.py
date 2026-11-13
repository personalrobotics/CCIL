"""Read every supported configuration version with its original meaning.

Migration renames retained keys, preserves changed defaults explicitly, and
records dropped or moved settings. Unsupported behavior fails with guidance.
See ``docs/config/README.md`` for the version history.
"""
from __future__ import annotations

from .params import REGISTRY

CONFIG_VERSION = 4
# Defaults that changed with a version: an older file that does not set the key keeps the older meaning.
CHANGED_DEFAULTS = {2: {"transition_model.lipschitz_coordinates": ("raw", "#125: version 2's default is "
                                                                          "model_normalized")},
                    3: {"generation.recovery_steps": ("1", "#167: version 3's default auto admits a recovery length "
                                                           "only through the data-only gates")}}

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
    "data.data_frac": "trajectory subsampling; superseded by M7 low-data subsets (#51-#55)",
    "state_encoder": "image-observation encoder from the fork image branch; deferred to M5",
    "policy.layers": "policy encoder width from label-filtering; ported with the policy stage in M3",
}
MOVED_TO = "policy training and evaluation moved to il-supervision-benchmarks (#168)"
# Legacy keys whose canonical keys moved with the policy harness (#168): dropped with a note.
MOVED_LEGACY = {"env": MOVED_TO, "eval.noise": MOVED_TO, "policy.lr": MOVED_TO, "policy.batch_size": MOVED_TO,
                "policy.train_epochs": MOVED_TO, "aug.balance_data": MOVED_TO}
# Version 1 to 3 keys kept under a new name (#168), and the sections dropped with the policy harness.
MOVED_V4 = {"policy.method": "supervision.source", "policy.demonstration_weight": "supervision.demonstration_weight",
            "policy.exclude_validation_episodes": "supervision.exclude_validation_episodes",
            "policy.trainer": "integrated.trainer"}
REMOVED_V4 = ("policy", "evaluation", "environment")
SOURCE_V4 = {"bc": "demonstrations", "ccil": "demonstrations_plus_selection"}
NOISE_BC = ("NoiseBC moved to il-supervision-benchmarks (#168): train it with ilsb policy train (supervision.source "
            "demonstrations, policy.noise); CCIL's supervision for it is supervision.source demonstrations")
# Keys withdrawn in favour of the certificate layer (#88) and of curation (#167).
WITHDRAWN = {
    "generation.max_label_error": "label-error filtering moved to certificates and curation: set certificate.profile "
                                  "(legacy_label_filtering for the branch's score) and curate.tolerance",
    "generation.label_error_quantile": "label-error filtering moved to certificates and curation: set "
                                       "certificate.profile and curate.quantile",
}
# Version 1 and 2 selection keys, read as their curate keys (#167).
MOVED_V3 = {"certificate.max_total_error": "curate.tolerance", "certificate.score_threshold": "curate.tolerance",
            "certificate.selection_quantile": "curate.quantile", "certificate.selection_population": "curate.population"}
WITHDRAWN.update({old: f"selection moved out of the certificate (#167): set {new}" for old, new in MOVED_V3.items()})
RENAMED = {legacy: p.key for p in REGISTRY.values() for legacy in p.legacy
           if legacy not in ("no_gpu", "dynamics.backward", "policy.naive", "policy.noise_bc", "data.pkl")}


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
        elif key in MOVED_LEGACY:
            notes.append(f"dropped {key}: {MOVED_LEGACY[key]}")
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
        elif key in ("policy.naive", "policy.noise_bc"):
            continue  # combined below
        elif key in RENAMED:
            values[RENAMED[key]] = v
            notes.append(f"{key} -> {RENAMED[key]}")
        else:
            errors.append(f"unknown legacy key {key!r}")
    if "aug.max_label_err" in flat or "aug.label_err_quantile" in flat:
        values["certificate.profile"] = "legacy_label_filtering"
        notes.append("aug.max_label_err / aug.label_err_quantile -> certificate.profile legacy_label_filtering "
                     "(the branch's score, empirical)")
    if "aug.label_err_quantile" in flat:
        values["curate.population"] = "produced"
        notes.append("aug.label_err_quantile -> curate.population produced (the branch took the quantile over every "
                     "generated label, independently of aug.max_label_err)")
    if "policy.naive" in flat or "policy.noise_bc" in flat:
        naive, noise = flat.get("policy.naive", False), flat.get("policy.noise_bc", 0.0)
        if naive is True and noise:
            errors.append(f"policy.naive with policy.noise_bc={noise!r}: {NOISE_BC}")
        elif naive is True:
            values["supervision.source"] = "demonstrations"
        elif naive is False and not noise:
            values["supervision.source"] = "demonstrations_plus_selection"
        else:
            errors.append(f"policy.naive={naive!r} with policy.noise_bc={noise!r} has no supported equivalent")
        notes.append("policy.naive + policy.noise_bc -> supervision.source")
    return values, notes, errors


def _move_selection(values: dict, version: int) -> tuple[dict, list[str], list[str]]:
    """A version 1 or 2 layer's selection keys, read as their curate keys (#167)."""
    out, notes, errors = {}, [], []
    for key, v in values.items():
        new = MOVED_V3.get(key)
        if new is None:
            out[key] = v
        elif new in out:
            errors.append(f"{key} and another key both set {new}")
        else:
            out[new] = v
            notes.append(f"{key} -> {new} (config_version {version}; #167 moved selection out of the certificate)")
    if values.get("certificate.reference_dynamics") is not None and "generation.evidence_mode" not in values:
        out["generation.evidence_mode"] = "reference_assisted"
        notes.append(f"generation.evidence_mode = 'reference_assisted': the meaning of config_version {version}, whose "
                     "certificate.reference_dynamics shaped the selection")
    return out, notes, errors


def _move_policy(values: dict, version: int) -> tuple[dict, list[str], list[str]]:
    """A version 1 to 3 layer's policy, evaluation and environment keys: renamed, or dropped (#168)."""
    out, notes, errors = {}, [], []
    for key, v in values.items():
        new = MOVED_V4.get(key)
        if new is None and key.split(".")[0] in REMOVED_V4:
            notes.append(f"dropped {key} (config_version {version}): {MOVED_TO}")
            continue
        if new is None:
            out[key] = v
            continue
        if key == "policy.method":
            if v == "noise_bc":
                errors.append(f"policy.method noise_bc (config_version {version}): {NOISE_BC}")
                continue
            v = SOURCE_V4.get(v, v)
        if new in values or new in out:
            errors.append(f"{key} and {new} both set {new}")
            continue
        out[new] = v
        notes.append(f"{key} -> {new} (config_version {version}; #168)")
    return out, notes, errors


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
        notes = ["read as legacy correct_il configuration (config_version 0)"] + notes
    elif version in range(1, CONFIG_VERSION + 1):
        values, notes, errors = flatten({k: v for k, v in raw.items() if k != "config_version"}), [], []
        if version < 3:
            values, notes, errors = _move_selection(values, version)
        if version < 4:
            values, more_notes, more_errors = _move_policy(values, version)
            notes, errors = notes + more_notes, errors + more_errors
    else:
        return {}, [], [f"config_version {version!r} is not supported (this version reads 0 to {CONFIG_VERSION})"]
    for changed, keys in CHANGED_DEFAULTS.items():
        if version < changed:
            for key, (old, why) in keys.items():
                if key not in values:
                    values[key] = old
                    notes.append(f"{key} = {old!r}: the meaning of config_version {version} ({why})")
    return values, notes, errors
