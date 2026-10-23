"""Non-executable demonstration datasets and the trusted legacy-pickle boundary.

Normal training and generation read only `ccil-trajectories/1` .npz files,
loaded with allow_pickle=False. Legacy pickle datasets enter only through
convert_legacy_pickle, which refuses to unpickle unless the caller proves the
file is trusted, because unpickling can execute arbitrary code. Structural
validation after loading checks the data, not the file's safety.
"""
import hashlib
import json
import os
import pickle

import numpy as np

from invariants import validate_trajectories

FORMAT = "ccil-trajectories/1"
CONVERTER_VERSION = "1"


class LegacyInputError(ValueError):
    """Legacy input was refused, or is not a valid dataset."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_trajectories(trajectories, path, meta, validate=True):
    """Write episodes as one .npz; atomic, so a failed write keeps any previous file.

    validate=False exists only to build malformed fixtures for tests.
    """
    if validate:
        validate_trajectories(trajectories)
    obs = [np.asarray(ep["observations"]) for ep in trajectories]
    act = [np.asarray(ep["actions"]) for ep in trajectories]
    act = [a[:, None] if a.ndim == 1 else a for a in act]
    meta = {"format": FORMAT, **meta}
    tmp = f"{path}.tmp-{os.getpid()}.npz"
    try:
        np.savez(tmp, observations=np.concatenate(obs), actions=np.concatenate(act),
                 episode_lengths=np.array([len(o) for o in obs], dtype=np.int64),
                 meta=np.array(json.dumps(meta, sort_keys=True)))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_trajectories(path):
    """Load and validate a ccil-trajectories .npz; returns (episodes, meta)."""
    if str(path).endswith(".pkl"):
        raise LegacyInputError(
            f"{path} is a legacy pickle; normal paths do not unpickle data because it can execute code. "
            f"Convert it once with: python correct_il/convert_legacy_dataset.py --help")
    with np.load(path, allow_pickle=False) as z:
        missing = {"observations", "actions", "episode_lengths", "meta"} - set(z.files)
        if missing:
            raise LegacyInputError(f"{path}: not a {FORMAT} file (missing {sorted(missing)})")
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != FORMAT:
            raise LegacyInputError(f"{path}: unsupported dataset format {meta.get('format')!r}, expected {FORMAT!r}")
        obs, act, lengths = z["observations"], z["actions"], z["episode_lengths"]
    if lengths.ndim != 1 or (lengths < 0).any() or lengths.sum() != len(obs) or len(obs) != len(act):
        raise LegacyInputError(f"{path}: episode_lengths {lengths.tolist()[:10]} do not match "
                               f"{len(obs)} observations and {len(act)} actions")
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    episodes = [{"observations": obs[b:e], "actions": act[b:e]} for b, e in zip(bounds[:-1], bounds[1:])]
    validate_trajectories(episodes)
    return episodes, meta


def convert_legacy_pickle(src, dst, expect_sha256=None, trust=False):
    """Convert a legacy pickle dataset after an explicit trust decision.

    Trust is established by expect_sha256 matching the file (checked before any
    unpickling) or by trust=True, the caller's acknowledgement that the file is
    trusted. The loaded structure is validated before anything is written.
    """
    digest = sha256_file(src)
    if expect_sha256 is not None:
        if digest != expect_sha256:
            raise LegacyInputError(f"{src}: sha256 {digest} does not match the expected {expect_sha256}; "
                                   f"refusing to unpickle it")
    elif not trust:
        raise LegacyInputError(f"refusing to unpickle {src}: pickle can execute arbitrary code. Pass an expected "
                               f"sha256 or explicitly acknowledge that the file is trusted")
    with open(src, "rb") as f:
        data = pickle.load(f)
    validate_trajectories(data)
    write_trajectories(data, dst, {"converter_version": CONVERTER_VERSION,
                                   "source": {"name": os.path.basename(src), "sha256": digest,
                                              "bytes": os.path.getsize(src)},
                                   "trust": "sha256" if expect_sha256 is not None else "acknowledged"})
    return digest
