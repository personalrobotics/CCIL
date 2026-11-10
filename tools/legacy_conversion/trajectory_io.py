"""Non-executable demonstration datasets and the trusted legacy-pickle boundary.

ccil never unpickles. Legacy pickle datasets enter only through
convert_legacy_pickle, which refuses to unpickle unless the caller proves the
file is trusted, because unpickling can execute arbitrary code, and writes the
M1 `ccil-trajectories/1` .npz that `ccil data convert` reads. Structural
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


def write_trajectories(trajectories, path, meta):
    """Write episodes as one .npz; atomic, so a failed write keeps any previous file."""
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
