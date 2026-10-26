"""The code-state gate every #24 run passes first (predeclaration §2).

tree_state(repo) is ok only when the working tree is clean (no modified,
staged or untracked file) and src, pyproject.toml and uv.lock, as checked
out, equal the frozen ccil commit. Uncommitted changes count: the frozen
paths are compared from the working tree, not from HEAD.
"""
from __future__ import annotations

import subprocess

CCIL_FROZEN = "847b06057e8d36a16d13aa2f3b2e375ceb9147e5"
FROZEN_PATHS = ("src", "pyproject.toml", "uv.lock")


def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def tree_state(repo) -> dict:
    differing = _git(repo, "diff", "--name-only", CCIL_FROZEN, "--", *FROZEN_PATHS).split()
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    return {"ok": not differing and not dirty, "head": _git(repo, "rev-parse", "HEAD").strip(),
            "ccil_frozen": CCIL_FROZEN, "differing_from_frozen_ccil": differing, "dirty": dirty}
