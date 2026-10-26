"""The code-state gate every #24 run passes first (predeclaration §2).

tree_state(repo) is ok only when the working tree is clean (no modified,
staged or untracked file) and src, pyproject.toml and uv.lock, as checked
out, equal the frozen ccil commit. Uncommitted changes count: the frozen
paths are compared from the working tree, not from HEAD.

gate(repo, real) adds the frozen tooling (§11). The freeze manifest,
tools/m3_pendulum/freeze.json, is added by the documentation-only freeze
pull request and names the tooling commit. Every path the study executes or
reads (TOOLING_PATHS) must equal that commit, apart from the manifest itself.
Real tuning, choice, study and analysis commands require it; smoke runs,
which precede the freeze, record it without requiring it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

CCIL_FROZEN = "847b06057e8d36a16d13aa2f3b2e375ceb9147e5"
FROZEN_PATHS = ("src", "pyproject.toml", "uv.lock")
FREEZE = "tools/m3_pendulum/freeze.json"
TOOLING_PATHS = ("tools/m3_pendulum", "tools/legacy_conversion", "correct_il", "data", "config",
                 "docs/artifacts/manifest.json", "docs/fixtures/evidence/legacy-conversion/artifacts/conversion.json")


def _git(repo, *args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def tree_state(repo) -> dict:
    differing = _git(repo, "diff", "--name-only", CCIL_FROZEN, "--", *FROZEN_PATHS).stdout.split()
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    return {"ok": not differing and not dirty, "head": _git(repo, "rev-parse", "HEAD").stdout.strip(),
            "ccil_frozen": CCIL_FROZEN, "differing_from_frozen_ccil": differing, "dirty": dirty}


def tooling_state(repo) -> dict:
    """Whether the tooling paths equal the commit the freeze manifest names."""
    path = Path(repo) / FREEZE
    if not path.exists():
        return {"ok": False, "problem": f"no freeze manifest ({FREEZE}); the tooling is not frozen"}
    commit = json.loads(path.read_text()).get("tooling_commit", "")
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode != 0:
        return {"ok": False, "tooling_commit": commit, "problem": "the freeze manifest names no commit of this repository"}
    differing = _git(repo, "diff", "--name-only", commit, "--", *TOOLING_PATHS, f":(exclude){FREEZE}").stdout.split()
    return {"ok": not differing, "tooling_commit": commit, "differing_from_frozen_tooling": differing,
            **({"problem": "the tooling differs from the frozen commit"} if differing else {})}


def gate(repo, real: bool) -> dict:
    """tree_state plus tooling_state; ok requires the frozen tooling only for real runs."""
    state, tooling = tree_state(repo), tooling_state(repo)
    return {**state, "ok": state["ok"] and (tooling["ok"] or not real), "real": real, "tooling": tooling}


def require(repo, real: bool) -> dict:
    state = gate(repo, real)
    if not state["ok"]:
        raise SystemExit(f"code-state gate refused: {json.dumps(state, sort_keys=True)}")
    return state
