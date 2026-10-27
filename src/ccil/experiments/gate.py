"""Code-state gates for predeclared studies (#112), generalized from the #24 tooling (tools/m3_pendulum/provenance.py).

A Freeze names what a study may run:
- the frozen ccil commit and the paths it covers (src, pyproject.toml, uv.lock by default);
- the study's tooling paths, which must equal the commit named in the freeze manifest;
- the manifest path, added by a documentation-only freeze pull request (it names its own tooling commit, so
  it is excluded from the comparison).

tree_state is ok only on a clean tree (no modified, staged or untracked file) whose frozen ccil paths, as
checked out, equal the frozen commit; uncommitted changes count. gate adds the tooling check, required for real
runs and recorded for smoke runs, which precede the freeze.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Freeze:
    ccil_commit: str
    tooling_paths: tuple[str, ...]
    manifest: str
    ccil_paths: tuple[str, ...] = ("src", "pyproject.toml", "uv.lock")


class GateRefused(SystemExit):
    """A study command refused to run because the code state is not the declared one."""


def _git(repo, *args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def tree_state(repo, freeze: Freeze) -> dict:
    differing = _git(repo, "diff", "--name-only", freeze.ccil_commit, "--", *freeze.ccil_paths).stdout.split()
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    return {"ok": not differing and not dirty, "head": _git(repo, "rev-parse", "HEAD").stdout.strip(),
            "ccil_frozen": freeze.ccil_commit, "differing_from_frozen_ccil": differing, "dirty": dirty}


def tooling_state(repo, freeze: Freeze) -> dict:
    """Whether the tooling paths equal the commit the freeze manifest names."""
    path = Path(repo) / freeze.manifest
    if not path.exists():
        return {"ok": False, "problem": f"no freeze manifest ({freeze.manifest}); the tooling is not frozen"}
    commit = json.loads(path.read_text()).get("tooling_commit", "")
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode != 0:
        return {"ok": False, "tooling_commit": commit, "problem": "the freeze manifest names no commit of this repository"}
    differing = _git(repo, "diff", "--name-only", commit, "--", *freeze.tooling_paths,
                     f":(exclude){freeze.manifest}").stdout.split()
    return {"ok": not differing, "tooling_commit": commit, "differing_from_frozen_tooling": differing,
            **({"problem": "the tooling differs from the frozen commit"} if differing else {})}


def gate(repo, freeze: Freeze, real: bool) -> dict:
    """tree_state plus tooling_state; ok requires the frozen tooling only for real runs."""
    state, tooling = tree_state(repo, freeze), tooling_state(repo, freeze)
    return {**state, "ok": state["ok"] and (tooling["ok"] or not real), "real": real, "tooling": tooling}


def require(repo, freeze: Freeze, real: bool) -> dict:
    state = gate(repo, freeze, real)
    if not state["ok"]:
        raise GateRefused(f"code-state gate refused: {json.dumps(state, sort_keys=True)}")
    return state
