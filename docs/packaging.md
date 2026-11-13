# Packaging

The `prlccil` distribution is a uv-managed project (#20). It installs the
`ccil` Python package and `ccil` console script.

| Item | Value |
|---|---|
| Definition | [`pyproject.toml`](../pyproject.toml), with build backend `uv_build`, distribution name `prlccil`, module name `ccil`, a src layout, and the `ccil` console script |
| Lock | [`uv.lock`](../uv.lock), committed. `.python-version` selects 3.11 for development. |
| Python | 3.10 and newer for `prlccil`; 3.11 and newer for `prlccil[torch]` and the `evidence` group (see [`security.md`](security.md)) |
| Core dependencies | `numpy>=1.24`, `pyyaml>=6.0`. Nothing else installs with the core. |
| Dependency groups | `evidence`: exact pins of matplotlib, pillow and torch, Python 3.11+, used only by the evidence tools |
| Optional extras | `torch`: `torch>=2.9.1` on Python 3.11+, for torch-backed models (#21). `lerobot`: `pyarrow>=21.0.0`, and `huggingface_hub>=1.0.0` on Python 3.11+, CCIL's LeRobot Dataset v3.0 format bridge. It does not install the LeRobot runtime ([`lerobot.md`](lerobot.md), #26). Image models (M5) will add their own. A missing extra fails only when its feature is requested, with install guidance. |

```bash
uv sync                              # development environment from the lock
uv run python tools/dependency_gate/check.py --out <dir>   # dependency gate; see security.md
uv run ccil --help
uv build                             # sdist and wheel into dist/
uv run --group evidence python tools/m2_schemas/check.py --out <dir>
```

## Release protocol

CCIL follows the same two-step release protocol as SSIK. The version in
`pyproject.toml` and the tag must agree exactly; the release workflow refuses
any mismatch. Release tags are `vX.Y.ZrcN` or `vX.Y.Z`.

1. Set the release-candidate version with `uv version X.Y.ZrcN`, merge that
   change, and run `scripts/release-precheck.sh` from a clean `main` checkout.
2. Tag that commit `vX.Y.ZrcN`. The workflow rebuilds and verifies the wheel
   and source archive, then publishes them to TestPyPI through the protected
   `testpypi-release` environment.
3. Exercise the TestPyPI wheel and the quickstart. Then set the stable version
   with `uv version X.Y.Z` in a new commit, merge it, and rerun the precheck.
4. Tag the new commit `vX.Y.Z`. The workflow publishes the same verified
   artifacts to PyPI through the protected `release` environment. Only after
   PyPI succeeds does it create the GitHub Release.

The RC and stable tags therefore point to different commits. Publishing uses
PyPI trusted publishing (OIDC), with no repository token or password. Configure
the PyPI and TestPyPI publishers for project `prlccil`, owner
`personalrobotics`, repository `CCIL`, workflow `release.yml`, and the matching
environment name above.

**Legacy packaging.** The retired legacy installer (`setup.py`,
`install_deps.txt`) is preserved at `archive/upstream-main-2026-10-22`, and the
legacy package with its pinned oracle environments at `surface-archive-v1`.

## Evidence

```bash
uv run python tools/m2_packaging/check.py --out <new-dir>
```

The check builds the artifacts twice, confirms they are byte-identical, and
inspects the wheel and sdist contents and metadata. It then installs the
**built wheel** into fresh Python 3.10 and 3.14 environments and runs the CLI
and `data prepare` from outside the repository. The report is at
[`fixtures/evidence/m2-packaging/report.json`](fixtures/evidence/m2-packaging/report.json).
