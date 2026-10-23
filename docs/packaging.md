# Packaging

The `ccil` core is a uv-managed project (#20).

| Item | Value |
|---|---|
| Definition | [`pyproject.toml`](../pyproject.toml), with build backend `uv_build`, a src layout, and the `ccil` console script |
| Lock | [`uv.lock`](../uv.lock), committed. `.python-version` selects 3.11 for development. |
| Python | 3.10 and newer |
| Core dependencies | `numpy>=1.24`, `pyyaml>=6.0`. Nothing else installs with the core. |
| Dependency groups | `evidence` (torch), used only by the M2 evidence tools |
| Optional extras | None yet. Adapters add them when they land: torch-backed models (M3), LeRobot (M4), image models (M5). A missing extra must fail only when its feature is requested, with install guidance. |

```bash
uv sync                              # development environment from the lock
uv run ccil --help
uv build                             # sdist and wheel into dist/
uv run --group evidence python tools/m2_schemas/check.py --out <dir>
```

**Legacy packaging.** The retired legacy installer (`setup.py`,
`install_deps.txt`) is preserved at `archive/upstream-main-2026-10-22`. The M1
legacy-harness tools (`tools/analytic_fixture`, `dynamics_checkpoint`,
`validation_split`, `invariants`, `legacy_serialization`) keep their
historically pinned `requirements.txt` files and run isolated from the project
lock:

```bash
uv run --no-project --python 3.11 --with-requirements tools/<tool>/requirements.txt python tools/<tool>/...
```

## Evidence

```bash
uv run python tools/m2_packaging/check.py --out <new-dir>
```

The check builds the artifacts twice, confirms they are byte-identical, and
inspects the wheel and sdist contents and metadata. It then installs the
**built wheel** into fresh Python 3.10 and 3.14 environments and runs the CLI
and `data prepare` from outside the repository. The report is at
[`fixtures/evidence/m2-packaging/report.json`](fixtures/evidence/m2-packaging/report.json).
