# Dependency and checkpoint security

This page sets the dependency boundary of the `ccil` package and the policy for
loading checkpoints (#84). [`tools/dependency_gate/`](../tools/dependency_gate/)
enforces them.

## Supported environments

| Environment | Python | Dependencies | Gate |
|---|---|---|---|
| `ccil` | 3.10+ | `numpy`, `pyyaml` | no known vulnerabilities in `uv.lock` |
| `prlccil[torch]` | 3.11+ | adds `torch>=2.9.1` | same |
| Development (`uv sync`), and `--group evidence` | 3.11+ for `evidence` | exact pins: `matplotlib==3.10.9`, `pillow==13.0.0`, `torch==2.14.1` | same |

- **Torch minimum.** 2.9.1 is the lowest release with no known critical, high
  or medium advisory. The remaining low-severity advisories need specific
  operators (`torch.jit.script`, `lstm_cell`) that ccil doesn't use. The gate
  installs and exercises `prlccil[torch]` at exactly 2.9.1 and at the locked
  version.
- **Why 3.11 for torch and evidence.** The fixed releases of `filelock`
  (≥ 4.2.1, a torch dependency) and `pillow` (≥ 13.0.0, a matplotlib
  dependency) require Python 3.11. On 3.10, `prlccil[torch]` therefore installs no
  torch, and torch-backed features fail with install guidance when requested.
  The core never needs either package.
- **The gate.** `uv audit` checks the whole lock, covering every group, extra
  and Python version, against OSV. Any vulnerability fails the gate unless its ID
  is listed under accepted advisories below. That list is currently empty for
  the lock. The gate is stricter than "no critical or high", because OSV
  results don't carry a reliable severity.

## Checkpoint policy

- **ccil checkpoints are not pickles.** Transition models, representations and
  policies are saved as `.npz` and read with `allow_pickle=False`, with their
  `component`, `kind` and `version` checked (#17). #21 stores the torch model the
  same way: its `state_dict` is converted to numpy arrays. ccil never calls
  `torch.load` or `torch.save`. A checkpoint from an untrusted source can
  therefore fail to load, but it can't run code.
- **Legacy pickles are trusted, hashed inputs only.** The one place that
  unpickles is `tools/legacy_conversion/trajectory_io.py`, which verifies each
  file's SHA-256 against [`artifacts/manifest.json`](artifacts/manifest.json)
  (or takes an explicit acknowledgement) before reading it. The legacy code
  that loaded pickled checkpoints, and its pinned pre-2.6 torch environments,
  are archived at `surface-archive-v1` (#169).

## Evidence

```bash
uv run python tools/dependency_gate/check.py --out <new-dir>
```

This needs network access, to PyPI and OSV. The report is at
[`fixtures/evidence/dependency-gate/report.json`](fixtures/evidence/dependency-gate/report.json).
