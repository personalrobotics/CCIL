# Dependency and checkpoint security

This page sets the dependency boundary of the `ccil` package, the policy for
loading checkpoints, and the advisories accepted in the legacy parity harness
(#84). [`tools/dependency_gate/`](../tools/dependency_gate/) enforces the
supported side.

## Supported environments

| Environment | Python | Dependencies | Gate |
|---|---|---|---|
| `ccil` | 3.10+ | `numpy`, `pyyaml` | no known vulnerabilities in `uv.lock` |
| `ccil[torch]` | 3.11+ | adds `torch>=2.9.1` | same |
| Development (`uv sync`), and `--group evidence` | 3.11+ for `evidence` | exact pins: `matplotlib==3.10.9`, `pillow==13.0.0`, `torch==2.14.1` | same |

- **Torch minimum.** 2.9.1 is the lowest release with no known critical, high
  or medium advisory. The remaining low-severity advisories need specific
  operators (`torch.jit.script`, `lstm_cell`) that ccil doesn't use. The gate
  installs and exercises `ccil[torch]` at exactly 2.9.1 and at the locked
  version.
- **Why 3.11 for torch and evidence.** The fixed releases of `filelock`
  (≥ 4.2.1, a torch dependency) and `pillow` (≥ 13.0.0, a matplotlib
  dependency) require Python 3.11. On 3.10, `ccil[torch]` therefore installs no
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
- **Pickle-based checkpoints and TorchScript archives are trusted, hashed
  inputs only.** This covers the legacy `.pt` files read by `correct_il` and the
  M1 tools (`torch.load`), and the TorchScript policies the legacy evaluation
  reads (`torch.jit.load` in `correct_il/eval_bc_policy.py` and
  `correct_il/envs/f1/src/eval_bc.py`). `weights_only=True`
  is not a safety boundary before torch 2.6 (GHSA-53q9-r3pm-6pq6), and the
  harness pins 2.1.0. Load only:
  - files the harness wrote itself in the same run; or
  - artifacts whose SHA-256 matches [`artifacts/manifest.json`](artifacts/manifest.json).

  The M1 case `legacy_pickle_not_executed` shows that a naive pickle payload is
  refused. It doesn't show that a crafted payload would be.

## Legacy parity harness: accepted advisories

Six manifests keep historical torch pins, because scientific parity with the
published code (#24) depends on them. They are isolated from the supported
environments:
- they run with `uv run --no-project --with-requirements …` and never share the
  project lock;
- they aren't part of the `ccil` package;
- they process only repository fixtures and their own outputs, under the
  checkpoint policy above.

| Manifest | Pin |
|---|---|
| `correct_il/envs/f1/requirements.txt` | `torch==2.0.1` |
| `tools/analytic_fixture/requirements.txt` | `torch==2.1.0` |
| `tools/dynamics_checkpoint/requirements.txt` | `torch==2.1.0` |
| `tools/invariants/requirements.txt` | `torch==2.1.0` |
| `tools/legacy_serialization/requirements.txt` | `torch==2.1.0` |
| `tools/validation_split/requirements.txt` | `torch==2.1.0` |

The same 11 torch advisories apply to each manifest, giving 66 alerts. They are
accepted under the conditions above and not dismissed. Retiring them, by
upgrading the pins once parity no longer needs them, is tracked separately.

| Advisory | Severity | Fixed in | Why acceptable in the harness |
|---|---|---|---|
| GHSA-53q9-r3pm-6pq6 | critical | 2.6.0 | `torch.load(weights_only=True)` RCE. Only trusted, hashed checkpoints are loaded. |
| GHSA-5pcm-hx3q-hm94 | high | 2.2.0 | Heap buffer overflow. It needs attacker-controlled tensors or model input, and the harness runs repository fixtures only. |
| GHSA-pg7h-5qx3-wjr3 | high | 2.2.0 | Use-after-free. Same input boundary. |
| GHSA-887c-mr87-cxwp | medium | 2.8.0 | Resource release. These are local, short-lived runs. |
| GHSA-vgrw-7cvw-pwgx | medium | 2.9.1 | `unpack_sequence` memory corruption. The harness doesn't use it. |
| GHSA-f4hp-rmr7-r7v8 | medium | none | `pad_packed_sequence` memory use. The harness doesn't use it. |
| GHSA-3749-ghw9-m3mg | low | 2.7.1 | Local denial of service. |
| GHSA-qfhq-4f3w-5fph | low | 2.10.0 | `lstm_cell` memory corruption. The harness doesn't use it. |
| GHSA-rrmf-rvhw-rf47 | low | 2.13.0 | `torch.jit.script` memory corruption. The harness compiles only its own policies and loads only TorchScript it wrote (`torch.jit.load`). |
| GHSA-c678-jfcj-6jmf | low | none | Tuple handler memory corruption. It needs crafted input. |
| GHSA-x3gm-94wq-g975 | low | none | Quantized sigmoid initialization. The harness doesn't use it. |

## Evidence

```bash
uv run python tools/dependency_gate/check.py --out <new-dir>
```

This needs network access, to PyPI and OSV. The report is at
[`fixtures/evidence/dependency-gate/report.json`](fixtures/evidence/dependency-gate/report.json).
