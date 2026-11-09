# Repository surface after #169

#169 removed everything CCIL no longer supports. Everything removed is at the
annotated tag `surface-archive-v1` (commit `4cbb8c5`, recorded in
[`history/refs.tsv`](../history/refs.tsv)).

## Before and after

| | `4cbb8c5` | after |
|---|---|---|
| Runtime Python (`src/`) | 11,315 lines | 11,282 lines |
| Verification tools (`tools/*.py`) | 54 files, 13,051 lines | 33 files, 8,735 lines |
| Committed evidence | 204 files, 2.4 MB | 42 files, 0.35 MB |
| Legacy `correct_il` package | 4,037 lines | none |
| Legacy pickles | 15 files, 177 MB | 2 files, 27 MB |
| Markdown, outside evidence | 5,087 lines | 3,916 lines |
| Tracked files | 475 | 194 |

The Git history (pack 194 MB) is unchanged.

## Package

The wheel holds 69 files: the `ccil` package (`config`, `core`, `data`,
`lerobot`, `models`, `pipelines`, `recipes`, `runs`, `schema`, the supervision
contract and the CLI) and its metadata. It has no data, run outputs, caches,
machine paths, legacy code or benchmark code, and installs with only `numpy`
and `pyyaml` on Python 3.10 and 3.14
([`m2-packaging`](../fixtures/evidence/m2-packaging/report.json)).

## Retained contracts and their checks

| Area | Contract | Check |
|---|---|---|
| Configuration | Provenance per value; failure before any output; legacy files and changed defaults keep their meaning; generated references match the code | `tools/m2_config` |
| Interfaces | The analytic oracle for every method; independent reasons; batch invariance; plug-in representations; persistence | `tools/m2_interfaces` |
| Pipelines | Replay and reuse by identity; the labels result; offline generation imports no trainer; the integrated reference flow and its resume | `tools/m2_pipelines` |
| Run directories | Atomic writes and recovery; classification of damaged runs | `tools/m2_runs`, `tools/m3_provenance` |
| Schemas | Lossless I/O; one refusal per rule; version compatibility; provenance | `tools/m2_schemas` |
| Labels | Closed forms and measurements; never refill; RNG streams; the model-misfit gate; action chunks | `tools/m3_labels` |
| Certificates and curation | Soundness against true dynamics; asserted values never bounds; selections verified; calibration; diagnosis | `tools/m3_certificates`, `tools/m45_quality` |
| Supervision | Rows and weights recomputed independently; mismatched artifacts fail closed; the documented recipe | `tools/m3_supervision` |
| Dynamics | The MLP's closed-form label invariants, Lipschitz types, persistence, determinism; batched fits equal single fits; clean hold-out; bounded-memory Jacobians; compile guard | `tools/m3_dynamics`, `tools/m3_fit_v2`, `tools/m3_jacobian_chunks`, `tools/m3_compile` |
| LeRobot | Ingest dispositions and endings; the per-release matrix; the overlay and its end-to-end consumer | `tools/m4_ingest`, `tools/m4_lerobot`, `tools/m4_overlay`, `tools/m4_roundtrip` |
| Legacy inputs | Hash-verified pickle conversion; tampered, unacknowledged and malformed pickles refused; dispositions of every legacy key | `tools/legacy_conversion` |
| Dependencies | A current, audited lock; the torch extra at its minimum and locked versions; `ccil` never unpickles | `tools/dependency_gate` |
| Repository | Ledger, inventory, artifact manifest, history refs, Markdown links | `scripts/check_*.py`, `scripts/verify_*` |

Each tool's docstring lists its cases. Every retained report was regenerated
once, at `29fdba1`; the `jacobian-chunks` CUDA case on siddh, and the Linux
LeRobot matrix (`06d98e0`) is kept as recorded.

## Removed

| Removed | Why |
|---|---|
| `ccil.training`, `ccil.experiments`, `ccil policy`, the former supervision names | Moved to il-supervision-benchmarks (#168); their refusal stubs ended. |
| `correct_il/` (except the pickle converter, now in `tools/legacy_conversion/`), legacy scripts and configs, 13 legacy pickles | The legacy implementation is no longer supported. The datasets keep their hash and archive locator in [`artifacts/manifest.json`](../artifacts/manifest.json). |
| `invariants`, `dynamics_checkpoint`, `validation_split`, `legacy_serialization`, the legacy half of `analytic_fixture`, the legacy cases of `m3_dynamics` | They tested only the legacy package. The trust boundary they shared is checked by `legacy_conversion`. |
| `m3_pendulum`, `m3_experiments`, `vector_env`, the study parts of `m3_fit_v2` and `m3_jacobian_chunks` | The Pendulum study is closed (il-supervision-benchmarks `docs/studies/pendulum.md`). |
| `m3_newton_batched` | A refactor-equivalence check of a private helper; `m2_interfaces` covers its failure mode. |
| Individual cases (about 30) | Tautologies, change-detectors, or duplicates of a behaviour-level case in another tool. |
| Evidence of removed checks; stale host, CUDA and study reports | Superseded; kept at the tag. |
