# CCIL: Continuity-based Data Augmentation for Imitation Learning

Official repository for CCIL (Ke et al., ICLR 2024, arXiv:2310.12972). The `ccil` package:
- trains dynamics models from demonstrations, with local Lipschitz regularization;
- generates corrective labels from them, with their quality evidence and curation;
- hands the resulting supervision to any downstream policy learner.

Policy training, NoiseBC, evaluation and benchmark studies are in
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks) (#168).

**Using CCIL:** the user contract, from demonstrations to consumable supervision, is [docs/contract.md](docs/contract.md).

## Setup

The `ccil` core is managed with [uv](https://docs.astral.sh/uv/) and supports Python 3.10 and newer:

```bash
git clone https://github.com/personalrobotics/CCIL.git
cd CCIL
uv sync                     # create .venv from uv.lock
uv run ccil --help          # schema, config, data, labels and pipeline commands
```

## Legacy inputs

Legacy datasets are pickles, which can execute code when loaded, so `ccil` never reads them. Convert one once
through the trust boundary, which verifies its SHA-256 against
[docs/artifacts/manifest.json](docs/artifacts/manifest.json) before unpickling it, then into the canonical schema:

```bash
uv run python tools/legacy_conversion/convert_legacy_dataset.py --shipped   # the two Pendulum datasets in data/
uv run ccil data convert data/pendulum_cont_100.npz pendulum_cont.npz
uv run ccil config convert config/pendulum_cont.yml --out canon.yml --manifest canon.json
```

To convert another pickle, pass `--expect-sha256 HEX` or acknowledge `--trust-legacy-pickle`. The conversion
contract is [docs/migration/legacy-conversion.md](docs/migration/legacy-conversion.md).

The legacy `correct_il` package, its scripts and configurations, and the other legacy datasets are archived at
the `surface-archive-v1` tag (#169). Every archived dataset keeps its SHA-256 and archive locator in the manifest.
The original installer is at `archive/upstream-main-2026-10-22`.
