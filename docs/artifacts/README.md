# CCIL artifacts and environment

Captured **2026-10-22**. This covers every non-source input that the
[reproducibility ledger](../ledger/README.md) needs: datasets, checkpoints,
code dependencies, system dependencies and the Python environment. Each one
has a stable identifier and a hash, or is listed below as unresolved with a
recovery action. The source of truth is [`manifest.json`](manifest.json). This
M0 manifest covers CCIL and its preserved forks. The actively changing
`image_ccil` repository is deferred to M5 issue #31, which must inventory its
artifacts and environments before image code migrates.

To verify:

```bash
git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'
python3 scripts/verify_artifacts.py            # offline: hashes, status rules, ledger cross-check
python3 scripts/verify_artifacts.py --network  # also fetch pinned commits, hash downloads, re-resolve the lock
```

[`verification-2026-10-22.log`](verification-2026-10-22.log) has the full
`--network` run. It reports 118 checks passed and 0 failures.

## Datasets

The 15 datasets live in Git at `archive/upstream-main-2026-10-22:data/`, and
their locator is `git:<tag>:<path>`. Nothing new was committed. The manifest
records each file's SHA-256, Git blob ID, size, trajectory count, horizon and
observation and action dimensions, plus its origin and license. For example,
`dataset:hopper-expert-v2_25` has 25 trajectories, horizon 531–1000,
observation dimension 11 and action dimension 3.

| Origin | Datasets | License | Origin verified |
|---|---|---|---|
| D4RL subset | ant, halfcheetah, hopper, walker2d | CC BY 4.0, attribution required | No. The subset selection is unrecorded. |
| Author-generated | pendulum ×2, F1, drones ×3, Meta-World ×4 | Unknown, since the repository has no license | No |
| Unknown | `pendulumswingup-v0.pkl` | Unknown | No. No config references it. |

What the data shows:

- **Counts differ from paper Table 2.** `pendulum_cont_100.pkl` holds 100
  trajectories (paper: 50) and `circle_50.pkl` holds 50 (paper: 30). The
  other counts match, including the Meta-World files that don't state a
  count in their names.
- **Drone and Meta-World demonstrations store rewards.** Their mean
  returns roughly match the paper's Expert row, except coffee-push: 1250 ±
  1546, against 4488 in the paper. That suggests its demonstrations include
  failed episodes.
- **The paper misattributes the Meta-World data.** App. D.2 says it comes
  from D4RL, but D4RL contains no Meta-World data.

## Environment

| File | Purpose |
|---|---|
| [`env/requirements.in`](env/requirements.in) | `install_deps.txt` plus `setup.py` `install_requires`, unchanged. |
| [`env/uv.toml`](env/uv.toml) | The minimum fixes needed for those requirements to resolve, each with its reason. |
| [`env/lock-linux-py38.txt`](env/lock-linux-py38.txt) | 95 pinned packages for Python 3.8 on x86_64 Linux, resolved as of 2024-07-15. |

No lock was recorded for the original runs. Resolving as of `main`'s last
code commit approximates the published environment; it does not prove it.

The declared requirements don't resolve as written. Each fix in `uv.toml`
addresses a real defect:

1. **d3rlpy fork.** `setup.py` imports numpy and Cython at build time
   without declaring them, which is why `install_deps.txt` exists. Modern pip
   builds packages in an isolated environment, so the README's two-step
   install fails the same way.
2. **gym 0.19.0.** Its published metadata (`opencv-python (>=3.)`) can't be
   parsed. It needs setuptools 65.5.0 and wheel 0.38.4 to build, which
   `install_deps.txt` pins without explaining why. Its real requirements are
   taken from its 0.19.0 tag.
3. **gym-pybullet-drones fork.** It declares only `cycler`, though its code
   imports `pybullet`, `gym`, `numpy` and `scipy`. It also builds against
   `poetry-core@main`, a moving target that no longer supports Python 3.8.
   The versions of `pybullet` and the others are resolved, not recorded, so
   they are assumptions.

System dependencies:

- **MuJoCo 2.1.0 binaries** (`system:mujoco-2.1.0`, Apache-2.0). Install
  them into `~/.mujoco/mujoco210`. The hash matches upstream's published
  `.sha256`.
- **Build tools.** A C compiler, GL/OSMesa headers and `patchelf`, all needed
  by `mujoco-py`.
- **A CUDA GPU.** Label generation and evaluation hard-code `cuda`, so there
  is no CPU-only stack.

**A clean install has not been verified.** Resolution succeeds, but nobody
has installed the lock and constructed each environment. GPU portability is
untested, and no GPU or driver model was ever recorded for the original runs.

## Unresolved artifacts

| Artifact | Needed by | Missing | Recovery action |
|---|---|---|---|
| `dataset:real-robot` | `real-*`, `internal-action-chunking-labels` | Demonstrations and trial logs for GraspCube, GearInsertion and GraspCoin | Ask the paper authors; check `image_ccil` when its integration resumes. |
| `checkpoint:expert_policy_pendulum` | Historical `aug.type: backward_expert` (no config uses it) | `output/expert_policy/pendulum.pt` | The method is explicitly unsupported. Recover the checkpoint and define its device and candidate contracts before re-enabling it. |
| `dataset:pendulum_img_50_noised_fix` | `deferred-image-pendulum` | The image pendulum dataset | Deferred with `image_ccil`. Regenerate with the fork's `gen_pendulum_data.py`. |

## Open verification

These checks need approval before they can run:

| Check | Requires | Why |
|---|---|---|
| Clean install of the lock, and constructing each task environment | A Linux host with GL/OSMesa headers and `patchelf` (needs `sudo`), plus about 3–4 GB of downloads | Proves the environment works, not just that it resolves. |
| D4RL subset origin | Downloading the four D4RL hdf5 files, about 950 MB | Proves the MuJoCo datasets are slices of D4RL, and which slices. |
| Tag the drone fork commit | Push an `archive/` tag to `personalrobotics/gym-pybullet-drones` | The pinned commit is reachable only from branch `fix-install`. |

## Machine-local paths

The manifest's `machine_local_paths` lists every hard-coded path, with its
replacement: a stable artifact ID, a repository-relative path, or an
out-of-scope ledger entry.
