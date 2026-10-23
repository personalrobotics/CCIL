# CCIL history and branch references

Captured **2026-10-22**. This file records every CCIL Git reference that must
stay recoverable while the state-based core is consolidated. The
machine-readable copy is [`refs.tsv`](refs.tsv). The actively changing
`image_ccil` repository is deliberately deferred to M5. Issue #31 owns its
stability-gated reference capture, branch reconciliation and provenance review
before any image code migrates. Deferral does not permit archiving, deleting or
rewriting that repository.

To check the live remotes against this manifest:

```bash
scripts/verify_history_manifest.sh
```

The script runs `git ls-remote` once per repository and prints one status per
recorded ref:

- `OK`: the ref still points at the recorded SHA.
- `ADVANCED`: a branch moved forward but its history still contains the
  recorded commit, as `main` does after consolidation work merges.
- `MOVED`: a tag changed, or a branch was rewritten so that it no longer
  contains the recorded commit.
- `MISSING`: the ref no longer exists.

It also prints `UNRECORDED` for any upstream branch or tag that this manifest
omits. It exits non-zero only on `MOVED` or `MISSING`. The script needs only
`git`; to decide `ADVANCED`, it fetches the history of the moved branch into a
temporary directory. It ran clean on the capture date.
An `UNRECORDED` line for a short-lived pull-request branch is expected and
does not fail the check.

## Repository

| Field | Value |
|---|---|
| Remote | <https://github.com/personalrobotics/CCIL> (public) |
| Default branch | `main` |
| Created | 2024-04-02 |
| Last push before capture | 2026-02-19 |
| GitHub Pages | `gh-pages` branch, served at <https://personalrobotics.cs.washington.edu/CCIL/> |
| Pull requests, releases, submodules, Git LFS | none |

## Upstream references

| Ref | SHA | Last commit | Relation to `main` | Purpose |
|---|---|---|---|---|
| `main` | `5a9e3c169874f876419953105fcba88cb5a3695e` | 2024-07-14, Quinn Pfeifer: "Remove extra numpy req in setup.py" | default branch, 6 commits | Published CCIL code and datasets that the paper and README reference. |
| `v1.0.0` (lightweight tag) | `5a9e3c169874f876419953105fcba88cb5a3695e` | same as `main` | same commit as `main` | Release marker for the published code. |
| `label-filtering` | `73d187dc233f73a0248c095da9d4b6fe3a48377a` | 2024-11-25, Abhay Deshpande: "Add label generation for action chunking policies" | 4 ahead, 0 behind | Post-publication research: IROS changes, simulation config updates, a sweep commit labeled "Dirty commit: sweeping", and label generation for action-chunking policies (`gen_aug_label_chunking.py`, ablation and sweep tabulation scripts). Touches 31 files (+640/−145). |
| `gh-pages` | `0650df8ee4f295b993e0903d94f03a98828872f7` | 2025-10-28, Abhay D: "Add unified preprint, update bibtex citations" | separate history: shares only the initial commit `09fbf948`, 37 ahead and 5 behind | Project website. Generated and static web content, not library code. |

## Archive tags

These annotated tags were pushed to `personalrobotics/CCIL` on 2026-10-22, so
that branch tips which could otherwise be lost stay reachable upstream. They
follow the preservation policy below.

| Tag | Tag object | Tagged commit | Preserves |
|---|---|---|---|
| `archive/upstream-main-2026-10-22` | `477ddb81c7adc3527c77b094c08181a98077eff3` | `5a9e3c169874f876419953105fcba88cb5a3695e` | Upstream `main` before consolidation commits. Pushed after the manifest merged. |
| `archive/upstream-label-filtering-2026-10-22` | `f076eebccad85f2b7f7ed3dd4423d01b7c69d468` | `73d187dc233f73a0248c095da9d4b6fe3a48377a` | Upstream `label-filtering` |
| `archive/abhaybd-image-2026-10-22` | `9d50f8f75446b6c742fa5e431283d5993a18c885` | `69aa85a37abf57421943f76cbcdb3922c55f3738` | Fork `abhaybd/CCIL` `image`. Its history also contains the fork's original `label-filtering` tip `dc7068a`. |
| `archive/abhaybd-main-2026-10-22` | `2d3a853ef731f1986e13413d83b95873f78aba68` | `a50f7922082c7aa4c0b9b1a57b2665900b5fecfa` | Fork `abhaybd/CCIL` `main` |

## Fork references

Six public forks exist. Five point at commits already upstream. The
`abhaybd/CCIL` fork held commits found nowhere upstream. They are now
preserved by the archive tags above.

| Fork ref | SHA | Status |
|---|---|---|
| `abhaybd/CCIL` `image` | `69aa85a37abf57421943f76cbcdb3922c55f3738` | **Fork-only.** Seven commits from 2024-09-11 to 2024-09-13 that add image observations to the pendulum environment, image dynamics training, and rendering. Built on the fork's `label-filtering`. This is early image-CCIL work. |
| `abhaybd/CCIL` `main` | `a50f7922082c7aa4c0b9b1a57b2665900b5fecfa` | **Fork-only.** One commit, "Add gitignore" (2024-09-11), on top of upstream `main`. |
| `abhaybd/CCIL` `label-filtering` | `dc7068af27d78f4a68a66d9789719a79074e3fea` | Superseded. Its three commits exist upstream on `label-filtering` under new SHAs with the same patch IDs (`git cherry` shows `-` for each), recommitted on 2024-11-25. |
| `alfred-cueva/CCIL-dev` `main` | `5a9e3c169874f876419953105fcba88cb5a3695e` | Same as upstream `main`. |
| `quinnmp/CCIL` `main` | `5a9e3c169874f876419953105fcba88cb5a3695e` | Same as upstream `main`. |
| `sidtalia/CCIL` `main` | `5a9e3c169874f876419953105fcba88cb5a3695e` | Same as upstream `main`. |
| `HoneyMack/CCIL` `main` | `5a9e3c169874f876419953105fcba88cb5a3695e` | Same as upstream `main`. |
| `LuziaKn/CCIL` `main` | `f656990a4cb9e94cee9726e19a20a7deef918002` | Upstream ancestor of `main`, two commits behind. |

The original issue also lists `merge`, `libero-merge`, `noise-bc` and
`caiyi_real_world`. Those branches belong to the actively changing `image_ccil`
repository. Their stability-gated capture and reconciliation are deferred to
M5 issue #31; this CCIL manifest does not claim to cover them.

## Content classification on `main`

| Path | Class | Notes |
|---|---|---|
| `correct_il/` (except `envs/f1/maps/`) | Project-owned code | Training, augmentation-label generation, dynamics models, evaluation. |
| `correct_il/envs/pendulum_env.py`, `correct_il/envs/f1/src/` | Project-owned environment code | The F1 environment was committed with the codebase in `a2ef17e`. It has its own `requirements.txt`, but no license or upstream attribution was found, so its provenance is **unknown**. |
| `correct_il/envs/f1/maps/` | Environment assets | Track images, YAML and centerline CSVs. |
| `config/*.yml`, `scripts/*.sh`, `scripts/launch.slurm` | Project-owned experiment configuration | Per-task hyperparameters and launch scripts. |
| `data/*.pkl` (15 files, about 177 MB) | Datasets | Expert demonstrations committed directly to Git, without LFS. The largest are `circle_50.pkl` and `flythrugate_50.pkl` at 48 MB each. |
| `setup.py`, `install_deps.txt` | Packaging | `install_deps.txt` pins Cython 0.29.32, NumPy 1.22.0, setuptools 65.5.0 and wheel 0.38.4. |
| `gh-pages` branch | Generated and static website | HTML, CSS, JS, images, videos and the paper PDF. |

No vendored third-party packages, submodules or checked-in experiment outputs
were found on `main`. `label-filtering` adds analysis scripts
(`plot_ablation.py`, `tabulate_sweep.py`) but no result files.

## Blockers and risks

| Item | Risk | Owner | Recovery action |
|---|---|---|---|
| `abhaybd/CCIL` `image` and `main` fork-only commits | **Resolved 2026-10-22.** Previously they would have been lost if the fork were deleted or force-pushed. | CCIL maintainers | Preserved upstream by `archive/abhaybd-image-2026-10-22` and `archive/abhaybd-main-2026-10-22`. |
| `label-filtering` contains a commit labeled "Dirty commit: sweeping" | The commit's contents are unreviewed. Later milestones must not treat it as a vetted baseline. | M0 reproducibility ledger (#7) | Classify it in the ledger. Do not rewrite it. |
| `v1.0.0` is a lightweight tag | It records no tagger or date and can be moved with `git push -f`. | CCIL maintainers | Keep it as is. The SHA recorded here, together with the archive tag below, is the durable reference. |
| `envs/f1` provenance | License and upstream origin unknown. | M0 capability inventory (#6) | Confirm the origin before any relicensing or release. |

There are no inaccessible references. Every ref listed above resolved on the capture date.

## Preservation policy

1. **No deletion or rewriting.** No branch, tag, fork-derived commit or
   artifact listed here is deleted, force-pushed or rebased during
   consolidation.
2. **Archive tags.** Before any branch is retired, an annotated tag
   `archive/<source>-<branch>-<YYYY-MM-DD>` is pushed to
   `personalrobotics/CCIL` pointing at the SHA recorded here. Examples are
   `archive/upstream-label-filtering-2026-10-22` and
   `archive/abhaybd-image-2026-10-22`. This uses the same `archive/` prefix that
   `image_ccil` already uses. Tags remain fetchable when a repository is archived
   on GitHub, and fork-only commits become upstream objects once they are tagged.
3. **Discoverability after archiving.** This file and `refs.tsv` stay on the
   default branch of the consolidated repository. The archived repository's
   README links to them.
4. **Re-verification.** Run `scripts/verify_history_manifest.sh` before each
   consolidation milestone closes. A `MOVED` or `MISSING` result blocks
   closure until the ref is explained here.

## How the SHAs were obtained

```bash
git ls-remote https://github.com/personalrobotics/CCIL
gh api repos/personalrobotics/CCIL --jq '{default_branch,created_at,pushed_at}'
gh api repos/personalrobotics/CCIL/branches --jq '.[]|[.name,.commit.sha]|@tsv'
gh api repos/personalrobotics/CCIL/tags --jq '.[]|[.name,.commit.sha]|@tsv'
gh api repos/personalrobotics/CCIL/forks --jq '.[].full_name'
git ls-remote https://github.com/<fork>          # for each fork
git fetch https://github.com/abhaybd/CCIL main image label-filtering
git rev-list --left-right --count origin/main...<sha>
git cherry -v origin/label-filtering dc7068af27d78f4a68a66d9789719a79074e3fea
git branch -r --contains <sha>                   # empty for the fork-only commits
```
