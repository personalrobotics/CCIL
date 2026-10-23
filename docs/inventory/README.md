# CCIL capability inventory

Captured **2026-10-22**. This maps each CCIL capability to its implementation on
each preserved ref, its provenance, and a proposed disposition. The
machine-readable source of truth is [`capabilities.json`](capabilities.json),
and every ref links back to an archive tag in the
[history manifest](../history/branch-references.md). `image_ccil` is out of
scope; its integration is deferred.

To check the inventory against history:

```bash
git fetch origin 'refs/tags/archive/*:refs/tags/archive/*'
python3 scripts/check_inventory.py
```

The check confirms that every listed path exists at its recorded commit, that
every commit matches the history manifest, that every capability has an owner
or `unknown` and a known disposition, and that every capability appears in the
matrix below.

## Refs

| Key | Commit | Preserved by |
|---|---|---|
| main | `5a9e3c1` | `archive/upstream-main-2026-10-22` (also `v1.0.0`) |
| label-filtering | `73d187d` | `archive/upstream-label-filtering-2026-10-22` |
| abhaybd-image | `69aa85a` | `archive/abhaybd-image-2026-10-22` |
| abhaybd-main | `a50f792` | `archive/abhaybd-main-2026-10-22` |
| gh-pages | `0650df8` | `gh-pages` branch |

## Capability matrix

The columns show where each capability exists: ● exists, ◐ differs from
`main`, and a blank means absent. Dispositions are proposals for later
milestones. "Reconcile" means the versions diverge and **none is canonical
yet**.

| Capability | main | label-filtering | abhaybd-image | Origin | Disposition |
|---|:-:|:-:|:-:|---|---|
| `dynamics-model` | ● | ◐ | ◐ | derived from mjrl | reconcile |
| `lipschitz-regularization` | ● | | | project | migrate |
| `spectral-norm` | ● | | | derived from PyTorch | migrate |
| `corrective-labels` | ● | ◐ | | project | reconcile |
| `label-filtering` | ● | ◐ | | project | reconcile |
| `action-chunking-labels` | | ● | | project | evaluate |
| `bc-policy` | ● | ◐ | | subclasses d3rlpy | reconcile |
| `baseline-vanilla-bc` | ● | | | project | migrate |
| `baseline-noise-bc` | ● | | | project | fix-then-migrate |
| `data-fraction-ablation` | | ● | | project | evaluate |
| `image-state-encoder` | | | ● | project | deferred |
| `diffusion-policy-eval` | | | ● | external | deferred |
| `env-pendulum` | ● | | ◐ | project | migrate |
| `env-f1tenth` | ● | | | unknown | migrate |
| `env-mujoco-d4rl` | ● | | | external | migrate |
| `env-metaworld` | ● | | | external | migrate |
| `env-drones` | ● | | | external (lab fork) | migrate |
| `datasets` | ● | | | unknown | migrate |
| `dynamics-training-tool` | ● | ◐ | | project | reconcile |
| `policy-evaluation` | ● | | | project | migrate |
| `f1-evaluation` | ● | ◐ | | project | reconcile |
| `results-analysis` | ● | ◐ | | project | evaluate |
| `rendering` | ● | | ◐ | project | evaluate |
| `config-system` | ● | | | project | migrate |
| `task-configs` | ● | ◐ | | project | reconcile |
| `pipeline-scripts` | ● | ◐ | | project | reconcile |
| `cluster-sweeps` | ● | ◐ | | project | exclude |
| `pendulum-data-generation` | | | ● | project | evaluate |
| `packaging` | ● | | | project | fix-then-migrate |
| `repo-hygiene` | | ● | | project | reconcile |
| `documentation` | ● | ◐ | | project | reconcile |
| `project-website` | gh-pages | | | project | archive-only |

`abhaybd-main` adds only a `.gitignore` (`repo-hygiene`). The JSON lists
unchanged inherited copies under `main` only.

## Blockers

These apply across the whole repository:

- **No license file.** No branch has a LICENSE. `setup.py` declares MIT, but
  GitHub reports no license. Code derived from mjrl and PyTorch needs its
  notices checked.
- **No designated maintainers.** Every capability's owner is `unknown`.
  Authors of record come from Git history and file headers only.
- **No mapping from experiments to code.** Nothing records which branch and
  config produced each published result. That belongs in the reproducibility
  ledger (#7).

Capability-specific findings for later milestones are listed below. Each is
recorded, not fixed.

- `f1-evaluation`: `main` reports average return, but `label-filtering`
  reports success rate under the same "Avg return" label.
- `baseline-noise-bc`: the injected noise is uniform on [0, 1), so it is
  non-negative rather than zero-mean.
- `action-chunking-labels`: the `chunk_size` condition is inverted, so a
  configured value is ignored.
- `bc-policy`: on `main`, an undefined `tycho` environment gets hard-coded
  loss weights. On `label-filtering`, a callable `loss_weights` is silently
  overwritten.
- `task-configs`: on `main`, two Meta-World configs write output to a literal
  `$HOME/CorrectiveIL/output/` directory.
- `datasets`: generation provenance is unknown except for F1, and
  `data/pendulumswingup-v0.pkl` is unreferenced (#8).
- `image-state-encoder`: its config needs a dataset that no preserved ref
  contains.

The JSON also lists generated or machine-specific files that should not
migrate, under `do_not_migrate`.
