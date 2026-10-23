# CCIL migration contract

**Status:** ratified 2026-10-22; amended 2026-10-23 to sequence image work behind
an explicit stability gate. Changing this contract requires a reviewed pull
request that edits this file. It is the binding scope and completion contract
for consolidating CCIL, and it draws on the
[history manifest](history/branch-references.md), the
[capability inventory](inventory/README.md), the
[reproducibility ledger](ledger/README.md) and the
[artifact manifest](artifacts/README.md).

## 1. Goal

Ship one maintained CCIL implementation that:

- generates state-based corrective labels behind stable interfaces first;
- adds end-to-end image CCIL in M5, including visual representation,
  corrective-label generation and downstream imitation-learning training;
- produces CCIL-augmented LeRobot datasets that standard trainers can consume
  unchanged;
- is evaluated on benchmarks the field currently uses, with expert data
  available.

Reproducing historical numbers is **not** a goal in itself. Historical code is
an oracle only where it is correct and runs on a maintained stack.

## 2. Repository and branches

- `personalrobotics/CCIL` is the canonical repository. All work lands on
  `main` through reviewed pull requests, one issue per pull request. There is
  no long-lived integration branch.
- The legacy branches (`label-filtering`, `gh-pages`) and the forks are frozen
  references. Their tips are preserved by the `archive/*` tags listed in the
  history manifest. Nothing is deleted or rewritten.
  `scripts/verify_history_manifest.sh` must pass before each milestone closes.
- `image_ccil` remains an active research repository while its implementation
  is in flux. Integration is deliberately deferred to M5. M5 starts only after
  maintainers declare a baseline stable; #31 then records its references,
  provenance, capabilities and artifacts before any image code migrates.
  Deferral is sequencing, not cancellation, and the repository must not be
  archived (see §9).

## 3. Benchmark scope (decision of 2026-10-22)

CCIL learns dynamics over the task state, so every in-scope benchmark must
expose low-dimensional state that includes object state, and must provide
expert demonstrations or a scripted expert.

| Benchmark | Decision | Milestone |
|---|---|---|
| Pendulum, continuous and discontinuous | **Keep.** This is the refactor-parity anchor (§5) and is tracked in #3. | M3 |
| MuJoCo Hopper, Walker2d, Ant, HalfCheetah | **Port** to Minari `mujoco/*/expert-v0` on Gymnasium v5. Results are new baselines. | M7 |
| Meta-World coffee-pull, button-press, coffee-push, drawer-close | **Port** to Meta-World v3, with demonstrations from its scripted experts. Results are new baselines. | M7 |
| robomimic low_dim (Lift, Can, Square, Transport, ToolHang) | **Add** | M7 |
| Push-T, state variant (`lerobot/pusht_keypoints`) | **Add.** It shares the dataset path with M4. | M4, M7 |
| ManiSkill3, state observation mode | **Add** | M7 |
| Drones (gym-pybullet-drones fork), F1TENTH | **Retire.** Niche, on an unmaintained stack, and the drone evaluation was invalid. | none |
| MOReL and MILO baselines | **Retire.** New suites compare BC, NoiseBC, BC+CCIL, and each suite's standard policy class. | none |
| ALOHA sim, LIBERO, RoboCasa365, RoboTwin 2.0 | **Defer** to image CCIL. Their published state lacks object state, or they are image-first. | M5 |
| Real-robot results (paper Figs. 8–13) | **Cite as published.** Reopen if the logged data is found. | none |

The legacy software stack (gym 0.19, mujoco-py, MuJoCo 2.1, the D4RL pickles,
and the lab forks of d3rlpy and gym-pybullet-drones) is a historical record
only. No milestone may depend on installing it, except for the minimal
Pendulum parity harness in §5.

## 4. Public interfaces

These are the boundaries the release must expose. The M2 issues own their
exact signatures; until those issues close, every ledger target command is
**provisional**.

| Interface | Owner | Contract |
|---|---|---|
| Python package `ccil` | #20 | Optional extras per benchmark suite. The core installs without any simulator. |
| Configuration schema | #4 | Versioned and validated. Every behavior-affecting value is named, and there are no hidden literals. |
| Trajectory, label and provenance schemas | #16 | Versioned. Each label records its source frame, generator, validator decision and parameters. |
| Transition-model, generator and validator interfaces | #17 | State-based in M2, then extended without breaking the core interfaces for image representations in M5. |
| Run artifacts | #18 | Written atomically, with commit, resolved config, seeds, environment and per-seed results. |
| Command-line interface | #17, #41 | One `ccil` entry point with subcommands. Names are fixed by #17. |
| LeRobot dataset output | M4 | An immutable augmented dataset with CCIL provenance, consumable by an unchanged LeRobot trainer. |
| Integrated image workflow | M5, #31–#37 | A stability-gated migration from image observations through CCIL labels to downstream policy training and evaluation. |

Out of scope for the first release is a compatibility layer for the legacy
scripts and their CLI. Image CCIL remains release scope, sequenced through M5.

## 5. Validation: two oracles

Every comparison is predeclared in the ledger entry or tracking issue before
any results are seen, and every seed is reported.

1. **Refactor parity** (blocking for M3). On Pendulum, continuous and
   discontinuous, run the legacy code at `archive/upstream-main-2026-10-22`
   with only the shims needed to import it without D4RL, then run the new
   implementation with the same data and seeds.
   - Deterministic stages must match within a stated numerical tolerance:
     dataset conversion, and label generation from fixed model weights.
   - Stochastic stages (model fitting and policy training, 10 seeds) report
     per-seed results. Parity holds when the 95% bootstrap interval of the
     mean difference lies within ±10% of the legacy mean. #25 may tighten
     this rule before running, but never after.
2. **Paper comparison** (informational). Legacy and new Pendulum results are
   compared with paper Fig. 4(c). A disagreement is reported, never tuned
   away. #3 is answered with this evidence, and "inconclusive" is an
   acceptable answer.

The M7 benchmarks have no historical oracle. They are new baselines with
predeclared protocols, seeds and statistics.

## 6. Correctness changes

A behavior change that fixes a defect is never folded into refactoring work.
It lands in its own pull request that:

- names the defect;
- shows before-and-after evidence on the affected ledger entries or fixture;
- sets the entries to `reproduced-with-correctness-change`.

Known defects that require this treatment are recorded in the inventory and
ledger. Examples are the inverted `chunk_size` condition, the NoiseBC noise
model, and the drone evaluation's observation replacement, although the drone
benchmark is now retired.

## 7. Data and trust boundaries

- **Legacy pickles are executable input.** They load only through the
  explicit trusted-input boundary (#15) and the one-way converter (#23), and
  only for retained datasets (Pendulum).
- **External datasets are validated at ingress against the schemas in #16.**
  This covers Minari, Hugging Face (LeRobot), robomimic and ManiSkill. They
  are fetched by stable identifier and revision and never committed to Git.
  Their licenses are recorded in the artifact manifest.
- **There is no license yet (blocker).** A LICENSE file must exist before any
  release (M6, #38).

## 8. Deprecation, blocked examples and release

- **Deprecation.** The legacy scripts, configs and CLI are not maintained.
  Their behavior is described by the ledger, and their source stays at the
  archive tags.
- **Blocked examples.** A blocked entry names the missing item, an owner and
  a next action. It is never counted as reproduced. A release may ship with
  blocked entries if they are listed in the release notes.
- **Release gates (M6).** A release needs all of the following:
  - every checker passes: `verify_history_manifest.sh`, `check_inventory.py`,
    `check_ledger.py` and `verify_artifacts.py`;
  - the clean-environment install is verified (#40);
  - the license and attribution review is complete (#38);
  - every ledger entry is terminal or blocked with an owner;
  - the M3 refactor-parity gate has passed.

## 9. Archive policy

Archival of `image_ccil` is **forbidden** until all of the following hold:

1. image integration (M5) has completed or been explicitly cancelled by an
   amendment to this contract;
2. its branches are recorded in the history manifest with `archive/*` tags;
3. the ledger has been reviewed, and every image entry is terminal;
4. its README links to this contract and the history manifest.

The legacy CCIL branches are never deleted. They stay discoverable through
the archive tags and the history manifest on `main`.

## 10. Ledger targets and terminal rules

Every ledger entry carries a `target_milestone` and a `terminal_rule`, and
`scripts/check_ledger.py` enforces both. As of this contract, 2 entries are
pending (Pendulum, M3), 3 are blocked with M3 targets, and 25 are out of
scope with rationales that cite §3.

## 11. Non-goals

- deadlines;
- rewriting history;
- promising reproduction where inputs are unavailable;
- requiring CCIL to beat behavior cloning as a closure condition;
- tuning to recover historical numbers;
- image CCIL before the M5 stability gate and #31 baseline capture are complete.
