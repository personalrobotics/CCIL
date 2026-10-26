# Pendulum study tooling (#24)

Tooling for the predeclared CCIL efficacy study,
[`docs/experiments/pendulum-ccil-study-predeclaration.md`](../../docs/experiments/pendulum-ccil-study-predeclaration.md).
It may never modify `src/ccil`, `pyproject.toml` or `uv.lock` (§2). Its own
dependencies are pinned in `requirements.txt`, and every command runs as:

```bash
uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt python tools/m3_pendulum/<script> ...
```

| File | Contents |
|---|---|
| `env.py` | The canonical environment (§3): `step`, `reward`, `initial_states`, `noise_draws`, and `rollout`, which evaluates a policy on a seed's 100 episodes and returns every per-step array; `save` writes them as a pickle-free npz |
| `audit.py` | The independent audit: recomputes rewards, returns and R from the saved states and actions, regenerates the initial states and noise draws, and replays every transition |
| `provenance.py` | The code-state gate every run passes first: a clean tree, and `src`, `pyproject.toml` and `uv.lock` equal to the frozen `ccil` commit (§2); for real runs, also the tooling paths equal to the commit `freeze.json` names (§11) |
| `study.py` | The run drivers: `prepare`, `sweep --phase tuning`, `choose`, `sweep --phase study`, and `run` for one unit |
| `analyze.py` | The analysis: re-audits every rollout, then the §8 decisions, intervals and tables |
| `check.py` | The §3 acceptance checks |
| `check_study.py` | Evidence for the drivers and the analysis: known answers, interval coverage, gates, and an end-to-end smoke run |

## Acceptance checks

```bash
uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \
    python tools/m3_pendulum/check.py --out <new-dir>
```

The checks train no policy, and the evaluator runs only fixed policies on
seed 9001, so they produce no tuning or study outcome. The report is
[`docs/fixtures/evidence/m3-pendulum/report.json`](../../docs/fixtures/evidence/m3-pendulum/report.json),
with `report-siddh.json` from the study host. It shows:
- **Transitions.** Both converted datasets have the predeclared SHA-256s.
  The environment reproduces every stored transition within 1e-6; the
  report gives the maximum error and the number of discontinuity crossings.
- **Evaluator audit.** For zero, ±1 and a fixed linear controller on both
  tasks, with and without evaluation noise, the audit reproduces every
  reward, the horizon, every return and R.
- **Tampering.** The audit refuses a changed reward, state, noise draw or R,
  and a dropped step, including inside a failed episode's finite prefix and
  after its failure.
- **Failures.** A non-finite action ends its episode, gives a non-finite R,
  and is recorded as a failure. Every step before it is still audited.
- **Repeatability.** Repeat evaluations are identical.
- **Code-state gate.** The tree is clean, and `src`, `pyproject.toml` and
  `uv.lock` equal commit `847b060`. A dirty tree fails the checks. On a
  scratch clone, the gate refuses an uncommitted `src` change and an
  untracked file.

The rollouts are written under `<new-dir>/rollouts/`, and each one can be
re-audited with `audit.audit(path)`.

## Running the study

After the tooling freeze (§11), on the study host, from the frozen commit:

```bash
P="uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt python tools/m3_pendulum"
$P/study.py sweep --root ROOT --phase tuning --jobs N --device cuda   # seeds 1-3, both grids, both tasks
$P/study.py choose --root ROOT                                        # ROOT/tuning-choice.json, made once
$P/study.py sweep --root ROOT --phase study --jobs N --device cuda    # seeds 101-120, every arm
$P/analyze.py --root ROOT --out ROOT/analysis.json
```

**The frozen tooling.** The freeze pull request adds `freeze.json`, naming
the tooling commit. Real sweeps, units, `choose` and the analysis refuse to
run unless every path the study executes or reads equals that commit:
`tools/m3_pendulum`, `tools/legacy_conversion`, `correct_il`, `data`,
`config`, and the conversion and artifact manifests. Smoke runs precede
the freeze, and record the tooling state without requiring it.

**Inputs** are verified on every use:
- each task's dataset and corrected configuration against the committed #23
  conversion;
- each data-size subset against its rebuild from `SeedSequence([seed, 9])`,
  byte for byte.

Every record carries the hashes it used.

**A unit** is one (task, arm, seed[, tuned value]). It:
1. passes the code-state gate and verifies its inputs;
2. runs `ccil policy train` on the #23 conversion of the paper configuration
   (`corrected`), with `policy.exclude_validation_episodes=false` and the
   arm's settings;
3. evaluates the policy at the arm's noise levels, saving and auditing every
   rollout.

Its record, `ROOT/results/<phase>/<task>/<arm>/s<seed>.json`, is written once.
A sweep skips any unit that already has a record, so an interrupted sweep
resumes. The units of one (task, seed) run in one worker and share their
stage store, so the arms reuse the same prepared data, dynamics model and
labels wherever their settings agree.

| Arm | Settings beyond the paper configuration | Noise levels |
|---|---|---|
| `bc`, `ccil` | `policy.method` | all |
| `noise_bc` | `policy.noise_scale` from the tuning choice | all |
| `ccil_tuned` | `generation.action_noise_std` from the tuning choice | all |
| `ccil_filtered` | `certificate.profile=disturbed_action_paper_proxy`, `certificate.selection_quantile=0.3` | 0 |
| `ccil_nolip` | `transition_model.lipschitz_type=none` | 0 |
| `bc_frac{10,25,50}`, `ccil_frac{10,25,50}` | the §7 nested episode subset | 0 |
| `zero` | no training: a = 0 | all |

**Failures (§9).** A training process killed by a signal, or out of memory,
is an infrastructure fault. It is recorded and rerun once, from a fresh
stage store; a second fault excludes the unit. Any other non-zero exit, a
non-finite training loss, or a non-finite return is an algorithmic failure,
and the record is the result. A configuration that `ccil` refuses is a
tooling defect: the unit stops without a record.

**GPU memory.** Certifying CCIL-filtered's labels scores them all at once:
about 60 GB of GPU memory for the discontinuous task, against at most 1 GB
for other units. A file lock (`ROOT/exclusive.lock`) lets one CCIL-filtered
unit train at a time, while the other units run in parallel.

**Balancing.** The discontinuous paper configuration balances noisy labels
(legacy `aug.balance_data`), which applies to CCIL only. The BC and NoiseBC
arms set `policy.balance_noisy_labels=false`, training on the demonstrations
as legacy BC did.

**Label accuracy (§7)** is measured on 10,000 accepted labels per CCIL unit,
drawn with `SeedSequence([seed, 10])`: ‖env.step(z_G, a_G) − y‖ in raw state
units.

**Verification** (`study.verify_phase`) runs before `choose` (tuning) and in
the analysis (both phases). It refuses:
- records that are not exactly the phase's units: missing, duplicate or
  extra units, or a record at another unit's path;
- a mix of smoke and real records, or of code states, or a record made
  without passing its gate;
- input hashes other than the verified inputs';
- a rollout whose bytes or full re-audit (with the replay) differ;
- a record whose R, failed episodes, task, seed or noise level disagree with
  its rollout's audit.

**The tuning choice** is computed from the verified tuning records. It
records their SHA-256s and, for CCIL-tuned, the label acceptance rates.

**The analysis** recomputes the tuning choice and refuses any difference,
and refuses a study record without the choice's SHA-256. Its tables use the
audit's recomputed R, never a stored one.

`--smoke` runs the same code at 1 dynamics epoch and one effective policy
epoch (1, or 20 for discontinuous CCIL arms, whose balanced epochs are divided by 20) on seeds
9001 (tuning) and 9002–9003 (study), in its own root. `check_study.py` uses
it, and records statuses and decisions, never smoke returns:

```bash
$P/check_study.py --out <new-dir> --device cuda --jobs 8
```

Its reports are `docs/fixtures/evidence/m3-pendulum/study-report.json`
(local, CPU) and `study-report-siddh.json` (CUDA).
