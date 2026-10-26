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
| `provenance.py` | The code-state gate every run passes first: a clean tree, and `src`, `pyproject.toml` and `uv.lock` equal to the frozen `ccil` commit (§2) |
| `check.py` | The §3 acceptance checks |

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
