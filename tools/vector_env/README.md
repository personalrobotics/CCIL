# Vectorized evaluation (#111)

`pendulum.py` advances many Pendulum episodes at once, and is numerically
equivalent to the canonical environment (`tools/m3_pendulum/env.py`).

| Function | What it does |
|---|---|
| `integrate` | scipy 1.16.3's RK45, vectorized over episodes, with each episode keeping its own adaptive steps |
| `step` | `integrate` plus the discontinuity |
| `reward` | The canonical reward |
| `rollout` | The canonical rollout: same initial states, noise draws, failure handling and arrays, so the #24 audit applies |
| `replay_mismatches` | A batched, bit-exact replay of a saved rollout |
| `vector_env` | A `gymnasium.vector.VectorEnv`, with autoreset disabled and truncation at the horizon |

```bash
uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
    python tools/vector_env/check.py --out <new-dir>
```

The reports are in [`docs/fixtures/evidence/vector-env/`](../../docs/fixtures/evidence/vector-env/).
They show:
- **Stored transitions:** every stored transition is reproduced within
  1e-6, with the canonical number of discontinuity crossings.
- **One step:** within 1e-12 of the canonical step on random states and
  actions.
- **Rollouts:** fixed-policy rollouts stay within 1e-12 of the canonical
  ones at every step.
- **Audit:** the rollouts pass the #24 audit, and the batched replay is
  exact.
- **Adapter:** the Gymnasium adapter reproduces the batched rollout exactly.
- **Timing:** each case reports canonical against batched time.
