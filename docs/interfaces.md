# Core interfaces

These are the algorithmic interfaces every CCIL method plugs into (#17). They
are implemented in [`src/ccil/core/`](../src/ccil/core/), and each module's
docstring is its normative contract.

```
dataset ─▶ Representation ─▶ TransitionBatch ─▶ Generator ─▶ CandidateBatch ─▶ Validators ─▶ Weighting ─▶ CandidateSet
                 (encode)       (z, a, z_next)      (propose)      (z, a, target)    (reject + measure)            (ccil.candidates)
                                      ▲
                              TransitionModel (fit / delta / jacobian_z / evaluate / save / load)
```

`generate_labels(dataset, representation, model, generator, validators, run_id, seed, weighting=None, batch_size=None)`
composes the stages.

## Contracts

| Interface | Inputs → outputs | State and ownership | Determinism | Failures |
|---|---|---|---|---|
| `Representation` | `encode(dict[modality → (N, …)]) → (N, D)` float. `output_modality` is `state` or `latent.<name>`. | A trainable representation owns its parameters. The integrated pipeline (#19, M5) fits it; generation never does. | Deterministic and batch-invariant | A wrong shape or dtype raises `InterfaceError`. |
| `TransitionModel` | `fit(z, a, z_next, seed) → metrics`; `delta(z, a) → (N, D)`; optional `jacobian_z → (N, D, D)`; `evaluate`; `save` / `load(path, device)` | The model owns its parameters, optimizer and device. Only `fit` changes state. | `delta` is side-effect free | Model-specific load errors, such as `CheckpointError` for torch models (#12). |
| `Generator` | `generate(TransitionBatch, model, seed) → CandidateBatch`, plus `name`, `version`, `params` | Stateless: `params` fully describe it | Randomness per transition from `(seed, crc32(episode_id), frame)`, so output is batch-invariant | It marks `failed` and never accepts or rejects. A malformed batch raises `InterfaceError`. |
| `Validator` | `check(batch, cands, model) → (reject (n,) bool, measurements)`, plus `reason` (a `ccil.candidates` reason code, or `None` for a validator that only measures) | Stateless | Deterministic | An unknown reason, a malformed mask, or a measuring-only validator that rejects raises `InterfaceError`. |
| `Weighting` | `weights(cands, accepted) → (n,)` | Stateless | Deterministic | Weights must be finite and ≥ 0, and 0 on rejected rows (checked by the schema). |

Arrays are **numpy** at every boundary. Torch-backed models and encoders convert
internally, which keeps the core at numpy + pyyaml. No gradients cross an
interface.

**Validators are independent.** All of them run, a candidate is accepted only if
none rejects it, and every rejecting validator records its reason. One label can
therefore be rejected as both `out_of_support` and `action_out_of_bounds`, with
its `distance` measured.

## Reference implementations

| Kind | Class | Behavior |
|---|---|---|
| Representation | `IdentityRepresentation` | Passes a modality through unchanged (vector CCIL). |
| Transition model | `LinearTransitionModel(A, B, c)` | f = Az + Ba + c, exact or fitted by least squares. |
| Generator | `BackwardEuler` | z − f(z, a) |
| Generator | `ExactPreimage` | Newton solve of z_G + f(z_G, a) = z. Convergence is judged on the returned iterate, which avoids the legacy stale-residual pattern. |
| Generator | `NoisyAction(k, σ)` | z' − f(z', a + δ) |
| Validators | `Finite`, `GenerationFailed`, `SupportRadius(ε)`, `ActionBounds`, `ModelResidual` | `SupportRadius` uses the legacy rule `distance < ε`. `ModelResidual` only measures. |
| Weighting | `Uniform` | 1 for accepted rows, 0 for rejected. |

M3 ports the torch `WorldModel` and the full legacy generator set (#21, #22)
onto these interfaces.

## The image path (M5)

A learned encoder implements `Representation` with
`output_modality = "latent.<name>"`, and a latent transition model implements
`TransitionModel`. Labels then live in latent space as `latent.<name>`
candidates. `ccil.candidates` 1.0 accepts that modality, so no schema change is
needed. Trajectories still store only `state` and `image` modalities.

## Evidence

```bash
uv run --group evidence python tools/m2_interfaces/check.py --out <new-dir>
```

[`fixtures/evidence/m2-interfaces/report.json`](fixtures/evidence/m2-interfaces/report.json)
has 6 cases:

- **The #10 oracle:** the new core passes the analytic oracle (I1–I5) for every
  method, and its run directory validates as complete.
- **Multiple reasons:** rows carry two independent reasons with measurements.
- **Batch invariance:** generation produces byte-identical output at batch size
  1 and at full size.
- **Image plug-in:** a stub trainable latent encoder with a fitted latent model
  goes through the unchanged schema path.
- **Interface failures:** four malformed-stage failures raise before anything is
  written.
