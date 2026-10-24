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
| `Representation` | `encode(dict[modality → (N, …)]) → (N, D)` float. `output_modality` is `state` or `latent.<name>`. `kind` and `version`. | A trainable representation owns its parameters. The integrated pipeline (#19, M5) fits it; generation never does. | Deterministic and batch-invariant | A wrong shape or dtype raises `InterfaceError`. |
| `TransitionModel` | `fit(z, a, z_next, seed) → metrics`; `delta(z, a) → (N, D)`; optional `jacobian_z → (N, D, D)`; `evaluate`; `save(path)`; classmethod `load(path, device)`; `kind` and `version` | The model owns its parameters, optimizer and device. Only `fit` changes state. | `delta` is side-effect free | A file that isn't a checkpoint of a known kind and version raises `CheckpointError`. Model-specific failures, such as an unavailable device, propagate unchanged. |
| `Generator` | `generate(TransitionBatch, model, seed) → CandidateBatch`, plus `name`, `version`, `params` | Stateless: `params` fully describe it | Randomness per transition from `(seed, crc32(episode_id), frame)`, so output is batch-invariant | It marks `failed` and never accepts or rejects. A malformed batch raises `InterfaceError`. |
| `Validator` | `check(batch, cands, model) → (reject (n,) bool, measurements)`, plus `reason`: a built-in reason code, a namespaced extension `<namespace>.<reason>`, or `None` for a validator that only measures | Stateless | Deterministic | A malformed or reserved (`ccil.`) reason, a malformed mask, or a measuring-only validator that rejects raises `InterfaceError` before anything is written. |
| `Weighting` | `weights(cands, accepted) → (n,)` | Stateless | Deterministic | Weights must be finite and ≥ 0, and 0 on rejected rows (checked by the schema). |

**Identity and persistence.**
- A transition model's `kind` is its `transition_model.kind` value, and a
  representation's is its `representation.kind` value. Policy trainers use
  `name` (the `policy.trainer` value).
- A component's `version` changes whenever its checkpoints or fitted results
  would change.
- A checkpoint records `component`, `kind` and `version`.
  `load_transition_model(path, device)` reads the kind and dispatches through
  `TRANSITION_MODELS`, a plain dict that #21 extends with `mlp`. There is no
  plugin discovery.

Arrays are **numpy** at every boundary. Torch-backed models and encoders convert
internally, which keeps the core at numpy + pyyaml. No gradients cross an
interface.

**Extension reasons.** A method-specific validator declares its own reason,
such as `reason = "example.low_confidence"`. The namespace and reason are each
`[a-z][a-z0-9_]*`, and `ccil` is reserved. `generate_labels` adds one column per
extension after the built-ins, sorted, so column order is deterministic. The
built-in codes keep their meanings. A consumer that doesn't know a namespace
still treats those rows as rejected, and can report the code as opaque.

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
| Generator | `ExactPreimage` | Newton solve of z_G + f(z_G, a) = z, row by row: each row stops at its own tolerance, so results are batch-invariant. A singular Newton system, a non-finite iterate, or non-convergence marks only that row `failed`. Convergence is judged on the returned iterate, which avoids the legacy stale-residual pattern. |
| Generator | `NoisyAction(k, σ)` | z' − f(z', a + δ) |
| Validators | `Finite`, `GenerationFailed`, `SupportRadius(ε)`, `ActionBounds`, `ModelResidual` | `SupportRadius` uses the legacy rule `distance < ε`. `ModelResidual` only measures. |
| Weighting | `Uniform` | 1 for accepted rows, 0 for rejected. |

M3 ports the torch `WorldModel` and the full legacy generator set (#21, #22)
onto these interfaces.

## The image path (M5)

A learned encoder implements `Representation` with
`output_modality = "latent.<name>"`, and a latent transition model implements
`TransitionModel`. Labels then live in latent space as `latent.<name>`
candidates. `ccil.candidates` 2.x accepts that modality, so no schema change is
needed. Trajectories still store only `state` and `image` modalities.

## Evidence

```bash
uv run --group evidence python tools/m2_interfaces/check.py --out <new-dir>
```

[`fixtures/evidence/m2-interfaces/report.json`](fixtures/evidence/m2-interfaces/report.json)
has 10 cases:

- **The #10 oracle:** the new core passes the analytic oracle (I1–I5) for every
  method, and its run directory validates as complete.
- **Multiple reasons:** rows carry two independent reasons with measurements.
- **Batch invariance:** generation produces byte-identical output at batch size
  1 and at full size.
- **Image plug-in:** a stub trainable latent encoder with a fitted latent model
  goes through the unchanged schema path.
- **Interface failures:** four malformed-stage failures raise before anything is
  written.
- **Per-candidate solver failures:** in one batch, `ExactPreimage` marks rows
  with a singular Newton system or non-finite model output as
  `generation_failed`. The solvable rows match the exact solution, with
  byte-identical output at batch sizes 1, 4 and all.
- **Persistence:** a fitted model's checkpoint records its kind and version,
  loads through `load_transition_model`, and re-saves byte for byte. Unknown
  kinds, other versions or components, and files that aren't checkpoints raise
  `CheckpointError`. An unavailable device raises the model's own `ValueError`.
- **Extension reasons:** a custom `example.low_confidence` validator writes a
  complete run, and its candidates (version 2.1) reload with the same reason
  order and row assignments. Built-in-only files stay 2.0. Ten malformed
  declarations, from validators or in a candidate set, fail before any artifact
  is written.
