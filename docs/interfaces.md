# Core interfaces

> The user-facing workflow (Diagnose, Generate, Curate, Inspect) and its artifact vocabulary are specified in [contract.md](contract.md).

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
| `Generator` | `generate(TransitionBatch, model, seed) → CandidateBatch`, plus `name`, `version`, `params`, and `measurement_specs` (definition, norm and units of each measurement it returns) | Stateless: `params` fully describe it | Randomness per transition from the seed and a 128-bit digest of `(episode_id, frame)` (`ccil.core.identity`), so distinct transitions never share a stream and output is batch-invariant: byte for byte with numpy models, and up to float32 kernel rounding (about 1e-8) with torch models, whose matrix products round differently at different batch shapes | It marks `failed` and never accepts or rejects. A malformed batch raises `InterfaceError`. |
| `Validator` | `check(batch, cands, model) → (reject (n,) bool, measurements)`, plus `measurement_specs` and `reason`: a built-in reason code, a namespaced extension `<namespace>.<reason>`, or `None` for a validator that only measures | Stateless | Deterministic | A malformed or reserved (`ccil.`) reason, a malformed mask, a measuring-only validator that rejects, or measurement specs that don't name exactly the returned measurements raise `InterfaceError` before anything is written. |
| `Weighting` | `weights(cands, accepted) → (n,)` | Stateless | Deterministic | Weights must be finite and ≥ 0, and 0 on rejected rows (checked by the schema). |

**Identity and persistence.**
- A transition model's `kind` is its `transition_model.kind` value, and a
  representation's is its `representation.kind` value. Policy trainers use
  `name` (the `integrated.trainer` value).
- A component's `version` changes whenever its checkpoints or fitted results
  would change.
- A checkpoint records `component`, `kind` and `version`.
  `load_transition_model(path, device)` reads the kind and dispatches through
  `TRANSITION_MODELS`, a plain dict. An entry may be a `"module:Class"` string,
  such as `mlp`, which is imported only when that kind is used and fails with
  install guidance if its extra is missing. There is no
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
| Generator | `NoisyAction(k, σ)` | z' − f(z', a + δ). It records `action_noise_norm`, `state_displacement_norm` and `draw_index`. With `action_bounds=(low, high)`, the draws are the first k proposals of each transition's stream whose stored action is inside the bounds (a truncated Gaussian), and `attempt_index` records each draw's raw position in the stream. |
| Generator | `NoisyActionPreimage(k, σ)` | Newton solve of z_G + f(z_G, a + δ) = z', from z. It records `action_noise_norm`, `state_displacement_norm`, `solver_residual` and `draw_index`, and takes `action_bounds` as `NoisyAction` does. A failed proposal is rejected, never refilled. |
| Validators | `Finite`, `GenerationFailed`, `SupportRadius(ε)`, `ActionBounds`, `ModelResidual`, `ModelMisfit(τ)` | `SupportRadius` uses the legacy rule `distance < ε`. `ModelResidual` only measures. `ModelMisfit` measures each anchor's one-step model error, and rejects the candidates of anchors with error ≥ τ. Label-error estimates, bounds and filters are certificates, a separate layer after generation: see [certificates.md](certificates.md) (#88). |
| Weighting | `Uniform` | 1 for accepted rows, 0 for rejected. |
| Transition model | `ccil.models.mlp.MLPTransitionModel` (`kind="mlp"`, needs `prlccil[torch]`) | The legacy `WorldModel` with every Lipschitz regularizer (#21). See [`migration/vector-dynamics.md`](migration/vector-dynamics.md). |

M3 ports the full legacy generator set onto these interfaces (#22).

## Library APIs for downstream harnesses

A benchmark harness (#168) may also use these. They are public: a change to
their signature or results is a versioned change, recorded here. Every other
module outside `ccil.core`, `ccil.schema`, `ccil.supervision` and
`ccil.lerobot` is internal.

| API | Contract |
|---|---|
| `ccil.models.mlp_batched.FitJob(z, a, z_next, seed)`, `fit_group(jobs, settings, *, device, compile, chunk_steps, deterministic)` | Fits many `MLPTransitionModel`s of one shape as one batched program (`transition_model.trainer_version 2`, [pipelines.md](pipelines.md)). Returns `[(model, fit record)]` in job order. Needs `prlccil[torch]`. A group of more than 128 fits, or an unsupported `lipschitz_type`, raises `ValueError`. |
| `ccil.models.mlp_batched.fit_identity(z, a, z_next, settings, seed)`, `model_sha256(model)` | Content identity of one fit (its rows, settings, seed and engine), and SHA-256 of a fitted model's payload. The labels stage compares both before it accepts a model fitted elsewhere. |
| `ccil.data.split.make_split(episode_ids, validation_fraction, seed)`, `SplitError` | The `ccil.split` 1.0 record that `data.prepare` writes: whole episodes, a seeded permutation. Raises `SplitError`, a `ValueError`, when either part would be empty. |
| `ccil.resolve_config`, `diagnose`, `generate`, `curate`, `inspect`, `ccil.InputError` | The golden path's Python entry points and the error every refused input raises ([contract.md](contract.md#commands)). |
| `ccil.generate-result`, `ccil.curate-result` 1.0 | What `ccil generate` and `ccil curate` wrote, with every artifact's path and SHA-256 ([schemas/reference.md](schemas/reference.md)). |
| `ccil labels generate --result PATH` | The run's stages, reuse and artifact paths as JSON, `ccil.labels-result` 1.0 ([pipelines.md](pipelines.md#offline-generation-ccil-diagnose-ccil-generate-ccil-curate)). |
| `ccil.core.transition.checkpoint_meta(path, component)`, `CheckpointError` | The meta of a component checkpoint, read without unpickling. A file that is not one raises `CheckpointError`. |

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

Reports: [`m2-interfaces/report.json`](fixtures/evidence/m2-interfaces/report.json).
Each case, and the contract it protects, is listed in the docstring of [`tools/m2_interfaces`](../tools/m2_interfaces/).
