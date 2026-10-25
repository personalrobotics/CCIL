# Label-error certificates

Certification is a separate step after label generation (#88):

    generate candidates → certify candidates → select / admit labels → train policy

It is implemented in
[`src/ccil/core/certificates.py`](../src/ccil/core/certificates.py) and run by
the `labels.certify` stage. The candidates file is an input and is never
modified. Training uses the rows where

    usable = candidate.accepted ∧ certificate.admissible ∧ certificate.selected

The file format, `ccil.certificates` 1.0, is specified in
[`schemas/reference.md`](schemas/reference.md).

## What a certificate claims

For a label (q, y), with q = (z_G, a_G), the quantity of interest is the true
label error ‖T(q) − y‖, where T(z, a) = z + f(z, a) is the true dynamics.

**Coordinates.** Norms are taken in scaled coordinates, z / `state_scale` and
a / `action_scale`. Errors are measured in the scaled state coordinates. The
modality, dimensions and scales are part of the descriptor (all scales
default to 1). Units are derived from them, never supplied as free text.

**Global terms.** Every certificate persists four terms, as diagnostics at
least:

| Term | Value |
|---|---|
| `state_displacement` | ‖z_G − z_t‖ from the anchor transition (z_t, a_t, z_{t+1}) |
| `action_displacement` | ‖a_G − a_t‖ |
| `anchor_model_error` | ‖z_t + f̂(z_t, a_t) − z_{t+1}‖ |
| `solver_error` | ‖T̂(q) − y‖, with the learned T̂ = z + f̂ |

**Numerical upper bounds.** The displacements are computed as upper bounds on
the exact norms of the stored values. The float64 forward error is added and
the result rounded up ([`src/ccil/core/verified.py`](../src/ccil/core/verified.py)).

The two model terms are upper bounds only when the model bounds its own
evaluation error (`delta_error_bound`). The float64 linear model does; the
descriptor then records `model_evaluation: bounded`. Otherwise it records
`assumed_exact`, and a profile whose total uses those terms is at most
`conditional`.

Totals are rounded up to enclose the float64 arithmetic of each formula.

**Demonstration residual.** Every bound below relies on a demonstration
transition. Stored transitions are exact only up to rounding, so a profile
that relies on one adds

    ρ_t = ‖S_z⁻¹ (T(z_t, a_t) − z_{t+1})‖

once to its total, as the required term `demonstration_residual`. It comes
from one of two specifications:
- `Provided("reference_residual", <reference>)` computes it per row from
  reference dynamics, as a verified upper bound, so it can be certified.
- `Declared(value, rationale)` bounds every ρ_t by the value, which makes the
  rows conditional.

**Profile terms.** A profile may declare more terms. It provides a computation
for every row, and a per-row recomputation that the source audit uses. A
profile also names the terms it *requires*: a row is supported only when its
required terms and its total are finite and the label lies in the profile's
neighborhood. A non-finite term that the profile does not require never makes
a row unsupported.

**Forms.** A profile declares one of three:

| Form | Total |
|---|---|
| **Additive** | `total = solver_error + model_error`, by ‖T(q) − y‖ ≤ ‖T̂(q) − y‖ + ‖f(q) − f̂(q)‖ |
| **Direct** | The profile bounds ‖T(q) − y‖ itself, so `model_error_value` is NaN. `solver_error` is still recorded as a diagnostic and is never added a second time. |
| **Score** | An empirical score, not a bound. `model_error_value` is NaN. |

## Constants are specified, never asserted

A profile's constants are given as one of two specifications:

| Specification | Meaning | Classification |
|---|---|---|
| `Declared(value, rationale)` | An assumption | Always `conditional`, whatever its value |
| `Provided(provider[, reference])` | Computed during certification by a registered bound provider, from artifacts whose SHA-256 the certificate records | The provider's source. An `analytic_bound` is verified in exact arithmetic and is `certified`. A `model_bound` holds for the model in exact real arithmetic and is `conditional`. |

A raw value cannot carry a bound source. The API refuses anything else, and
the schema refuses a provided source without provenance or a declared
constant with provenance. The source audit recomputes every provided
constant.

Each profile role says what its constant bounds: the true f, the learned f̂,
or the discrepancy f − f̂. A provider must bound the same thing.

**Registered providers.** A reference is a `ccil.reference-dynamics` 1.0 file
holding a known linear system f = A z + B a (see `write_reference_dynamics`).

| Provider | Source | Bounds | Value (state; action) |
|---|---|---|---|
| `linear_discrepancy` | `analytic_bound` | f − f̂ | ‖S_z⁻¹ (A − Â) S_z‖₂; ‖S_z⁻¹ (B − B̂) S_a‖₂, for a reference and a linear model |
| `reference_lipschitz` | `analytic_bound` | the true f | ‖S_z⁻¹ A S_z‖₂; ‖S_z⁻¹ B S_a‖₂, for a reference |
| `model_lipschitz` | `analytic_bound` for a linear model, `model_bound` for an MLP | f̂ | The model's own `lipschitz_bound`: the same verified norm of Â or B̂, or the MLP bound below |
| `reference_residual` | `analytic_bound` | the demonstration | ρ_t per row, for a reference |

**Verified norms.** An `analytic_bound` norm is a float s for which
s²I − MᵀM is proven positive semidefinite in exact rational arithmetic, on the
exact values of the stored floats. That makes s an upper bound on ‖M‖₂, not an
estimate of it. The audit verifies the persisted value again.

**The MLP bound.** f̂ = M (out_scale + 1e-8) g(x) + out_shift, where x is the
normalized input and g = W_L σ(… σ(W_1 x + b_1) …) + b_L. In scaled
coordinates:

    K ≤ ‖S_z⁻¹ M D_out W_L‖ · ‖W_{L−1}‖ ⋯ ‖W_2‖ · ‖W_1[:, cols] D_in S‖ · L_σ^{L−1}

- **Scales.** D_in = 1 / (scale + 1e-8) and D_out = out_scale + 1e-8, as the
  float32 forward pass computes them. M is the output mask.
- **Activation.** σ is ReLU or tanh, so L_σ = 1. Any other activation is
  refused.
- **Norms.** Every norm is an exact spectral norm (an SVD in float64) of the
  weight the eval-mode forward pass applies. For spectral normalization that
  weight is `weight_orig / max(σ̂ / c, 1)` with the stored u and v. Neither
  the power-iteration estimate σ̂ nor the constraint c is ever taken as a
  bound, because a stale estimate leaves a layer above c.
- **Classification.** The bound holds for the network in exact real
  arithmetic. Neither the network's float32 evaluation nor the float64
  rounding of its norms is enclosed, so it is a `model_bound` and
  `conditional`. MLP model terms are likewise `assumed_exact`.

## Classification and admission

Each row's classification is the weakest of what it relies on.

| Classification | When | `is_upper_bound` |
|---|---|---|
| `certified` | Every constant is an `analytic_bound`, the demonstration residual is provided, and any model terms the total uses are `bounded`. | true |
| `conditional` | Something rests on an assumption: a declared constant or demonstration bound, a `model_bound`, or `assumed_exact` model terms. | true, under that assumption |
| `calibrated` | Reserved for #92. | true, at its stated confidence |
| `empirical` | A score profile. | false |
| `unsupported` | A required term or the total is not finite, or the label is outside the neighborhood. | false |

**Admission** is a policy, chosen explicitly, and separate from
classification:

| Mode | Admits |
|---|---|
| `upper_bound` (default) | Bound rows with `total_error_value ≤ threshold` (when a threshold is set). Supported rows that are not bounds get `certificate.not_upper_bound`. |
| `score` | Any supported row with `total_error_value ≤ threshold`. Its rows keep their classification and `is_upper_bound`, so a score is never presented as a bound. |

`require_upper_bound(cert)` raises unless admission is `upper_bound` and
every supported row is a bound.

**Selection** is a budget choice:
- With a `selection_quantile`, the selected rows are the accepted, admissible
  rows whose total is at most that quantile of their totals. If no row is
  eligible, none is selected and the threshold is null.
- Without one, every row is selected.

Selection never changes a candidate's acceptance or a row's classification.

**Reasons** are independent:

| Reason | Set when |
|---|---|
| `certificate.unsupported` | The row is unsupported. |
| `certificate.not_upper_bound` | Under `upper_bound` admission, a supported row is not a bound. |
| `certificate.exceeds_threshold` | A row that the admission mode considers is over the threshold. |
| `certificate.not_selected` | An eligible row is not selected. |

## Profiles

**`continuity` (additive).**

    model_error = anchor_model_error + ρ_t + C_z ‖Δz‖ + C_a ‖Δa‖

C_z and C_a bound the variation of f − f̂ over the neighborhood. When C comes
from separate bounds on the true and learned dynamics, it is
C = K_true + K_learned. Since ‖(f − f̂)(z_t, a_t)‖ ≤ anchor_model_error + ρ_t,
the bound needs no exact demonstration. Rows outside the optional
neighborhood radii are unsupported. Version 2 added ρ_t.

**`backtrack` (Ke et al., Thm 4.2; additive).** It certifies the labels of
the backtrack generators (`backward_euler`, `exact_preimage`).

    model_error = anchor_model_error + ρ_t + (K̂_z + K_z) ‖Δz‖

K̂_z bounds the learned f̂ and K_z the true f, both in state. A BackTrack
label keeps the anchor's action, so no action constant enters. A row whose
action differs from its anchor's is outside the neighborhood, and
unsupported. Because K̂_z + K_z ≥ ‖A − Â‖, it is never tighter than
`continuity` with the discrepancy constant.

**`disturbed_action` (direct).** This is a direct label-error bound that
follows from the proof of Ke et al., Thm 4.3. The theorem itself is stated for
the model discrepancy. It certifies the labels of the disturbed-action
generators (`noisy_action`, `noisy_action_preimage`).

    total = K_a ‖Δa‖ + (1 + K_z) ‖Δz‖ + ρ_t + target_offset

K_z and K_a bound the true f. Since
T(q) − y = Δz + f(z_G, a_G) − f(z_t, a_t) + (T(z_t, a_t) − z_{t+1}) + (z_{t+1} − y),
the bound needs no learned model:
- `solver_error` is recorded, never added;
- `target_offset` = ‖y − z_{t+1}‖ is a profile term, and is 0 for these
  generators.

**Score profiles (empirical).** These are point-Jacobian scores of f̂ at the
anchor, from the model's exact `jacobian_za`, never finite differences.
Without exact Jacobians the scores are NaN and the rows unsupported, never 0.
They assume a small anchor model error, which `generation.model_error_threshold`
enforces at generation. Every row is `empirical` with
`is_upper_bound = false`, `upper_bound` admission admits none, and
`require_upper_bound` refuses them.

| Profile | Score | Labels |
|---|---|---|
| `backtrack_paper_proxy` | ‖J_S f̂‖ ‖Δz‖ (Ke et al., Def. 4.5) | backtrack |
| `disturbed_action_paper_proxy` | ‖J_A f̂‖ ‖Δa‖ + (1 + ‖J_S f̂‖) ‖Δz‖ (Def. 4.6) | disturbed action |
| `legacy_label_filtering` | ‖J_{z+f̂}‖ ‖(Δz, Δa)‖, the label-filtering branch's `label_error` (73d187d), opt-in for parity | any |

A profile names the generator family it applies to, and certifying another
family's candidates raises. A profile is a `Profile` subclass registered with
`register_profile`.

**Which profile for which method.**

| Generation method | Bound | Paper score | Legacy score |
|---|---|---|---|
| `backward_euler`, `backward_euler_fast`, `forward_euler` | `backtrack` (or `continuity`) | `backtrack_paper_proxy` | `legacy_label_filtering` |
| `noisy_action`, `noisy_action_fe` | `disturbed_action` (or `continuity`) | `disturbed_action_paper_proxy` | `legacy_label_filtering` |

## Verification

| Level | Uses | Checks |
|---|---|---|
| **Structural** (`verify_structure`) | The certificate and its candidates file | The schema, the candidates' SHA-256 and row identity, and the registered profile and version. The descriptor must be exactly what the profile writes for its constants and coordinates, so edited units are caught. Every provider must be registered and match its claimed source. Every value, classification, mask and threshold is recomputed from the persisted terms. It cannot tell whether the terms or provided constants are right. |
| **Source audit** (`audit_sources`) | Also the sources (`Sources`: the candidates, dataset, transition model, representation, any other inputs, and references) | Every supplied file's SHA-256 must match the recorded one, and nothing may be missing or extra. The representation's kind, version and modality and the model's kind and version must match the record, as must the dimensions. It then recomputes every term one row at a time, including profile terms. Each term must match to a row-local tolerance: 64 ulps of that row's operand magnitude, in the precision the term was evaluated in (float32 for an MLP's model terms, float64 otherwise). An extreme state elsewhere therefore never loosens a row. Provided constants are recomputed, with identity fields equal and values within 1e-9 relative, since float64 norms differ by ulps across BLAS. Every `analytic_bound` value is verified again exactly. |

Hashes establish artifact identity, not mathematical correctness.

## The `labels.certify` stage

`ccil.pipelines.offline.certify_stage(cfg, prepared, fitted, labeled,
representation, profile, admission="upper_bound", threshold=None,
selection_quantile=None, references=None, state_scale=None,
action_scale=None)`.

- **Inputs:** the dataset, the transition model checkpoint, the labels
  stage's candidates, and each reference as `reference.<name>`.
- **Identity:** the profile's specification, the admission, the selection
  and the scales are part of the stage identity, so a changed constant
  specification, threshold or reference produces a new run.
- **Metrics:** counts by classification, and the admissible, selected and
  usable rows.

**Configuration.** With `certificate.profile` set, `ccil labels generate` runs
`labels.certify` after `labels.generate`. The parameters are listed in
[`config/parameters.md`](config/parameters.md).

| Key | Use |
|---|---|
| `certificate.profile` | The profile, or null for no certificate |
| `certificate.reference_dynamics` | True dynamics: `reference_lipschitz` true constants, `reference_residual` demonstration residuals, and `linear_discrepancy` for continuity with a linear model |
| `certificate.true_state_lipschitz`, `certificate.true_action_lipschitz` | Declared true constants, without a reference |
| `certificate.demonstration_residual_bound` | Declared bound on every demonstration residual, without a reference |
| `certificate.learned_lipschitz_source`, `certificate.learned_state_lipschitz` | `model` (default) uses `model_lipschitz`; `declared` uses the value |
| `certificate.discrepancy_state_lipschitz`, `certificate.discrepancy_action_lipschitz` | Declared continuity constants, without an exact discrepancy |
| `certificate.neighborhood_state`, `certificate.neighborhood_action` | Neighborhood radii |
| `certificate.max_total_error` | Bound profiles: `upper_bound` admission threshold |
| `certificate.score_threshold` | Score profiles: `score` admission threshold |
| `certificate.selection_quantile` | Budget selection |

**Consistency rules.** The configuration rule `certificate` refuses:
- a profile for the other generator family;
- a score profile with constants, a reference, or `max_total_error`;
- a bound profile with `score_threshold`;
- a bound profile without a reference or declared demonstration bound;
- missing or unused declared constants;
- `certificate.*` settings without a profile.

The integrated pipeline refuses `certificate.profile` until its policy stage
consumes certificates (#89).

**Migration.**
- Legacy `aug.max_label_err` and `aug.label_err_quantile` migrate to
  `certificate.profile: legacy_label_filtering`, with
  `certificate.score_threshold` and `certificate.selection_quantile`.
- `aug.model_err_thresh` stays `generation.model_error_threshold`.
- `generation.max_label_error` and `generation.label_error_quantile` are
  withdrawn, and fail with guidance.
- The `LabelError` validator is removed. Its definitions are the paper
  proxies.

## Evidence

```bash
uv run --group evidence python tools/m3_certificates/check.py --out <new-dir>
```

The report is
[`fixtures/evidence/m3-certificates/report.json`](fixtures/evidence/m3-certificates/report.json),
with `report-siddh.json` from the CUDA host. It uses the analytic double
integrator, with the true label error computed independently from the true
dynamics.

- **Oracle coverage.** Provider constants give a certified, tight bound with
  the exact model. With a wrong linear model, every row of both generators is
  covered, in unit and non-unit scales.
- **Asserted values.** A raw value cannot become a bound. A declared zero is
  conditional and fails coverage. A forged certified zero passes the
  structural check but not the audit.
- **Neighborhood.** Labels outside it are unsupported.
- **Admission.** Admission is separate from classification, for an empirical
  test profile under both modes.
- **Masks.** They follow their rules, including a quantile with no eligible
  row.
- **Profile terms.** They round-trip and are audited. Only required terms
  gate support.
- **Two verification levels.** They detect different corruptions, and the
  audit binds input hashes and component identity.
- **Schema.** The reader and writer enforce it, each malformed case failing on
  its intended rule.
- **API.** Malformed constants, policies and scales are refused.
- **Pipeline.** The pipeline writes a complete, covered and audited
  `labels.certify` run next to unchanged candidates, and reuses or re-runs it
  correctly.
- **Row-local audit.** With one state at 10⁶ elsewhere in the dataset, a
  10⁻⁶ corruption of another row's term is rejected. Its tolerance is
  5·10⁻¹⁶, where the previous dataset-wide tolerance was 1.0.

The theorem-profile cases are in
[`tools/m3_certificates/theorems.py`](../tools/m3_certificates/theorems.py).
Every bound includes ρ_t, and coverage is asserted directly, as
true error ≤ total with no tolerance.

- **`backtrack`.** It is certified and covered for both backtrack generators.
  It has no action term: a model with 100× the action sensitivity gets the
  same K̂_z, and rows that move the action are unsupported.
- **`disturbed_action`.** It is certified and covered for both
  disturbed-action generators. Its total equals the direct formula, and
  adding the solver error would inflate it by up to 47%. Declared zero
  constants fail coverage.
- **MLP `model_lipschitz`.**
  - It covers every sampled exact Jacobian and finite difference of ReLU,
    tanh, spectral-normalized and output-masked MLPs, in unit and non-unit
    scales.
  - For actions in milli-units, the raw weight product falls short of the
    sampled action sensitivity.
  - A stale power-iteration estimate leaves layers above the constraint, so
    the constraint-based bound is broken while the exact bound covers.
  - An unsupported activation is refused.
- **MLP models.** `backtrack`, which uses f̂'s bound and terms, is
  `conditional`. `disturbed_action`, which uses no learned quantity, stays
  `certified`. Both are covered and pass the structural check and the audit.
- **Provider kinds and families.** Mismatched provider kinds, forged
  provider kinds, and the wrong generator family are refused. The pipeline
  writes complete runs for both theorem profiles.

Part 3's cases are in
[`tools/m3_certificates/proxies.py`](../tools/m3_certificates/proxies.py).

- **Parity.** The paper proxies equal Defs. 4.5 and 4.6. `legacy_label_filtering`
  equals a transcription of the branch's `label_error`, for a linear model
  and an MLP.
- **Scores are never bounds.** Score admission filters by `score_threshold`.
- **Selection.** It never changes acceptance. Rows rejected by the
  model-error gate are never selected, and the quantile is over accepted,
  admissible rows.
- **Unresolved derivatives.** They give unsupported rows, never 0.
- **Configuration.** Migration, the withdrawn keys, the consistency rules,
  and the integrated pipeline's refusal all behave as described.
- **Pipeline.** `ccil labels generate` with certificate settings writes
  complete runs, and its candidates are read unchanged by the reader that
  predates #88 (`8663348`).

**Cross-host audit.** `check.py` also writes `bundle/`: each certificate with
copies of its sources and a manifest. Its report lists the SHA-256 of every
bundle file.

```bash
uv run --group evidence python tools/m3_certificates/audit_bundle.py --bundle <bundle> --out <new-dir>
```

Each host audits the other's bundle:
- `audit-siddh-of-mac.json` is the CUDA host auditing the bundle behind
  `report.json`;
- `audit-mac-of-siddh.json` is the reverse, for `report-siddh.json`.

The SHA-256 in each audit report match the generating report's
`bundle_sha256`, so the same bytes were audited.
