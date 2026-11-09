# Label-error certificates

Certification is a separate step after label generation (#88):

    generate candidates → certify candidates → curate a selection → train policy

It is implemented in
[`src/ccil/core/certificates.py`](../src/ccil/core/certificates.py) and run by
the `labels.certify` stage. The candidates file is an input and is never
modified. A certificate is evidence only: which rows to train on is a separate
`ccil.selection`, written by the `labels.curate` stage
([Curation](#curation)). Training uses the rows where

    usable = candidate.accepted ∧ selection.mask

[supervision.md](supervision.md) defines how a trainer consumes these rows.

The file formats, `ccil.certificates` 2.0 and `ccil.selection` 1.0, are
specified in [`schemas/reference.md`](schemas/reference.md). Version 1.0
certificates, which stored their own admission and selection, are still read
and verified; training on them uses `accepted ∧ admissible ∧ selected`.

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
| `calibrated` | An accepted `empirical` row outside a calibration's episodes, made so by curation with that calibration (below); certification never writes it. | true, as a coverage statement at the calibration's level |
| `empirical` | A score profile. | false |
| `unsupported` | A required term or the total is not finite, or the label is outside the neighborhood. | false |

**Reasons** are independent:

| Reason | Set when |
|---|---|
| `certificate.unsupported` | The row is unsupported. |
| `certificate.not_upper_bound` | A supported row is not a bound. |

`require_upper_bound(cert)` raises unless every supported row is a bound.

## Curation

A curation policy ([`src/ccil/core/curate.py`](../src/ccil/core/curate.py))
reads the candidates and their certificates and writes a `ccil.selection`. It
modifies neither, so a new tolerance writes a new selection while both hashes
stay the same.

| Policy (`admit`) | Selects |
|---|---|
| `accepted` | Every accepted row, without certificates. Recorded as uncertified. |
| `bound` | Accepted bound rows (certified, conditional, calibrated) with `total_error_value ≤ tolerance`, when a tolerance is set. Supported rows that are not bounds get `selection.not_bound`. |
| `score` | Accepted supported rows with `total_error_value ≤ tolerance`. The selection records that its error is not a bound. |

- **Admission follows the profile:** the pipeline curates a bound profile's
  certificates with `bound` and a score profile's with `score`.
- **Quantile:** with `quantile`, the selected rows are the admitted rows whose
  total is at most that quantile of a population:
  - `admitted` (the default): the accepted, admitted rows;
  - `produced`: every supported row a legacy run would have generated (not
    rejected for generation failure, non-finite values or model misfit),
    independently of the tolerance and the support radius, as the
    label-filtering branch took it. Migrated legacy configurations use it.

  If no row is in the population, none is selected and the threshold is null.
- **Fail closed:** unsupported rows are never selected. Curation never changes
  acceptance or classification, and a lower tolerance or quantile selects a
  subset.
- **Per row,** the selection records the raw error and its units, the ratio
  error / tolerance, the rank of the error within the named population, and
  every reason against the row: `selection.not_accepted`, `unsupported`,
  `not_bound`, `exceeds_tolerance`, `not_selected`. The tolerance is strictly
  positive. A ratio of at most one means exactly that the recorded error is
  within the tolerance; it is a bound only when the error is. The ratio is NaN
  exactly where there is no tolerance or no finite error.
- **Caveats:** every policy here compares one quality term, the error, so the
  selection records that fidelity alone favours labels close to the
  demonstrations.
- **Structure first:** curation verifies the certificate's structure against
  the candidates (`verify_structure`) before it decides anything, so a
  certificate whose totals, classification or reasons do not follow from its
  terms never produces a selection.
- **Evidence mode:** constants or demonstration residuals computed from
  reference dynamics are `reference_assisted` evidence. `generation.evidence_mode`
  is the allowance: under `data_only`, the default, curation refuses such
  evidence. The selection records both the allowance (`evidence_allowed`) and
  the provenance of what it used (`evidence_mode`), which is `data_only`
  whenever the certificate is data-only or absent
  ([contract.md](contract.md#reference-dynamics)).
- **Calibration:** with a calibration of the same certificate file, a `bound`
  policy also admits its calibrated rows, at their calibrated bound (next
  section). The selection, then `ccil.selection` 1.1, records the
  calibration's SHA-256 and marks each calibrated row.
- **Verification:** `verify_selection` verifies the certificate's structure,
  then recomputes every mask, weight, error, ratio, rank, reason, threshold,
  calibrated row and the evidence provenance from the selection's inputs.

## Calibration and the reference audit

Both read reference dynamics (`ccil.reference-dynamics`), so both are
declared evidence ([contract.md](contract.md#reference-dynamics)):
[`src/ccil/core/calibrate.py`](../src/ccil/core/calibrate.py).

**Calibration** (`calibrate(candidates, certificates, reference, episodes,
level)`, `ccil.calibration` 1.0) gives one bound on the label error observed
under the reference, ‖(T(z_G, a_G) − y) / state_scale‖, in the certificate's
units, for a score profile's rows.
- **Eligible rows:** accepted rows the certificate classifies `empirical`.
- **Unit: the episode.** Rows of one trajectory, and the labels of one
  transition, are not independent, so rows are not the conformal unit. Each
  calibration episode with an eligible row is scored by the largest observed
  error over those rows. The pipeline calibrates on the split's held-out
  episodes.
- **Bound:** the k-th smallest of the n episode scores,
  k = ⌈(n + 1) · level⌉, the split-conformal rule, or none when k > n. At
  level 0.9 that needs at least 9 held-out episodes with eligible rows.
- **What it guarantees.** Two conditions must hold:
  - the calibration episodes are a random draw from the dataset's
    episodes, independent of the labels;
  - the labels and certificate do not depend on the draw, as with
    `transition_model.fit_episodes all`, which the configuration requires.

  Then the expected fraction of the other episodes whose every eligible row
  has true error at most the bound is at least `level`.
  - The probability is over the draw. It is not conditional on one split, a
    row or a score.
  - It covers every eligible row of an episode at once, so it holds for any
    subset of them, including the rows curation selects.
  - It is not a worst-case bound. It does not transfer to other datasets,
    tasks, model families or certificate files.
  - With `fit_episodes train` the label model is fitted on the rows the
    bound is applied to but not on the calibration rows, so the two errors
    are not exchangeable.
- **Which rows become calibrated:** an eligible row of the same certificate
  file whose episode is not a calibration episode, when the bound exists.
  Every other row keeps its class: the calibration's own episodes,
  ineligible rows, and every row of another certificate file.
- **Curation:**
  - A calibration needs `admit bound` and a `reference_assisted` allowance.
    `data_only` curation refuses it, in the library and in the configuration.
  - Curate first recomputes everything in the calibration that does not
    depend on the reference: the certificate, population, n and k. A
    calibration that disagrees with its certificate is refused.
  - `verify_calibration` recomputes the observed errors from the reference.
- **Supervision:** `build_supervision` needs the calibration of a calibrated
  selection, verifies the selection with it, and classifies those rows
  `calibrated`.
- **What it does not establish:** that the calibration transfers to another
  task, model family or certificate file.

**The reference audit** (`reference_audit(selection, candidates, reference,
certificates, calibration)`, `ccil.reference-audit` 1.0) verifies a selection
against its inputs, then records the observed error beside it:
- the selected and the accepted-but-unselected rows' error quantiles;
- where the error is a bound, how many selected rows exceed it;
- with a tolerance, how many selected rows lie within it;
- on calibrated rows, how many the bound covers, by row and by episode. This
  is an observation of one draw, not evidence for the guarantee.

It is never an input to admission, generation, evidence or selection, so it
is allowed in `data_only` mode.

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
| `backtrack_paper_proxy` | ‖J_S f̂‖ ‖Δz‖ (Ke et al., Def. 4.5). It has no action term, so a row whose action differs from its anchor's is unsupported, and it takes no action neighborhood. | backtrack |
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
| **Structural** (`verify_structure`) | The certificate and its candidates file | The schema, the candidates' SHA-256 and row identity, and the registered profile and version. The descriptor must be exactly what the profile writes for its constants and coordinates, so edited units are caught. Every provider must be registered and match its claimed source. Every value and classification (and, for a 1.0 file, every mask and threshold) is recomputed from the persisted terms. It cannot tell whether the terms or provided constants are right. |
| **Source audit** (`audit_sources`) | Also the sources (`Sources`: the candidates, dataset, transition model, representation, any other inputs, and references) | Every supplied file's SHA-256 must match the recorded one, and nothing may be missing or extra. The representation's kind, version and modality and the model's kind and version must match the record, as must the dimensions. It then recomputes every term one row at a time, including profile terms. Each term must match to a row-local tolerance: 64 ulps of that row's operand magnitude, in the precision the term was evaluated in (float32 for an MLP's model terms, float64 otherwise). An extreme state elsewhere therefore never loosens a row. Provided constants are recomputed, with identity fields equal and values within 1e-9 relative, since float64 norms differ by ulps across BLAS. Every `analytic_bound` value is verified again exactly. |

Hashes establish artifact identity, not mathematical correctness.

## The `labels.certify` stage

`ccil.pipelines.offline.certify_stage(cfg, prepared, fitted, labeled,
representation, profile, references=None, state_scale=None,
action_scale=None)`.

- **Inputs:** the dataset, the transition model checkpoint, the labels
  stage's candidates, and each reference as `reference.<name>`.
- **Identity:** the profile's specification and the scales are part of the
  stage identity, so a changed constant specification or reference produces a
  new run. A changed tolerance does not.
- **Metrics:** counts by classification.

`curate_stage(cfg, labeled, certified, policy, calibrated=None)`
(`labels.curate`) follows it and writes `labels/selection.npz`. Its identity
is the policy and its input hashes; its metrics are the selected and
calibrated rows and the count of each reason.

Two optional stages read reference dynamics:
- **`labels.calibrate`** (`calibrate_stage`, with `calibration.level`) runs
  between certification and curation and writes `labels/calibration.json`.
  It needs:
  - a score profile;
  - `certificate.reference_dynamics`;
  - `generation.evidence_mode reference_assisted`;
  - held-out episodes;
  - `generation.recovery_steps 1`: the diagnosis admits from data alone, and
    a score profile gives its trust gate no off-data evidence.

  The pipeline then curates that profile with `admit bound`.
- **`labels.audit`** (`audit_stage`, with `audit.reference_dynamics`) runs
  last and writes `labels/audit.json`. No other stage reads it, so adding it
  reuses every other run.

**Configuration.** With `certificate.profile` set, `ccil labels generate` runs
`labels.certify` after `labels.generate`; it always ends with `labels.curate`.
The parameters are listed in [`config/parameters.md`](config/parameters.md).

| Key | Use |
|---|---|
| `certificate.profile` | The profile, or null for no certificate |
| `certificate.reference_dynamics` | True dynamics: `reference_lipschitz` true constants, `reference_residual` demonstration residuals, and `linear_discrepancy` for continuity with a linear model |
| `certificate.true_state_lipschitz`, `certificate.true_action_lipschitz` | Declared true constants, without a reference |
| `certificate.demonstration_residual_bound` | Declared bound on every demonstration residual, without a reference |
| `certificate.learned_lipschitz_source`, `certificate.learned_state_lipschitz` | `model` (default) uses `model_lipschitz`; `declared` uses the value |
| `certificate.discrepancy_state_lipschitz`, `certificate.discrepancy_action_lipschitz` | Declared continuity constants, without an exact discrepancy |
| `certificate.neighborhood_state`, `certificate.neighborhood_action` | Neighborhood radii |
| `curate.tolerance` | The tolerance on the certificate's error |
| `curate.quantile` | Budget selection |
| `curate.population` | `admitted` (default) or `produced` (the legacy population) |
| `generation.evidence_mode` | `data_only` (default) or `reference_assisted`, required with `certificate.reference_dynamics` |

**Consistency rules.** The configuration rule `certificate` refuses:
- a profile for the other generator family;
- a score profile with constants or a reference;
- a bound profile without a reference or declared demonstration bound;
- missing or unused declared constants;
- `learned_lipschitz_source: declared` outside the `backtrack` bound profile;
- an action neighborhood for either BackTrack profile;
- `certificate.*` settings, including non-default choices, without a profile.

The rule `curate` refuses a tolerance or quantile without a profile, a
`produced` population without a quantile, and reference dynamics in
`data_only` mode.

The integrated pipeline refuses `certificate.profile` until its policy stage
consumes certificates (#89).

**Migration.**
- Legacy `aug.max_label_err` and `aug.label_err_quantile` migrate to
  `certificate.profile: legacy_label_filtering`, with `curate.tolerance` and
  `curate.quantile`. The quantile migrates with `curate.population: produced`,
  which reproduces the branch's selection exactly.
- In a `config_version` 1 or 2 file, `certificate.max_total_error` and
  `score_threshold` are read as `curate.tolerance`, `selection_quantile` as
  `curate.quantile` and `selection_population` as `curate.population`, each
  with a note. Such a file's `certificate.reference_dynamics` resolves
  `generation.evidence_mode` to `reference_assisted`. Version 3 files and the
  command line refuse the old keys with guidance.
- `aug.model_err_thresh` stays `generation.model_error_threshold`.
- `generation.max_label_error` and `generation.label_error_quantile` are
  withdrawn, and fail with guidance.
- The `LabelError` validator is removed. Its definitions are the paper
  proxies.

## Evidence

```bash
uv run --group evidence python tools/m45_quality/check.py --out <new-dir>
uv run --group evidence python tools/m3_certificates/check.py --out <new-dir>
uv run --group evidence python tools/m3_certificates/audit_bundle.py --bundle <bundle> --out <new-dir>
```

Reports: [`m45-quality/report.json`](fixtures/evidence/m45-quality/report.json), [`m3-certificates/report.json`](fixtures/evidence/m3-certificates/report.json).
Each case, and the contract it protects, is listed in the docstrings of [`tools/m45_quality`](../tools/m45_quality/), [`tools/m3_certificates`](../tools/m3_certificates/).
