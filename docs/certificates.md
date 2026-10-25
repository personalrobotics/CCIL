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
| `Provided(provider, reference)` | Computed during certification by a registered bound provider, from artifacts whose SHA-256 the certificate records | The provider's source: `analytic_bound` or `model_bound`, both `certified` |

A raw value cannot carry a bound source. The API refuses anything else, and
the schema refuses a provided source without provenance or a declared
constant with provenance. The source audit recomputes every provided
constant.

**Registered provider.** `linear_discrepancy` (`analytic_bound`). It takes a
`ccil.reference-dynamics` 1.0 file (true linear dynamics A, B; see
`write_reference_dynamics`) and a linear transition model. It returns the
exact operator norms of the scaled differences:
- state: ‖S_z⁻¹ (A − Â) S_z‖₂;
- action: ‖S_z⁻¹ (B − B̂) S_a‖₂.

Providers for neural models (`model_bound`) arrive with the theorem profiles.

## Classification and admission

Each row's classification is the weakest of what it relies on.

| Classification | When | `is_upper_bound` |
|---|---|---|
| `certified` | Every constant is provided. | true |
| `conditional` | Some constant is declared. | true, under that assumption |
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

    model_error = anchor_model_error + C_z ‖Δz‖ + C_a ‖Δa‖

C_z and C_a bound the variation of f − f̂ over the neighborhood. When C comes
from separate bounds on the true and learned dynamics, it is
C = K_true + K_learned. The bound also assumes that the demonstration
transition is exact. Rows outside the optional neighborhood radii are
unsupported.

The CCIL theorem profiles (`backtrack`, `disturbed_action`) and the paper's
proxies follow in later parts of #88. A profile is a `Profile` subclass
registered with `register_profile`.

## Verification

| Level | Uses | Checks |
|---|---|---|
| **Structural** (`verify_structure`) | The certificate and its candidates file | The schema, the candidates' SHA-256 and row identity, and the registered profile and version. The descriptor must be exactly what the profile writes for its constants and coordinates, so edited units are caught. Every provider must be registered and match its claimed source. Every value, classification, mask and threshold is recomputed from the persisted terms. It cannot tell whether the terms or provided constants are right. |
| **Source audit** (`audit_sources`) | Also the sources (`Sources`: the candidates, dataset, transition model, representation, any other inputs, and references) | Every supplied file's SHA-256 must match the recorded one, and nothing may be missing or extra. The representation's kind, version and modality and the model's kind and version must match the record, as must the dimensions. It then recomputes every term per row (including profile terms) and every provided constant. |

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

Configuration keys (`certificate.*`) arrive with the theorem profiles.

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
