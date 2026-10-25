# Label-error certificates

Certification is a separate step after label generation (#88):

    generate candidates → certify candidates → select / admit labels → train policy

It is implemented in
[`src/ccil/core/certificates.py`](../src/ccil/core/certificates.py) and run by
the `labels.certify` stage. The candidates file is an input and is never
modified. Training uses the rows where

    usable = candidate.accepted ∧ certificate.admissible ∧ certificate.selected

## What a certificate claims

For a label (q, y), with q = (z_G, a_G), the quantity of interest is the true
label error ‖T(q) − y‖, where T(z, a) = z + f(z, a) is the true dynamics. Every
certificate row persists the terms it used:

| Term | Value |
|---|---|
| `state_displacement` | ‖z_G − z_t‖ from the anchor transition (z_t, a_t, z_{t+1}) |
| `action_displacement` | ‖a_G − a_t‖ |
| `anchor_model_error` | ‖z_t + f̂(z_t, a_t) − z_{t+1}‖ |
| `solver_error` | ‖T̂(q) − y‖, with the learned T̂ = z + f̂ |

A row also persists `model_error_value`, `total_error_value`, `classification`,
`is_upper_bound`, `in_neighborhood`, the `admissible` and `selected` masks, and
its reasons.

A profile declares one of two derivation forms:

| Form | Total |
|---|---|
| **Additive** | `total = solver_error + model_error`, by the triangle inequality ‖T(q) − y‖ ≤ ‖T̂(q) − y‖ + ‖f(q) − f̂(q)‖ |
| **Direct** | The profile bounds ‖T(q) − y‖ itself, so `model_error_value` is not used (NaN). `solver_error` is still recorded as a diagnostic and is never added a second time. |

## Classification

Each row's classification is the weakest of what it relies on.

| Classification | Meaning | `is_upper_bound` |
|---|---|---|
| `certified` | Every constant is an `analytic_bound` or a proven `model_bound`, over the declared neighborhood. | true |
| `conditional` | Some constant is `declared`: an assumption, with its rationale. | true, under that assumption |
| `calibrated` | Reserved for #92. | true, at its stated confidence |
| `empirical` | A score: for example, point Jacobians in place of constants. | false |
| `unsupported` | A non-finite term, or a label outside the neighborhood. | false |

A score is never presented as a bound. Only bound rows can be admissible.

Every constant records its value, source, rationale and units. The sources are
`analytic_bound`, `model_bound`, `declared` and `point_jacobian`.

## Profiles

**`continuity` (additive).**

    model_error = anchor_model_error + C_z ‖Δz‖ + C_a ‖Δa‖

C_z and C_a bound the variation of f − f̂ over the neighborhood. When C comes
from separate bounds on the true and learned dynamics, it is
C = K_true + K_learned. Rows outside the neighborhood radii are unsupported.
The bound also assumes that the demonstration transition is exact. A point
Jacobian is not a constant over a neighborhood, so `continuity` refuses one.

The CCIL theorem profiles (`backtrack`, `disturbed_action`) and the paper's
empirical proxies follow in later parts of #88.

## Masks and reasons

**`admissible`.** The row is a bound, and `total_error_value ≤ max_total_error`
when that threshold is set.

**`selected`.** A budget choice. When `selection_quantile` is set, a row is
selected if it is accepted, admissible, and its total is at most that quantile
of the totals of accepted, admissible rows. When the quantile is not set,
every row is selected. Selection never changes a candidate's acceptance or a
row's classification.

**Reasons.** They are independent, and namespaced so they cannot be confused
with candidate rejection reasons:

| Reason | Set when |
|---|---|
| `certificate.unsupported` | The row is unsupported. |
| `certificate.exceeds_bound` | A bound row is over `max_total_error`. |
| `certificate.not_selected` | An eligible row is not selected. |

## Verification

| Level | Uses | Checks |
|---|---|---|
| **Structural** (`verify_structure`) | The certificate and its candidates | The candidates' SHA-256 and row identity, then every value, classification, mask and threshold, recomputed from the persisted terms and descriptor. It cannot tell whether the terms themselves are right. |
| **Source audit** (`audit_sources`) | Also the dataset, representation and transition model | Recomputes every source term per row, independently of the batch path, and compares it with the persisted terms. |

Hashes establish artifact identity, not mathematical correctness.

## The `ccil.certificates` 1.0 file

`labels/certificates.npz` has one row per candidate row, in the same order,
with `source_episode_id` and `source_frame` repeated for joining.

| Field | Contents |
|---|---|
| `term/<name>` | The four terms above |
| `model_error_value`, `total_error_value` | float64 |
| `classification` | int8 codes into `meta.classification_codes` |
| `is_upper_bound`, `in_neighborhood`, `admissible`, `selected` | bool |
| `reasons` | bool, (N, 3), in the order of `meta.reason_codes` |

The meta records the profile `descriptor`: the profile and version, the form,
constants, neighborhood, norms, units and assumptions. It also records the
`thresholds` (`max_total_error`, `selection_quantile`, the resulting
`selection_threshold`), the candidates' SHA-256, and the SHA-256 of every stage
input.

The reader rejects:
- other major versions;
- missing fields;
- unknown classification or reason codes;
- out-of-range codes;
- arrays that disagree on the row count.

## The `labels.certify` stage

`ccil.pipelines.offline.certify_stage(cfg, prepared, fitted, labeled,
representation, profile, max_total_error=None, selection_quantile=None)`.

- **Inputs:** the dataset, the transition model checkpoint, and the labels
  stage's candidates.
- **Identity:** the profile's descriptor and the thresholds are part of the
  stage identity, so a changed constant produces a new run.
- **Metrics:** counts by classification, and the admissible, selected and
  usable rows.

Configuration keys (`certificate.*`) arrive with the theorem profiles.

## Evidence

```bash
uv run --group evidence python tools/m3_certificates/check.py --out <new-dir>
```

The report is
[`fixtures/evidence/m3-certificates/report.json`](fixtures/evidence/m3-certificates/report.json),
with `report-siddh.json` from the CUDA host. It covers the following, on the
analytic double integrator:

- **Oracle coverage.** The true label error is computed independently from the
  true dynamics.
  - With an exact model and C = 0, every row is certified and tight.
  - With a wrong linear model and C = ‖A − Â‖, ‖B − B̂‖, every row is
    covered, for both generators.
- **Underestimated constant.** An underestimated declared constant fails
  coverage.
- **Neighborhood.** Labels outside it are unsupported.
- **Masks and acceptance.** The masks follow their rules, and
  `candidate.accepted` is unchanged.
- **Two verification levels.** Structural recomputation detects arithmetic,
  classification, mask, descriptor and candidate-file corruptions. Only the
  source audit detects a falsified term that is consistent with the totals.
- **Reader.** Malformed files are rejected.
- **Pipeline.** The pipeline writes a complete `labels.certify` run beside the
  unchanged candidates. A rerun is reused, and a changed constant produces a
  new run.
