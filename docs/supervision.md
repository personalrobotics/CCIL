# The supervision contract

> The user-facing workflow (Diagnose, Generate, Curate, Inspect) and its artifact vocabulary are specified in [contract.md](contract.md).

CCIL produces synthetic supervision. Its ordinary workflow turns
demonstrations into immutable artifacts:
- the prepared dataset;
- the candidate labels (`labels/candidates.npz`);
- optionally, their certificates (`labels/certificates.npz`);
- a selection (`labels/selection.npz`), written by one curation policy
  ([certificates.md](certificates.md#curation)).

Users train downstream policies with whatever trainer they choose. This
document defines exactly which rows such a trainer may use, and with which
weights. [`src/ccil/supervision.py`](../src/ccil/supervision.py) implements
it as `build_supervision`. It trains nothing and imports no trainer.

The vector-policy trainers that consume this contract moved to
il-supervision-benchmarks (#168): `ilsb policy train` and `ilsb.learners.vector`.
Using CCIL does not require them.

## Sources

`build_supervision(dataset, source, ...)` takes what the policy learns from:

| `source` | Rows |
|---|---|
| `demonstrations` | The demonstrations only |
| `demonstrations`, `perturbation="noise_bc"` | The demonstrations only, declared to be perturbed online by the consumer's trainer at every draw (NoiseBC, in il-supervision-benchmarks), so nothing synthetic is stored. |
| `demonstrations_plus_selection`, with a selection | The demonstrations, and every candidate with `accepted ∧ selection.mask`, at the selection's weights. With certificates, each row keeps its certificate classification, except that a calibrated selection's calibrated rows are `calibrated`. The manifest records the profile and the selection's policy, threshold, evidence mode and caveats, and the calibration's SHA-256 when there is one. |
| `demonstrations_plus_selection`, no selection or certificates | The demonstrations, and every candidate with `accepted`. Recorded as `certification: uncertified`, with each synthetic row classified `uncertified`. |
| `demonstrations_plus_selection`, legacy 1.0 certificates | The demonstrations, and every candidate with `accepted ∧ admissible ∧ selected`, the selection a 1.0 certificate stored. Recorded with `legacy: true`. |

The manifest's `method` records the arm as before: `bc`, `noise_bc` or `ccil`.

**Former names (#168).** The source names `bc`, `noise_bc` and `ccil`, and the
`method=` keyword, are refused with a `SupervisionError` naming the replacement.

| Former | Now |
|---|---|
| `bc` | `demonstrations` |
| `noise_bc` | `demonstrations`, `perturbation="noise_bc"` |
| `ccil` | `demonstrations_plus_selection` |

**Usable rows only.** A rejected or unselected candidate never appears,
whatever its weight. There is no weight-only path into the loss.

## Rows and weights

**Demonstration rows** are the transition frames t ≤ T − 2 of the training
episodes: (z_t, a_t), with z_t encoded by the representation. These are the
rows legacy `load_data` trained on.
- They are weighted by `demonstration_weight` (default 1), which is explicit
  and recorded.
- Their `source` is `demonstration`, and their `candidate_row` is −1.

**Synthetic rows** are the candidate's observation and action, in the
representation's output modality.
- They keep the weight recorded in the candidates file.
- Their `source` is `synthetic`, and `candidate_row` indexes the candidates
  file.

**Split.** With a split (`ccil.split`), rows whose source episode is held out
are excluded, and counted.

**Order.** Demonstrations come first, in dataset order, then synthetic rows in
candidate order.

**Representation.** A fitted representation, such as `pca`, is bound to its
checkpoint (`representation_checkpoint`):
- the object must serialize to exactly the checkpoint's bytes;
- the checkpoint's SHA-256 must be the one the candidates and certificates
  record;
- the manifest records it.

Otherwise demonstrations could be encoded in coordinates other than the ones
the labels were generated in. A fixed representation (`identity`) takes no
checkpoint.

## Failing closed

`SupervisionError` lists every reason the artifacts do not belong together:
- the candidates' recorded dataset hash is not the dataset file's;
- the candidates fail the source-identity check against the dataset;
- the certificates certify a different candidates file, or were computed
  from a different dataset;
- the certificate rows differ from the candidate rows in identity or order;
- the certificate fails structural verification
  ([certificates.md](certificates.md#verification));
- the split's episodes are not the dataset's;
- the candidate modality is not the representation's, or the candidate
  observation or action shape is not the demonstrations';
- a fitted representation is given without its checkpoint, or differs from
  it, or the candidates or certificates were made in another representation;
- a 2.0 certificate is given without a selection, or a 1.0 certificate with
  one;
- the selection selects from other candidates, was curated from other
  certificates (or from none, when certificates are given), was curated with
  a calibration other than the one given (or none), or does not recompute
  from its inputs (`verify_selection`);
- certificates or a selection are given without candidates, or any of them
  with the `demonstrations` source;
- the `noise_bc` perturbation is given with `demonstrations_plus_selection`;
- the candidates carry action chunks that the consumer has not declared it
  will consume.

## Action chunks

A consumer declares a chunk policy:
- **`refuse` (default).** Chunked candidates raise. A trainer that predicts
  single actions must not silently use only a chunk's first action.
- **`consume`.** Every row carries a (K, A) chunk and a mask.
  - A synthetic row's chunk is the stored one.
  - A demonstration row's chunk is a_t at position 0 and a_{t+k} at position
    k, masked and zero past the episode's last frame. This is the label-chunk
    alignment ([`core/chunks.py`](../src/ccil/core/chunks.py)), with target
    frame t + 1.
  - K is the candidates' chunk size, or `chunk_size` (an integer ≥ 1) for
    demonstration-only arms.
  - The handoff is `chunk_arrays()`, which returns observations, chunks,
    masks and weights. The single-action `arrays()` refuses chunked
    supervision.

Only position 0 of a label's chunk carries its one-step CCIL claim.

## Manifest

`Supervision.manifest()` (`ccil.supervision-manifest` 1.0, or 1.1 with a
selection, which adds `selection`) records:
- the arm, the certification, the demonstration weight and frames;
- the chunk policy and representation;
- the SHA-256 of every input;
- the **row-count reconciliation**.

| Reconciliation count | Rule |
|---|---|
| Demonstration rows | Training plus held-out rows equals the transitions. |
| Synthetic rows | Training plus held-out rows equals the usable candidates (`selected`). |
| Total rows | Demonstration plus synthetic rows equals the row count. |

Construction fails unless the counts close.

## Nested-K views

A study that varies the synthetic dose K generates one candidate set at K_max and derives smaller doses from it. `ccil.core.nested_view(candidates, k, parent_sha256)` returns the rows a direct generation at `labels_per_transition = k` would produce: the rows whose recorded draw ordinal (`generator.draw_index`) is below k.

**Why the views are exact.**
- The `noisy_action` generators draw each transition's actions from that transition's own stream, so its first k draws are the same at any K. Each row records its ordinal in that stream as the measurement `draw_index`, so a view selects draws by identity, not by file position. With action bounds, the stream skips proposals outside the bounds; the k draws are the first k feasible proposals, `attempt_index` records where each fell in the raw stream, and a view refuses attempt indices that do not increase within a transition.
- Every built-in validator and the uniform weighting act row by row.

**Refused:**
- a set without draw ordinals (one written by an earlier generator version);
- rows that are not K_max consecutive draws per transition, with ordinals 0 to K_max − 1 in order (so a file whose draws were reordered within a transition is refused, not silently given a different view);
- a generator without nested draws;
- a validator or weighting not known to be row-local, because a population rule would change under a view;
- k outside [1, K_max].

**Certificates are not views.** Certify each view itself, because a quantile selection depends on the population.

**Recorded:** the view records the K_max file's hash as the input `nested_view_of`.

## Without `ccil`: the recipe

An independent consumer needs the candidates, the selection and the split.
The evidence executes this exact block.

```python
import json

import numpy as np

from ccil.schema import read_candidates, read_selection

c = read_candidates(candidates_path)
s = read_selection(selection_path)         # its candidates_sha256 must be this file's
train = json.loads(open(split_path).read())["train"]
usable = c.accepted & s.mask               # without a selection: c.accepted
usable &= np.isin(c.source_episode_id, train)      # never train on held-out episodes' labels
rows = np.flatnonzero(usable)    # c.observations[...][rows], c.actions[rows], s.weight[rows]
```

Before relying on that, check two things:
- the selection's rows equal the candidates' (`source_episode_id`,
  `source_frame`, same order), and its recorded candidate and certificate
  hashes are the files' (`s.error_spec["is_bound"]` says whether its errors
  are bounds);
- for a fitted representation, the demonstrations are encoded by the same
  checkpoint the candidates record.

## Evidence

```bash
uv run --group evidence python tools/m3_supervision/check.py --out <new-dir>
```

The report is
[`fixtures/evidence/m3-supervision/report.json`](fixtures/evidence/m3-supervision/report.json),
with `report-siddh.json` from the CUDA host, both produced at `c8a4712`, before
selections; the tool now reads the certified arm through its selection. It writes each arm's manifest to
`manifests/`. The artifacts come from the real offline pipeline. The report
shows:
- **Arms.** Each arm equals an independent recomputation from the files.
- **Exclusion.** Accepted rows that are inadmissible or unselected, with
  positive weights, never reach an instrumented trainer.
- **Weights and splits.** Weights, splits and reconciliation hold.
- **Failing closed.** Every mismatched-artifact case fails with its
  own reason.
- **Chunks.** They are consumed or refused, never reduced.
- **Recipe.** The recipe above reproduces the usable mask.
- **Integrated pipeline.** It trains through the contract.
