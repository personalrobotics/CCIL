# The supervision contract

CCIL produces synthetic supervision. Its ordinary workflow turns
demonstrations into immutable artifacts:
- the prepared dataset;
- the candidate labels (`labels/candidates.npz`);
- optionally, their certificates (`labels/certificates.npz`).

Users train downstream policies with whatever trainer they choose. This
document defines exactly which rows such a trainer may use, and with which
weights. [`src/ccil/supervision.py`](../src/ccil/supervision.py) implements
it as `build_supervision`. It trains nothing and imports no trainer.

The maintained vector-policy trainer (#89,
[policy-harness.md](policy-harness.md)) is experimental reference
infrastructure. It exists to reproduce the paper's experiments and to show
that the artifacts are consumable, and it consumes this contract like any
other trainer. Using CCIL does not require it.

## Arms

| `method` | Rows |
|---|---|
| `bc` | The demonstrations only |
| `noise_bc` | The demonstrations only. The trainer perturbs them online at every draw ([`training/noise_bc.py`](../src/ccil/training/noise_bc.py)), so nothing synthetic is stored. |
| `ccil`, no certificates | The demonstrations, and every candidate with `accepted`. Recorded as `certification: uncertified`, with each synthetic row classified `uncertified`. |
| `ccil`, with certificates | The demonstrations, and every candidate with `accepted ∧ admissible ∧ selected`. Each row keeps its certificate classification, and the manifest records the profile, admission and selection. |

**Usable rows only.** A rejected, inadmissible or unselected candidate never
appears, whatever its weight. There is no weight-only path into the loss.

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
- certificates are given without candidates, or candidates or certificates
  with `bc` or `noise_bc`;
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

`Supervision.manifest()` (`ccil.supervision-manifest` 1.0) records:
- the arm, the certification, the demonstration weight and frames;
- the chunk policy and representation;
- the SHA-256 of every input;
- the **row-count reconciliation**.

| Reconciliation count | Rule |
|---|---|
| Demonstration rows | Training plus held-out rows equals the transitions. |
| Synthetic rows | Training plus held-out rows equals the usable candidates. |
| Total rows | Demonstration plus synthetic rows equals the row count. |

Construction fails unless the counts close.

## Without `ccil`: the recipe

An independent consumer needs the two label files and the split. The
evidence executes this exact block.

```python
import json

import numpy as np

from ccil.schema import read_candidates, read_certificates

c = read_candidates(candidates_path)
k = read_certificates(certificates_path)    # its candidates_sha256 must be this file's
train = json.loads(open(split_path).read())["train"]
usable = c.accepted & k.admissible & k.selected    # without certificates: c.accepted
usable &= np.isin(c.source_episode_id, train)      # never train on held-out episodes' labels
rows = np.flatnonzero(usable)    # c.observations[...][rows], c.actions[rows], c.weight[rows]
```

Before relying on that, check two things:
- the certificate's rows equal the candidates' (`source_episode_id`,
  `source_frame`, same order), and the hashes match;
- for a fitted representation, the demonstrations are encoded by the same
  checkpoint the candidates record.

## Evidence

```bash
uv run --group evidence python tools/m3_supervision/check.py --out <new-dir>
```

The report is
[`fixtures/evidence/m3-supervision/report.json`](fixtures/evidence/m3-supervision/report.json),
with `report-siddh.json` from the CUDA host. It writes each arm's manifest to
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
