# The vector-policy harness (moved)

The experimental vector-policy harness moved to
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks)
(#168):

- `ccil policy train` is replaced by `ilsb policy train CONFIG`, an
  `ilsb.policy-train` 1 configuration whose `labels` section runs
  `ccil labels generate`. `ccil policy train` now exits 2 naming it.
- `ccil.training` (`vector_bc`, `vector_bc_v2`, `noise_bc`) is replaced by
  `ilsb.learners.vector`, with the same trainer names, versions and training
  identities, so checkpoints written by `ccil policy train` load and resume
  there. Importing `ccil.training` raises `ImportError` naming it.

CCIL ends at its supervision artifacts, which any trainer consumes
([supervision.md](supervision.md)). The migration table is in
[contract.md](contract.md#dispositions-of-the-current-surface). This page as it
described the harness before the move is at commit 8903b24.
