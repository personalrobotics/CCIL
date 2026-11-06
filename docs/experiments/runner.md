# Predeclared studies (moved)

`ccil.experiments`, the predeclared-study runner, moved to
[il-supervision-benchmarks](https://github.com/personalrobotics/il-supervision-benchmarks)
(#168), whose study harness (`ilsb run SPEC`) replaces it. Importing
`ccil.experiments` raises `ImportError` naming it.

This page as it described the runner before the move is at commit 8903b24. The
complete historical Pendulum study is preserved at tag
`pendulum-evidence-archive-v1`.
