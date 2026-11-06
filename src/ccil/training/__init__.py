"""Policy-training components (#89 adds the vector BC trainer). Nothing here touches label generation.

Moved to il-supervision-benchmarks (#168): ilsb.learners.vector holds these trainers, with the same names, versions
and training identities. This package is removed in the next release.
"""
import warnings

warnings.warn("ccil.training moved to il-supervision-benchmarks (#168): use ilsb.learners.vector (vector_bc, "
              "vector_bc_v2, noise_bc) from personalrobotics/il-supervision-benchmarks, f1e2b30 or later; "
              "ccil.training is removed in the next release", FutureWarning, stacklevel=2)
