"""Experimental study infrastructure (#112): plan, gate, record, verify and analyze predeclared studies.

Reference infrastructure for running CCIL's own experiments (#24, M7), extracted from the #24 tooling. Like the
vector-policy harness, it is not part of CCIL's supervision contract, and its interfaces may change.

Moved to il-supervision-benchmarks (#168), whose study harness (ilsb run) replaces it. This package is removed in
the next release.
"""
import warnings

warnings.warn("ccil.experiments moved to il-supervision-benchmarks (#168): its study harness (ilsb run SPEC, "
              "personalrobotics/il-supervision-benchmarks) replaces it; ccil.experiments is removed in the next "
              "release", FutureWarning, stacklevel=2)
