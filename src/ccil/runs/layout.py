"""Run-directory layout and the schemas this version of ccil can read.

Every path is relative to the run directory. Accepted and rejected candidates
live in one ccil.candidates file, so they share the run ID and source IDs.
"""
MANIFEST = "manifest.json"
CONFIG = "config.json"
METRICS = "metrics.json"
DATASET = "data/dataset.npz"
SPLIT = "data/split.json"
CANDIDATES = "labels/candidates.npz"
CHECKPOINTS = "checkpoints"   # checkpoints/<name>.pt
PLOTS = "plots"               # plots/<name>.png
LOG = "logs/run.log"
STAGE = "stage.json"           # pipeline stages: the identity their reuse decision used

MANIFEST_SCHEMA = "ccil.run-manifest"
MANIFEST_VERSION = (2, 0)
METRICS_SCHEMA = "ccil.metrics"

# Component schema -> highest major version this ccil can read.
READABLE = {
    "ccil.trajectories": 1,
    "ccil.candidates": 2,
    "ccil.config": 1,
    "ccil.split": 1,
    METRICS_SCHEMA: 1,
}

STAGING_MARKER = ".tmp-"  # staging directories are named .<run_id>.tmp-<pid>
