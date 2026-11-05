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
CERTIFICATES = "labels/certificates.npz"  # labels.certify; candidates are never modified
SELECTION = "labels/selection.npz"  # labels.curate; candidates and certificates are never modified
DIAGNOSIS = "diagnosis/diagnosis.json"  # labels.diagnose: the data-only recovery-length decision
CALIBRATION = "labels/calibration.json"  # labels.calibrate: reference-assisted score calibration
AUDIT = "labels/audit.json"  # labels.audit: reference dynamics annotating the selection; never an input
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
    "ccil.certificates": 2,
    "ccil.selection": 1,
    "ccil.diagnosis": 1,
    "ccil.calibration": 1,
    "ccil.reference-audit": 1,
    "ccil.config": 1,
    "ccil.split": 1,
    METRICS_SCHEMA: 1,
}

STAGING_MARKER = ".tmp-"  # staging directories are named .<run_id>.tmp-<pid>
