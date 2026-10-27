#!/usr/bin/env python3
"""Evidence for bounded-memory exact Jacobians (#109, first slice of #110).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_jacobian_chunks/check.py --out DIR [--device cuda] [--task discontinuous]

On the task's real labels (the ordinary pipeline at seed 501, as the #24 drivers configure it):
1. Peak device memory of jacobian_za over every accepted label, chunked (the default, JACOBIAN_CHUNK rows)
   and unchunked (one call), with the time of each.
2. Agreement: the rows whose Jacobian differs between the two, how many and by how much (absolute and
   relative). A ReLU network's exact Jacobian is discontinuous where a pre-activation is 0, so a different
   kernel can flip a row lying within rounding of a kink; this is measured, not assumed away.
3. Certificates: labels.certify with disturbed_action_paper_proxy (selection quantile 0.3), chunked and
   unchunked: the rows whose admission, selection or score differ.
Pass: chunked peak memory is at most a tenth of the unchunked peak, and no certificate decision differs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "m3_pendulum"))
SEED = 501


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="discontinuous")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import study
    import torch

    from ccil.config import resolve
    from ccil.core import IdentityRepresentation
    from ccil.core.transition import load_transition_model
    from ccil.models import mlp
    from ccil.pipelines import offline
    from ccil.schema.io import read_candidates
    study.prepare(out)
    data = out / "inputs" / "data" / f"{study.DATASETS[a.task]}.npz"
    sets = [f"run.seed={SEED}", f"compute.device={a.device}", f"data.path={data}", f"run.output_dir={out / 'stages'}",
            "policy.exclude_validation_episodes=false", "certificate.profile=disturbed_action_paper_proxy",
            "certificate.selection_quantile=0.3"]
    os.chdir(out / "inputs")
    config = str(out / "inputs" / "config" / study.CONFIGS[a.task])
    prepared, fitted, labeled = offline.generate_offline(resolve(config=config, sets=sets[:5]))[:3]
    model = load_transition_model(fitted.out(offline.MODEL_CHECKPOINT), device=a.device)
    cands = read_candidates(labeled.out("labels/candidates.npz"))
    acc = np.asarray(cands.accepted, bool)
    z, act = cands.observations["state"][acc], cands.actions[acc]

    def run_jacobian(chunk):
        mlp.JACOBIAN_CHUNK = chunk
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        j = model.jacobian_za(z, act)
        torch.cuda.synchronize()
        return j, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    default = mlp.JACOBIAN_CHUNK
    j_chunk, t_chunk, m_chunk = run_jacobian(default)
    j_full, t_full, m_full = run_jacobian(10**12)
    d = np.abs(j_chunk - j_full)
    rows = d.reshape(len(d), -1).max(axis=1) > 0
    rel = d / np.maximum(np.abs(j_full), 1e-30)

    def certify(chunk, tag):
        """labels.certify into its own output directory, so neither run reuses the other's stage."""
        mlp.JACOBIAN_CHUNK = chunk
        cfg = resolve(config=config, sets=[*sets[:3], f"run.output_dir={out / f'stages-{tag}'}", *sets[4:]])
        profile, kwargs = offline.build_certificate(cfg.values)
        res = offline.certify_stage(cfg, prepared, fitted, labeled, IdentityRepresentation("state"), profile, **kwargs)
        with np.load(res.out("labels/certificates.npz"), allow_pickle=False) as f:
            return {k: f[k] for k in ("admissible", "selected", "total_error_value")}
    cert_chunk = certify(default, "chunked")
    cert_full = certify(10**12, "unchunked")
    mlp.JACOBIAN_CHUNK = default
    cert_diff = {k: int((~((cert_chunk[k] == cert_full[k]) | (np.isnan(cert_chunk[k]) & np.isnan(cert_full[k])))).sum())
                 if cert_chunk[k].dtype.kind == "f" else int((cert_chunk[k] != cert_full[k]).sum()) for k in cert_chunk}
    decisions_equal = all(cert_diff.get(k, 0) == 0 for k in ("admissible", "selected"))
    report = {"all_pass": bool(m_chunk <= 0.1 * m_full and decisions_equal),
              "task": a.task, "seed": SEED, "accepted_labels": int(acc.sum()), "jacobian_chunk_rows": default,
              "memory_gb": {"chunked": m_chunk, "unchunked": m_full}, "seconds": {"chunked": t_chunk, "unchunked": t_full},
              "jacobian_rows_differing": int(rows.sum()), "jacobian_max_abs_difference": float(d.max()),
              "jacobian_max_relative_difference_at_differing_rows": float(rel[rows].max()) if rows.any() else 0.0,
              "certificate_fields_differing": cert_diff, "certificate_decisions_equal": decisions_equal,
              "environment": {"code_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip(),
                              "code_dirty": bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                                                 "--untracked-files=all"], capture_output=True,
                                                                text=True).stdout.strip()),
                              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                              "platform": platform.platform()}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=1, default=str))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
