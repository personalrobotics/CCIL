#!/usr/bin/env python3
"""Evidence for bounded-memory exact Jacobians (#109, first slice of #110).

    uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \\
        python tools/m3_jacobian_chunks/check.py --out DIR [--device cuda] [--parts cpu,cuda] [--task discontinuous]

jacobian_za runs in balanced chunks whose row count is derived from a device-memory budget
(compute.jacobian_memory_mb) and the model's dimensions and width.

cpu (any host):
1. Chunk boundaries: on seeded fixtures (D = 3, A = 1, [256, 256], 30,000 rows; D = 64, A = 7, [256, 256], 3,000
   rows), a budget forcing many chunks gives Jacobians bit-identical to one unchunked call on the CPU.
2. Derivation: at a fixed budget the rows per chunk fall as D grows, and the chunks are balanced: every chunk
   but the last has ceil(N / chunks) rows.

cuda (on the task's real labels: the ordinary pipeline at seed 501, as the #24 drivers configure it):
3. Peak device memory of jacobian_za over every accepted label at the default budget and unchunked, with the
   time of each; the chunked peak must be within the budget and at most a tenth of the unchunked peak.
4. Agreement: the rows whose Jacobian differs between the two, how many and by how much. A ReLU network's exact
   Jacobian is discontinuous where a pre-activation is 0, so a different kernel can flip a row lying within
   rounding of a kink; this is measured, not assumed away.
5. Certificates: labels.certify with disturbed_action_paper_proxy (selection quantile 0.3), at the default budget
   and unchunked: admission, selection and total_error_value must be identical for every row.
6. Width: at D = 64 and D = 512 (A = 7, [256, 256]), a 256 MiB budget keeps the peak within the budget.
7. A budget smaller than one estimated row (D = 512 under 1 MiB) is refused, naming the minimum.
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


def _fixture(D, A, n, seed):
    import torch

    from ccil.models.mlp import MLPTransitionModel
    torch.manual_seed(seed)
    m = MLPTransitionModel([256, 256], epochs=1, batch_size=512, device="cpu")
    m._build(D, A)
    r = np.random.default_rng(seed)
    return m, r.standard_normal((n, D)), r.standard_normal((n, A))


def cpu_part() -> dict:
    """Forced multi-chunk Jacobians against one call, bit for bit, on the CPU."""
    from ccil.models import mlp
    cases = []
    for D, A, n in ((3, 1, 30000), (64, 7, 3000)):
        m, z, a = _fixture(D, A, n, 7)
        m.jacobian_memory_bytes = 1 << 60
        full = m.jacobian_za(z, a)
        one = m.jacobian_chunking
        rows = 997 if D == 3 else 211  # deliberately not a divisor of n
        m.jacobian_memory_bytes = rows * mlp.jacobian_row_bytes(D, A, [256, 256])  # a budget of exactly `rows` rows
        chunked = m.jacobian_za(z, a)
        ch = m.jacobian_chunking
        sizes_ok = ch["chunks"] > 1 and ch["rows_per_chunk"] * (ch["chunks"] - 1) < n <= ch["rows_per_chunk"] * ch["chunks"]
        cases.append({"D": D, "A": A, "rows": n, "unchunked": one, "chunked": ch,
                      "rows_differing": int((chunked != full).reshape(n, -1).any(axis=1).sum()),
                      "identical": bool(np.array_equal(chunked, full)), "balanced": bool(sizes_ok)})
    per_row = {D: mlp.jacobian_chunk_rows(D, 1, [256, 256], 1 << 30) for D in (3, 16, 64, 512)}
    falls = all(per_row[x] > per_row[y] for x, y in ((3, 16), (16, 64), (64, 512)))
    # A budget below one estimated row fails closed, naming the minimum, rather than running a one-row chunk over it.
    m, z, a = _fixture(512, 7, 4, 3)
    m.jacobian_memory_bytes = 1 * 2**20
    try:
        m.jacobian_za(z, a)
        refusal = None
    except mlp.JacobianBudgetError as e:
        refusal = str(e)
    row_mib = mlp.jacobian_row_bytes(512, 7, [256, 256]) / 2**20
    return {"pass": all(c["identical"] and c["balanced"] for c in cases) and falls and refusal is not None,
            "cases": cases, "max_rows_per_chunk_at_1_GiB": per_row,
            "below_one_row": {"D": 512, "A": 7, "budget_mib": 1, "estimated_row_mib": row_mib, "refusal": refusal}}


def width_part(device) -> dict:
    """Peak device memory within a 256 MiB budget for wider models: D = 64 and D = 512 (A = 7, [256, 256])."""
    import torch
    rows = []
    for D, n in ((64, 20000), (512, 2000)):
        m, z, a = _fixture(D, 7, n, 11)
        m.device = device
        m.net.to(device)
        m.jacobian_memory_bytes = 256 * 2**20
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        m.jacobian_za(z, a)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        rows.append({"D": D, "A": 7, "rows": n, "budget_bytes": m.jacobian_memory_bytes, "peak_bytes": int(peak),
                     "chunking": m.jacobian_chunking,
                     "ok": bool(peak <= m.jacobian_memory_bytes and m.jacobian_chunking["chunks"] > 1)})
    return {"pass": all(r["ok"] for r in rows), "cases": rows}


def cuda_part(out: Path, task: str, device: str) -> dict:
    import study
    import torch

    from ccil.config import resolve
    from ccil.core import IdentityRepresentation
    from ccil.core.transition import load_transition_model
    from ccil.models import mlp
    from ccil.pipelines import offline
    from ccil.schema.io import read_candidates
    study.prepare(out)
    data = out / "inputs" / "data" / f"{study.DATASETS[task]}.npz"
    sets = [f"run.seed={SEED}", f"compute.device={device}", f"data.path={data}", f"run.output_dir={out / 'stages'}",
            "policy.exclude_validation_episodes=false", "certificate.profile=disturbed_action_paper_proxy",
            "certificate.selection_quantile=0.3"]
    cwd = os.getcwd()
    os.chdir(out / "inputs")
    config = str(out / "inputs" / "config" / study.CONFIGS[task])
    prepared, fitted, labeled = offline.generate_offline(resolve(config=config, sets=sets[:5]))[:3]
    model = load_transition_model(fitted.out(offline.MODEL_CHECKPOINT), device=device)
    cands = read_candidates(labeled.out("labels/candidates.npz"))
    acc = np.asarray(cands.accepted, bool)
    z, act = cands.observations["state"][acc], cands.actions[acc]
    budget = mlp.JACOBIAN_MEMORY_BYTES
    unbounded = 1 << 60

    def run_jacobian(b):
        model.jacobian_memory_bytes = b
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        j = model.jacobian_za(z, act)
        torch.cuda.synchronize()
        return j, time.time() - t0, (torch.cuda.max_memory_allocated() - base), dict(model.jacobian_chunking)
    j_chunk, t_chunk, m_chunk, chunking = run_jacobian(budget)
    j_full, t_full, m_full, _ = run_jacobian(unbounded)
    d = np.abs(j_chunk - j_full)
    rows = d.reshape(len(d), -1).max(axis=1) > 0
    rel = d / np.maximum(np.abs(j_full), 1e-30)

    def certify(mb, tag):
        """labels.certify into its own output directory, so neither run reuses the other's stage."""
        cfg = resolve(config=config, sets=[*sets[:3], f"run.output_dir={out / f'stages-{tag}'}", *sets[4:],
                                           f"compute.jacobian_memory_mb={mb}"])
        profile, kwargs = offline.build_certificate(cfg.values)
        res = offline.certify_stage(cfg, prepared, fitted, labeled, IdentityRepresentation("state"), profile, **kwargs)
        with np.load(res.out("labels/certificates.npz"), allow_pickle=False) as f:
            return {k: f[k] for k in ("admissible", "selected", "total_error_value")}
    cert_chunk = certify(budget // 2**20, "chunked")
    cert_full = certify(unbounded // 2**20, "unchunked")
    os.chdir(cwd)
    cert_diff = {k: int((~((cert_chunk[k] == cert_full[k]) | (np.isnan(cert_chunk[k]) & np.isnan(cert_full[k])))).sum())
                 if cert_chunk[k].dtype.kind == "f" else int((cert_chunk[k] != cert_full[k]).sum()) for k in cert_chunk}
    identical = all(v == 0 for v in cert_diff.values())
    return {"pass": bool(m_chunk <= budget and m_chunk <= 0.1 * m_full and identical),
            "criterion": "chunked peak within the budget and at most a tenth of the unchunked peak; every certificate "
                         "field (admissible, selected, total_error_value) identical",
            "task": task, "seed": SEED, "accepted_labels": int(acc.sum()), "budget_bytes": budget, "chunking": chunking,
            "peak_gb": {"chunked": m_chunk / 2**30, "unchunked": m_full / 2**30},
            "seconds": {"chunked": t_chunk, "unchunked": t_full},
            "jacobian_rows_differing": int(rows.sum()), "jacobian_max_abs_difference": float(d.max()),
            "jacobian_max_relative_difference_at_differing_rows": float(rel[rows].max()) if rows.any() else 0.0,
            "certificate_fields_differing": cert_diff, "certificates_identical": identical}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parts", default="cpu,cuda")
    ap.add_argument("--task", default="discontinuous")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    parts = a.parts.split(",")
    cases = {}
    if "cpu" in parts:
        cases["cpu_chunk_boundaries"] = cpu_part()
    if "cuda" in parts:
        cases["cuda_real_labels"] = cuda_part(out, a.task, a.device)
        cases["cuda_width"] = width_part(a.device)
    for name, c in cases.items():
        print(f"{'PASS' if c['pass'] else 'FAIL'} {name}", flush=True)
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases.values()), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
                              "torch": torch.__version__,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              "platform": platform.platform()}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"{sum(c['pass'] for c in cases.values())}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
