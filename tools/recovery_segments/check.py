#!/usr/bin/env python3
"""Evidence for #171 Stage 1: one-step CCIL is exactly the M = 1 recovery segment (docs/recovery-segments.md).

    uv run --group evidence python tools/recovery_segments/check.py --out NEW_DIR

No product code is involved beyond running it: the quickstart's demonstrations go through `ccil generate` for each
one-step generator (backward_euler, forward_euler, noisy_action, noisy_action_fe) under a linear and an MLP
transition model, with action chunks. The diagnosis's displacement and novelty gates are set to 0 so that every
generator yields rows: admission is not what this checks. A reference encoder, kept in this tool, derives a ccil.segments record from
each ccil.candidates file and its transition model. Then:
- the candidates file is unchanged by the derivation, and the record names it by SHA-256;
- the pointwise view at M = 1 equals the candidates' observations, actions, accepted, reasons and weight bit for bit,
  with each segment's total weight W inherited from its candidate row, and the target equals the candidates' target;
- the terminal target residual, recomputed from the stored arrays, equals the candidates' `residual` measurement;
  the per-step dynamics defect is zero up to the stored dtype's rounding; the solver convergence residual equals
  `generator.solver_residual` where the generator reports one and is absent otherwise;
- the chunk view at M = 1 equals the candidates' action_chunks and action_chunk_mask bit for bit;
- on M = 2 and M = 4 segments rolled forward from the same starts (a schema exercise, not a recovery claim), the
  pointwise weights sum to W per segment, every row reconciles to one (segment_id, offset), and the chunk view
  returns U;
- the reference reader refuses an offset outside its segment, weights that do not sum to W, a terminal claim on a
  pointwise row, and a candidates file whose SHA-256 differs from the one recorded.
Writes NEW_DIR/report.json, which holds no paths or times: two runs on one host are byte-identical.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ccil.core.chunks import TARGET_OFFSET
from ccil.core.transition import load_transition_model
from ccil.schema import read_candidates, read_trajectories

REPO = Path(__file__).resolve().parents[2]
QUICKSTART = REPO / "docs" / "quickstart.md"
CHUNK = 3
METHODS = {"backward_euler": "backtrack", "forward_euler": "backtrack",
           "noisy_action": "disturbed_action", "noisy_action_fe": "disturbed_action"}
MLP = {"kind": "mlp", "hidden_layers": [64, 64], "epochs": 50, "batch_size": 64}
POINTWISE_KEYS = ("segment_id", "offset", "observation", "action", "weight")  # never a segment-level claim
TERMINAL_KEYS = ("target", "terminal_residual", "accepted_segment")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quickstart(work: Path) -> dict:
    """Write the quickstart's demonstrations into work; return its configuration."""
    import yaml
    files = {name: body for _, name, body in re.findall(r"```(\w+) quickstart ?(\S*)\n(.*?)```",
                                                         QUICKSTART.read_text(), re.S) if name}
    (work / "demos.py").write_text(files["demos.py"])
    subprocess.run([sys.executable, "demos.py"], cwd=work, check=True)
    return yaml.safe_load(files["ccil.yaml"])


def generate(work: Path, base: dict, method: str, model: str) -> Path:
    """Run `ccil generate` for one case; return its generate-result.json."""
    cfg = json.loads(json.dumps(base))
    name = f"{model}-{method}"
    cfg["run"]["output_dir"] = f"runs-{name}"
    cfg["generation"] = {**cfg["generation"], "method": method, "action_chunk_size": CHUNK}
    cfg["diagnose"] = {"min_displacement": 0.0, "min_novelty": 0.0}
    if model == "mlp":
        cfg["transition_model"] = dict(MLP)
    if METHODS[method] == "backtrack":
        cfg["certificate"] = {k: v for k, v in cfg["certificate"].items() if k != "true_action_lipschitz"}
        cfg["certificate"]["profile"] = "backtrack"
    (work / f"{name}.yaml").write_text(json.dumps(cfg, indent=1))  # JSON is YAML
    r = subprocess.run([sys.executable, "-m", "ccil", "generate", "--config", f"{name}.yaml"], cwd=work,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ccil generate {name}: exit {r.returncode}\n{r.stderr[-2000:]}")
    return work / cfg["run"]["output_dir"] / "generate-result.json"


# --- reference encoder and reader for ccil.segments 1.0 (docs/recovery-segments.md) -------------------------------

def from_candidates(cands, cand_sha, model, profile, solver_residual):
    """The M = 1 segments of a ccil.candidates file: x_0 = z_G, u_0 = a_G, x_1 = F_hat(x_0, u_0), y = target."""
    x0, u = cands.observations["state"], cands.actions
    y = cands.target_observations["state"]
    rollout = x0 + model.delta(x0, u)
    n = len(u)
    seg = {"meta": {"schema": "ccil.segments", "version": "1.0", "mode": "backward", "candidates_sha256": cand_sha,
                    "target_frame_offset": TARGET_OFFSET[profile]},
           "M": np.ones(n, np.int64), "row_offset": np.arange(n, dtype=np.int64),
           "candidate_row": np.arange(n, dtype=np.int64),
           "source_episode_id": cands.source_episode_id, "source_frame": cands.source_frame,
           "target_frame": cands.source_frame + TARGET_OFFSET[profile], "target": y,
           "terminal_residual": np.linalg.norm(rollout - y, axis=1),
           "accepted": cands.accepted, "reasons": cands.reasons, "W": cands.weight,
           "states": np.stack([x0, rollout.astype(x0.dtype)], 1).reshape(2 * n, -1),  # x_0, x_1 per segment
           "actions": u, "dynamics_defect": None}
    if solver_residual is not None:
        seg["solver_residual"] = solver_residual
    seg["dynamics_defect"] = defects(seg, model)
    return seg


def rolled(seg, model, dataset, M):
    """M-step segments from the same starts: u_0 = a_G, then the source episode's actions. Not a recovery claim."""
    eps = {e.episode_id: e for e in dataset.episodes}
    keep = [i for i, (e, f) in enumerate(zip(seg["source_episode_id"], seg["target_frame"]))
            if f + M - 1 <= eps[str(e)].length - 1]
    n, A = len(keep), seg["actions"].shape[1]
    x0 = seg["states"].reshape(-1, 2, seg["states"].shape[1])[keep, 0]
    U = np.zeros((n, M, A), seg["actions"].dtype)
    U[:, 0] = seg["actions"][keep]
    for j, i in enumerate(keep):
        f = int(seg["target_frame"][i])
        U[j, 1:] = eps[str(seg["source_episode_id"][i])].actions[f:f + M - 1]
    xs = [x0]
    for k in range(M):
        xs.append((xs[-1] + model.delta(xs[-1], U[:, k])).astype(x0.dtype))
    out = {"meta": dict(seg["meta"]), "M": np.full(n, M, np.int64), "row_offset": np.arange(n, dtype=np.int64) * M,
           "candidate_row": seg["candidate_row"][keep], "source_episode_id": seg["source_episode_id"][keep],
           "source_frame": seg["source_frame"][keep], "target_frame": seg["target_frame"][keep] + M - 1,
           "W": seg["W"][keep], "accepted": seg["accepted"][keep],
           "states": np.stack(xs, 1).reshape(n * (M + 1), -1), "actions": U.reshape(n * M, A)}
    out["dynamics_defect"] = defects(out, model)
    return out


def state_rows(seg):
    """Index of each segment's x_0 in the flat states array: the M_i + 1 states of segment i are contiguous."""
    return seg["row_offset"] + np.arange(len(seg["M"]))


def defects(seg, model):
    """e_k = ||F_hat(x_k, u_k) - x_(k+1)|| for every action row."""
    s0 = state_rows(seg)
    xk = np.concatenate([np.arange(a, a + m) for a, m in zip(s0, seg["M"])]).astype(np.int64)
    x, u, nxt = seg["states"][xk], seg["actions"], seg["states"][xk + 1]
    return np.linalg.norm((x + model.delta(x, u)).astype(np.float64) - nxt.astype(np.float64), axis=1)


def pointwise(seg):
    """Rows (x_k, u_k), k < M, each with weight W / M: the weights of one segment sum to W whatever M is."""
    s0 = state_rows(seg)
    sid = np.repeat(np.arange(len(seg["M"])), seg["M"])
    off = np.arange(len(sid)) - np.repeat(seg["row_offset"], seg["M"])
    return {"segment_id": sid, "offset": off, "observation": seg["states"][s0[sid] + off],
            "action": seg["actions"], "weight": np.repeat(seg["W"] / seg["M"], seg["M"])}


def chunk_view(seg, dataset, K):
    """(x_0, U), then the source episode's continuation from the frame the segment reaches; masked past its end."""
    n, A = len(seg["M"]), seg["actions"].shape[1]
    eps = {e.episode_id: e for e in dataset.episodes}
    chunks, mask = np.zeros((n, K, A), seg["actions"].dtype), np.zeros((n, K), bool)
    for i in range(n):
        M, a = int(seg["M"][i]), int(seg["row_offset"][i])
        m = min(M, K)
        chunks[i, :m], mask[i, :m] = seg["actions"][a:a + m], True
        ep, f = eps[str(seg["source_episode_id"][i])], int(seg["target_frame"][i])
        for k in range(M, K):
            if f + k - M <= ep.length - 1:
                chunks[i, k], mask[i, k] = ep.actions[f + k - M], True
    return seg["states"][state_rows(seg)], chunks, mask


def problems(seg, view=None, candidates_path=None):
    """Why the reference reader refuses a segment record or its pointwise view (empty when it accepts)."""
    out = []
    M, off = seg["M"], seg["row_offset"]
    if (M < 1).any() or off[0] != 0 or not np.array_equal(off[1:], np.cumsum(M)[:-1]):
        out.append("row_offset does not partition the action rows into the segments")
    if view is not None:
        if set(view) != set(POINTWISE_KEYS):
            out.append(f"pointwise rows carry {sorted(set(view) - set(POINTWISE_KEYS))}: a segment-level claim on a row")
        sid, k = view.get("segment_id"), view.get("offset")
        if sid is not None and k is not None and ((k < 0) | (k >= M[sid])).any():
            out.append("a pointwise row's offset lies outside its segment")
        if sid is not None and "weight" in view:
            total = np.bincount(sid, weights=view["weight"], minlength=len(M))
            if not np.allclose(total, seg["W"], rtol=1e-12, atol=0):
                out.append("pointwise weights do not sum to the segment's W")
    if candidates_path is not None and sha(candidates_path) != seg["meta"]["candidates_sha256"]:
        out.append("the candidates file differs from the one this record was derived from")
    return out


# --- the cases --------------------------------------------------------------------------------------------------

def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def check_case(result: Path, method: str):
    doc = json.loads(result.read_text())
    root = result.parent
    art = {k: root / v["path"] for k, v in doc["artifacts"].items()}
    dataset = read_trajectories(art["dataset"])
    cpath = art["candidates"]
    before = sha(cpath)
    cands = read_candidates(cpath, dataset)
    if not len(cands.actions):
        return False, {"rows": 0}, None, None, None, None, cpath
    model = load_transition_model(art["transition_model"])
    meas = cands.measurements
    seg = from_candidates(cands, before, model, METHODS[method], meas.get("generator.solver_residual"))
    view = pointwise(seg)
    x0, chunks, mask = chunk_view(seg, dataset, CHUNK)
    eps32 = float(np.finfo(seg["states"].dtype).eps)
    scale = np.maximum(1.0, np.abs(seg["states"].astype(np.float64)).max())
    reduction = {
        "rows": int(len(cands.actions)),
        "observations": bitwise(view["observation"], cands.observations["state"]) and bitwise(x0, cands.observations["state"]),
        "actions": bitwise(view["action"], cands.actions),
        "weight": bitwise(view["weight"], cands.weight),
        "accepted": bitwise(seg["accepted"], cands.accepted),
        "reasons": bitwise(seg["reasons"], cands.reasons),
        "target": bitwise(seg["target"], cands.target_observations["state"]),
        "offsets_zero": bool((view["offset"] == 0).all()),
        "terminal_residual_equals_residual": bitwise(seg["terminal_residual"], meas["residual"]),
        "terminal_residual_max_abs_diff": float(np.abs(seg["terminal_residual"] - meas["residual"]).max()),
        "dynamics_defect_max": float(seg["dynamics_defect"].max()),
        "dynamics_defect_bound": eps32 * scale * np.sqrt(seg["states"].shape[1]),
        "solver_residual": ("absent" if "generator.solver_residual" not in meas else
                            bitwise(seg["solver_residual"], meas["generator.solver_residual"])),
        "chunks": bitwise(chunks, cands.action_chunks), "chunk_mask": bitwise(mask, cands.action_chunk_mask),
        "candidates_unchanged": sha(cpath) == before,
        "reader_accepts": problems(seg, view, cpath) == [],
    }
    want_solver = method in ("forward_euler", "noisy_action_fe")
    ok = all(reduction[k] is True for k in ("observations", "actions", "weight", "accepted", "reasons", "target",
                                            "offsets_zero", "terminal_residual_equals_residual", "chunks",
                                            "chunk_mask", "candidates_unchanged", "reader_accepts")) \
        and reduction["dynamics_defect_max"] <= reduction["dynamics_defect_bound"] \
        and (reduction["solver_residual"] is True if want_solver else reduction["solver_residual"] == "absent")
    reduction["candidates_sha256"] = before
    reduction["accepted_rows"] = int(cands.accepted.sum())
    return ok, reduction, seg, view, model, dataset, cpath


def check_exposure(seg, model, dataset):
    out, ok = {}, True
    for M in (2, 4):
        s = rolled(seg, model, dataset, M)
        v = pointwise(s)
        totals = np.bincount(v["segment_id"], weights=v["weight"], minlength=len(s["M"]))
        pairs = set(zip(v["segment_id"].tolist(), v["offset"].tolist()))
        _, chunks, mask = chunk_view(s, dataset, M + 1)
        U = s["actions"].reshape(len(s["M"]), M, -1)
        r = {"segments": int(len(s["M"])), "rows": int(len(v["weight"])),
             "weights_sum_to_W": bool(np.allclose(totals, s["W"], rtol=1e-12, atol=0)),
             "rows_reconcile_one_to_one": len(pairs) == len(v["weight"]) == int(s["M"].sum())
                                          and all(0 <= k < M for _, k in pairs),
             "chunk_view_returns_U": bitwise(chunks[:, :M], U) and bool(mask[:, :M].all()),
             "dynamics_defect_max": float(s["dynamics_defect"].max()),
             "reader_accepts": problems(s, v) == []}
        out[f"M={M}"] = r
        ok = ok and r["weights_sum_to_W"] and r["rows_reconcile_one_to_one"] and r["chunk_view_returns_U"] \
            and r["reader_accepts"]
    return ok, out


def check_refusals(seg, view, cpath, work):
    bad_offset = dict(view, offset=view["offset"] + 1)
    bad_weight = dict(view, weight=view["weight"] * 2)
    claim = dict(view, terminal_residual=seg["terminal_residual"])
    other = work / "changed-candidates.npz"
    raw = bytearray(Path(cpath).read_bytes())
    raw[len(raw) // 2] ^= 1
    other.write_bytes(bytes(raw))
    got = {"offset_outside_segment": problems(seg, bad_offset),
           "weights_not_summing_to_W": problems(seg, bad_weight),
           "terminal_claim_on_a_row": problems(seg, claim),
           "candidates_sha256_mismatch": problems(seg, None, other)}
    want = {"offset_outside_segment": "outside its segment", "weights_not_summing_to_W": "do not sum",
            "terminal_claim_on_a_row": "segment-level claim", "candidates_sha256_mismatch": "differs"}
    ok = all(len(got[k]) == 1 and want[k] in got[k][0] for k in want)
    return ok, got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)

    base = quickstart(work)
    first = None
    for model_kind in ("linear", "mlp"):
        for method in METHODS:
            ok, red, seg, view, model, dataset, cpath = check_case(generate(work, base, method, model_kind), method)
            record(f"m1_reduction[{model_kind},{method}]", ok, **red)
            if seg is None:
                continue
            if first is None:
                first = (seg, view, model, dataset, cpath)
            if method == "noisy_action_fe":
                ok, ex = check_exposure(seg, model, dataset)
                record(f"exposure_and_views_at_M_above_1[{model_kind},{method}]", ok, **ex)
    ok, got = check_refusals(first[0], first[1], first[4], work)
    record("reader_refusals", ok, refusals=got)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    import torch
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "quickstart_sha256": sha(QUICKSTART), "chunk_size": CHUNK, "mlp": MLP,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "torch": torch.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
