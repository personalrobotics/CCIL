#!/usr/bin/env python3
"""Recovery-segment diagnostic, phase 2: the external audit (docs/experiments/pendulum-recovery-diagnostic.md, v1).

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_recovery/audit.py"
    $R --data DATASET.npz --phase1 DIR --capture RUN [--resume]

Refuses unless DIR holds a complete phase 1 (manifest.json, every file's sha256 verifying) and RUN is the positive
control run (il-supervision-benchmarks runs/20261102T003101Z-capture-control) whose capture sets and splits match the
recorded hashes and phase 1's splits. Opens the exact map, the corrected demonstrator and the capture sets only here.
Writes DIR/audit/s{seed}.json per seed (resumable), the exact-dynamics reference segments, then DIR/audit/audit.json
with the six questions, the per-seed outcomes and the decision. Nothing here changes M, the segments or a threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "pendulum_correction"))
import generate as P1  # noqa: E402
import scan  # noqa: E402  (the exact model; it imports the oracle's exact map and corrected demonstrator)

O = scan.O  # noqa: E741  (the oracle module, named as in scan.py)
log, Heartbeat, atomic_json, atomic_npz, sha = P1.log, P1.Heartbeat, P1.atomic_json, P1.atomic_npz, P1.sha
LQR, ANNULUS, CONSISTENCY = 0.1, 0.5, 0.3
ROLLOUT, HOLD, HOLD_E, HOLD_W = 500, 100, 0.1, 0.5
MIN_SEGMENTS, COVERAGE_MIN, DISTINCT_MIN, AGREE_MIN, CONFIRM_MIN = 100, 0.5, 0.5, 0.9, 0.95
MIN_UNCOVERED, MIN_SEEDS = 100, 8
CAPTURE_SHA256 = {  # il-supervision-benchmarks run 20261102T003101Z-capture-control, recorded before generation
    3311: "e2d858168c7ad71b81512b1c971b2d71e1bdb371896db8f53adcc881dfe7c741",
    3312: "ddb42f5ee2d68f2052995a48543dc44dbce1c4155fe0bc65f875074273145752",
    3313: "143a5938384160e092515f3c1bb58ffd4840f4a5d795e9ef217dfc819e781986",
    3314: "4998a37c61414c6bf8c255e23b46181ed02d1063f23501a9a5223008290e73cc",
    3315: "208b51a0b27861a49ae866d5bb646fe17e3add0b7c7a3e076ae5a2f4711a9f92",
    3316: "6fe22e26b7c52438a910713172b740e5e4010745672cafae5152ddba2517024c",
    3317: "1b0e2325201ae6cfdb49ddf8ab5685c8c1ea1d85df0b59d6430ed2b2b667f4ed",
    3318: "c6888278512fc27d23d88f503e334fdad6b42daeaede2686387b13fcf87c22a9",
    3319: "4547f0535cc0c1b0a3e8e8c4e28197bee01a0acba6169ea8e437f468feb67dbf",
    3320: "ab320051f98a64d38b561999e4ff00ca41d945fac1f3afc0fc25a04e77895a71"}


class Exact:
    """The exact map as a transition model, as Gate 3A's ExactModel."""

    def delta(self, z, a):
        return scan.MODEL.delta(z, np.asarray(a).reshape(len(z)))


def captured(z: np.ndarray) -> np.ndarray:
    """|e| < 0.1 and pi_D captures within ROLLOUT steps (|e| <= 0.1, |omega| <= 0.5 over the last HOLD)."""
    e, w = O.ew_of(z)
    ok = np.abs(e) < LQR
    idx = np.flatnonzero(ok)
    for lo in range(0, len(idx), 20_000):
        part = idx[lo:lo + 20_000]
        s, _ = O.rollout(O.demonstrator, O.s3_of(e[part], w[part]), ROLLOUT)
        ee, ww = O.ew_of(s[:, -HOLD:])
        ok[part] = ((np.abs(ee) <= HOLD_E) & (np.abs(ww) <= HOLD_W)).all(1)
    return ok


def exact_replay(X, U, target):
    """Per segment: the largest exact one-step error d(F(x_i, u_i), x_{i+1}) and the exact composed error to target."""
    one = np.stack([P1.dist(O.step3(X[:, i], U[:, i]), X[:, i + 1]) for i in range(U.shape[1])], 1).max(1)
    z = X[:, 0]
    for i in range(U.shape[1]):
        z = O.step3(z, U[:, i])
    return one, P1.dist(z, target)


def questions(seg: dict, pw: dict, u: dict, cap: dict, rho: float) -> dict:
    """The six questions for one seed's segments (learned or exact) against its capture sets."""
    M, tau = int(seg["M"]), u["tau_demo"]
    acc = np.flatnonzero(seg["accepted"])
    obs, act, sid = pw["observations"], pw["actions"][:, 0], pw["segment"]
    q = (lambda v: np.quantile(v, [0.05, 0.5, 0.9, 0.95]).tolist() if len(v) else None)  # noqa: E731
    e, _ = O.ew_of(obs)
    inr = captured(obs)
    ann = (np.abs(e) >= LQR) & (np.abs(e) <= ANNULUS)
    dis = np.abs(act - O.demonstrator(obs))
    d_demo = u["tree"].query(obs)
    # Q2: coverage of the collection capture states not already covered by the demonstrations
    out = {"M": M, "accepted_segments": int(len(acc)), "rows": int(len(obs))}
    for name in ("collection", "diagnostic"):
        xc = cap[name]
        demo_cov = u["tree"].query(xc) <= rho
        syn_cov = (P1.Tree(obs).query(xc) <= rho) if len(obs) else np.zeros(len(xc), bool)
        unc = ~demo_cov
        out[f"q2_{name}"] = {"capture_states": int(len(xc)), "demonstration_covered": int(demo_cov.sum()),
                             "demonstration_uncovered": int(unc.sum()),
                             "newly_covered": int((syn_cov & unc).sum()),
                             "coverage_of_uncovered": float((syn_cov & unc).sum() / unc.sum()) if unc.any() else None,
                             "synthetic_covered_all": float(syn_cov.mean())}
    one, comp = exact_replay(seg["X"][acc], seg["U"][acc], seg["target"][acc]) if len(acc) else (np.zeros(0),) * 2
    conf = (one <= 0.25 * tau) & (comp <= 0.25 * tau)
    out.update({
        "q1_segments_entering_region": int(len(np.unique(sid[inr]))), "q1_rows_in_region": int(inr.sum()),
        "q1_rows_lqr_branch": int((np.abs(e) < LQR).sum()),
        "q3_fraction_in_region_distinct": float((d_demo[inr] >= rho).mean()) if inr.any() else None,
        "q3_d_demo_in_region_q05_q50_q90_q95": q(d_demo[inr]),
        "q4_fraction_in_region_within_0.3": float((dis[inr] <= CONSISTENCY).mean()) if inr.any() else None,
        "q4_disagreement_in_region_q05_q50_q90_q95": q(dis[inr]),
        "q5_annulus_rows": int(ann.sum()), "q5_annulus_disagreement_q05_q50_q90_q95": q(dis[ann]),
        "q5_annulus_fraction_within_0.3": float((dis[ann] <= CONSISTENCY).mean()) if ann.any() else None,
        "q6_fraction_confirmed": float(conf.mean()) if len(acc) else None,
        "q6_one_step_q50_q95_max": [*np.quantile(one, [0.5, 0.95]).tolist(), float(one.max())] if len(acc) else None,
        "q6_composed_q50_q95_max": [*np.quantile(comp, [0.5, 0.95]).tolist(), float(comp.max())] if len(acc) else None})
    q2 = out["q2_collection"]
    out["criteria"] = {
        "selected_M": M > 0,
        "q1": out["q1_segments_entering_region"] >= MIN_SEGMENTS,
        "q2": q2["coverage_of_uncovered"] is not None and q2["coverage_of_uncovered"] >= COVERAGE_MIN,
        "q3": out["q3_fraction_in_region_distinct"] is not None and out["q3_fraction_in_region_distinct"] >= DISTINCT_MIN,
        "q4": out["q4_fraction_in_region_within_0.3"] is not None and out["q4_fraction_in_region_within_0.3"] >= AGREE_MIN,
        "q6": out["q6_fraction_confirmed"] is not None and out["q6_fraction_confirmed"] >= CONFIRM_MIN}
    out["passes"] = all(out["criteria"].values())
    return out


def empty_view():
    return ({"M": np.array(0), "accepted": np.zeros(0, bool), "X": np.zeros((0, 1, 3)), "U": np.zeros((0, 0)),
             "target": np.zeros((0, 3))},
            {"observations": np.zeros((0, 3)), "actions": np.zeros((0, 1)), "segment": np.zeros(0, np.int64)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    man_p = a.phase1 / "manifest.json"
    if not man_p.exists():
        raise SystemExit("refused: phase 1 is not complete (no manifest.json)")
    man = json.loads(man_p.read_text())
    bad = [f for f, h in man["files"].items() if sha(a.phase1 / f) != h]
    if bad:
        raise SystemExit(f"refused: phase 1 files changed since the manifest: {bad}")
    for s, h in CAPTURE_SHA256.items():
        if s in P1.SEEDS and sha(a.capture / "select" / f"s{s}.npz") != h:
            raise SystemExit(f"refused: the capture set for seed {s} is not the recorded one")
    for s in P1.SEEDS:
        theirs = json.loads((a.capture / "data" / f"s{s}" / "split.json").read_text())
        if {k: sorted(theirs[k]) for k in ("train", "validation")} != {k: sorted(v) for k, v in
                                                                         man["splits"][str(s)].items()}:
            raise SystemExit(f"refused: seed {s}'s split differs from the positive control's")
    d = a.phase1 / "audit"
    if d.exists() and (d / "audit.json").exists():
        raise SystemExit(f"refused: {d / 'audit.json'} exists")
    if d.exists() and not a.resume:
        raise SystemExit(f"refused: {d} exists (an incomplete audit takes --resume)")
    d.mkdir(exist_ok=True)
    t0 = time.time()
    U = P1.units(a.data)
    log("audit inputs verified; phase 1", man["code_commit"][:7], "selected M", man["selected_M"])
    seeds = {}
    for i, s in enumerate(P1.SEEDS):
        path = d / f"s{s}.json"
        if path.exists():
            seeds[s] = json.loads(path.read_text())
            continue
        with Heartbeat(f"audit s{s} ({i + 1} of {len(P1.SEEDS)})"):
            f = np.load(a.capture / "select" / f"s{s}.npz")
            cap = {n: f[f"{n}/x"].astype(np.float64) for n in ("collection", "diagnostic")}
            rho = float(np.quantile(P1.Tree(cap["collection"]).query(cap["diagnostic"]), 0.95))
            seg, pw = (dict(np.load(a.phase1 / f"segments-s{s}.npz")), dict(np.load(a.phase1 / f"pointwise-s{s}.npz")))
            if bool(seg["empty"]):
                seg, pw = empty_view()
            u = U[s]
            rec = {"seed": s, "rho_cap": rho, "tau_demo": u["tau_demo"], "rho_over_tau": rho / u["tau_demo"],
                   "learned": questions(seg, pw, u, cap, rho)}
            M = int(seg["M"])
            if M:  # exact-dynamics reference segments at the already selected M (reported; not in the decision)
                g = P1.segments(Exact(), Exact(), u, M, 1)
                acc = np.flatnonzero(g["accepted"])
                sid, off = np.repeat(acc, M), np.tile(np.arange(M), len(acc))
                epw = {"observations": g["X"][sid, off], "actions": g["U"][sid, off][:, None], "segment": sid,
                       "offset": off}
                atomic_npz(d / f"exact-segments-s{s}.npz", M=np.array(M), **g)
                atomic_npz(d / f"exact-pointwise-s{s}.npz", M=np.array(M), **epw)
                rec["exact_reference"] = questions({"M": np.array(M), **g}, epw, u, cap, rho)
            atomic_json(path, rec)
            seeds[s] = rec
            L = rec["learned"]
            log("audit", s, f"M={L['M']}", json.dumps({k: v for k, v in L["criteria"].items()}), "passes",
                L["passes"])
    unc = sum(r["learned"]["q2_collection"]["demonstration_uncovered"] >= MIN_UNCOVERED for r in seeds.values())
    passing = [s for s, r in seeds.items() if r["learned"]["passes"]]
    if unc < MIN_SEEDS:
        decision = "coverage is not the missing mechanism at rho_cap"
    else:
        decision = "passes" if len(passing) >= MIN_SEEDS else "fails"
    doc = {"phase": 2, "predeclaration": "docs/experiments/pendulum-recovery-diagnostic.md (version 1)",
           "audit_commit": P1.git("rev-parse", "HEAD"), "audit_dirty": bool(P1.git("status", "--porcelain")),
           "phase1_manifest_sha256": sha(man_p), "phase1_commit": man["code_commit"],
           "capture_sha256": {str(k): v for k, v in CAPTURE_SHA256.items()},
           "selected_M": man["selected_M"], "seeds": {str(s): r for s, r in seeds.items()},
           "seeds_with_enough_uncovered": unc, "passing_seeds": passing, "decision": decision,
           "seconds": round(time.time() - t0, 1)}
    atomic_json(d / "audit.json", doc)
    log("decision", decision, "passing", passing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
