#!/usr/bin/env python3
"""The recovery-segment diagnostic's descriptive figure (docs/experiments/pendulum-recovery-diagnostic.md, "Figure").

    R="uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/pendulum_recovery/figure.py"
    $R --data DATASET.npz --phase1 DIR --capture RUN --out FILE.svg [--seed S] [--check]

For one seed (default: the first): (a) the axis contract, e in [-pi, pi) periodic and omega in [-8, 8], with the
training demonstrations and a sample of accepted learned-model segments from x0 to the target; (b) the near-upright
inset (e in [-0.5, 0.5], omega in [-3, 3]) with the LQR branch shaded and the collection capture states, covered or
not within rho_cap by a pointwise row; (c) the inset again, rows colored by |u - pi_D|. Requires a complete audit.
--check exits 1 unless FILE equals a regeneration.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit as A  # noqa: E402

P1, O = A.P1, A.O  # noqa: E741  (the oracle module, named as in scan.py)
INK, MUTED, GRID, SURFACE, DEMO, COVERED, UNCOVERED = ("#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#1baf7a",
                                                       "#2a6fdb", "#d1495b")
SAMPLE = 300


def setup():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "pendulum-recovery-diagnostic", "svg.fonttype": "none",
                         "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def ew(z):
    return O.ew_of(np.asarray(z, np.float64).reshape(-1, 3))


def lines(ax, states, color, lw=0.4, alpha=0.6):
    for s in states:
        e, w = ew(s)
        jump = np.flatnonzero(np.abs(np.diff(e)) > np.pi)
        ax.plot(np.insert(e, jump + 1, np.nan), np.insert(w, jump + 1, np.nan), color=color, lw=lw, alpha=alpha)


def frame(ax, title, near):
    if near:
        ax.set_xlim(-0.5, 0.5), ax.set_ylim(-3, 3)
        ax.axvspan(-A.LQR, A.LQR, color=GRID, alpha=0.6, lw=0, zorder=0)
    else:
        ax.set_xlim(-np.pi, np.pi), ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0 (upright)", "π/2", "π"])
    ax.set_xlabel("e = θ − π (rad)" + ("; shaded: the LQR branch |e| < 0.1" if near else ", periodic"))
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)


def figure(data: Path, p1: Path, capture: Path, seed: int | None) -> str:
    doc = json.loads((p1 / "audit" / "audit.json").read_text())
    s = seed or int(next(iter(doc["seeds"])))
    rec = doc["seeds"][str(s)]
    u = P1.units(data)[s]
    demos = [u["S"][e] for e in u["split"]["train"]]
    seg, pw = dict(np.load(p1 / f"segments-s{s}.npz")), dict(np.load(p1 / f"pointwise-s{s}.npz"))
    M = int(seg["M"])
    xc = np.load(capture / "select" / f"s{s}.npz")["collection/x"].astype(np.float64)
    obs = pw["observations"]
    covered = (P1.Tree(obs).query(xc) <= rec["rho_cap"]) if len(obs) else np.zeros(len(xc), bool)
    plt = setup()
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), layout="constrained")
    for ax, near in ((axs[0], False), (axs[1], True)):
        lines(ax, demos, DEMO, lw=0.5, alpha=0.7)
        if M:
            acc = np.flatnonzero(seg["accepted"])
            pick = acc[np.unique(np.linspace(0, len(acc) - 1, min(SAMPLE, len(acc))).astype(int))] if len(acc) else acc
            lines(ax, seg["X"][pick], INK, lw=0.35, alpha=0.5)
    L = rec["learned"]
    frame(axs[0], f"(a) seed {s}: training demonstrations (green)\n"
                  f"{'a sample of accepted segments x̃₀ → target (black), M = ' + str(M) if M else 'empty synthetic view'}",
          False)
    for c, color, label in ((covered, COVERED, "covered"), (~covered, UNCOVERED, "not covered")):
        e, w = ew(xc[c])
        axs[1].plot(e, w, ls="none", marker=".", ms=1.6, color=color, alpha=0.7, rasterized=True, label=label)
    axs[1].legend(loc="upper right", fontsize=7, markerscale=4, frameon=False)
    q2 = L["q2_collection"]
    frame(axs[1], f"(b) near upright: collection capture states, covered within ρ_cap = {rec['rho_cap']:.3f} by a "
                  f"pointwise row\ncoverage of demonstration-uncovered states: "
                  f"{q2['coverage_of_uncovered'] if q2['coverage_of_uncovered'] is None else round(q2['coverage_of_uncovered'], 3)}",
          True)
    if len(obs):
        dis = np.abs(pw["actions"][:, 0] - O.demonstrator(obs))
        e, w = ew(obs)
        sc = axs[2].scatter(e, w, c=dis, cmap="viridis", vmin=0, vmax=3, s=1.2, rasterized=True)
        fig.colorbar(sc, ax=axs[2], shrink=0.8, label="|ũ − π_D(x̃)| (rad/s²)")
    frame(axs[2], "(c) near upright: pointwise rows by disagreement with the corrected demonstrator\n"
                  "(audit only; the demonstrator never entered generation)", True)
    fig.suptitle(f"Pendulum recovery-segment diagnostic (descriptive; learned-model segments from demonstrations only). "
                 f"Decision: {doc['decision']}", x=0.01, ha="left", color=INK, fontsize=10)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.phase1 / "audit" / "audit.json").exists():
        raise SystemExit("refused: the audit did not complete; no figure is drawn from it")
    text = figure(a.data, a.phase1, a.capture, a.seed)
    if a.check:
        same = a.out.exists() and a.out.read_text() == text
        print("identical" if same else "differs")
        return 0 if same else 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
