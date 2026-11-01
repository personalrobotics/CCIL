#!/usr/bin/env python3
"""The Gate 3A correction-geometry portrait (docs/experiments/pendulum-correction.md, "Figure"): descriptive only.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python tools/pendulum_correction/figure.py --scan DIR --data pendulum_cont_corrected.npz --out FILE.svg [--check]

DIR is a completed scan.py output (correction.json and its row files). One row of panels per r, at its treatment
cell, or at the largest sigma labeled "no treatment". No value, optimal action or task regret is drawn: the DP oracle
did not converge. --check exits 1 unless FILE equals a regeneration.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "pendulum_oracle"))
import oracle as O  # noqa: E402

E_LIM, W_LIM = (-np.pi, np.pi), (-8.0, 8.0)
INK, MUTED, GRID, SURFACE, DEMO = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#1baf7a"
ARROWS = 400  # at most this many source-to-synthetic arrows per panel, every k-th accepted row (deterministic)
CONSISTENCY_TOL = 0.3


def setup():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "pendulum-correction", "svg.fonttype": "none", "font.family": "DejaVu Sans",
                         "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def axes_contract(ax, title):
    ax.set_xlim(*E_LIM)
    ax.set_ylim(*W_LIM)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0 (upright)", "π/2", "π"])
    ax.set_xlabel("e = θ − π (rad), periodic")
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)


def background(ax, E, W, ud, H):
    ax.pcolormesh(E, W, ud, cmap="RdBu_r", vmin=-3, vmax=3, shading="nearest", rasterized=True, alpha=0.3)
    ax.contour(E, W, H, levels=np.arange(-18, 40, 6), colors=MUTED, linewidths=0.3)
    ax.contour(E, W, H, levels=[0.0], colors=INK, linewidths=0.8)
    ax.axvspan(-0.1, 0.1, color=MUTED, alpha=0.18, lw=0)


def demos(ax, states, alpha=0.5):
    e, w = O.ew_of(states)
    ax.plot(e, w, ls="none", marker=".", ms=0.6, color=DEMO, alpha=alpha, rasterized=True)


def figure(scan: Path, data: Path) -> str:
    doc = json.loads((scan / "correction.json").read_text())
    from ccil.schema import read_trajectories
    ds = {e.episode_id: e.observations["state"].astype(np.float64) for e in read_trajectories(data).episodes}
    e = np.linspace(-np.pi, np.pi, 241)
    w = np.linspace(*W_LIM, 161)
    E, W = np.meshgrid(e, w, indexing="ij")
    ud = O.demonstrator(O.s3_of(E.ravel(), W.ravel())).reshape(E.shape)
    H = O.energy(E, W)
    rs = sorted(doc["treatment"], key=float)
    plt = setup()
    fig, axs = plt.subplots(len(rs), 3, figsize=(15.5, 5.0 * len(rs)), sharex=True, sharey=True, layout="constrained",
                            squeeze=False)
    for row, r in enumerate(rs):
        t = doc["treatment"][r]
        sigma, solver = (t["sigma"], t["solver"]) if t["sigma"] is not None else (max(doc["sigma_grid"]), "a")
        label = (f"r = {r}: treatment cell σ = {sigma:g}, solver {solver}" if t["sigma"] is not None
                 else f"r = {r}: no treatment; drawn at σ = {sigma:g}, solver a")
        f = np.load(scan / f"rows-{solver}-sigma{sigma:g}.npz")
        train = np.isin(f["episode"], doc["splits"][r]["train"])
        tau = doc["tau_demo"][r]
        acc, on = train & f["accepted"], f["manifold_error"] <= 6.7e-7
        near = np.isin(f["class"], (0, 1))
        sup = (acc & on & (f["construction_residual"] <= 1e-3) & f["bounded"] & (f["consistency"] <= CONSISTENCY_TOL)
               & (f["switch_margin"] >= 0.01) & (f[f"d_demo_r{float(r):g}"] >= tau) & (f["region"] > 0)
               & (f["recovered"] == 1))
        demo_states = np.concatenate([ds[i] for i in doc["splits"][r]["train"]])
        xs, ws = O.ew_of(f["z_source"])
        xt, wt = f["e"], f["w"]
        # (a) the demonstrator's action field, energy, the LQR window, demonstrations and correction arrows
        ax = axs[row, 0]
        background(ax, E, W, ud, H)
        demos(ax, demo_states)
        arrow = np.flatnonzero(acc & on & ~near)
        arrow = arrow[::max(1, len(arrow) // ARROWS)]
        de = O.wrap(xt[arrow] - xs[arrow])
        q = ax.quiver(xs[arrow], ws[arrow], de, wt[arrow] - ws[arrow], f["consistency"][arrow], cmap="Oranges",
                      clim=(0, 3 * CONSISTENCY_TOL), angles="xy", scale_units="xy", scale=1, width=0.0025, zorder=4)
        ax.scatter(xt[arrow], wt[arrow], c=f["consistency"][arrow], cmap="Oranges", vmin=0, vmax=3 * CONSISTENCY_TOL,
                   s=2, lw=0, zorder=4, rasterized=True)  # endpoints, so short corrections stay visible
        ax.plot(xt[sup], wt[sup], ls="none", marker="o", ms=1.6, mfc="none", mec=INK, mew=0.4, zorder=5)
        axes_contract(ax, f"(a) Corrections over the demonstrator action field; energy, H = 0 bold; LQR window (band)\n"
                          f"arrows: accepted, on-manifold, moved rows, colored by |ã − π_D(z̃)|; circles: supported\n{label}")
        fig.colorbar(q, ax=ax, shrink=0.7, label="|ã − π_D(z̃)| (rad/s²)")
        # (b) coverage: where the supported rows sit relative to the demonstrations
        ax = axs[row, 1]
        background(ax, E, W, ud, H)
        demos(ax, demo_states)
        dd = f[f"d_demo_r{float(r):g}"]
        sc = ax.scatter(xt[sup], wt[sup], c=dd[sup], cmap="Purples", vmin=0, vmax=max(4 * tau, 1e-9), s=3, lw=0,
                        zorder=5, rasterized=True)
        axes_contract(ax, f"(b) Supported rows by distance to the nearest training demonstration\n"
                          f"τ_demo = {tau:.3g}; supported {int(sup.sum())} rows")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="d_demo (physical metric)")
        # (c) excluded rows, never drawn as corrections
        ax = axs[row, 2]
        demos(ax, demo_states, alpha=0.2)
        rej = train & ~f["accepted"]
        ax.plot(xt[rej & on], wt[rej & on], ls="none", marker=".", ms=1, color="#d03b3b", alpha=0.5, rasterized=True,
                label=f"rejected ({int(rej.sum())})")
        ax.plot(xs[acc & near], ws[acc & near], ls="none", marker=".", ms=1.5, color=INK, alpha=0.6, rasterized=True,
                label=f"near-identity, at source ({int((acc & near).sum())})")
        off = train & ~on
        ax.plot(xt[off], wt[off], ls="none", marker="x", ms=2, color="#4a3aa7", rasterized=True,
                label=f"off-manifold, angle only ({int(off.sum())})")
        ax.legend(loc="upper right", fontsize=6.5, frameon=False, markerscale=4)
        axes_contract(ax, "(c) Excluded rows: rejected, near-identity (returned at initialization\n"
                          "or unmoved) and off-manifold; demonstrations (green)")
    fig.suptitle("Pendulum correction geometry (exact-dynamics CCIL; descriptive; no task-optimality)", x=0.01,
                 ha="left", color=INK, fontsize=10)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = figure(a.scan, a.data)
    if a.check:
        same = a.out.exists() and a.out.read_text() == text
        print("identical" if same else "differs")
        return 0 if same else 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
