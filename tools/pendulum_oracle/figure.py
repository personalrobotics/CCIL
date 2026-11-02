#!/usr/bin/env python3
"""Panel 1, "task anatomy", from the oracle's fields (docs/experiments/pendulum-oracle.md, "Artifacts").

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python tools/pendulum_oracle/figure.py --run DIR --data pendulum_cont_corrected.npz --out FILE.svg [--check]

DIR is a check.py output (fields.npz and oracle.json). The figure is drawn at the region beta recorded there (the
selected beta, or 1e-2 under "no compatible beta", as labeled). --check exits 1 unless FILE equals a regeneration.
Every later phase portrait uses the same axis contract: e in [-pi, pi) horizontal and periodic, omega in [-8, 8].
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle as O  # noqa: E402

E_LIM, W_LIM = (-np.pi, np.pi), (-8.0, 8.0)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
STARTS = {"S1": (np.pi, 0.0), "S2": (np.pi - 0.3, 0.0), "S3": (np.pi / 2, 0.0), "S4": (-np.pi / 2, 0.0),
          "S5": (np.pi, 2.0), "S6": (2.0, -3.0), "S7": (0.6, 0.0), "S8": (0.0, 4.0)}


def setup():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "pendulum-oracle", "svg.fonttype": "none", "font.family": "DejaVu Sans",
                         "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    return buf.getvalue()


def axes_contract(ax, title):
    ax.set_xlim(*E_LIM)
    ax.set_ylim(*W_LIM)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0 (upright)", "π/2", "π"])
    ax.set_xlabel("e = θ − π (rad), periodic")
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(title, loc="left", color=INK, fontsize=9)


def periodic_lines(ax, e, w, **kw):
    """Trajectories on the cylinder, broken where they wrap across e = ±π."""
    for ei, wi in zip(e, w):
        x, y = ei.astype(float).copy(), wi.astype(float).copy()
        jump = np.flatnonzero(np.abs(np.diff(x)) > np.pi)
        x, y = np.insert(x, jump + 1, np.nan), np.insert(y, jump + 1, np.nan)
        ax.plot(x, y, **kw)


def mark_starts(ax):
    for k, (e, w) in STARTS.items():
        e = O.wrap(e)
        ax.plot(e, w, marker="o", ms=3.5, mfc=SURFACE, mec=INK, mew=0.8, ls="none", zorder=6)
        ax.annotate(k, (e, w), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=INK, zorder=6)


def omega_c_outline(ax, P, c):
    t = np.linspace(0, 2 * np.pi, 400)
    z = np.stack([np.cos(t), np.sin(t)])
    x = np.sqrt(c) * np.linalg.solve(np.linalg.cholesky(P).T, z)
    ax.plot(x[0], x[1], color=INK, lw=1.0, zorder=5)


def figure(run: Path, data: Path) -> str:
    f = np.load(run / "fields.npz")
    doc = json.loads((run / "oracle.json").read_text())
    beta = float(f["region/beta"])
    tag = f"beta{beta:g}"
    e, w = f["e"], f["w"]
    ne, nw = len(e), len(w)
    V = f[f"{tag}/V"].reshape(ne, nw)
    u = f[f"{tag}/greedy_u"].reshape(ne, nw)
    tie = f[f"{tag}/tie"].reshape(ne, nw)
    P, c = f[f"{tag}/P"], float(f[f"{tag}/c"])
    keep = np.abs(w) <= W_LIM[1]
    w_k = w[keep]
    E, W = np.meshgrid(e, w_k, indexing="ij")
    Vk, uk, tk = V[:, keep], u[:, keep], tie[:, keep]
    d = np.load(data)
    st = d["obs/state"].astype(np.float64).reshape(-1, 500, 3)
    de, dw = O.ew_of(st)
    ud = O.demonstrator(O.s3_of(E.ravel(), W.ravel())).reshape(E.shape)
    H = O.energy(E, W)

    plt = setup()
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True, layout="constrained")
    selected = doc["beta_selection"]["selected_beta"] is not None
    label = (f"demonstrator-compatible DP objective, β = {beta:g}" if selected
             else f"no compatible β; drawn at β = {beta:g}")
    # (a) cost-to-go, phase flow under the DP action, energy, ties, switching, Omega_c, demonstrations
    ax = axs[0]
    fin = np.where(Vk < O.BIG / 2, Vk, np.nan)
    im = ax.pcolormesh(E, W, np.log10(fin + 1e-3), cmap="Blues_r", shading="nearest", rasterized=True)
    ax.contour(E, W, H, levels=np.arange(-18, 40, 6), colors=MUTED, linewidths=0.3)
    ax.contour(E, W, H, levels=[0.0], colors=INK, linewidths=0.8)
    periodic_lines(ax, de[::5], dw[::5], color="#1baf7a", lw=0.4, alpha=0.8)
    s = (slice(None, None, max(1, ne // 24)), slice(None, None, max(1, len(w_k) // 16)))
    acc = -O.G * np.sin(E[s] + np.pi) + uk[s]
    n = np.hypot(W[s], acc / O.OMEGA0) + 1e-9
    ax.quiver(E[s], W[s], W[s] / n, acc / O.OMEGA0 / n, color=INK, width=0.0022, scale=40, zorder=4)
    ax.plot(E[tk], W[tk], ls="none", marker="s", ms=1.2, color="#eb6834", zorder=5)
    omega_c_outline(ax, P, c)
    mark_starts(ax)
    axes_contract(ax, f"(a) Cost-to-go V (log10); phase flow under the DP action;\nenergy contours, H = 0 bold; Ω_c; ties (orange); demonstrations\n{label}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log10(V + 1e-3)")
    # (b) the DP action field
    ax = axs[1]
    reach = Vk < O.BIG / 2
    ax.pcolormesh(E, W, np.where(reach, np.nan, 1.0), cmap="Greys", vmin=0, vmax=4, shading="nearest",
                  rasterized=True)
    im = ax.pcolormesh(E, W, np.where(reach, uk, np.nan), cmap="RdBu_r", vmin=-3, vmax=3, shading="nearest",
                       rasterized=True)
    ax.contour(E, W, (np.abs(uk) >= 3) & reach, levels=[0.5], colors=INK, linewidths=0.4)
    ax.plot(E[tk], W[tk], ls="none", marker="s", ms=1.2, color=INK, zorder=5)
    omega_c_outline(ax, P, c)
    mark_starts(ax)
    axes_contract(ax, "(b) DP greedy action u (rad/s²)\nsaturation boundary; ties (black); Ω_c\nunreachable, V = ∞ (gray)")
    # (c) the corrected demonstrator's action field, with the demonstrations
    ax = axs[2]
    ax.pcolormesh(E, W, ud, cmap="RdBu_r", vmin=-3, vmax=3, shading="nearest", rasterized=True)
    ax.contour(E, W, (np.abs(ud) >= 3).astype(float), levels=[0.5], colors=INK, linewidths=0.4)
    periodic_lines(ax, de[::5], dw[::5], color="#1baf7a", lw=0.4, alpha=0.8)
    ax.axvspan(-0.1, 0.1, color=MUTED, alpha=0.15, lw=0)
    mark_starts(ax)
    axes_contract(ax, "(c) Corrected demonstrator action π_D (rad/s²)\nsaturation boundary; LQR window |e| < 0.1 (band)\n"
                      "demonstrations (every 5th episode, green)")
    fig.colorbar(im, ax=axs[1:], shrink=0.7, label="u (rad/s²), clipped at ±3")
    out = svg(fig)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not json.loads((a.run / "oracle.json").read_text()).get("all_pass"):
        raise SystemExit("refused: the oracle run did not pass every check; no figure is published from it")
    text = figure(a.run, a.data)
    if a.check:
        same = a.out.exists() and a.out.read_text() == text
        print("identical" if same else "differs")
        return 0 if same else 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
