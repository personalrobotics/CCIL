#!/usr/bin/env python3
"""The figures of the Pendulum study report (#25), drawn from the committed analysis.

    uv run --group evidence python tools/pendulum_report/plots.py           # rewrite docs/experiments/figures/
    uv run --group evidence python tools/pendulum_report/plots.py --check   # exit 1 if a committed figure differs

Every value plotted is read from docs/fixtures/evidence/pendulum-study-v6/analysis.json, the same input as the
tables (report.py), whose per-seed values link to the records. Nothing here computes an interval: the analysis did.
The SVGs are deterministic for a given matplotlib (pinned in the evidence group).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "experiments" / "figures"
ANALYSIS = REPO / "docs" / "fixtures" / "evidence" / "pendulum-study-v6" / "analysis.json"
TASKS = ("continuous", "discontinuous")
NAMES = {"bc": "BC", "noise_bc": "NoiseBC", "ccil": "CCIL", "ccil_tuned": "CCIL-tuned",
         "ccil_filtered": "CCIL-filtered", "ccil_nolip": "CCIL-noLip", "bc_modern": "BC-modern",
         "ccil_modern": "CCIL-modern", "bc_frac10": "BC 10%", "bc_frac25": "BC 25%", "bc_frac50": "BC 50%",
         "ccil_frac10": "CCIL 10%", "ccil_frac25": "CCIL 25%", "ccil_frac50": "CCIL 50%", "zero": "zero action"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")  # reference palette, slots 1-6
MARKERS = ("o", "s", "^", "D", "v", "P")

plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "ccil-pendulum-v6", "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def name(c: dict) -> str:
    return f"{NAMES[c['A']]} − {NAMES[c['B']]}"


def interval_rows(ax, rows, lim, color=SERIES[0]):
    """One row per comparison: per-seed differences as dots, the mean, and the decision interval.
    Values beyond the axis limits are drawn at the edge and labeled, so no seed disappears."""
    lo, hi = lim
    for y, (label, c) in enumerate(rows):
        seeds = list(c["per_seed"].values())
        inside = [v for v in seeds if lo <= v <= hi]
        jitter = [(i % 5 - 2) * 0.06 for i in range(len(inside))]
        ax.scatter(inside, [y + j for j in jitter], s=9, color=MUTED, alpha=0.55, linewidths=0, zorder=2)
        for side, out in ((hi, sorted(v for v in seeds if v > hi)), (lo, sorted(v for v in seeds if v < lo))):
            if out:
                right = side == hi
                text = f"{out[0]:+.0f}" if len(out) == 1 else f"{len(out)} seeds, {out[0]:+.0f} to {out[-1]:+.0f}"
                ax.annotate(text, (side, y), xytext=(-3 if right else 3, 7), textcoords="offset points",
                            ha="right" if right else "left", fontsize=7, color=MUTED)
                ax.plot([side], [y], marker=">" if right else "<", color=MUTED, ms=5, zorder=2, clip_on=False)
        a, b = c["interval"]
        ax.plot([max(a, lo), min(b, hi)], [y, y], color=color, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([c["mean"]], [y], "o", color=color, ms=6, mec=SURFACE, mew=1.5, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(lo, hi)


def primary(a: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.4), sharey=True, layout="constrained")
    for ax, t in zip(axes, TASKS):
        p = a["tasks"][t]["primary"]
        rows = [("CCIL − BC", p["comparison"]), ("Control: BC − BC 10%", p["positive_control"]["comparison"])]
        interval_rows(ax, rows, (-1500, 3000))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("difference in return, noise-free (higher favors the first arm)")
    fig.suptitle("Primary decision: per-seed differences (gray), mean (dot) and 97.5% union interval (bar), "
                 "20 paired seeds", x=0.01, ha="left", fontsize=10)
    return fig


def comparisons(a: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True, layout="constrained")
    for ax, t in zip(axes, TASKS):
        T = a["tasks"][t]
        rows = [(name(c), c) for c in T["secondary"]]
        rows += [(f"{name(c)}, eval noise {c['noise']}", c) for c in T["by_noise"]]
        rows += [(name(c), c) for c in T["by_data_size"]]
        interval_rows(ax, rows, (-2500, 2000))
        for y in (len(T["secondary"]) - 0.5, len(T["secondary"]) + len(T["by_noise"]) - 0.5):
            ax.axhline(y, color=GRID, lw=0.8)
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("difference in return")
    fig.suptitle("Secondary comparisons (descriptive): per-seed differences, mean and 95% union interval. "
                 "\nNoise-free unless marked. Groups: arms; evaluation noise; demonstration count",
                 x=0.01, ha="left", fontsize=10)
    return fig


def arms(a: dict):
    order = ["zero", "ccil_tuned", "ccil_nolip", "ccil", "ccil_frac50", "ccil_modern", "bc_modern", "noise_bc",
             "bc", "ccil_frac25", "bc_frac25", "bc_frac50", "ccil_filtered", "ccil_frac10", "bc_frac10"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="constrained")
    for ax, t in zip(axes, TASKS):
        A = a["tasks"][t]["arms"]
        rows = [(NAMES[k], {"per_seed": A[k]["0.0"]["per_seed"], "mean": A[k]["0.0"]["mean"],
                            "interval": [A[k]["0.0"]["mean"]] * 2}) for k in order]
        interval_rows(ax, rows, (-5000, -1500), color=INK)
        ax.axvline(A["zero"]["0.0"]["mean"], color=SERIES[0], lw=1, ls="--", zorder=1)
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("return, noise-free (dashed: zero-action policy)")
    fig.suptitle("Every arm, noise-free: per-seed returns (gray) and mean (dot), 20 seeds each",
                 x=0.01, ha="left", fontsize=10)
    return fig


def noise(a: dict):
    keys = ("bc", "noise_bc", "ccil", "ccil_tuned", "bc_modern", "ccil_modern")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
    for ax, t in zip(axes, TASKS):
        A = a["tasks"][t]["arms"]
        levels = ["0.0", "0.01", "0.05", "0.1"]
        for k, color, m in zip(keys, SERIES, MARKERS):
            ys = [A[k][lv]["mean"] for lv in levels]
            ax.plot(range(len(levels)), ys, color=color, lw=2, marker=m, ms=6, mec=SURFACE, mew=1, label=NAMES[k])
        ax.axhline(A["zero"]["0.0"]["mean"], color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(levels)), levels)
        ax.grid(axis="y", color=GRID)
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("evaluation action noise")
    axes[0].set_ylabel("mean return over 20 seeds")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", fontsize=8, frameon=False)
    fig.suptitle("Mean return against evaluation noise (dashed: zero-action policy, noise-free; BC and NoiseBC "
                 "nearly coincide).\n"
                 "Intervals for each difference are in the comparisons figure and the tables",
                 x=0.01, ha="left", fontsize=10)
    return fig


FIGURES = {"pendulum-v6-primary.svg": primary, "pendulum-v6-comparisons.svg": comparisons,
           "pendulum-v6-arms.svg": arms, "pendulum-v6-noise.svg": noise}


def render(a: dict) -> dict[str, bytes]:
    out = {}
    for fname, draw in FIGURES.items():
        fig = draw(a)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None})
        plt.close(fig)
        out[fname] = buf.getvalue()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    figures = render(json.loads(ANALYSIS.read_text()))
    if ap.parse_args().check:
        bad = [f for f, data in figures.items() if not (OUT / f).exists() or (OUT / f).read_bytes() != data]
        print("figures match" if not bad else f"differ from a regeneration: {', '.join(bad)}")
        return 1 if bad else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for f, data in figures.items():
        (OUT / f).write_bytes(data)
        print(f"wrote {OUT / f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
