#!/usr/bin/env python3
"""The tables and figures of the M3 Pendulum closeout (#25), generated from committed evidence only.

    uv run --group evidence python tools/pendulum_report/closeout.py           # rewrite the tables and figures
    uv run --group evidence python tools/pendulum_report/closeout.py --check   # exit 1 if a regeneration differs

Nothing here trains, evaluates or computes an interval: every number is read from a committed report, and each
table names the report it came from. Writes docs/experiments/pendulum-m3-tables.md and two SVG figures under
docs/experiments/figures/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "experiments"
OUT = DOCS / "pendulum-m3-tables.md"
FIGS = DOCS / "figures"
EV = REPO / "docs" / "fixtures" / "evidence"
V6, V2 = EV / "pendulum-study-v6", EV / "pendulum-eval-v2"
COR, CNT = EV / "pendulum-corrected-demonstrator", EV / "pendulum-demonstration-count"
TASKS = ("continuous", "discontinuous")
SERIES = {"continuous": "#2a78d6", "discontinuous": "#eb6834"}  # categorical slots 1-2 (validated)
ARM_STYLE = {"BC": ("#2a78d6", "o"), "CCIL": ("#eb6834", "s"), "Corrected demonstrator": ("#1baf7a", "D")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
OUTCOME = {"ccil_improves_on_bc": "CCIL improves on BC", "ccil_is_worse_than_bc": "CCIL is worse than BC",
           "increases": "increases", "decreases": "decreases", "no_detectable_difference": "no detectable difference",
           "inconclusive": "inconclusive"}
EVIDENCE = [V6 / "analysis.json", V2 / "profiles.json", V2 / "part-a.json", V2 / "part-b.json",
            V2 / "part-b-audit.json", V2 / "part-b-clip-sensitivity.json", V2 / "expert-reconstruction.json",
            V2 / "expert-diagnosis.json", COR / "acceptance.json", COR / "generation.json",
            COR / "canonical-datasets.json", COR / "cross-host-comparison.json", COR / "replay-macos.json",
            COR / "study-smoke.json", COR / "study-preflight.json", COR / "study-audit.json",
            COR / "study-evaluate.json", COR / "study-evaluate-audit.json", CNT / "study-smoke.json",
            CNT / "study-preflight.json", CNT / "study-audit.json", CNT / "study-evaluate.json",
            CNT / "study-evaluate-audit.json"]


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return "../" + str(p.relative_to(REPO / "docs"))


def iv(x, d=3) -> str:
    return f"[{x[0]:+.{d}f}, {x[1]:+.{d}f}]"


def effects() -> list[dict]:
    """Every CCIL - BC upright-occupancy estimate that decided something, in study order."""
    b, c, n = load(V2 / "part-b.json"), load(COR / "study-evaluate.json"), load(CNT / "study-evaluate.json")
    rows = []
    for t in TASKS:
        p = b["primary"][t]
        rows.append({"study": "Historical demonstrations (Part B)", "task": t, "demos": 100 if t == "continuous" else 500,
                     "mean": p["comparison"]["mean"], "interval": p["comparison"]["interval"],
                     "level": p["comparison"]["confidence"], "outcome": p["outcome"], "source": V2 / "part-b.json"})
    for t in TASKS:
        p = c["primary"][t]["corrected_ccil_minus_bc"]
        rows.append({"study": "Corrected demonstrations", "task": t, "demos": 100 if t == "continuous" else 500,
                     "mean": p["mean"], "interval": p["interval"], "level": p["confidence"], "outcome": p["outcome"],
                     "source": COR / "study-evaluate.json"})
    p = n["primary"]["ccil_minus_bc_100"]
    rows.append({"study": "Corrected demonstrations, 100-of-500 subsets", "task": "discontinuous", "demos": 100,
                 "mean": p["mean"], "interval": p["interval"], "level": p["confidence"], "outcome": p["outcome"],
                 "source": CNT / "study-evaluate.json"})
    return rows


def arm_means() -> list[dict]:
    """Noise-free mean upright occupancy and final hold of BC, CCIL and the demonstrator, per condition."""
    b, c, n = load(V2 / "part-b.json"), load(COR / "study-evaluate.json"), load(CNT / "study-evaluate.json")
    out = []
    for t in TASKS:
        demos = 100 if t == "continuous" else 500
        out.append({"condition": f"{t}, historical, {demos} demos", "source": V2 / "part-b.json",
                    "arms": {"BC": b["arm_summaries"][f"{t}/bc/0.0"], "CCIL": b["arm_summaries"][f"{t}/ccil/0.0"]}})
        out.append({"condition": f"{t}, corrected, {demos} demos", "source": COR / "study-evaluate.json",
                    "arms": {"BC": c["arm_summaries"][f"{t}/bc/0.0"], "CCIL": c["arm_summaries"][f"{t}/ccil/0.0"],
                             "Corrected demonstrator": c["arm_summaries"][f"{t}/corrected_demonstrator/0.0"]}})
    out.append({"condition": "discontinuous, corrected, 100 demos", "source": CNT / "study-evaluate.json",
                "arms": {"BC": n["arm_summaries"]["discontinuous/bc_100/0.0"],
                         "CCIL": n["arm_summaries"]["discontinuous/ccil_100/0.0"]}})
    return out


# --- figures ----------------------------------------------------------------------------------------------------------
def _setup():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "pendulum-m3", "svg.fonttype": "none", "font.family": "DejaVu Sans",
                         "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    return buf.getvalue()


def effects_figure(rows) -> str:
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        col = SERIES[r["task"]]
        ax.plot(r["interval"], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot([r["mean"]], [y], "o", color=col, ms=7, mec=SURFACE, mew=2)
        ax.annotate(f"{r['mean']:+.3f}", (r["interval"][1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['study']}\n{r['task']}, {r['demos']} demos, {r['level'] * 100:g}% interval"
                        for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("CCIL − BC, upright occupancy (noise-free, fresh starts, 20 paired seeds)")
    ax.set_xlim(-0.12, 0.34)
    ax.legend(handles=[plt.Line2D([], [], color=SERIES[t], marker="o", lw=2, label=t) for t in TASKS],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
    return _svg(fig)


def arms_figure(groups) -> str:
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ys = list(range(len(groups)))[::-1]
    for y, g in zip(ys, groups):
        for k, (arm, (col, mk)) in enumerate(ARM_STYLE.items()):
            if arm in g["arms"]:
                ax.plot([g["arms"][arm]["upright_occupancy"]], [y + (k - 1) * 0.18], mk, color=col, ms=7,
                        mec=SURFACE, mew=1.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([g["condition"] for g in groups], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Mean upright occupancy over seeds (noise-free, fresh starts; zero action scores 0.000)")
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, ls="", label=a) for a, (c, m) in ARM_STYLE.items()],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8)
    return _svg(fig)


# --- tables -----------------------------------------------------------------------------------------------------------
def tables(rows, groups) -> str:
    v6, a, b = load(V6 / "analysis.json"), load(V2 / "part-a.json"), load(V2 / "part-b.json")
    clip, aud, diag = load(V2 / "part-b-clip-sensitivity.json"), load(V2 / "part-b-audit.json"), \
        load(V2 / "expert-diagnosis.json")
    c, n = load(COR / "study-evaluate.json"), load(CNT / "study-evaluate.json")
    L = ["# Pendulum M3 closeout: generated tables", "",
         "Generated by `tools/pendulum_report/closeout.py` from committed evidence; do not edit by hand. "
         "`--check` verifies that this file and the figures equal a regeneration. The narrative is "
         "[`pendulum-m3-closeout.md`](pendulum-m3-closeout.md).", ""]

    L += ["## T1. Version 6 under the legacy return (historical; the return is defective as a task metric)", "",
          f"Source: [`analysis.json`]({rel(V6 / 'analysis.json')}). CCIL − BC in the legacy quadratic return, "
          "noise-free, 97.5% union interval, 20 seeds.", "",
          "| Task | CCIL − BC | Interval | Outcome | Positive control |", "|---|---|---|---|---|"]
    for t in TASKS:
        p = v6["tasks"][t]["primary"]
        ctl = p.get("positive_control", {}).get("passes")
        L.append(f"| {t} | {p['comparison']['mean']:+.0f} | {iv(p['comparison']['interval'], 0)} | "
                 f"{OUTCOME[p['outcome']]} | {'passes' if ctl else 'fails' if ctl is False else 'n/a'} |")
    L += ["", "Under this return the zero-action policy outscores every arm except CCIL-tuned on continuous "
          "([`pendulum-ccil-study-results.md`](pendulum-ccil-study-results.md)).", ""]

    L += ["## T2. CCIL − BC in upright occupancy, every deciding estimate", "",
          "| Study | Task | Demos | Mean | Interval | Level | Outcome | Source |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['study']} | {r['task']} | {r['demos']} | {r['mean']:+.3f} | {iv(r['interval'])} | "
                 f"{r['level'] * 100:g}% | {OUTCOME[r['outcome']]} | [`{r['source'].name}`]({rel(r['source'])}) |")
    L += ["", "Part A rescoring of the stored v6 rollouts (post hoc, not confirmatory, 95%; "
          f"[`part-a.json`]({rel(V2 / 'part-a.json')})):", ""]
    for t in TASKS:
        x = a["comparisons_95"][f"{t}/ccil-bc/0.0"]["upright_occupancy"]
        L.append(f"- {t}: {x['mean']:+.3f} {iv(x['interval'])}")
    L += ["", "Paired changes (98.75% for the corrected study, 97.5% for the count study):", "",
          "| Change | Task | Mean | Interval | Outcome |", "|---|---|---|---|---|"]
    for t in TASKS:
        p = c["primary"][t]["change_in_ccil_minus_bc"]
        L.append(f"| corrected − historical CCIL − BC | {t} | {p['mean']:+.3f} | {iv(p['interval'])} | "
                 f"{OUTCOME[p['outcome']]} |")
    p = n["primary"]["change_100_minus_500"]
    L += [f"| (CCIL − BC) at 100 − at 500 demos | discontinuous | {p['mean']:+.3f} | {iv(p['interval'])} | "
          f"{OUTCOME[p['outcome']]} |", ""]
    for t in TASKS:
        ctl = b["primary"][t]["positive_control"]
        L.append(f"- Part B positive control (BC − BC 10%), {t}: {ctl['comparison']['mean']:+.3f} "
                 f"{iv(ctl['comparison']['interval'])}, {'passes' if ctl['passes'] else 'fails'}.")
    L.append("")

    L += ["## T3. Noise-free arm means", "", "| Condition | Arm | Upright occupancy | Final 2 s hold | Height | "
          "Source |", "|---|---|---|---|---|---|"]
    for g in groups:
        for arm, s in g["arms"].items():
            L.append(f"| {g['condition']} | {arm} | {s['upright_occupancy']:.3f} | {s['final_two_second_hold']:.3f} | "
                     f"{s['mean_normalized_height']:.3f} | [`{g['source'].name}`]({rel(g['source'])}) |")
    L.append("")

    L += ["## T4. Evaluation noise, channels separated (descriptive, 95%)", "",
          "CCIL − BC in upright occupancy. The observation mixture is not a physically scaled sensor model.", "",
          "| Study | Task | Channel | Level | Mean | Interval |", "|---|---|---|---|---|---|"]
    for name, rep, key in (("Historical (Part B)", b, "{t}/ccil-bc/{ch}/{lv}"),
                           ("Corrected", c, "{t}/ccil-bc/{ch}/{lv}")):
        for t in TASKS:
            for ch in ("observation_only", "action_only"):
                for lv in ("0.01", "0.05", "0.1"):
                    x = rep["noise_95"][key.format(t=t, ch=ch, lv=lv)]
                    x = x.get("upright_occupancy", x)
                    L.append(f"| {name} | {t} | {ch.replace('_', ' ')} | {lv} | {x['mean']:+.3f} | "
                             f"{iv(x['interval'])} |")
    L.append("")

    L += ["## T5. Action bound", "",
          f"Commanded actions beyond ±3, noise-free ([`part-b-audit.json`]({rel(V2 / 'part-b-audit.json')}), "
          f"[`study-evaluate.json`]({rel(COR / 'study-evaluate.json')}), "
          f"[`study-evaluate.json`]({rel(CNT / 'study-evaluate.json')})):", "",
          "| Study | Task | Arm | Fraction of steps | Largest excess |", "|---|---|---|---|---|"]
    for t in TASKS:
        for arm in ("bc", "ccil"):
            u = aud["action_bound"]["by_unit"][f"{t}/{arm}/0.0"]["commanded"]
            L.append(f"| Historical (Part B) | {t} | {arm.upper()} | {u['fraction_over']:.3f} | {u['max_excess']:.1e} |")
    for t in TASKS:
        for arm in ("bc", "ccil"):
            s = c["arm_summaries"][f"{t}/{arm}/0.0"]
            L.append(f"| Corrected | {t} | {arm.upper()} | {s['commanded_out_of_range_fraction']:.3f} | — |")
    for arm in ("bc_100", "ccil_100"):
        s = n["arm_summaries"][f"discontinuous/{arm}/0.0"]
        L.append(f"| Count (100 demos) | discontinuous | {arm.upper()} | {s['commanded_out_of_range_fraction']:.3f} | — |")
    L += ["", f"Clipped to ±3 and re-evaluated (Part B units, [`part-b-clip-sensitivity.json`]"
          f"({rel(V2 / 'part-b-clip-sensitivity.json')})):", ""]
    for t in TASKS:
        x = clip["primary_rule_on_clipped"][t]
        L.append(f"- {t}: CCIL − BC {x['comparison']['mean']:+.3f} {iv(x['comparison']['interval'])}, "
                 f"{OUTCOME[x['outcome']]}.")
    big = max((v["upright_occupancy"], k) for k, v in clip["max_abs_change_by_arm"].items())
    L += [f"- Largest per-seed change in any arm's occupancy: {big[0]:.3f} ({big[1]}).", ""]

    L += ["## T6. Historical demonstrator wrapping defect (post hoc diagnosis)", "",
          f"Source: [`expert-diagnosis.json`]({rel(V2 / 'expert-diagnosis.json')}).", "",
          "| Task | Window states | Affected-side states | Stored action there | Episodes visiting | "
          "Occupancy, visiting | Occupancy, not visiting |", "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        w, o = diag["tasks"][t]["claim2_wrapping_inconsistency"], diag["tasks"][t]["claim4_descriptive_occupancy"]
        L.append(f"| {t} | {w['window_states']} | {w['affected_side_states']} | "
                 f"{', '.join(f'{v:g}' for v in w['affected_side_stored_action_values'])} | "
                 f"{o['episodes_visiting_affected_side']} of {o['episodes']} | "
                 f"{o['visiting']['mean_upright_occupancy']:.3f} | {o['not_visiting']['mean_upright_occupancy']:.3f} |")
    L += ["", "The occupancy split is a descriptive association, not a causal effect.", ""]

    L += ["## T7. Evidence files", "", "| File | sha256 |", "|---|---|"]
    for p in EVIDENCE:
        L.append(f"| [`{p.relative_to(EV)}`]({rel(p)}) | `{hashlib.sha256(p.read_bytes()).hexdigest()}` |")
    return "\n".join(L) + "\n"


def render() -> dict:
    rows, groups = effects(), arm_means()
    return {OUT: tables(rows, groups), FIGS / "pendulum-m3-effects.svg": effects_figure(rows),
            FIGS / "pendulum-m3-arms.svg": arms_figure(groups)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = render()
    if args.check:
        stale = [str(p.relative_to(REPO)) for p, text in files.items() if not p.exists() or p.read_text() != text]
        print("stale: " + ", ".join(stale) if stale else "up to date")
        return 1 if stale else 0
    FIGS.mkdir(parents=True, exist_ok=True)
    for p, text in files.items():
        p.write_text(text)
        print("wrote", p.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
