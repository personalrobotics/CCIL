#!/usr/bin/env python3
"""The tables of the Pendulum study report (#25), generated from the committed analysis and records.

    python tools/pendulum_report/report.py            # rewrite docs/experiments/pendulum-ccil-study-tables.md
    python tools/pendulum_report/report.py --check    # exit 1 if the committed tables differ from a regeneration

Every number comes from docs/fixtures/evidence/pendulum-study-v6/analysis.json (the predeclared analysis,
tools/m3_experiments/analyze_v6.py at the frozen commit) or, for the superseded version 4 run, from its records.
Each per-seed value links to the record it came from. Nothing here computes an interval: the analysis did.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "experiments" / "pendulum-ccil-study-tables.md"
V6 = REPO / "docs" / "fixtures" / "evidence" / "pendulum-study-v6"
V4 = REPO / "docs" / "fixtures" / "evidence" / "pendulum-study-v4"
TASKS = ("continuous", "discontinuous")
OUTCOMES = {"ccil_improves_on_bc": "CCIL improves on BC", "ccil_is_worse_than_bc": "CCIL is worse than BC",
            "no_detectable_difference": "No detectable difference", "inconclusive": "Inconclusive"}


def rel(path: Path) -> str:
    return "../" + str(path.relative_to(REPO / "docs"))


def iv(pair) -> str:
    return f"[{pair[0]:.0f}, {pair[1]:.0f}]"


def arm_dir(root: Path, phase: str, task: str, arm: str) -> Path:
    """The record directory of an arm: tuned arms carry their chosen value (arm@value)."""
    plain = root / "results" / phase / task / arm
    if plain.is_dir():
        return plain
    found = sorted((root / "results" / phase / task).glob(f"{arm}@*"))
    if len(found) != 1:
        raise SystemExit(f"no unique record directory for {phase}/{task}/{arm}: {found}")
    return found[0]


def record_link(task: str, arm: str, seed) -> str:
    return rel(arm_dir(V6, "study", task, arm) / f"s{seed}.json")


def comparison_row(c: dict) -> str:
    return (f"| {c['A']} − {c['B']} | {c['noise']} | {c['n']} | {c['mean']:+.0f} | {iv(c['t'])} | {iv(c['bca'])} "
            f"| **{iv(c['interval'])}**" + (f" ({c['note']})" if c.get("note") else "") + " |")


def v6_tables(a: dict) -> list[str]:
    L = ["## Version 6", "",
         f"Analysis: [`analysis.json`]({rel(V6 / 'analysis.json')}) ({a['version']}), code `{a['code_commit']}`, "
         f"frozen tooling `{a['tooling_commit']}`, numpy {a['numpy']}, scipy {a['scipy']}, {a['resamples']} "
         f"resamples, rng seed {a['rng_seed']}. Rollouts re-audited: {a['rollouts_reaudited']}. Algorithmic "
         f"failures: {len(a['failures'])}.", "",
         "### Primary decision (§8): CCIL − BC, noise-free, c = 0.975", "",
         "| Task | n | d̄ | paired t | BCa | union | outcome | relative effect (BCa) | positive control: BC − BC 10%, union |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        p = a["tasks"][t]["primary"]
        c, pc, re = p["comparison"], p["positive_control"], p["relative_effect"]
        ok = pc.get("passes", pc["comparison"]["interval"][0] > 0)
        L.append(f"| {t} | {c['n']} | {c['mean']:+.0f} | {iv(c['t'])} | {iv(c['bca'])} | **{iv(c['interval'])}** "
                 f"| **{OUTCOMES.get(p['outcome'], p['outcome'])}** | {re['value']:+.3f} "
                 f"[{re['bca'][0]:+.3f}, {re['bca'][1]:+.3f}] | {pc['comparison']['mean']:+.0f} "
                 f"{iv(pc['comparison']['interval'])}: {'passes' if ok else 'fails'} |")
    L += ["", "Inconclusive reasons: " + "; ".join(
        f"{t}: {', '.join(a['tasks'][t]['primary']['inconclusive_reasons']) or 'none'}" for t in TASKS) + ".", ""]

    L += ["### Per-seed primary differences, CCIL − BC (noise-free)", ""]
    for t in TASKS:
        c = a["tasks"][t]["primary"]["comparison"]
        L += [f"**{t}**", "", "| Seed | BC | CCIL | CCIL − BC |", "|---|---|---|---|"]
        arms = a["tasks"][t]["arms"]
        for s in c["seeds"]:
            bc, cc = arms["bc"]["0.0"]["per_seed"][str(s)], arms["ccil"]["0.0"]["per_seed"][str(s)]
            L.append(f"| {s} | [{bc:.1f}]({record_link(t, 'bc', s)}) | [{cc:.1f}]({record_link(t, 'ccil', s)}) "
                     f"| {c['per_seed'][str(s)]:+.1f} |")
        L.append("")

    L += ["### Arms: mean ± std of R over seeds, per evaluation noise level", ""]
    for t in TASKS:
        arms = a["tasks"][t]["arms"]
        levels = sorted({lv for v in arms.values() for lv in v}, key=float)
        L += [f"**{t}**", "", "| Arm | " + " | ".join(f"noise {lv}" for lv in levels) + " |",
              "|---|" + "---|" * len(levels)]
        for arm in sorted(arms):
            cells = [f"{arms[arm][lv]['mean']:.0f} ± {arms[arm][lv]['std']:.0f} (n {arms[arm][lv]['n']})"
                     if lv in arms[arm] else "" for lv in levels]
            L.append(f"| [{arm}]({rel(arm_dir(V6, 'study', t, arm))}) | " + " | ".join(cells) + " |")
        L.append("")

    head = ["| Comparison | noise | n | d̄ | paired t | BCa | union |", "|---|---|---|---|---|---|---|"]
    for title, key in (("Secondary comparisons (descriptive, c = 0.95)", "secondary"),
                       ("CCIL − BC per evaluation noise level (descriptive, c = 0.95)", "by_noise"),
                       ("Per demonstration count (descriptive, c = 0.95)", "by_data_size"),
                       ("Against the zero-action reference (descriptive, c = 0.95)", "zero_reference")):
        L += [f"### {title}", ""]
        for t in TASKS:
            L += [f"**{t}**", ""] + head + [comparison_row(c) for c in a["tasks"][t][key]] + [""]

    L += ["### Paper comparison (informational, Fig. 4(c))", "",
          "| Task | Arm | paper | this study: mean ± std (n) |", "|---|---|---|---|"]
    for t in TASKS:
        for arm, v in a["tasks"][t]["paper_comparison"].items():
            s = v["this_study"]
            L.append(f"| {t} | {arm} | {v['paper']} | {s['mean']:.0f} ± {s['std']:.0f} ({s['n']}) |")
    L.append("")

    L += ["### Tuning choice (§6)", "",
          f"[`tuning-choice.json`]({rel(V6 / 'tuning-choice.json')}), sha256 `{a['tuning_choice_sha256']}`.", ""]
    for t in TASKS:
        for arm, v in a["tuning_choice"][t].items():
            L += [f"**{t} / {arm}**: chose **{v['value']}** ({v['rule']}).", "",
                  "| Value | eligible | criterion per seed | mean |", "|---|---|---|---|"]
            for val, row in sorted(v["table"].items(), key=lambda kv: float(kv[0])):
                if "mean_validation_loss" in row:
                    per, mean = row["validation_loss"], f"{row['mean_validation_loss']:.3g}"
                    cells = ", ".join(f"{k}: {x:.3g}" for k, x in per.items())
                else:
                    per, mean = row["R"], f"{row['mean_R']:.0f}"
                    cells = ", ".join(f"{k}: {x:.0f}" for k, x in per.items())
                L.append(f"| {val} | {row['eligible']} | {cells} | {mean} |")
            L.append("")

    L += ["### Label accuracy (CCIL arms; true one-step error of accepted labels)", "",
          "| Task / arm | units | acceptance rate (min) | mean error, mean over seeds | 95th percentile, max over seeds |",
          "|---|---|---|---|---|"]
    groups: dict[str, list] = {}
    for key, v in a["label_accuracy"].items():
        task, arm, _ = key.split("/")
        groups.setdefault(f"{task} / {arm}", []).append(v)
    for g, vs in sorted(groups.items()):
        L.append(f"| {g} | {len(vs)} | {min(v['acceptance_rate'] for v in vs):.4f} "
                 f"| {statistics.fmean(v['true_label_error']['mean'] for v in vs):.2e} "
                 f"| {max(v['true_label_error']['quantiles']['0.95'] for v in vs):.2e} |")
    L += ["", "Raw state units, one step of the batched evaluation integrator; per-unit values are in the "
          "analysis (`label_accuracy`) and in each record.", ""]
    return L


def v4_tables() -> list[str]:
    choice = json.loads((V4 / "tuning-choice.json").read_text())
    L = ["## Version 4 (incomplete and superseded)", "",
         "Frozen tooling `cba34fe`, `ccil` `847b060`. The tuning phase completed; the operator stopped the study "
         "after 21 of 520 units to move to the faster stack (predeclaration, History). These records are reported "
         "as they are and are never mixed with version 6's. No interval or decision is computed from them.", "",
         f"### Tuning choice: [`tuning-choice.json`]({rel(V4 / 'tuning-choice.json')})", ""]
    for t in TASKS:
        for arm, v in choice[t].items():
            L += [f"**{t} / {arm}**: chose **{v['value']}** ({v['rule']}).", "",
                  "| Value | eligible | R per seed | mean R |", "|---|---|---|---|"]
            for val, row in sorted(v["table"].items(), key=lambda kv: float(kv[0])):
                L.append(f"| {val} | {row['eligible']} | "
                         + ", ".join(f"{k}: {x:.0f}" for k, x in row["R"].items()) + f" | {row['mean_R']:.0f} |")
            L.append("")
    L += ["### The 21 study units that completed", "", "| Task | Arm | Seed | status | R, noise-free |",
          "|---|---|---|---|---|"]
    for p in sorted((V4 / "results" / "study").rglob("s*.json")):
        r = json.loads(p.read_text())
        R = (r.get("evaluations") or {}).get("0.0", {}).get("R")
        L.append(f"| {r['task']} | {p.parent.name} | {r['seed']} | {r['status']} | "
                 f"[{'n/a' if R is None else f'{R:.1f}'}]({rel(p)}) |")
    return L + [""]


def render() -> str:
    a = json.loads((V6 / "analysis.json").read_text())
    head = ["# Pendulum CCIL efficacy study: tables (#25)", "",
            "Generated by `tools/pendulum_report/report.py`; do not edit. The narrative is "
            "[`pendulum-ccil-study-results.md`](pendulum-ccil-study-results.md).", ""]
    return "\n".join(head + v6_tables(a) + v4_tables())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    text = render()
    if ap.parse_args().check:
        same = OUT.exists() and OUT.read_text() == text
        print("tables match" if same else f"{OUT} differs from a regeneration")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
