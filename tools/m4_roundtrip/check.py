#!/usr/bin/env python3
"""The deterministic LeRobot round trip (#29): a LeRobot dataset through CCIL and back as an augmented bundle.

    uv run --group evidence --extra lerobot python tools/m4_roundtrip/check.py --out DIR

The fixture, docs/fixtures/lerobot-roundtrip/, was written by LeRobot 0.6.1 (tools/m4_roundtrip/lerobot_side.py
fixture): a double integrator with four episodes (12, 9, 15 and 6 frames), a 2-d state, an 8x8 RGB image and a 1-d
action. It is synthetic, so it is safe to redistribute. LeRobot's own writes are not byte-deterministic (a random
fingerprint in the Parquet metadata, sampled image statistics), so the fixture is committed and checked by hash.

The pipeline, run twice from the fixture:
1. convert (#27), naming the image in ignore: CCIL never reads images until M5;
2. `ccil labels generate` (the analytic double-integrator recipe, 3-step action chunks);
3. build_supervision, consuming the chunks;
4. export_overlay (#28): the fixture, unchanged and images included, plus the ccil/ overlay;
and once more with a support radius that accepts nothing (the zero-accepted case). Checks:
- the two runs give identical conversion manifests, candidates, supervision manifests and overlays;
- the bundle's LeRobot portion is the fixture byte for byte, and reloading it gives the fixture's episodes, frames,
  states, actions, timestamps and tasks;
- the overlay's labels are exactly the run's accepted candidates on the training episodes, with their provenance;
  rejected candidates and the held-out episodes' labels appear only in the tables;
- chunks that would cross an episode's end are masked there, never filled from the next episode;
- an unchanged LeRobot consumer (LeRobot 0.6.1's dataset and a stock DataLoader) iterates the bundle and decodes
  its images; CCIL's training dataset sees every label with state-only keys, and refuses a policy that reads the
  image, which labels do not have.
Writes DIR/report.json, DIR/accepted.csv, DIR/rejected.csv, DIR/quality.json and DIR/consumer-*.json.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURE = REPO / "docs" / "fixtures" / "lerobot-roundtrip"
CHUNK = 5  # a chunk position k >= 1 holds the action at frame t + k - 1: K = 5 reaches past an episode's end
# The recipe's 0.01 accepts only exact labels on this fixture, whose label distances span 0 to 0.10 (median about
# 0.08): 0.08 gives both accepted and rejected labels, the cases #29 asks for.
SUPPORT_RADIUS = 0.08
# No held-out episodes: the recipe holds out half, which on four episodes keeps every accepted label near an
# episode end out of the overlay; the chunk boundary is the case #29 asks for.
VALIDATION_FRACTION = 0.0
CONSUMER = "0.6.1"

from ccil.core import IdentityRepresentation  # noqa: E402
from ccil.lerobot import convert, read_dataset  # noqa: E402
from ccil.lerobot.overlay import export_overlay, read_overlay  # noqa: E402
from ccil.schema import REASON_CODES, read_candidates, read_trajectories  # noqa: E402
from ccil.supervision import build_supervision  # noqa: E402


def tree(d: Path, skip=None) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file() and not (skip and p.relative_to(d).parts[0] == skip)}


def pipeline(work: Path, extra=()) -> dict:
    work.mkdir(parents=True)
    conv = work / "conversion"
    conv.mkdir()
    convert(FIXTURE, conv / "trajectories.npz", conv / "manifest.json", ignore=["observation.image"])
    args = [sys.executable, "-m", "ccil", "labels", "generate", "--recipe", "analytic_double_integrator",
            "--set", f"data.path={conv / 'trajectories.npz'}", "--set", f"run.output_dir={work / 'runs'}",
            "--set", f"generation.action_chunk_size={CHUNK}", "--set", f"generation.support_radius={SUPPORT_RADIUS}",
            "--set", f"data.validation_fraction={VALIDATION_FRACTION}",
            *[x for kv in extra for x in ("--set", kv)]]
    r = subprocess.run(args, capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    runs = {line.split()[0]: Path(line.split()[2]) for line in r.stdout.splitlines() if len(line.split()) == 3}
    runs = {k: (v if v.is_absolute() else work / v) for k, v in runs.items()}
    prepared, labelled = runs["data.prepare"], runs["labels.generate"]
    cands_path = labelled / "labels" / "candidates.npz"
    dataset = read_trajectories(prepared / "data" / "dataset.npz")
    cands = read_candidates(cands_path, dataset)
    split_path = prepared / "data" / "split.json"  # absent when no episode is held out
    sup = build_supervision(prepared / "data" / "dataset.npz", "ccil", representation=IdentityRepresentation("state"),
                            candidates=cands_path, split=split_path if split_path.exists() else None,
                            action_chunks="consume")
    meta = export_overlay(FIXTURE, work / "bundle", sup, conversion_manifest=conv / "manifest.json",
                          candidates=cands_path, prepared_run=prepared)
    split = json.loads(split_path.read_text()) if split_path.exists() else {"validation": []}
    return {"work": work, "conversion": json.loads((conv / "manifest.json").read_text()), "cands": cands,
            "cands_sha": hashlib.sha256(cands_path.read_bytes()).hexdigest(), "sup": sup, "meta": meta,
            "dataset": dataset, "split": split, "bundle": work / "bundle"}


def tables(out: Path, run: dict) -> dict:
    """accepted.csv and rejected.csv from the candidates file, and the quality report."""
    cands, split = run["cands"], run["split"]
    held = set(split.get("validation", []))
    _, rows = read_overlay(run["bundle"])
    in_overlay = set(rows["candidate_row"].tolist())
    names = sorted(cands.measurements)
    lengths = {ep.episode_id: ep.length for ep in run["dataset"].episodes}
    masked = (~cands.action_chunk_mask).sum(1) if cands.action_chunk_mask is not None else np.zeros(len(cands.accepted))
    with open(out / "accepted.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")  # LF: the committed tables stay clean under git diff --check
        w.writerow(["candidate_row", "episode", "frame", "weight", "in_overlay", "chunk_positions_masked", *names])
        for i in np.flatnonzero(cands.accepted):
            w.writerow([i, cands.source_episode_id[i], int(cands.source_frame[i]), f"{cands.weight[i]:.17g}",
                        int(i in in_overlay), int(masked[i]), *[f"{cands.measurements[n][i]:.17g}" for n in names]])
    with open(out / "rejected.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["candidate_row", "episode", "frame", "reasons", *names])
        for i in np.flatnonzero(~cands.accepted):
            reasons = ";".join(REASON_CODES[j] for j in np.flatnonzero(cands.reasons[i]))
            w.writerow([i, cands.source_episode_id[i], int(cands.source_frame[i]), reasons,
                        *[f"{cands.measurements[n][i]:.17g}" for n in names]])
    per_episode = {}
    for e in lengths:
        sel = cands.source_episode_id == e
        per_episode[e] = {"frames": lengths[e], "candidates": int(sel.sum()), "accepted": int(cands.accepted[sel].sum()),
                          "held_out": e in held}
    reasons = {REASON_CODES[j]: int(cands.reasons[:, j].sum()) for j in range(len(REASON_CODES)) if cands.reasons[:, j].any()}
    near_end = [int(masked[i]) for i in np.flatnonzero(cands.accepted) if masked[i]]
    dist = cands.measurements.get("distance")
    return {"support_radius": SUPPORT_RADIUS, "distance_quantiles": None if dist is None else
            {q: float(np.quantile(dist, float(q))) for q in ("0", "0.25", "0.5", "0.75", "1")},
            "candidates": int(len(cands.accepted)), "accepted": int(cands.accepted.sum()),
            "rejected": int((~cands.accepted).sum()), "rejection_reasons": reasons, "per_episode": per_episode,
            "overlay_labels": len(in_overlay), "accepted_on_held_out_episodes":
            int(sum(cands.accepted[i] and cands.source_episode_id[i] in held for i in range(len(cands.accepted)))),
            "chunk_size": CHUNK, "accepted_labels_with_masked_chunk_positions": len(near_end),
            "masked_positions_histogram": {str(k): near_end.count(k) for k in sorted(set(near_end))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:600]}", flush=True)

    recorded = json.loads((HERE / "fixture.json").read_text())
    fixture_tree = tree(FIXTURE)
    record("fixture_is_the_recorded_one", fixture_tree == recorded["files"], files=fixture_tree,
           written_by=recorded["written_by"])

    first, second = pipeline(out / "run-a"), pipeline(out / "run-b")
    same = {"conversion_manifest": first["conversion"] == second["conversion"],
            "candidates": first["cands_sha"] == second["cands_sha"],
            "supervision_manifest": first["sup"].manifest() == second["sup"].manifest(),
            "overlay_meta": first["meta"] == second["meta"],
            "bundle": tree(first["bundle"]) == tree(second["bundle"])}
    record("repeated_runs_are_identical", all(same.values()), **same, tolerance="exact (CPU)",
           overlay_shards=first["meta"]["shards"], candidates_sha256=first["cands_sha"])

    src = read_dataset(FIXTURE, ignore=["observation.image"])
    back = read_dataset(first["bundle"], ignore=["observation.image"])
    record("reloaded_source_records_preserved",
           tree(first["bundle"], skip="ccil") == fixture_tree
           and [ep.episode_id for ep in back.episodes] == [ep.episode_id for ep in src.episodes]
           and all(np.array_equal(x.observations["state"], y.observations["state"]) and np.array_equal(x.actions, y.actions)
                   and np.array_equal(x.timestamps, y.timestamps) and x.task == y.task
                   for x, y in zip(back.episodes, src.episodes)),
           episodes=[ep.length for ep in src.episodes], tasks=sorted({ep.task for ep in src.episodes}))

    cands = first["cands"]
    m, rows = read_overlay(first["bundle"])
    held = set(first["split"].get("validation", []))
    expected = np.array([i for i in np.flatnonzero(cands.accepted) if cands.source_episode_id[i] not in held])
    order = np.argsort(rows["candidate_row"])
    masks = rows["action_chunk_mask"][order]
    chunks = rows["action_chunk"][order].reshape(len(order), CHUNK, -1)
    record("labels_and_provenance_match_the_run",
           np.array_equal(np.sort(rows["candidate_row"]), expected)
           and np.array_equal(rows["source_frame_index"][order], cands.source_frame[expected])
           and np.array_equal(rows["weight"][order], cands.weight[expected])
           and np.array_equal(masks, cands.action_chunk_mask[expected])
           and bool(np.all(chunks[~masks] == 0)) and m["run_id"] == cands.run_id
           and m["inputs"]["candidates"] == first["cands_sha"],
           labels=int(len(expected)), held_out_episodes=sorted(held), run_id=m["run_id"])

    quality = tables(out, first)
    (out / "quality.json").write_text(json.dumps(quality, indent=1, sort_keys=True) + "\n")
    mask = cands.action_chunk_mask
    prefix = bool(np.all(mask[:, :-1] >= mask[:, 1:]))  # true positions first: a chunk never re-opens
    zero = bool(np.all(cands.action_chunks[~mask] == 0))  # masked positions hold zeros, never the next episode
    record("quality_report_and_tables", quality["accepted"] > 0 and quality["rejected"] > 0
           and quality["accepted_labels_with_masked_chunk_positions"] > 0 and prefix and zero
           and int((~masks).sum()) > 0, chunk_masks_are_prefixes=prefix, masked_positions_are_zero=zero,
           overlay_masked_positions=int((~masks).sum()), **quality)

    none = pipeline(out / "run-none", extra=("generation.support_radius=1e-12",))
    mn, rn = read_overlay(none["bundle"])
    record("zero_accepted_gives_an_empty_overlay", mn["rows"] == 0 and len(rn["candidate_row"]) == 0
           and int(none["cands"].accepted.sum()) == 0 and tree(none["bundle"], skip="ccil") == fixture_tree,
           candidates=int(len(none["cands"].accepted)))

    # The consumer: LeRobot's dataset and a stock DataLoader, in the release's own environment, with ccil installed.
    spec = importlib.util.spec_from_file_location("m4_lerobot_check", REPO / "tools" / "m4_lerobot" / "check.py")
    m4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m4)
    python, freeze = m4.env_for(out, CONSUMER)
    dist = out / "dist"
    subprocess.run(["uv", "build", "-q", "--wheel", "--out-dir", str(dist), str(REPO)], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), str(next(dist.glob("ccil-*.whl")))], check=True)
    transcripts = {}
    for name, run in (("labels", first), ("zero", none)):
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_LEROBOT_HOME": str(out / "hf-home")}
        r = subprocess.run([str(python), str(HERE / "lerobot_side.py"), "consume", str(run["bundle"]),
                            str(out / f"consumer-{name}.json")], capture_output=True, text=True, env=env)
        transcripts[name] = json.loads((out / f"consumer-{name}.json").read_text()) if r.returncode == 0 else {"error": r.stderr[-800:]}
    t, z = transcripts["labels"], transcripts["zero"]
    frames = sum(ep.length for ep in src.episodes)
    record("unchanged_lerobot_consumer_iterates_the_bundle",
           t.get("stock", {}).get("frames") == frames and t["stock"]["images_decoded"] == frames
           and t["stock"]["episodes"] == len(src.episodes) and t.get("image_policy_refused") is True
           and sorted(t["ccil"]["label_candidate_rows"]) == sorted(expected.tolist())
           and z.get("ccil", {}).get("labels") == 0 and z["stock"]["content_sha256"] == t["stock"]["content_sha256"],
           consumer=CONSUMER, transcripts=transcripts,
           installed={l.split("==")[0]: l.split("==")[1] for l in freeze.split() if l.split("==")[0].lower() in
                      ("lerobot", "torch", "pyarrow", "numpy")})

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "command": "uv run --group evidence --extra lerobot python tools/m4_roundtrip/check.py --out DIR",
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml",
                                                     "uv.lock", "docs/fixtures/lerobot-roundtrip")),
                              "python": platform.python_version(), "platform": platform.platform()}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
