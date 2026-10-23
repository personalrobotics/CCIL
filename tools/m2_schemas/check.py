#!/usr/bin/env python3
"""Evidence for the ccil data schemas (#16).

    PYTHONPATH=src python tools/m2_schemas/check.py --out DIR

Valid examples (vector, multimodal image, action chunks, multi-reason
candidates) round-trip exactly; one invalid example per rule fails with the
expected diagnostic and is left unmodified; the analytic fixture travels from
trajectories through closed-form candidates to accepted labels with identity
and measurements intact; all 15 shipped datasets convert from the M1 format
exactly. Writes DIR/report.json.
"""
import argparse
import copy
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "tools" / "analytic_fixture"):
    sys.path.insert(0, str(p))
from ccil.schema import (REASON_CODES, CandidateSet, Episode, SchemaError, TrajectoryDataset,  # noqa: E402
                         read_candidates, read_trajectories, write_candidates, write_trajectories)
from ccil.schema.io import _meta  # noqa: E402
from ccil.schema.legacy import from_m1_npz  # noqa: E402
from oracle import episodes_to_transitions  # noqa: E402
from system import DoubleIntegrator  # noqa: E402

R = {c: i for i, c in enumerate(REASON_CODES)}


def same(a, b):
    """Exact structural equality of datasets or candidate sets (NaN-equal, dtype-checked)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape \
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if hasattr(a, "__dataclass_fields__"):
        return type(a) is type(b) and all(same(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)
    return a == b


def fixture_dataset():
    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    by_ep = {}
    for tr in episodes_to_transitions(config):
        by_ep.setdefault(tr["episode"], []).append(tr)
    eps = []
    for name, trs in by_ep.items():
        if name == "corrupt":
            continue  # the NaN episode is invalid at the dataset level by design
        obs = np.stack([t["s"] for t in trs] + [trs[-1]["s_next"]]).astype(np.float32)
        act = np.vstack([t["a"] for t in trs] + [np.zeros(1, np.float32)]).astype(np.float32)
        eps.append(Episode(episode_id=name, observations={"state": obs}, actions=act, terminated=False))
    return TrajectoryDataset(eps, {"source": "tools/analytic_fixture/config.json"}), config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(section, name, ok, **detail):
        cases.append({"section": section, "case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {section}.{name} {'' if ok else detail}")

    rng = np.random.default_rng(0)
    T = 6

    def vector_ep(i, **kw):
        return Episode(episode_id=f"ep{i}", observations={"state": rng.normal(size=(T, 3)).astype(np.float32)},
                       actions=rng.normal(size=(T, 2)).astype(np.float32), **kw)

    # --- A: valid examples round-trip exactly ------------------------------
    vector = TrajectoryDataset([vector_ep(0, terminated=True), vector_ep(1)], {"example": "vector"})
    image = TrajectoryDataset([Episode(
        episode_id="cam0",
        observations={"state": rng.normal(size=(T, 3)).astype(np.float32),
                      "image.top": rng.integers(0, 256, size=(T, 8, 8, 3), dtype=np.uint8)},
        actions=rng.normal(size=(T, 2)).astype(np.float32), timestamps=np.arange(T) * 0.05, task="push the block")],
        {"example": "multimodal"})
    K = 3
    mask = (np.arange(T)[:, None] + np.arange(K)[None, :]) < T
    act = rng.normal(size=(T, 2)).astype(np.float32)
    chunks = np.zeros((T, K, 2), np.float32)
    for t in range(T):
        for k in range(K):
            if mask[t, k]:
                chunks[t, k] = act[t + k]
    chunked = TrajectoryDataset([Episode(episode_id="chunk0", observations={"state": rng.normal(size=(T, 3)).astype(np.float32)},
                                         actions=act, action_chunks=chunks, action_chunk_mask=mask)], {"example": "chunks"})
    for name, ds in (("vector", vector), ("multimodal_image", image), ("action_chunks", chunked)):
        path = out / f"{name}.npz"
        write_trajectories(ds, path)
        back = read_trajectories(path)
        record("A", f"valid_{name}_round_trips", same(ds, back), episodes=len(ds.episodes),
               modalities=sorted(ds.episodes[0].observations))

    reasons = np.zeros((4, len(REASON_CODES)), bool)
    reasons[1, R["out_of_support"]] = True
    reasons[2, [R["out_of_support"], R["action_out_of_bounds"]]] = True
    reasons[3, R["non_finite"]] = True
    cand = CandidateSet(
        run_id="example", generator={"name": "example", "version": "1", "params_hash": "0" * 16},
        source_episode_id=np.array(["ep0", "ep0", "ep1", "ep1"]), source_frame=np.array([0, 1, 0, 4]),
        observations={"state": np.vstack([rng.normal(size=(3, 3)), np.full((1, 3), np.nan)]).astype(np.float32)},
        actions=rng.normal(size=(4, 2)).astype(np.float32),
        target_observations={"state": rng.normal(size=(4, 3)).astype(np.float32)},
        accepted=np.array([True, False, False, False]), reasons=reasons,
        measurements={"distance": np.array([0.1, 0.9, 1.2, np.nan])}, weight=np.array([1.0, 0, 0, 0]))
    cand.validate(vector)
    write_candidates(cand, out / "candidates.npz")
    back = read_candidates(out / "candidates.npz", vector)
    record("A", "valid_multi_reason_candidates_round_trip", same(cand, back),
           row_reasons=[back.row_reasons(i) for i in range(4)])

    # --- B: invalid examples, one per rule ---------------------------------
    def expect(name, build, rule, where=None, validate=lambda x: x.validate()):
        obj = build()
        snapshot = copy.deepcopy(obj)
        try:
            validate(obj)
            got = None
        except SchemaError as e:
            got = e.violations
        hit = got is not None and any(v["rule"] == rule and (where is None or where in v["where"]) for v in got)
        record("B", name, hit and same(obj, snapshot), expected_rule=rule, expected_where=where, violations=got)

    def mutated(fn, base=vector):
        ds = copy.deepcopy(base)
        fn(ds)
        return ds

    expect("nan_state", lambda: mutated(lambda d: d.episodes[1].observations["state"].__setitem__((4, 0), np.nan)),
           "non_finite", "episode 1 (ep1), state, frame 4")
    expect("wrong_rank", lambda: mutated(lambda d: d.episodes[0].observations.__setitem__("state", np.zeros(T, np.float32))), "rank")
    expect("integer_actions", lambda: mutated(lambda d: setattr(d.episodes[0], "actions", np.zeros((T, 2), np.int64))), "dtype")
    expect("non_increasing_timestamps", lambda: mutated(lambda d: d.episodes[0].timestamps.__setitem__(3, 0.0), image),
           "timestamps", "frame 3")
    expect("chunk_crosses_episode", lambda: mutated(lambda d: d.episodes[0].action_chunk_mask.__setitem__((T - 1, 1), True), chunked),
           "chunk_crosses_episode", f"frame {T - 1}")
    expect("duplicate_episode_id", lambda: mutated(lambda d: setattr(d.episodes[1], "episode_id", "ep0")), "episode_id")
    expect("modality_shape_mismatch", lambda: mutated(lambda d: d.episodes[1].observations.__setitem__(
        "state", np.zeros((T, 4), np.float32))), "dimension_mismatch")
    expect("bad_modality_name", lambda: mutated(lambda d: d.episodes[0].observations.__setitem__(
        "rgb", d.episodes[0].observations.pop("state"))), "modality_name")
    expect("action_out_of_bounds", lambda: copy.deepcopy(vector), "action_out_of_bounds",
           validate=lambda x: x.validate(action_low=-0.1, action_high=0.1))

    def cmut(fn):
        c = copy.deepcopy(cand)
        fn(c)
        return c

    expect("missing_source_identity", lambda: cmut(lambda c: c.source_episode_id.__setitem__(0, "")), "source_identity")
    expect("source_frame_not_a_transition", lambda: cmut(lambda c: c.source_frame.__setitem__(0, T - 1)), "source_identity",
           "row 0", validate=lambda c: c.validate(vector))
    expect("accepted_row_with_reason", lambda: cmut(lambda c: c.reasons.__setitem__((0, R["out_of_support"]), True)),
           "accepted_reasons", "row 0")
    expect("rejected_row_weight", lambda: cmut(lambda c: c.weight.__setitem__(1, 0.5)), "weight")
    expect("unknown_reason_code", lambda: cmut(lambda c: setattr(c, "reason_codes", REASON_CODES[:-1] + ("too_far",))),
           "reason_codes")

    # File-level compatibility rules.
    base = out / "vector.npz"

    def variant(name, edit):
        with np.load(base, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        edit(arrays)
        path = out / name
        np.savez(path, **arrays)
        try:
            read_trajectories(path)
            return None
        except SchemaError as e:
            return [v["detail"] for v in e.violations]

    def meta_edit(**changes):
        def edit(a):
            m = json.loads(str(a["meta"]))
            m.update(changes)
            a["meta"] = np.array(json.dumps(m))
        return edit

    got = variant("unknown_field.npz", lambda a: a.__setitem__("foo", np.zeros(3)))
    record("B", "unknown_field_rejected", got is not None and any("unknown field 'foo'" in d for d in got), errors=got)
    got = variant("ext_field.npz", lambda a: a.__setitem__("ext/notes", np.array("free-form")))
    record("B", "ext_field_ignored", got is None, errors=got)
    got = variant("major2.npz", meta_edit(version="2.0"))
    record("B", "unknown_major_version_rejected", got is not None and any("major version 2" in d for d in got), errors=got)
    got = variant("minor_declared.npz", lambda a: (a.__setitem__("future_field", np.zeros(3)),
                                                    meta_edit(version="1.1", optional_fields=["future_field"])(a)))
    record("B", "newer_minor_declared_optional_field_ignored", got is None, errors=got)
    got = variant("minor_undeclared.npz", lambda a: (a.__setitem__("future_field", np.zeros(3)), meta_edit(version="1.1")(a)))
    record("B", "newer_minor_undeclared_field_rejected", got is not None, errors=got)
    got = variant("missing_required.npz", lambda a: a.pop("terminated"))
    record("B", "missing_required_field_rejected", got is not None and any("terminated" in d for d in got), errors=got)

    # --- C: analytic fixture end to end -----------------------------------
    ds, config = fixture_dataset()
    ds.validate()
    write_trajectories(ds, out / "fixture.npz")
    ds = read_trajectories(out / "fixture.npz")
    sys_ = DoubleIntegrator(config["h"])
    s, a, sp, ids, frames = ds.transitions("state")
    s64, a64, sp64 = (x.astype(np.float64) for x in (s, a, sp))
    eps_fast = config["methods"]["backward_euler_fast"]["epsilon"]
    s_g = sys_.backward_euler(s64, a64)
    dist = np.linalg.norm(s_g - s64, axis=1)
    resid = sys_.residual(s_g, a64, s64)
    reasons = np.zeros((len(s), len(REASON_CODES)), bool)
    reasons[:, R["out_of_support"]] = ~(dist < eps_fast)
    reasons[:, R["action_out_of_bounds"]] = (np.abs(a64) > 0.1).any(axis=1)  # a second, independent reason
    fixture_cands = CandidateSet(
        run_id="analytic-fixture", generator={"name": "backward_euler_closed_form", "version": "1", "params_hash": "eps=0.01"},
        source_episode_id=ids, source_frame=frames.astype(np.int64), observations={"state": s_g.astype(np.float32)},
        actions=a, target_observations={"state": s}, accepted=~reasons.any(axis=1), reasons=reasons,
        measurements={"distance": dist, "residual": resid})
    write_candidates(fixture_cands, out / "fixture_candidates.npz")
    back = read_candidates(out / "fixture_candidates.npz", ds)
    labels = back.accepted_labels()
    exp_acc = (dist < eps_fast) & ~(np.abs(a64) > 0.1).any(axis=1)
    record("C", "analytic_fixture_identity_and_measurements_survive", same(fixture_cands, back)
           and np.array_equal(labels.source_frame, frames[exp_acc]) and np.array_equal(labels.source_episode_id, ids[exp_acc])
           and np.array_equal(labels.measurements["residual"], resid[exp_acc]),
           transitions=int(len(s)), accepted=int(exp_acc.sum()),
           multi_reason_rows=int((reasons.sum(axis=1) > 1).sum()),
           accepted_sources=[[str(e), int(f)] for e, f in zip(labels.source_episode_id, labels.source_frame)])

    # --- D: shipped datasets convert from the M1 format exactly -------------
    m1 = out / "m1"
    r = subprocess.run([sys.executable, str(REPO / "correct_il/convert_legacy_dataset.py"), "--shipped", "--out-dir", str(m1)],
                       capture_output=True, text=True)
    rows = []
    for f in sorted(m1.glob("*.npz")):
        conv = from_m1_npz(f)
        write_trajectories(conv, out / f"v1-{f.name}")
        back = read_trajectories(out / f"v1-{f.name}")
        with np.load(f, allow_pickle=False) as z:
            exact = np.array_equal(np.concatenate([e.observations["state"] for e in back.episodes]), z["observations"]) \
                and np.array_equal(np.concatenate([e.actions for e in back.episodes]), z["actions"]) \
                and [e.length for e in back.episodes] == z["episode_lengths"].tolist() \
                and back.episodes[0].observations["state"].dtype == z["observations"].dtype
        rows.append({"dataset": f.name, "episodes": len(back.episodes), "exact": bool(exact and same(conv, back)),
                     "source_sha256": back.meta["converted_from"]["sha256"]})
    record("D", "shipped_datasets_convert_exactly", r.returncode == 0 and len(rows) == 15 and all(x["exact"] for x in rows),
           datasets=rows)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
