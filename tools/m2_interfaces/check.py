#!/usr/bin/env python3
"""Evidence for the core interfaces (#17).

    uv run --group evidence python tools/m2_interfaces/check.py --out DIR

1. The new core, through IdentityRepresentation -> LinearTransitionModel (exact)
   -> generators -> validators, passes the #10 analytic oracle (I1-I5) for every
   method, and its run directory validates as complete.
2. Validators reject independently: rows carry several reasons with measurements.
3. Generation is batch-invariant: batch size 1 and one batch give identical bytes.
4. An image-style method plugs in a trainable latent representation and a fitted
   latent model without any schema change.
5. Malformed stage outputs raise InterfaceError and leave no run directory.
6. ExactPreimage isolates failures per candidate: in one batch, rows with a
   singular Newton system or non-finite model output are marked failed
   (generation_failed) while the solvable rows match the exact linear solution,
   byte for byte and at every batch size.
7. A fitted transition model's checkpoint records its kind and version, loads
   through the generic loader, and re-saves byte-identically; unknown kinds,
   other versions or components, and non-checkpoints raise CheckpointError,
   while an unavailable device surfaces as the model's own error.
8. A custom validator with a namespaced reason (example.low_confidence) writes
   a complete run whose candidates reload with the same reason order and row
   assignments (2.2: generated files carry measurement specs, #22); a
   built-in-only file without specs is still written as 2.0. Malformed reason
   declarations, from validators or in a candidate set, fail before any
   artifact is written.
Writes DIR/report.json.
"""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
import oracle  # noqa: E402  (tools/analytic_fixture/oracle.py)
from system import DoubleIntegrator  # noqa: E402

from ccil.config import resolve  # noqa: E402
from ccil.core import (ActionBounds, BackwardEuler, CandidateBatch, ExactPreimage, Finite, GenerationFailed,  # noqa: E402
                       IdentityRepresentation, InterfaceError, LinearTransitionModel, ModelResidual, NoisyAction,
                       SupportRadius, Uniform, generate_labels)
from ccil.runs import RunWriter, layout, validate_run  # noqa: E402
from ccil.schema.io import _atomic_savez  # noqa: E402
from ccil.schema import (REASON_CODES, Episode, SchemaError, TrajectoryDataset, read_candidates, write_candidates,  # noqa: E402
                         write_trajectories)

SEED = 20261023


def fixture(config):
    """The analytic fixture as a ccil.trajectories dataset (the NaN episode is an ingress failure, #14)."""
    by_ep = {}
    for tr in oracle.episodes_to_transitions(config):
        by_ep.setdefault(tr["episode"], []).append(tr)
    eps = []
    for name, trs in by_ep.items():
        obs = np.stack([t["s"] for t in trs] + [trs[-1]["s_next"]]).astype(np.float32)
        act = np.vstack([t["a"] for t in trs] + [np.zeros(1, np.float32)]).astype(np.float32)
        eps.append(Episode(name, {"state": obs}, act))
    return TrajectoryDataset(eps)


def to_oracle(cands, dataset):
    """Group candidate rows by source transition, in dataset order, in the oracle's record format."""
    order = [(ep.episode_id, t) for ep in dataset.episodes for t in range(ep.length - 1)]
    rows = {key: [] for key in order}
    for i in range(len(cands.accepted)):
        rows[(str(cands.source_episode_id[i]), int(cands.source_frame[i]))].append(i)
    out = []
    for key in order:
        recs = []
        for i in rows[key]:
            rs = cands.row_reasons(i)
            reason = next((r for r in ("out_of_support", "non_finite", "generation_failed", "action_out_of_bounds") if r in rs), None)
            recs.append({"s_G": cands.observations["state"][i], "a_G": cands.actions[i],
                         "target": cands.target_observations["state"][i], "accepted": bool(cands.accepted[i]), "reason": reason})
        gen_failed = sum(r["reason"] == "generation_failed" for r in recs)
        out.append({"attempted": len(recs), "candidates": recs,
                    "reported": {"accepted": sum(r["accepted"] for r in recs),
                                 "rejected": len(recs) - sum(r["accepted"] for r in recs) - gen_failed,
                                 "generation_failed": gen_failed}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}")

    config = json.loads((REPO / "tools/analytic_fixture/config.json").read_text())
    config["episodes"] = [e for e in config["episodes"] if not e["corrupt_actions"]]
    dataset = fixture(config).validate()
    sys_ = DoubleIntegrator(config["h"])
    model = LinearTransitionModel(sys_.A, sys_.B)
    rep = IdentityRepresentation("state")

    # 1. The #10 oracle, I1-I5 per method.
    methods = {
        "backward_euler_fast": BackwardEuler(),
        "backward_euler": BackwardEuler(),
        "forward_euler": ExactPreimage(max_iterations=config["methods"]["forward_euler"]["max_iter"]),
        "noisy_action": NoisyAction(config["methods"]["noisy_action"]["num_labels"], config["methods"]["noisy_action"]["delta"]),
    }
    results, sets = {}, {}
    for key, gen in methods.items():
        eps = config["methods"][key].get("epsilon")
        validators = [Finite(), GenerationFailed(), ModelResidual()] + ([SupportRadius(eps)] if eps else [])
        sets[key] = generate_labels(dataset, rep, model, gen, validators, run_id=f"analytic-{key}", seed=SEED)
        results[key] = to_oracle(sets[key], dataset)
    expected = oracle.episodes_to_transitions(config)
    _, verdicts = oracle.evaluate(config, expected, results)
    verdicts = {k: v for k, v in verdicts.items() if k != "I6_episode_boundaries"}
    flat = {f"{inv}[{m}]": r["pass"] for inv, per in verdicts.items() for m, r in per.items()}
    classes = {m: {c: sum(1 for dp in results[m] for x in dp["candidates"] if x["expected_class"] == c)
                   for c in ("accepted", "out_of_support", "boundary")} for m in results}
    record("analytic_oracle_I1_to_I5_pass_for_every_method", all(flat.values()), verdicts=flat, expected_classes=classes)

    cfg = resolve(overrides={"run.seed": SEED, "run.output_dir": str(out / "runs"), "data.validation_fraction": 0.0})
    with RunWriter(out / "runs", "analytic-noisy-action", "core.generate_labels", cfg) as run:
        write_trajectories(dataset, run.path(layout.DATASET))
        c = copy.copy(sets["noisy_action"])
        c.run_id = "analytic-noisy-action"
        write_candidates(c, run.path(layout.CANDIDATES))
        run.metrics({"candidates": int(len(c.accepted)), "accepted": int(c.accepted.sum())})
    status, problems = validate_run(out / "runs" / "analytic-noisy-action")
    record("analytic_run_directory_complete", status == "complete", status=status, problems=problems)

    # 2. Independent validators: multiple reasons per row, with measurements.
    multi = generate_labels(dataset, rep, model, NoisyAction(4, 0.05),
                            [Finite(), SupportRadius(0.005), ActionBounds(-0.1, 0.1), ModelResidual()],
                            run_id="multi-reason", seed=SEED, weighting=Uniform())
    both = [i for i in range(len(multi.accepted))
            if {"out_of_support", "action_out_of_bounds"} <= set(multi.row_reasons(i))]
    record("rows_carry_multiple_independent_reasons", len(both) > 0 and set(multi.measurements) >= {"distance", "residual"}
           and np.array_equal(multi.weight, multi.accepted.astype(float)),
           rows_with_both=len(both), reason_counts={r: int(multi.reasons[:, j].sum()) for j, r in enumerate(REASON_CODES)},
           example=[{"episode": str(multi.source_episode_id[i]), "frame": int(multi.source_frame[i]),
                     "reasons": multi.row_reasons(i), "distance": float(multi.measurements["distance"][i]),
                     "action": multi.actions[i].tolist()} for i in both[:3]])

    # 3. Batch invariance: byte-identical candidate files.
    digests = {}
    for name, gen in (("noisy_action", NoisyAction(4, 0.05)), ("exact_preimage", ExactPreimage())):
        files = []
        for bs in (1, None):
            cs = generate_labels(dataset, rep, model, gen, [Finite(), GenerationFailed(), SupportRadius(0.01), ModelResidual()],
                                 run_id="batch", seed=SEED, batch_size=bs)
            path = out / f"batch-{name}-{bs}.npz"
            write_candidates(cs, path)
            files.append(hashlib.sha256(path.read_bytes()).hexdigest())
        digests[name] = files
    record("generation_is_batch_invariant", all(a == b for a, b in digests.values()), sha256=digests)

    # 4. Image-style plug-in: trainable latent representation + fitted latent model.
    rng = np.random.default_rng(0)
    render = rng.normal(size=(3, 64))

    class StubImageEncoder:
        """Fixed random projection of pixels; stands in for a learned encoder (no image-CCIL parity claimed)."""
        output_modality, trainable = "latent.stub", True

        def __init__(self, dim=4):
            self.W = np.random.default_rng(1).normal(size=(64, dim)) / 8.0

        def encode(self, observations):
            img = observations["image.top"].astype(np.float64) / 255.0
            return img.reshape(len(img), -1) @ self.W

        def state_dict(self):
            return {"W": self.W}

        def load_state_dict(self, state):
            self.W = state["W"]

    img_eps = []
    for e in range(3):
        s = np.cumsum(rng.normal(scale=0.1, size=(12, 3)), axis=0)
        pix = np.clip(128 + 40 * (s @ render), 0, 255).astype(np.uint8).reshape(12, 8, 8, 1)
        img_eps.append(Episode(f"img{e}", {"state": s.astype(np.float32), "image.top": pix},
                               rng.normal(size=(12, 2)).astype(np.float32)))
    img_ds = TrajectoryDataset(img_eps).validate()
    enc = StubImageEncoder()
    from ccil.core import transitions
    tb = transitions(img_ds, enc)
    latent_model = LinearTransitionModel()
    fit_metrics = latent_model.fit(tb.z, tb.a, tb.z_next, seed=0)
    latent = generate_labels(img_ds, enc, latent_model, BackwardEuler(),
                             [Finite(), SupportRadius(0.5), ModelResidual()], run_id="latent-stub", seed=SEED)
    write_candidates(latent, out / "latent.npz")
    back = read_candidates(out / "latent.npz", img_ds)
    record("image_style_latent_plug_in_uses_unchanged_schema",
           set(back.observations) == {"latent.stub"} and back.observations["latent.stub"].shape[1] == 4
           and back.accepted.any(), latent_dim=4, candidates=int(len(back.accepted)), accepted=int(back.accepted.sum()),
           fit=fit_metrics)

    # 5. Interface failures raise InterfaceError before anything is written.
    class BadRep(IdentityRepresentation):
        def encode(self, observations):
            return observations["state"][:, 0]

    class BadGen(BackwardEuler):
        def generate(self, batch, model, seed):
            c = super().generate(batch, model, seed)
            return CandidateBatch(c.source, c.z[:, :1], c.a, c.target, c.failed)

    class BadReason(Finite):
        reason = "too_far"

    class SneakyMeasure(ModelResidual):
        def check(self, batch, cands, model):
            r, m = super().check(batch, cands, model)
            return np.ones_like(r), m

    failures = {}
    for name, args in (("representation_rank", (BadRep("state"), BackwardEuler(), [Finite()])),
                       ("generator_shape", (rep, BadGen(), [Finite()])),
                       ("unknown_reason", (rep, BackwardEuler(), [BadReason()])),
                       ("measuring_validator_rejects", (rep, BackwardEuler(), [SneakyMeasure()]))):
        dest = out / "runs" / f"fail-{name}"
        try:
            with RunWriter(out / "runs", f"fail-{name}", "core.generate_labels", cfg) as run:
                cs = generate_labels(dataset, args[0], model, args[1], args[2], run_id=f"fail-{name}", seed=SEED)
                write_candidates(cs, run.path(layout.CANDIDATES))
            failures[name] = "no error"
        except InterfaceError as e:
            failures[name] = None if dest.exists() else str(e)[:200]
    leftovers = [p.name for p in (out / "runs").iterdir() if p.name.startswith(".")]
    record("malformed_stage_outputs_raise_before_writing", all(failures.values()) and not leftovers,
           errors=failures, staging_left=leftovers)

    # 6. ExactPreimage failure isolation in a mixed batch.
    class MixedModel:
        """Exact linear dynamics, except delta = -z (I + df/dz singular) where a > 0.1
        and NaN where a < -0.04."""

        def __init__(self, exact):
            self.exact = exact

        def masks(self, a):
            return a[:, 0] > 0.1, a[:, 0] < -0.04

        def delta(self, z, a):
            singular, nan = self.masks(a)
            d = self.exact.delta(z, a)
            d[singular] = -z[singular]
            d[nan] = np.nan
            return d

        def save(self, path):
            _atomic_savez(path, {"A": self.exact.A, "B": self.exact.B,
                                 "kind": np.array("mixed: -z where a > 0.1, NaN where a < -0.04")})

        def jacobian_z(self, z, a):
            singular, nan = self.masks(a)
            J = np.array(self.exact.jacobian_z(z, a))
            J[singular] = -np.eye(z.shape[1])
            J[nan] = np.nan
            return J

    mixed, gen = MixedModel(model), ExactPreimage()
    vals = [Finite(), GenerationFailed(), ModelResidual()]
    try:
        runs = {bs: generate_labels(dataset, rep, mixed, gen, vals, run_id="mixed", seed=SEED, batch_size=bs)
                for bs in (1, 4, None)}
        error = None
    except Exception as e:  # noqa: BLE001  (the defect being guarded against is an uncaught solver error)
        runs, error = {}, f"{type(e).__name__}: {e}"
    if runs:
        ref = generate_labels(dataset, rep, model, gen, vals, run_id="mixed", seed=SEED)
        full = runs[None]
        singular, nan = mixed.masks(full.actions)
        bad = singular | nan
        gf = full.reasons[:, REASON_CODES.index("generation_failed")]
        mixed_digests = set()
        for bs, cs in runs.items():
            path = out / f"mixed-{bs}.npz"
            write_candidates(cs, path)
            mixed_digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
        ok = (bad.any() and (~bad).any() and np.array_equal(gf, bad) and np.array_equal(full.accepted, ~bad)
              and np.array_equal(full.observations["state"][~bad], ref.observations["state"][~bad])
              and len(mixed_digests) == 1)
        detail = {"rows": int(len(bad)), "singular_rows": int(singular.sum()), "non_finite_rows": int(nan.sum()),
                  "generation_failed": int(gf.sum()), "accepted": int(full.accepted.sum()),
                  "solvable_rows_match_exact_model": bool(np.array_equal(full.observations["state"][~bad],
                                                                         ref.observations["state"][~bad])),
                  "batch_sizes": [1, 4, "all"], "distinct_sha256": sorted(mixed_digests)}
    else:
        ok, detail = False, {"error": error}
    record("exact_preimage_isolates_singular_and_non_finite_rows", ok, **detail)

    # 7. Transition-model persistence: identity in the checkpoint, generic loading, visible failures.
    from ccil.core import TRANSITION_MODELS, TransitionModel, load_transition_model
    fitted = LinearTransitionModel()
    tb = transitions(dataset, rep)
    fitted.fit(tb.z, tb.a, tb.z_next, seed=SEED)
    ck = out / "transition_model.npz"
    fitted.save(ck)
    with np.load(ck, allow_pickle=False) as z:
        ck_meta = json.loads(str(z["meta"]))
    loaded = load_transition_model(ck)
    loaded.save(out / "transition_model_resaved.npz")
    same_bytes = ck.read_bytes() == (out / "transition_model_resaved.npz").read_bytes()
    same_delta = np.array_equal(loaded.delta(tb.z, tb.a), fitted.delta(tb.z, tb.a))

    def rewrite_meta(name, **changes):
        with np.load(ck, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["meta"] = np.array(json.dumps({**ck_meta, **changes}))
        np.savez(out / name, **arrays)
        return out / name

    (out / "not_a_checkpoint.npz").write_bytes(b"not a checkpoint")
    failures = {}
    for name, call in (("unknown_kind", lambda: load_transition_model(rewrite_meta("ck_kind.npz", kind="spline"))),
                       ("future_version", lambda: load_transition_model(rewrite_meta("ck_ver.npz", version="2"))),
                       ("other_component", lambda: load_transition_model(rewrite_meta("ck_comp.npz", component="policy"))),
                       ("not_a_checkpoint", lambda: load_transition_model(out / "not_a_checkpoint.npz")),
                       ("unavailable_device", lambda: load_transition_model(ck, device="cuda"))):
        try:
            call()
            failures[name] = "loaded"
        except Exception as e:  # noqa: BLE001  (recording which error type each failure surfaces as)
            failures[name] = f"{type(e).__name__}: {e}"
    expected = {"unknown_kind": "CheckpointError", "future_version": "CheckpointError", "other_component": "CheckpointError",
                "not_a_checkpoint": "CheckpointError", "unavailable_device": "ValueError"}
    record("transition_model_persistence_contract",
           isinstance(fitted, TransitionModel) and ck_meta == {"component": "transition_model", "kind": "linear", "version": "1"}
           and TRANSITION_MODELS["linear"] is LinearTransitionModel and same_bytes and same_delta
           and all(failures[k].split(":")[0] == v for k, v in expected.items()),
           checkpoint_meta=ck_meta, save_load_save_identical=same_bytes, delta_identical=same_delta,
           failures={k: v.replace(str(out), "<out>") for k, v in failures.items()})

    # 8. Namespaced extension reasons: end to end through a run, and refused at ingress when malformed.
    class LowConfidence:
        """Rejects candidates whose action norm exceeds 0.1; a stand-in for a method-specific test."""
        reason = "example.low_confidence"
        measurement_specs = {"example.action_norm": {"definition": "||a_G||", "norm": "l2", "units": "action units"}}

        def check(self, batch, cands, model):
            norm = np.linalg.norm(cands.a, axis=1)
            return norm > 0.1, {"example.action_norm": norm}

    ext_validators = [Finite(), GenerationFailed(), SupportRadius(0.01), LowConfidence(), ModelResidual()]
    ext = generate_labels(dataset, rep, model, NoisyAction(4, 0.05), ext_validators, run_id="analytic-extension", seed=SEED)
    with RunWriter(out / "runs", "analytic-extension", "core.generate_labels", cfg) as run:
        write_trajectories(dataset, run.path(layout.DATASET))
        write_candidates(ext, run.path(layout.CANDIDATES))
    ext_path = out / "runs" / "analytic-extension" / layout.CANDIDATES
    ext_back = read_candidates(ext_path, dataset)
    with np.load(ext_path, allow_pickle=False) as z:
        ext_version = json.loads(str(z["meta"]))["version"]
    with np.load(out / "batch-noisy_action-None.npz", allow_pickle=False) as z:
        builtin_version = json.loads(str(z["meta"]))["version"]
    # A built-in run carries measurement specs (2.2, #22); without specs it is written exactly as 2.0.
    bare = copy.copy(read_candidates(out / "batch-noisy_action-None.npz"))
    bare.measurement_specs = None
    write_candidates(bare, out / "builtin-without-specs.npz")
    with np.load(out / "builtin-without-specs.npz", allow_pickle=False) as z:
        bare_meta = json.loads(str(z["meta"]))
    col = ext_back.reason_codes.index("example.low_confidence")
    norm = np.linalg.norm(ext_back.actions.astype(np.float64), axis=1)
    record("namespaced_reason_round_trips_through_a_run",
           ext_back.reason_codes == REASON_CODES + ("example.low_confidence",) and ext_version == "2.2"
           and builtin_version == "2.2" and bare_meta["version"] == "2.0" and "measurement_specs" not in bare_meta
           and np.array_equal(ext_back.reasons, ext.reasons)
           and np.array_equal(ext_back.reasons[:, col], norm > 0.1) and ext_back.reasons[:, col].any()
           and np.array_equal(ext_back.accepted, ~ext_back.reasons.any(axis=1))
           and validate_run(out / "runs" / "analytic-extension")[0] == "complete",
           artifact="<out>/runs/analytic-extension/labels/candidates.npz", reason_codes=list(ext_back.reason_codes),
           file_version=ext_version, builtin_only_file_version=builtin_version,
           builtin_without_specs_file_version=bare_meta["version"],
           rows_by_reason={c: int(ext_back.reasons[:, i].sum()) for i, c in enumerate(ext_back.reason_codes)},
           accepted=int(ext_back.accepted.sum()), candidates=int(len(ext_back.accepted)))

    ingress = {}
    for name, code in (("unnamespaced_unknown", "low_confidence"), ("uppercase_namespace", "Example.low"),
                       ("nested_namespace", "a.b.c"), ("reserved_namespace", "ccil.low"), ("non_string", 3)):
        class Bad(LowConfidence):
            reason = code
        dest = out / "runs" / f"ingress-{name}"
        try:
            with RunWriter(out / "runs", f"ingress-{name}", "core.generate_labels", cfg) as run:
                write_candidates(generate_labels(dataset, rep, model, BackwardEuler(), [Bad()], run_id="x", seed=SEED),
                                 run.path(layout.CANDIDATES))
            ingress[f"validator:{name}"] = "written"
        except InterfaceError as e:
            ingress[f"validator:{name}"] = "written" if dest.exists() else f"InterfaceError: {e}"

    def declared(codes, reasons=None):
        c = copy.copy(ext)
        c.reason_codes = codes
        if reasons is not None:
            c.reasons = reasons
        return c

    dup = REASON_CODES + ("example.low_confidence", "example.low_confidence")
    wrong_order = ("example.low_confidence",) + REASON_CODES
    unsorted_ext = REASON_CODES + ("zeta.b", "alpha.a")
    two = np.hstack([ext.reasons, ext.reasons[:, -1:]])
    for name, cands in (("duplicate", declared(dup, two)), ("not_canonical_order", declared(wrong_order)),
                        ("extensions_unsorted", declared(unsorted_ext, two)),
                        ("non_string", declared(REASON_CODES + (7,))), ("not_a_sequence", declared("non_finite"))):
        path = out / f"ingress-{name}.npz"
        try:
            write_candidates(cands, path)
            ingress[f"declared:{name}"] = "written"
        except SchemaError as e:
            ingress[f"declared:{name}"] = "written" if path.exists() else f"SchemaError: {sorted(e.counts)}"
    leftovers = [p.name for p in (out / "runs").iterdir() if p.name.startswith(("ingress-", "."))]
    record("malformed_reason_codes_fail_before_any_artifact",
           all(v.startswith(("InterfaceError", "SchemaError")) for v in ingress.values()) and not leftovers,
           results=ingress, artifacts_left=leftovers)

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
