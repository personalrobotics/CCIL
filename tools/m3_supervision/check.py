#!/usr/bin/env python3
"""Evidence for #89 part 1: the trainer-neutral supervision contract.

    uv run --group evidence python tools/m3_supervision/check.py --out DIR

Artifacts come from the real offline pipeline (generate_offline, with and
without certificates, with action chunks) on rollouts of the analytic double
integrator with a train/validation split. For each arm the rows are recomputed
independently from the files and compared with build_supervision:
- bc and noise_bc are the training demonstrations only; ccil without
  certificates adds every accepted candidate, labelled uncertified; ccil with
  certificates adds exactly accepted & admissible & selected;
- accepted candidates that are inadmissible or unselected, with positive
  weights, never reach a trainer: an instrumented trainer sees exactly the
  usable rows;
- synthetic weights are the recorded ones, the demonstration weight is
  explicit, and the row counts reconcile;
- held-out episodes contribute no rows and are counted;
- mismatched artifacts fail closed, each with its own reason;
- chunked candidates are refused unless consumed; consumed chunks equal the
  stored ones and demonstration chunks follow the label-chunk alignment;
- an independent consumer reading only the candidates and certificates files
  reproduces the usable mask;
- the integrated pipeline trains through the contract.
Writes DIR/report.json and DIR/manifests/<arm>.json (the inspectable artifacts).
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
sys.path.insert(0, str(REPO / "tools" / "analytic_fixture"))
from system import DoubleIntegrator  # noqa: E402

from ccil.config import resolve  # noqa: E402
from ccil.core import IdentityRepresentation  # noqa: E402
from ccil.core.certificates import write_reference_dynamics  # noqa: E402
from ccil.pipelines.offline import generate_offline  # noqa: E402
from ccil.runs import layout  # noqa: E402
from ccil.schema import (Episode, TrajectoryDataset, read_candidates, read_certificates, read_trajectories,  # noqa: E402
                         write_certificates, write_trajectories)
from ccil.supervision import SupervisionError, build_supervision  # noqa: E402

SEED, K, SIGMA = 20261026, 4, 0.05


def rollouts(sys_, seed, episodes=8, length=10, shift=0.0):
    rng = np.random.default_rng(seed)
    eps = []
    for k in range(episodes):
        z = [rng.uniform(-1, 1, 2) + shift]
        a = rng.uniform(-1, 1, (length, 1))
        for t in range(length - 1):
            z.append(sys_.step_np(z[-1][None], a[t:t + 1])[0])
        eps.append(Episode(f"ep{k:02d}", {"state": np.array(z, np.float32)}, a.astype(np.float32)))
    return TrajectoryDataset(eps).validate()


def outcome(fn):
    try:
        fn()
        return "accepted"
    except SupervisionError as e:
        return "SupervisionError: " + " | ".join(p[:150] for p in e.problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifests").mkdir()
    cases = []

    def record(name, ok, **detail):
        cases.append({"case": name, "pass": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else json.dumps(detail, default=str)[:900]}", flush=True)

    sys_ = DoubleIntegrator(0.1)
    rep = IdentityRepresentation("state")
    write_trajectories(rollouts(sys_, SEED), out / "rollouts.npz")
    write_trajectories(rollouts(sys_, SEED, shift=0.01), out / "rollouts-other.npz")
    write_reference_dynamics(out / "true-dynamics.npz", sys_.A, sys_.B, "analytic double integrator")

    def pipeline(name, data="rollouts.npz", extra=()):
        cfg = resolve(recipe="analytic_double_integrator", sets=[
            f"data.path={out / data}", f"run.output_dir={out / 'runs' / name}", "data.validation_fraction=0.25",
            "generation.method=noisy_action", f"generation.labels_per_transition={K}",
            f"generation.action_noise_std={SIGMA}", *extra])
        res = generate_offline(cfg)
        paths = {"dataset": res[0].out(layout.DATASET), "split": res[0].out(layout.SPLIT),
                 "candidates": res[2].out(layout.CANDIDATES)}
        if len(res) == 4:
            paths["certificates"] = res[3].out(layout.CERTIFICATES)
        return paths

    certified = ["generation.support_radius=0.08", "certificate.profile=disturbed_action",
                 f"certificate.reference_dynamics={out / 'true-dynamics.npz'}", "certificate.max_total_error=0.015",
                 "certificate.selection_quantile=0.6"]
    P = pipeline("certified", extra=certified)
    PC = pipeline("chunks", extra=["generation.action_chunk_size=3"])
    PO = pipeline("other", data="rollouts-other.npz", extra=certified)

    ds = read_trajectories(P["dataset"])
    split = json.loads(Path(P["split"]).read_text())
    cands = read_candidates(P["candidates"], ds)
    cert = read_certificates(P["certificates"])
    train = set(split["train"])

    # Independent recomputation from the files.
    def demonstrations():
        rows = [(ep.episode_id, t) for ep in ds.episodes if ep.episode_id in train for t in range(ep.length - 1)]
        eps = {ep.episode_id: ep for ep in ds.episodes}
        obs = np.array([eps[e].observations["state"][t] for e, t in rows])
        act = np.array([eps[e].actions[t] for e, t in rows])
        return rows, obs, act

    def synthetic(mask):
        rows = [i for i in np.flatnonzero(mask) if str(cands.source_episode_id[i]) in train]
        return rows

    d_rows, d_obs, d_act = demonstrations()
    accepted = np.asarray(cands.accepted)
    usable_cert = accepted & cert.admissible & cert.selected
    expect = {"bc": [], "noise_bc": [], "ccil_uncertified": synthetic(accepted), "ccil_certified": synthetic(usable_cert)}
    arms = {"bc": dict(method="bc"), "noise_bc": dict(method="noise_bc"),
            "ccil_uncertified": dict(method="ccil", candidates=P["candidates"]),
            "ccil_certified": dict(method="ccil", candidates=P["candidates"], certificates=P["certificates"])}
    sups, arm_results = {}, {}
    for arm, kw in arms.items():
        sup = build_supervision(P["dataset"], representation=rep, split=P["split"], **kw)
        sups[arm] = sup
        rows = expect[arm]
        obs = np.concatenate([d_obs, cands.observations["state"][rows]]) if rows else d_obs
        act = np.concatenate([d_act, cands.actions[rows]]) if rows else d_act
        w = np.concatenate([np.ones(len(d_rows)), cands.weight[rows].astype(np.float64)]) if rows else np.ones(len(d_rows))
        cls = (["demonstration"] * len(d_rows)
               + ([str(cert.classification[i]) for i in rows] if arm == "ccil_certified" else ["uncertified"] * len(rows)))
        ok = (np.array_equal(sup.observations, obs) and np.array_equal(sup.actions, act) and np.array_equal(sup.weights, w)
              and sup.classification.tolist() == cls and sup.candidate_row.tolist() == [-1] * len(d_rows) + list(rows))
        (out / "manifests" / f"{arm}.json").write_text(json.dumps(sup.manifest(), indent=1, sort_keys=True) + "\n")
        arm_results[arm] = {"ok": bool(ok), "rows": int(len(sup.weights)), "demonstration_rows": len(d_rows),
                            "synthetic_rows": len(rows), "certification": sup.manifest()["certification"]}
    record("arms_match_independent_recomputation", all(v["ok"] for v in arm_results.values())
           and sups["ccil_certified"].manifest()["certification"]["profile"] == "disturbed_action"
           and sups["ccil_uncertified"].manifest()["certification"] == "uncertified"
           and sups["noise_bc"].manifest()["online_perturbation"] is not None,
           arms=arm_results, manifests="<out>/manifests/<arm>.json")

    # Nothing unusable reaches the loss.
    class Recorder:
        def fit(self, z, a, w):
            self.z, self.a, self.w = z, a, w
            return self
    rec = Recorder().fit(*sups["ccil_certified"].arrays())
    excluded = accepted & ~(cert.admissible & cert.selected)
    seen = {tuple(map(float, r)) for r in rec.z}
    leaked = [i for i in np.flatnonzero(excluded) if tuple(map(float, cands.observations["state"][i])) in seen]
    inadmissible, unselected = accepted & ~cert.admissible, accepted & cert.admissible & ~cert.selected
    record("unusable_rows_never_reach_the_trainer",
           inadmissible.any() and unselected.any() and (cands.weight[excluded] > 0).all() and not leaked
           and len(rec.w) == len(d_rows) + len(expect["ccil_certified"]),
           accepted_but_inadmissible=int(inadmissible.sum()), admissible_but_unselected=int(unselected.sum()),
           their_weights_positive=bool((cands.weight[excluded] > 0).all()), leaked=len(leaked),
           rows_seen_by_trainer=int(len(rec.w)))

    # Weights, explicit demonstration weight, and reconciliation.
    s2 = build_supervision(P["dataset"], "ccil", representation=rep, candidates=P["candidates"],
                           certificates=P["certificates"], split=P["split"], demonstration_weight=2.0)
    syn = s2.source == "synthetic"
    c = s2.manifest()["counts"]
    record("weights_and_reconciliation",
           np.array_equal(s2.weights[syn], cands.weight[s2.candidate_row[syn]].astype(np.float64))
           and (s2.weights[~syn] == 2.0).all() and s2.manifest()["demonstration_weight"] == 2.0
           and c["demonstration_rows"] + c["demonstration_rows_held_out"] == c["demonstration_transitions"]
           and c["synthetic_rows"] + c["usable_rows_held_out"] == c["accepted_admissible_and_selected"]
           and c["rows"] == c["demonstration_rows"] + c["synthetic_rows"],
           counts=c)

    # Split.
    held = set(split["validation"])
    record("held_out_episodes_contribute_no_rows",
           not any(e in held for e in sups["ccil_certified"].episode_id) and c["demonstration_rows_held_out"] > 0
           and c["usable_rows_held_out"] > 0,
           validation_episodes=sorted(held), demonstration_rows_held_out=c["demonstration_rows_held_out"],
           usable_rows_held_out=c["usable_rows_held_out"])

    # Fail closed.
    reordered = copy.deepcopy(cert)
    perm = np.random.default_rng(SEED).permutation(len(reordered.source_frame))
    for name in ("source_episode_id", "source_frame", "model_error_value", "total_error_value", "classification",
                 "is_upper_bound", "in_neighborhood", "admissible", "selected", "reasons"):
        setattr(reordered, name, getattr(reordered, name)[perm])
    reordered.terms = {k: v[perm] for k, v in reordered.terms.items()}
    write_certificates(reordered, out / "certificates-reordered.npz")
    other_dataset = copy.deepcopy(cert)
    other_dataset.inputs = {**other_dataset.inputs, "dataset": "0" * 64}
    write_certificates(other_dataset, out / "certificates-other-dataset.npz")
    flipped = copy.deepcopy(cert)
    i = int(np.flatnonzero(flipped.admissible & ~flipped.selected)[0]) if (flipped.admissible & ~flipped.selected).any() else 0
    flipped.selected = flipped.selected.copy()
    flipped.selected[i] = ~flipped.selected[i]
    flipped.reasons = flipped.reasons.copy()
    flipped.reasons[i, 3] = ~flipped.selected[i] & flipped.reasons[i, 3]
    write_certificates(flipped, out / "certificates-flipped.npz")
    bad_split = dict(split, episode_ids=split["episode_ids"][:-1] + ["foreign"],
                     train=sorted([e for e in split["train"] if e != split["episode_ids"][-1]] + (
                         ["foreign"] if split["episode_ids"][-1] in split["train"] else [])),
                     validation=sorted([e for e in split["validation"] if e != split["episode_ids"][-1]] + (
                         ["foreign"] if split["episode_ids"][-1] in split["validation"] else [])))
    (out / "split-foreign.json").write_text(json.dumps(bad_split))
    base = dict(representation=rep, split=P["split"])
    refused = {
        "candidates_from_another_dataset": outcome(lambda: build_supervision(P["dataset"], "ccil", candidates=PO["candidates"], **base)),
        "certificates_for_other_candidates": outcome(lambda: build_supervision(
            P["dataset"], "ccil", candidates=PC["candidates"], certificates=P["certificates"], action_chunks="consume", **base)),
        "certificate_rows_reordered": outcome(lambda: build_supervision(
            P["dataset"], "ccil", candidates=P["candidates"], certificates=out / "certificates-reordered.npz", **base)),
        "certificate_from_another_dataset": outcome(lambda: build_supervision(
            P["dataset"], "ccil", candidates=P["candidates"], certificates=out / "certificates-other-dataset.npz", **base)),
        "certificate_selection_tampered": outcome(lambda: build_supervision(
            P["dataset"], "ccil", candidates=P["candidates"], certificates=out / "certificates-flipped.npz", **base)),
        "split_of_another_dataset": outcome(lambda: build_supervision(P["dataset"], "bc", representation=rep,
                                                                      split=out / "split-foreign.json")),
        "representation_modality_mismatch": outcome(lambda: build_supervision(
            P["dataset"], "ccil", candidates=P["candidates"], representation=IdentityRepresentation("velocity"), split=P["split"])),
        "bc_with_candidates": outcome(lambda: build_supervision(P["dataset"], "bc", candidates=P["candidates"], **base)),
        "noise_bc_with_certificates": outcome(lambda: build_supervision(P["dataset"], "noise_bc", certificates=P["certificates"], **base)),
        "certificates_without_candidates": outcome(lambda: build_supervision(P["dataset"], "ccil", certificates=P["certificates"], **base)),
        "ccil_without_candidates": outcome(lambda: build_supervision(P["dataset"], "ccil", **base)),
        "negative_demonstration_weight": outcome(lambda: build_supervision(P["dataset"], "bc", demonstration_weight=-1.0, **base)),
        "chunked_candidates_refused_by_default": outcome(lambda: build_supervision(P["dataset"], "ccil", candidates=PC["candidates"],
                                                                                  representation=rep, split=PC["split"])),
    }
    reason = {"candidates_from_another_dataset": "candidates were generated from dataset",
              "certificates_for_other_candidates": "certificates certify candidates",
              "certificate_rows_reordered": "do not match the candidate rows in identity and order",
              "certificate_from_another_dataset": "certificates were computed from dataset",
              "certificate_selection_tampered": "certificate structure: selected",
              "split_of_another_dataset": "split episode ids", "representation_modality_mismatch": "not the representation's",
              "bc_with_candidates": "bc trains on the demonstrations only",
              "noise_bc_with_certificates": "noise_bc trains on the demonstrations only",
              "certificates_without_candidates": "certificates need the candidates file",
              "ccil_without_candidates": "ccil needs a candidates file",
              "negative_demonstration_weight": "demonstration_weight", "chunked_candidates_refused_by_default": "action chunks"}
    record("mismatched_artifacts_fail_closed",
           set(reason) == set(refused) and all(v.startswith("SupervisionError") and reason[k] in v for k, v in refused.items()),
           refused=refused)

    # Chunks: consumed, never silently reduced.
    dsc = read_trajectories(PC["dataset"])
    cc = read_candidates(PC["candidates"], dsc)
    sc = build_supervision(PC["dataset"], "ccil", representation=rep, candidates=PC["candidates"], split=PC["split"],
                           action_chunks="consume")
    syn = sc.source == "synthetic"
    eps = {ep.episode_id: ep for ep in dsc.episodes}
    demo_ok = True
    for i in np.flatnonzero(~syn):
        ep, t = eps[str(sc.episode_id[i])], int(sc.frame[i])
        for k in range(3):
            inside = t + k <= ep.length - 1
            demo_ok &= bool(sc.action_chunk_mask[i, k] == inside)
            demo_ok &= bool(np.array_equal(sc.action_chunks[i, k], ep.actions[t + k] if inside else np.zeros(1, np.float32)))
    bc_chunks = build_supervision(PC["dataset"], "bc", representation=rep, split=PC["split"], action_chunks="consume",
                                  chunk_size=3)
    record("action_chunks_consumed_or_refused",
           np.array_equal(sc.action_chunks[syn], cc.action_chunks[sc.candidate_row[syn]])
           and np.array_equal(sc.action_chunk_mask[syn], cc.action_chunk_mask[sc.candidate_row[syn]])
           and demo_ok and (~sc.action_chunk_mask[~syn]).any() and np.array_equal(sc.action_chunks[:, 0], sc.actions)
           and bc_chunks.action_chunks.shape[1] == 3
           and outcome(lambda: build_supervision(PC["dataset"], "bc", representation=rep, action_chunks="consume")).startswith("SupervisionError")
           and refused["chunked_candidates_refused_by_default"].startswith("SupervisionError"),
           chunk_size=int(sc.action_chunks.shape[1]), demonstration_rows=int((~syn).sum()), synthetic_rows=int(syn.sum()),
           demonstration_positions_masked=int((~sc.action_chunk_mask[~syn]).sum()))

    # An independent consumer: the recipe in docs/supervision.md, from the two files alone.
    c2, k2 = read_candidates(P["candidates"]), read_certificates(P["certificates"])
    recipe = c2.accepted & k2.admissible & k2.selected
    in_train = np.isin(c2.source_episode_id.astype(str), sorted(train))
    rows = sups["ccil_certified"].candidate_row[sups["ccil_certified"].source == "synthetic"]
    record("independent_consumer_reproduces_the_usable_mask",
           np.array_equal(np.flatnonzero(recipe & in_train), rows), usable=int(recipe.sum()), in_training_episodes=int(len(rows)))

    # The integrated pipeline trains through the contract.
    cfg = resolve(recipe="analytic_double_integrator", sets=[
        f"data.path={out / 'rollouts.npz'}", f"run.output_dir={out / 'runs' / 'integrated'}", "data.validation_fraction=0.25",
        "representation.kind=pca", "representation.dim=2", "generation.method=backward_euler",
        "generation.support_radius=null"])
    from ccil.pipelines.integrated import run_integrated
    res = run_integrated(cfg)
    train_stage = next(r for r in res if r.stage == "policy.train")
    metrics = json.loads((train_stage.run_dir / "metrics.json").read_text())["values"]
    record("integrated_pipeline_trains_through_the_contract",
           metrics["supervision"]["method"] == "ccil" and metrics["supervision"]["certification"] == "uncertified"
           and metrics["supervision"]["representation"]["modality"] == "latent.pca"
           and metrics["supervision"]["counts"]["rows"] > metrics["supervision"]["counts"]["demonstration_rows"],
           supervision_counts=metrics["supervision"]["counts"])

    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {"all_pass": all(c["pass"] for c in cases), "cases": cases,
              "environment": {"code_commit": git("rev-parse", "HEAD"),
                              "code_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                              "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    (out / "report.json").write_text(text.replace(str(out), "<out>").replace(str(REPO), "<repo>"))
    for p in (out / "manifests").iterdir():
        p.write_text(p.read_text().replace(str(out), "<out>"))
    print(f"{sum(c['pass'] for c in cases)}/{len(cases)} cases pass; report: {out / 'report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
