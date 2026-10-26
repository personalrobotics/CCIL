"""Integrated pipeline: CCIL owns representation, dynamics, and labels, and
orchestrates downstream policy training and evaluation.

Stages: data.prepare -> representation.fit -> transition.fit -> labels.generate
-> policy.train -> policy.evaluate. Stages 1-4 are CCIL-owned and identical for
any downstream trainer; replacing the trainer reuses them. The reference
components here (PCA representation, linear or mean-action policy, offline
action-error evaluation) stand in for M5's image encoders and real policies and
claim no image-CCIL parity.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import ConfigError
from ..core import transitions
from ..core.transition import check_identity, checkpoint_meta
from ..data.prepare import prepare_stage
from ..runs import layout
from ..schema import read_trajectories
from ..schema.io import _atomic_savez
from ..supervision import build_supervision
from .offline import fit_transition_stage, labels_stage, preflight
from .policy import TRAINERS
from .stages import run_stage

REPRESENTATION_CHECKPOINT = f"{layout.CHECKPOINTS}/representation.npz"
POLICY_CHECKPOINT = f"{layout.CHECKPOINTS}/policy.npz"


class PCARepresentation:
    """Deterministic PCA of one modality (state, or flattened image pixels in [0, 1])."""

    kind, version = "pca", "1"
    trainable = True
    output_modality = "latent.pca"

    def __init__(self, modality: str, dim: int):
        self.modality, self.dim = modality, dim
        self.mean = self.components = None

    def _flat(self, x):
        x = x.astype(np.float64) / (255.0 if x.dtype == np.uint8 else 1.0)
        return x.reshape(len(x), -1)

    def fit(self, dataset) -> dict:
        X = np.concatenate([self._flat(ep.observations[self.modality]) for ep in dataset.episodes])
        if self.dim > min(X.shape):
            raise ConfigError([f"representation.dim {self.dim} exceeds the data rank bound {min(X.shape)}"])
        self.mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean, full_matrices=False)
        # Fix each component's sign so the decomposition is deterministic across platforms.
        signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
        self.components = (vt * signs[:, None])[: self.dim]
        return {"explained_variance_ratio": float((s[: self.dim] ** 2).sum() / (s ** 2).sum())}

    def encode(self, observations):
        return (self._flat(observations[self.modality]) - self.mean) @ self.components.T

    def state_dict(self):
        return {"mean": self.mean, "components": self.components}

    def load_state_dict(self, state):
        self.mean, self.components = state["mean"], state["components"]

    def save(self, path):
        meta = {"component": "representation", "kind": self.kind, "version": self.version, "modality": self.modality}
        _atomic_savez(path, {"mean": self.mean, "components": self.components,
                             "meta": np.array(json.dumps(meta, sort_keys=True))})

    @classmethod
    def load(cls, path, modality, dim):
        check_identity(path, checkpoint_meta(path, "representation"), cls)
        rep = cls(modality, dim)
        with np.load(path, allow_pickle=False) as z:
            rep.load_state_dict({"mean": z["mean"], "components": z["components"]})
        return rep


def run_integrated(cfg, trainer=None):
    """Run all six stages; trainer overrides policy.trainer (any PolicyTrainer)."""
    v = cfg.values
    preflight(cfg)
    if v["representation.kind"] != "pca":
        raise ConfigError(["the integrated pipeline fits a learned representation; set representation.kind=pca"])
    if v["policy.method"] == "noise_bc":
        raise ConfigError(["policy.method noise_bc is ported in M3"])
    if v["certificate.profile"] is not None:
        raise ConfigError(["certificate.profile: the integrated pipeline's policy stage does not consume certificates "
                           "yet (#89); certify with ccil labels generate"])
    trainer = trainer or TRAINERS[v["policy.trainer"]]()
    modality, dim = v["representation.modality"], v["representation.dim"]
    results = [prepare_stage(cfg)]
    prepared = results[0]

    def fit_representation(run):
        rep = PCARepresentation(modality, dim)
        metrics = rep.fit(read_trajectories(prepared.out(layout.DATASET)))
        rep.save(run.path(REPRESENTATION_CHECKPOINT))
        run.metrics(metrics)
    rep_stage = run_stage("representation.fit", cfg, {"dataset": prepared.out(layout.DATASET)}, fit_representation,
                          components={"representation": PCARepresentation(modality, dim)})
    results.append(rep_stage)
    rep = PCARepresentation.load(rep_stage.out(REPRESENTATION_CHECKPOINT), modality, dim)
    rep_input = {"representation": rep_stage.out(REPRESENTATION_CHECKPOINT)}
    fitted = fit_transition_stage(cfg, prepared, rep, extra_inputs=rep_input)
    labeled = labels_stage(cfg, prepared, fitted, rep, extra_inputs=rep_input)
    results += [fitted, labeled]

    split_path = prepared.out(layout.SPLIT)
    split = json.loads(split_path.read_text()) if split_path.exists() else None
    policy_inputs = {"dataset": prepared.out(layout.DATASET), "candidates": labeled.out(layout.CANDIDATES), **rep_input}
    if split is not None:
        policy_inputs["split"] = split_path

    def train(run):
        # The trainer-neutral supervision contract (ccil.supervision) decides which rows reach the loss.
        sup = build_supervision(policy_inputs["dataset"], v["policy.method"], representation=rep,
                                representation_checkpoint=policy_inputs["representation"],
                                candidates=policy_inputs["candidates"] if v["policy.method"] == "ccil" else None,
                                split=policy_inputs.get("split"))
        trainer.fit(*sup.arrays())
        trainer.save(run.path(POLICY_CHECKPOINT))
        run.metrics({"trainer": trainer.name, "supervision": sup.manifest()})
    trained = run_stage("policy.train", cfg, policy_inputs, train, components={"trainer": trainer})
    results.append(trained)

    eval_inputs = {**policy_inputs, "policy": trained.out(POLICY_CHECKPOINT)}

    def evaluate(run):
        # Load the policy.train checkpoint whether that stage was written or reused.
        policy = trainer.load(eval_inputs["policy"])
        dataset = read_trajectories(prepared.out(layout.DATASET))
        batch = transitions(dataset, rep)
        held = np.isin(batch.episode_id, split["validation"]) if split else np.ones(len(batch), bool)
        err = np.linalg.norm(policy.predict(batch.z[held]) - batch.a[held], axis=1)
        run.metrics({"held_out_transitions": int(held.sum()), "action_l2_mean": float(err.mean()),
                     "action_l2_max": float(err.max()), "note": "offline action error; simulator evaluation is M7"})
    results.append(run_stage("policy.evaluate", cfg, eval_inputs, evaluate, components={"trainer": trainer}))
    return results
