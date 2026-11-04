"""Data-only label geometry: the demonstration metric, tau_demo, and novelty (#167).

Every distance here is in one declared metric, computed from the demonstrations
alone: the Euclidean norm of z / s, where s is the per-coordinate standard
deviation of the training episodes' demonstration states in the
representation (1 for a coordinate that never varies). Actions are not
coordinates of it. The metric is recorded with every quantity it measures, so
a distance is never read without its denominator.

- d_source: the distance of a label state from the demonstration state of its
  source transition.
- d_demo: its distance to the nearest training demonstration state.
- tau_demo: the demonstrations' own spacing, a quantile (0.95 by default) of
  the distance from each held-out demonstration state to the nearest training
  state. It needs held-out episodes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .representation import transitions

UNITS = "scaled state (z / s, s: per-coordinate std of the training demonstration states)"
DEFINITION = "||(x - y) / s||_2 in the representation, s the per-coordinate std of the training demonstration states"
BLOCK = 2 ** 24  # floats in one block of nearest-distance differences


@dataclass(frozen=True)
class DemoMetric:
    """The training demonstration states and the scale of the metric they define."""

    train_states: np.ndarray   # (M, D) float64
    held_out_states: np.ndarray  # (H, D) float64
    scale: np.ndarray          # (D,) float64, > 0
    train_episodes: tuple
    held_out_episodes: tuple

    def spec(self) -> dict:
        return {"definition": DEFINITION, "units": UNITS, "scale": [float(x) for x in self.scale],
                "train_episodes": len(self.train_episodes), "train_states": int(len(self.train_states)),
                "held_out_episodes": len(self.held_out_episodes), "held_out_states": int(len(self.held_out_states))}


def demo_metric(dataset, split: dict | None, representation) -> DemoMetric:
    """The metric of a dataset and its split record (ccil.split); without a split, every episode trains."""
    train = set(split["train"]) if split is not None else {ep.episode_id for ep in dataset.episodes}
    batch = transitions(dataset, representation)
    # Every recorded state of an episode: its transitions' sources plus the last next state.
    states = {}
    for i, ep in enumerate(batch.episode_id.astype(str)):
        states.setdefault(ep, []).append(batch.z[i])
    last = {}
    for i, ep in enumerate(batch.episode_id.astype(str)):
        last[ep] = batch.z_next[i]
    rows = {ep: np.vstack(v + [last[ep]]).astype(np.float64) for ep, v in states.items()}
    order = [ep.episode_id for ep in dataset.episodes if ep.episode_id in rows]
    tr = [rows[e] for e in order if e in train]
    ho = [rows[e] for e in order if e not in train]
    D = batch.z.shape[1]
    train_states = np.vstack(tr) if tr else np.zeros((0, D))
    held_out = np.vstack(ho) if ho else np.zeros((0, D))
    std = train_states.std(axis=0) if len(train_states) else np.ones(D)
    scale = np.where(std > 0, std, 1.0)
    return DemoMetric(train_states, held_out, scale, tuple(e for e in order if e in train),
                      tuple(e for e in order if e not in train))


def nearest_distance(points, reference, scale) -> np.ndarray:
    """Distance from each point to its nearest reference point, in the metric of scale; inf without references.

    Differences are taken directly, not through ||p||^2 + ||r||^2 - 2 p.r, whose cancellation would swamp the
    small distances that matter here (a label next to its source). Blocks bound the memory to BLOCK floats.
    """
    points = np.asarray(points, np.float64) / scale
    ref = np.asarray(reference, np.float64) / scale
    out = np.full(len(points), np.inf)
    if not len(ref) or not len(points):
        return out
    rows = max(1, BLOCK // (len(ref) * ref.shape[1]))
    for start in range(0, len(points), rows):
        diff = points[start:start + rows, None, :] - ref[None, :, :]
        out[start:start + rows] = np.sqrt((diff * diff).sum(axis=2).min(axis=1))
    return out


def tau_demo(metric: DemoMetric, quantile: float = 0.95) -> float | None:
    """The demonstrations' spacing: a quantile of held-out-to-training nearest distances; None without both."""
    if not len(metric.held_out_states) or not len(metric.train_states):
        return None
    return float(np.quantile(nearest_distance(metric.held_out_states, metric.train_states, metric.scale), quantile))


class SourceDisplacement:
    """Measures d_source, the label state's distance from its source demonstration state, in the demonstration metric."""

    reason = None

    def __init__(self, metric: DemoMetric):
        self.scale = metric.scale.copy()
        self.measurement_specs = {"quality.d_source": {
            "definition": "distance of the label state from the demonstration state of its source transition: " + DEFINITION,
            "norm": "l2", "units": UNITS}}

    def check(self, batch, cands, model):
        d = np.linalg.norm((cands.z.astype(np.float64) - batch.z[cands.source].astype(np.float64)) / self.scale, axis=1)
        return np.zeros(len(cands), bool), {"quality.d_source": d}


class NearestDemonstration:
    """Measures d_demo, the label state's distance to the nearest training demonstration state.

    A brute-force search over every training state: opt-in, since its cost is candidates x training states.
    """

    reason = None

    def __init__(self, metric: DemoMetric):
        self._metric = metric
        self.scale = metric.scale.copy()
        self.reference_sha256 = hashlib.sha256(np.ascontiguousarray(metric.train_states).tobytes()).hexdigest()
        self.measurement_specs = {"quality.d_demo": {
            "definition": "distance of the label state to the nearest training demonstration state: " + DEFINITION,
            "norm": "l2", "units": UNITS}}

    def check(self, batch, cands, model):
        d = nearest_distance(cands.z, self._metric.train_states, self._metric.scale)
        return np.zeros(len(cands), bool), {"quality.d_demo": d}


__all__ = ["DemoMetric", "demo_metric", "nearest_distance", "tau_demo", "SourceDisplacement", "NearestDemonstration",
           "UNITS", "DEFINITION"]
