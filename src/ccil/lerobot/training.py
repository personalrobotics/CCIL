"""CCIL supervision as one training dataset over a LeRobot dataset and its overlay (#28). Needs torch.

CCILTrainingDataset wraps a LeRobot dataset object the caller constructs (CCIL never imports lerobot) together
with the ccil/ overlay of the same directory, verified to have been made from that directory's dataset. Its rows are exactly CCIL's supervision: the demonstration frames
(t <= T - 2 of the overlay's training episodes) and every usable synthetic label, each with its weight under
"ccil.weight"; sampler() draws them by weight, optionally balancing demonstrations against labels. A stock
LeRobot trainer that ignores ccil.weight is not CCIL-correct, and this module never claims it is.

Temporal windows fail closed. Demonstration rows are the LeRobot dataset's own items, with its ordinary windows.
A synthetic label is an independent record: an observation history for it is refused (no semantically valid
construction exists until M5), and future actions come only from the label's stored action chunk, never from
neighbouring labels; without a stored chunk, a future-action window is refused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .overlay import read_overlay


class TemporalWindowRefused(ValueError):
    """The LeRobot dataset asks for a temporal window a synthetic label cannot supply."""


def _deltas(lerobot_dataset) -> dict:
    d = getattr(lerobot_dataset, "delta_timestamps", None)
    return dict(d) if d else {}


class CCILTrainingDataset:
    """Demonstration frames and synthetic labels as one indexable dataset, with weights."""

    def __init__(self, lerobot_dataset, *, keys=None):
        import torch
        self._torch = torch
        self.base = lerobot_dataset
        # The overlay is read from the LeRobot dataset's own directory, never from a separate root: read_overlay
        # verifies that directory's LeRobot portion against the source the labels were made from, so demonstrations
        # and labels always come from one dataset.
        root = getattr(lerobot_dataset, "root", None)
        if root is None:
            raise ValueError("the LeRobot dataset has no root directory to read its CCIL overlay from")
        self.meta, self.labels = read_overlay(Path(root))
        feature = self.meta["feature_map"]["observation"]["lerobot_feature"]
        self.feature = feature
        features = getattr(getattr(lerobot_dataset, "meta", None), "features", {}) or {}
        obs_keys = sorted(k for k in features if k.startswith("observation."))
        self.keys = list(keys) if keys is not None else obs_keys
        uncovered = [k for k in self.keys if k.startswith("observation.") and k != feature]
        if uncovered:
            raise ValueError(f"the labels give {feature} only; a policy reading {uncovered} cannot consume them "
                             f"(pass keys= with the observation features the policy uses)")
        deltas = _deltas(lerobot_dataset)
        history = {k: v for k, v in deltas.items() if k.startswith("observation.") and list(v) != [0]}
        if history:
            raise TemporalWindowRefused(f"observation windows {sorted(history)} are refused for synthetic labels: "
                                        f"a label has no observation history (until M5 defines one)")
        self.action_deltas = deltas.get("action")
        if self.action_deltas is not None and list(self.action_deltas) != [0]:
            if "action_chunk" not in self.labels:
                raise TemporalWindowRefused("a future-action window needs the labels' stored action chunks; this "
                                            "overlay has single actions, and neighbouring labels are never adjacent")
            k_needed = len(self.action_deltas)
            k_stored = self.labels["action_chunk_mask"].shape[1]
            if list(self.action_deltas) != list(np.arange(k_needed) / lerobot_dataset.fps)[:k_needed] \
                    or k_needed > k_stored:
                raise TemporalWindowRefused(f"action window {list(self.action_deltas)} is not the stored chunk's "
                                            f"first {k_needed} of {k_stored} steps at 1 / fps")
        # Demonstration rows: frames t <= T - 2 of the overlay's training episodes, in the LeRobot dataset's order.
        eps = getattr(lerobot_dataset.meta, "episodes")
        demo = []
        for e in self.meta["demonstrations"]["episodes"]:
            row = eps[e]
            demo.extend(range(int(row["dataset_from_index"]), int(row["dataset_to_index"]) - 1))
        self.demo = np.asarray(demo, dtype=np.int64)
        counts = (self.meta["supervision"].get("counts") or {})
        if "demonstration_rows" in counts and counts["demonstration_rows"] != len(self.demo):
            raise ValueError(f"{len(self.demo)} demonstration frames, the supervision recorded "
                             f"{counts['demonstration_rows']}")
        self.demo_weight = float(self.meta["demonstrations"]["weight"])

    def __len__(self) -> int:
        return len(self.demo) + len(self.labels["candidate_row"])

    def is_synthetic(self, i: int) -> bool:
        return i >= len(self.demo)

    def __getitem__(self, i: int) -> dict:
        torch = self._torch
        if i < len(self.demo):
            item = dict(self.base[int(self.demo[i])])
            item["ccil.weight"] = torch.tensor(self.demo_weight, dtype=torch.float64)
            item["ccil.synthetic"] = torch.tensor(False)
            return item
        j = i - len(self.demo)
        L = self.labels
        item = {self.feature: torch.as_tensor(L[self.feature][j])}
        if self.action_deltas is not None and list(self.action_deltas) != [0]:
            k = len(self.action_deltas)
            a = L["action_chunk"][j].reshape(L["action_chunk_mask"].shape[1], -1)[:k]
            item["action"] = torch.as_tensor(a)
            item["action_is_pad"] = torch.as_tensor(~L["action_chunk_mask"][j][:k])
        elif "action" in L:
            item["action"] = torch.as_tensor(L["action"][j])
        else:
            item["action"] = torch.as_tensor(L["action_chunk"][j].reshape(L["action_chunk_mask"].shape[1], -1)[0])
        item["ccil.weight"] = torch.tensor(float(L["weight"][j]), dtype=torch.float64)
        item["ccil.synthetic"] = torch.tensor(True)
        item["ccil.source_index"] = torch.tensor(int(L["source_index"][j]))
        item["ccil.candidate_row"] = torch.tensor(int(L["candidate_row"][j]))
        return item

    def weights(self, balance: bool = False) -> np.ndarray:
        """Per-row sampling weights: the recorded weights; with balance, demonstrations and labels each carry half
        the total mass."""
        w = np.concatenate([np.full(len(self.demo), self.demo_weight), self.labels["weight"].astype(np.float64)])
        if balance and len(self.demo) and len(self.labels["weight"]):
            d, s = w[:len(self.demo)].sum(), w[len(self.demo):].sum()
            if d > 0 and s > 0:
                w[:len(self.demo)] *= 0.5 / d
                w[len(self.demo):] *= 0.5 / s
        return w

    def sampler(self, num_samples: int | None = None, balance: bool = False, generator=None):
        torch = self._torch
        w = torch.as_tensor(self.weights(balance))
        return torch.utils.data.WeightedRandomSampler(w, num_samples or len(self), replacement=True,
                                                      generator=generator)
