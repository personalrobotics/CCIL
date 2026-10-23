"""Run the working-tree generators and label collection against the fixture.

Calls the repository's own choose_augmentation and collect_labels, loaded from
correct_il/ without importing its gym/d4rl module graph, and only regroups the
returned per-candidate arrays by transition for the oracle.
"""
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from legacy import load_definitions

SOURCES = {
    "correct_il/gen_aug_label.py": [
        "_candidates", "_root_find_one",
        "gen_forward_euler", "gen_backward_euler", "gen_backward_euler_fast",
        "gen_noisy_action_be", "gen_noisy_action_fe", "choose_augmentation", "collect_labels"],
    "correct_il/utils.py": ["root_finder_s", "build_dataset"],
}
AUG_TYPE = {"backward_euler_fast": "backward_euler_fast", "backward_euler": "backward_euler",
            "forward_euler": "forward_euler", "noisy_action": "noisy_action"}


def run(config, model, trajectories):
    impl = load_definitions(SOURCES, extra={"partial": partial, "tqdm": lambda x: x})
    np_s, np_a, np_sp = impl["build_dataset"](trajectories)
    transitions = [{"s": s, "a": a, "s_next": sp} for s, a, sp in zip(np_s, np_a, np_sp)]
    data = tuple(torch.as_tensor(x).float() for x in (np_s, np_a, np_sp))

    results = {}
    for method, params in config["methods"].items():
        torch.manual_seed(config["seed"])  # main() seeds once before generating
        generator = impl["choose_augmentation"](SimpleNamespace(type=AUG_TYPE[method], **params))
        labels = impl["collect_labels"](generator, model, data, np_s, len(np_s))
        out = []
        for i in range(len(np_s)):
            idx = np.flatnonzero(labels["datapoint"] == i)
            reasons = [labels["reason"][k] for k in idx]
            out.append({
                "attempted": len(idx),
                "reported": {"accepted": int(labels["accepted"][idx].sum()),
                             "rejected": sum(r in ("non_finite", "out_of_support") for r in reasons),
                             "generation_failed": sum(r == "generation_failed" for r in reasons)},
                "candidates": [{"s_G": labels["s"][k], "a_G": labels["a"][k], "target": labels["s_next"][k],
                                "accepted": bool(labels["accepted"][k]), "reason": labels["reason"][k]} for k in idx],
            })
        results[method] = out
    return transitions, results, {}
