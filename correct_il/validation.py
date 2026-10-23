"""Trajectory-level validation splits and held-out transition-model metrics.

The label model is always fitted on every trajectory. Held-out metrics come from
a separate evaluation model fitted on the train split with the same settings, so
they validate the fitting procedure without changing the labels.
"""
import json
import os

import numpy as np
import torch

SPLIT_FORMAT = "ccil-split/1"


class SplitError(Exception):
    """A valid trajectory split cannot be formed or a persisted split is malformed."""


def make_split(n_trajectories, validation_fraction, seed):
    """Assign whole trajectories to train or validation; never splits an episode."""
    n_val = int(round(n_trajectories * validation_fraction))
    if not 1 <= n_val <= n_trajectories - 1:
        raise SplitError(
            f"dataset has {n_trajectories} trajectories; validation_fraction={validation_fraction} "
            f"gives {n_val} validation trajectories, but each split needs at least one. "
            f"Use more trajectories, change dynamics.validation_fraction, or set it to 0 "
            f"to skip held-out evaluation.")
    perm = np.random.default_rng(seed).permutation(n_trajectories)
    split = {"format": SPLIT_FORMAT, "seed": int(seed), "validation_fraction": float(validation_fraction),
             "n_trajectories": int(n_trajectories),
             "train": sorted(int(i) for i in perm[n_val:]), "validation": sorted(int(i) for i in perm[:n_val])}
    check_disjoint(split)
    return split


def check_disjoint(split):
    train, val = set(split["train"]), set(split["validation"])
    if train & val or train | val != set(range(split["n_trajectories"])):
        raise SplitError(f"split is not a partition of {split['n_trajectories']} trajectories: "
                         f"overlap {sorted(train & val)}")


def load_or_create_split(path, n_trajectories, validation_fraction, seed):
    """Reuse the split at path when it matches the configuration; otherwise write a new one.

    Returns (split, reused). The file is replaced atomically.
    """
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        if split.get("format") != SPLIT_FORMAT:
            raise SplitError(f"{path}: unsupported split format {split.get('format')!r}")
        if (split["n_trajectories"], split["validation_fraction"], split["seed"]) == \
                (n_trajectories, float(validation_fraction), int(seed)):
            check_disjoint(split)
            return split, True
    split = make_split(n_trajectories, validation_fraction, seed)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(split, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return split, False


def transitions(trajectories, backward=False):
    """(s, a, s_target) per trajectory, never crossing an episode boundary.

    A backward model maps (s_{t+1}, a_t) to s_t.
    """
    out = []
    for traj in trajectories:
        obs = np.asarray(traj["observations"], dtype=np.float32)
        act = np.asarray(traj["actions"], dtype=np.float32)
        act = act[:, None] if act.ndim == 1 else act
        s, a, sp = obs[:-1], act[:-1], obs[1:]
        out.append((sp, a, s) if backward else (s, a, sp))
    return out


@torch.no_grad()
def one_step_errors(model, trajectories, indices, backward=False):
    """Per-trajectory one-step L2 errors ||f(s, a) - s_target||_2, in state units.

    relative_l2 divides by the mean true displacement ||s_target - s||_2 of the
    same trajectory, so it is unitless and comparable across tasks. indices are
    the trajectories' positions in the dataset, recorded with each row.
    """
    rows = []
    for i, (s, a, target) in zip(indices, transitions(trajectories, backward)):
        pred = model.predict(s, a).cpu().numpy()
        err = np.linalg.norm(pred - target, axis=1)
        disp = float(np.mean(np.linalg.norm(target - s, axis=1)))
        rows.append({"trajectory": int(i), "transitions": int(len(err)), "mean_l2": float(np.mean(err)),
                     "max_l2": float(np.max(err)), "relative_l2": float(np.mean(err) / disp) if disp > 0 else None})
    return rows


def aggregate(rows):
    """Transition-weighted (pooled) and per-trajectory (macro) means of the one-step L2 error."""
    n = sum(r["transitions"] for r in rows)
    return {
        "trajectories": len(rows),
        "transitions": n,
        "pooled_mean_l2": float(sum(r["mean_l2"] * r["transitions"] for r in rows) / n),
        "macro_mean_l2": float(np.mean([r["mean_l2"] for r in rows])),
        "max_l2": float(max(r["max_l2"] for r in rows)),
    }


def held_out_evaluation(trajectories, split, fit, backward=False):
    """Fit an evaluation model on the train split and score it per trajectory.

    fit(trajectories) -> model must use the label model's settings and seed.
    Validation metrics use only split["validation"] trajectories; train-split
    errors are reported separately for comparison.
    """
    check_disjoint(split)
    train = [trajectories[i] for i in split["train"]]
    val = [trajectories[i] for i in split["validation"]]
    model = fit(train)
    held = one_step_errors(model, val, split["validation"], backward)
    seen = one_step_errors(model, train, split["train"], backward)
    return {
        "metric": "one-step L2 error ||f(s, a) - s_target||_2 in state units; relative_l2 divides by mean true displacement",
        "evaluation_model": "fitted on the train split with the label model's settings and seed; not saved",
        "label_model": "fitted on all trajectories",
        "held_out": {"aggregate": aggregate(held), "per_trajectory": held},
        "train_split": {"aggregate": aggregate(seen), "per_trajectory": seen},
    }
