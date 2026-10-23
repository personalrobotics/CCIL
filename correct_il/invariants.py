"""Ingress validation for demonstration datasets.

A dataset is a list of episodes, each a dict with "observations" (T, state_dim)
and "actions" (T, act_dim) or (T,) of equal length T >= 2. Transition t pairs
observations[t], actions[t] and observations[t + 1]; the last action is unused.
Validation reports every violation with its episode and frame and never
truncates, broadcasts, clamps, or coerces a value.
"""
import numpy as np

FLOAT32_MAX = float(np.finfo(np.float32).max)
MAX_EXAMPLES = 20  # per invariant; the count is always exact


class DataValidationError(ValueError):
    """The dataset violates one or more invariants; .violations lists them."""

    def __init__(self, violations, counts):
        self.violations, self.counts = violations, counts
        lines = [f"{v['invariant']}: episode {v['episode']}, frame {v['frame']}: {v['detail']}" for v in violations]
        summary = ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
        super().__init__(f"dataset failed validation ({summary}):\n  " + "\n  ".join(lines))


def validate_trajectories(trajectories, action_low=None, action_high=None):
    """Raise DataValidationError unless every invariant holds; return (state_dim, act_dim).

    action_low/action_high: optional per-dimension bounds (scalars broadcast by
    declaration, not by the data). Without them actions are not range-checked.
    """
    violations, counts = [], {}

    def bad(invariant, episode, frame, detail):
        counts[invariant] = counts.get(invariant, 0) + 1
        if counts[invariant] <= MAX_EXAMPLES:
            violations.append({"invariant": invariant, "episode": episode, "frame": frame, "detail": detail})

    if (action_low is None) != (action_high is None):
        raise ValueError("declare both action_low and action_high, or neither")
    if not isinstance(trajectories, (list, tuple)) or not trajectories:
        raise DataValidationError([{"invariant": "dataset_structure", "episode": None, "frame": None,
                                    "detail": f"expected a non-empty list of episodes, got {type(trajectories).__name__}"}],
                                  {"dataset_structure": 1})
    dims = None
    for e, ep in enumerate(trajectories):
        if not isinstance(ep, dict) or "observations" not in ep or "actions" not in ep:
            bad("episode_structure", e, None, "episode must be a dict with 'observations' and 'actions'")
            continue
        obs, act = np.asarray(ep["observations"]), np.asarray(ep["actions"])
        ok = True
        for name, x, ndims in (("observations", obs, (2,)), ("actions", act, (1, 2))):
            if not np.issubdtype(x.dtype, np.floating):
                bad("dtype", e, None, f"{name} dtype {x.dtype} is not a floating type")
                ok = False
            if x.ndim not in ndims:
                bad("shape", e, None, f"{name} has shape {x.shape}; expected {' or '.join(f'{n}-D' for n in ndims)}")
                ok = False
        if not ok:
            continue
        act = act[:, None] if act.ndim == 1 else act
        if len(obs) != len(act):
            bad("length_mismatch", e, None, f"{len(obs)} observations but {len(act)} actions")
            continue
        if len(obs) < 2:
            bad("too_short", e, None, f"{len(obs)} frames; a transition needs at least 2")
            continue
        ep_dims = (obs.shape[1], act.shape[1])
        if dims is None:
            dims = ep_dims
        elif ep_dims != dims:
            bad("dimension_mismatch", e, None, f"(state_dim, act_dim) = {ep_dims}, but episode 0 has {dims}")
            continue
        used_act = act[:-1]  # the last action never enters a transition
        for name, x in (("observations", obs), ("actions", used_act)):
            for f in np.flatnonzero(~np.isfinite(x).all(axis=1)):
                bad("non_finite", e, int(f), f"{name}[{f}] = {x[f].tolist()}")
            big = np.isfinite(x).all(axis=1) & (np.abs(x) > FLOAT32_MAX).any(axis=1)
            for f in np.flatnonzero(big):
                bad("float32_overflow", e, int(f), f"{name}[{f}] exceeds the float32 range")
        if action_low is not None:
            low = np.broadcast_to(np.asarray(action_low, dtype=float), (dims[1],))
            high = np.broadcast_to(np.asarray(action_high, dtype=float), (dims[1],))
            with np.errstate(invalid="ignore"):
                out = ((used_act < low) | (used_act > high)).any(axis=1)
            for f in np.flatnonzero(out):
                bad("action_out_of_bounds", e, int(f), f"actions[{f}] = {used_act[f].tolist()} outside [{low.tolist()}, {high.tolist()}]")
    if violations:
        raise DataValidationError(violations, counts)
    return dims


def transition_index(trajectories):
    """Episode and frame of each transition, aligned with utils.build_dataset."""
    lengths = [len(ep["observations"]) - 1 for ep in trajectories]
    episode = np.repeat(np.arange(len(lengths)), lengths)
    frame = np.concatenate([np.arange(n) for n in lengths]) if lengths else np.zeros(0, dtype=int)
    return episode, frame


def declared_action_bounds(config_data):
    """(low, high) from data.action_low/data.action_high, or None when undeclared."""
    keys = [k for k in ("action_low", "action_high") if k in config_data]
    if not keys:
        return None
    if len(keys) == 1:
        raise ValueError("declare both data.action_low and data.action_high, or neither")
    return config_data["action_low"], config_data["action_high"]
