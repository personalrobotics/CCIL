"""Trajectory schema `ccil.trajectories` 1.0.

An episode is T frames. Frame t has one entry per observation modality and an
action; a transition is (frame t, action t, frame t+1), identified by
(episode_id, t) for t <= T-2. Every action frame must be finite because
policies also train on the final (observation, action) pair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .fields import ArraySpec, Violations

SCHEMA = "ccil.trajectories"
VERSION = (1, 1)  # 1.1 adds the optional per-episode termination_known (#27)

MODALITY = re.compile(r"^(state|image)(\.[a-z0-9_]+)?$")
STATE = ArraySpec("observations[state*]", "float", ("T", "D"), "task-defined",
                  "Low-dimensional state per frame; a modality named state or state.<name>.")
IMAGE = ArraySpec("observations[image*]", "uint8", ("T", "H", "W", "C"), "pixel intensity 0-255",
                  "Camera frames per frame; a modality named image.<camera>, C in {1, 3, 4}.")
ACTIONS = ArraySpec("actions", "float", ("T", "A"), "task-defined", "Action applied at each frame.")
TIMESTAMPS = ArraySpec("timestamps", "float", ("T",), "seconds",
                       "Frame times; strictly increasing. Given for every episode or none.", required=False)
CHUNKS = ArraySpec("action_chunks", "float", ("T", "K", "A"), "task-defined",
                   "action_chunks[t, k] is the action intended at frame t + k. Given for every episode or none, with one K.",
                   required=False)
CHUNK_MASK = ArraySpec("action_chunk_mask", "bool", ("T", "K"), "-",
                       "True exactly where t + k < T; required with action_chunks. A chunk never crosses the episode end.",
                       required=False)

EPISODE_FIELDS = [
    ("episode_id", "str", "Unique, non-empty identifier of the episode."),
    ("observations", "dict[str, array]", "Named modalities matching state, state.<name>, or image.<camera>."),
    ("actions", "array", ACTIONS.description),
    ("terminated", "bool", "True if the episode ended in a terminal state; False if it was truncated, or if its ending "
                           "is unknown (termination_known False)."),
    ("termination_known", "bool | None", "Whether the source records why the episode ended (1.1). False: unknown, "
                                         "and terminated must be False; None: not recorded (1.0 files). Set for every "
                                         "episode or for none."),
    ("timestamps", "array | None", TIMESTAMPS.description),
    ("task", "str | None", "Task identifier or instruction."),
    ("action_chunks", "array | None", CHUNKS.description),
    ("action_chunk_mask", "array | None", CHUNK_MASK.description),
]
ARRAY_SPECS = [STATE, IMAGE, ACTIONS, TIMESTAMPS, CHUNKS, CHUNK_MASK]


@dataclass
class Episode:
    episode_id: str
    observations: dict
    actions: np.ndarray
    terminated: bool = False
    termination_known: bool | None = None
    timestamps: np.ndarray | None = None
    task: str | None = None
    action_chunks: np.ndarray | None = None
    action_chunk_mask: np.ndarray | None = None

    @property
    def length(self) -> int:
        return len(self.actions) if isinstance(self.actions, np.ndarray) else 0


@dataclass
class TrajectoryDataset:
    episodes: list
    meta: dict = field(default_factory=dict)

    def validate(self, action_low=None, action_high=None) -> "TrajectoryDataset":
        """Raise SchemaError listing every violation; return self when valid."""
        out = Violations(SCHEMA)
        if not isinstance(self.episodes, list) or not self.episodes:
            out.add("structure", "dataset", "episodes must be a non-empty list")
            out.raise_if_any()
        if (action_low is None) != (action_high is None):
            raise ValueError("declare both action_low and action_high, or neither")
        if len({ep.termination_known is None for ep in self.episodes if isinstance(ep, Episode)}) > 1:
            out.add("termination", "dataset", "termination_known must be set for every episode or for none")
        seen, shared, bounds_reported = set(), {}, False
        modalities = None
        for e, ep in enumerate(self.episodes):
            where = f"episode {e}"
            if not isinstance(ep, Episode):
                out.add("structure", where, f"expected Episode, got {type(ep).__name__}")
                continue
            if not isinstance(ep.episode_id, str) or not ep.episode_id:
                out.add("episode_id", where, "episode_id must be a non-empty string")
            elif ep.episode_id in seen:
                out.add("episode_id", where, f"duplicate episode_id {ep.episode_id!r}")
            else:
                seen.add(ep.episode_id)
                where = f"episode {e} ({ep.episode_id})"
            if type(ep.terminated) is not bool:
                out.add("type", where, f"terminated must be bool, got {type(ep.terminated).__name__}")
            if ep.termination_known is not None and type(ep.termination_known) is not bool:
                out.add("type", where, f"termination_known must be bool or None, got {type(ep.termination_known).__name__}")
            elif ep.termination_known is False and ep.terminated is True:
                out.add("termination", where, "terminated is True but termination_known is False: an unknown ending "
                                              "cannot be terminal")
            if ep.task is not None and not isinstance(ep.task, str):
                out.add("type", where, f"task must be str or None, got {type(ep.task).__name__}")
            dims = {}
            if not ACTIONS.check(ep.actions, where, dims, out):
                continue
            T = dims["T"]
            if T < 2:
                out.add("too_short", where, f"{T} frames; a transition needs at least 2")
            if "A" in shared and dims["A"] != shared["A"]:
                out.add("dimension_mismatch", where, f"action dim {dims['A']}, other episodes have {shared['A']}")
            shared.setdefault("A", dims["A"])
            if not isinstance(ep.observations, dict) or not ep.observations:
                out.add("structure", where, "observations must be a non-empty dict of modalities")
                continue
            if modalities is None:
                modalities = set(ep.observations)
            elif set(ep.observations) != modalities:
                out.add("modality_mismatch", where, f"modalities {sorted(ep.observations)}, others have {sorted(modalities)}")
            for name, x in ep.observations.items():
                m = MODALITY.match(name) if isinstance(name, str) else None
                if not m:
                    out.add("modality_name", where, f"modality {name!r} must match state[.name] or image.<camera>")
                    continue
                spec = STATE if m.group(1) == "state" else IMAGE
                mdims = {"T": T}
                if not spec.check(x, f"{where}, {name}", mdims, out):
                    continue
                if spec is IMAGE and mdims["C"] not in (1, 3, 4):
                    out.add("shape", f"{where}, {name}", f"image channels {mdims['C']} not in (1, 3, 4)")
                trailing = x.shape[1:]
                if shared.setdefault(name, trailing) != trailing:
                    out.add("dimension_mismatch", f"{where}, {name}", f"shape {trailing}, other episodes have {shared[name]}")
            if ep.timestamps is not None and TIMESTAMPS.check(ep.timestamps, where, {"T": T}, out):
                bad = np.flatnonzero(np.diff(ep.timestamps) <= 0)
                for f in bad[:5]:
                    out.add("timestamps", f"{where}, frame {int(f) + 1}", "timestamps must strictly increase")
            if (ep.action_chunks is None) != (ep.action_chunk_mask is None):
                out.add("structure", where, "action_chunks and action_chunk_mask must be given together")
            elif ep.action_chunks is not None:
                cdims = {"T": T, "A": dims["A"]}
                ok = CHUNKS.check(ep.action_chunks, where, cdims, out, finite=False)
                ok = CHUNK_MASK.check(ep.action_chunk_mask, where, cdims, out) and ok
                if ok:
                    K = cdims["K"]
                    if shared.setdefault("K", K) != K:
                        out.add("dimension_mismatch", where, f"action chunk length {K}, other episodes have {shared['K']}")
                    within = (np.arange(T)[:, None] + np.arange(K)[None, :]) < T
                    cross = ep.action_chunk_mask & ~within
                    for t in np.flatnonzero(cross.any(axis=1))[:5]:
                        out.add("chunk_crosses_episode", f"{where}, frame {int(t)}",
                                "action_chunk_mask marks entries past the episode end as valid")
                    holes = ~ep.action_chunk_mask & within
                    for t in np.flatnonzero(holes.any(axis=1))[:5]:
                        out.add("chunk_mask", f"{where}, frame {int(t)}", "in-episode chunk entries must be marked valid")
                    vals = ep.action_chunks[ep.action_chunk_mask]
                    if not np.isfinite(vals).all():
                        out.add("non_finite", where, "action_chunks has non-finite values at valid entries")
            if action_low is not None:
                sizes = {np.size(action_low), np.size(action_high)}
                if not sizes <= {1, dims["A"]}:
                    if not bounds_reported:
                        out.add("dimension_mismatch", "action bounds", f"bounds have {sorted(sizes)} values but actions "
                                                                       f"have {dims['A']} dimensions; give 1 or {dims['A']}")
                    bounds_reported = True
                    continue
                low = np.broadcast_to(np.asarray(action_low, float), (dims["A"],))
                high = np.broadcast_to(np.asarray(action_high, float), (dims["A"],))
                with np.errstate(invalid="ignore"):
                    oob = ((ep.actions < low) | (ep.actions > high)).any(axis=1)
                for f in np.flatnonzero(oob):
                    out.add("action_out_of_bounds", f"{where}, frame {int(f)}", f"actions {ep.actions[f].tolist()} outside bounds")
        # Optional per-frame fields are stored as one array across episodes, so each is all-or-none.
        eps = [ep for ep in self.episodes if isinstance(ep, Episode)]
        for name in ("timestamps", "action_chunks"):
            have = [getattr(ep, name) is not None for ep in eps]
            if any(have) and not all(have):
                missing = [ep.episode_id for ep, h in zip(eps, have) if not h]
                out.add("optional_field_presence", "dataset", f"{name} is given for {sum(have)} of {len(have)} episodes; "
                                                              f"give it for every episode or none (missing: {missing[:5]})")
        out.raise_if_any()
        return self

    def transitions(self, modality: str = "state"):
        """(s, a, s_next, episode_ids, frames) for every transition, never crossing an episode."""
        s, a, sp, ids, frames = [], [], [], [], []
        for ep in self.episodes:
            x, T = ep.observations[modality], ep.length
            s.append(x[:-1]); sp.append(x[1:]); a.append(ep.actions[:-1])
            ids.append(np.full(T - 1, ep.episode_id)); frames.append(np.arange(T - 1))
        return tuple(np.concatenate(v) for v in (s, a, sp, ids, frames))
