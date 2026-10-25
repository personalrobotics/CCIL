"""Action-chunk labels: a corrective action followed by the demonstration's continuation (#22).

A label (z_G, a_G) arrives at frame tau of its source episode: tau = t for
backtrack labels (backward Euler, exact preimage) and t + 1 for
disturbed_action labels (noisy action). Its chunk of size K holds a_G at
position 0 and the episode's actions a_{tau + k - 1} at positions k >= 1, which
the demonstrator took from the state the label reaches. Positions past the
episode's last frame are masked and zero: a chunk is never padded and never
reads another episode.

Only position 0 carries the label's claim. The CCIL results are one-step
(Ke et al., Thms. 4.2 and 4.3); positions k >= 1 are demonstration actions with
no corrective-label guarantee.
"""
from __future__ import annotations

import numpy as np

from .batch import InterfaceError

TARGET_OFFSET = {"backtrack": 0, "disturbed_action": 1}
SPEC = {
    "alignment": "position 0 is the label action a_G; position k >= 1 is the source episode's action at frame "
                 "chunk_target_frame + k - 1, taken from the state the label reaches",
    "guarantee": "one-step: only position 0 carries the label's CCIL claim; positions k >= 1 are demonstration actions",
    "truncation": "positions past the source episode's last frame are masked and zero; never padded",
}
TARGET_FRAME_SPEC = {"definition": "frame the label arrives at in its source episode: t for backtrack labels, t + 1 for "
                                   "disturbed_action labels; chunk position k >= 1 holds the action at this frame + k - 1",
                     "norm": None, "units": "frames"}


def action_chunks(dataset, generator, episode_ids, frames, label_actions, chunk_size: int):
    """(chunks (N, K, A), mask (N, K), target frames (N,)) for labels of the given source transitions."""
    profile = getattr(generator, "label_error_profile", None)
    if profile not in TARGET_OFFSET:
        raise InterfaceError(f"{type(generator).__name__} declares no label_error_profile, so the frame its labels "
                             "arrive at, and their chunk alignment, is undefined")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise InterfaceError(f"action_chunk_size must be an integer >= 1, got {chunk_size!r}")
    K = int(chunk_size)
    n, A = label_actions.shape
    chunks = np.zeros((n, K, A), dtype=label_actions.dtype)
    mask = np.zeros((n, K), bool)
    chunks[:, 0], mask[:, 0] = label_actions, True
    tau = np.asarray(frames, np.int64) + TARGET_OFFSET[profile]
    episodes = {ep.episode_id: ep for ep in dataset.episodes}
    for eid in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == eid)
        ep = episodes[str(eid)]
        pos = tau[rows, None] + np.arange(K - 1)[None, :]  # frame of positions 1..K-1
        inside = pos <= ep.length - 1
        mask[rows, 1:] = inside
        block = chunks[rows]  # a copy (fancy indexing); written back below
        block[:, 1:][inside] = ep.actions[pos[inside]].astype(label_actions.dtype)
        chunks[rows] = block
    return chunks, mask, tau.astype(np.float64), TARGET_OFFSET[profile]
