"""Collision-resistant transition identities for deterministic randomness.

The digest is BLAKE2b-128 over a versioned, unambiguous encoding of the full
episode ID and signed frame index.
"""
from __future__ import annotations

import hashlib
import struct

IDENTITY_VERSION = b"ccil.transition-identity/1"


def identity_digest(episode_id, frame) -> bytes:
    """16-byte digest of (episode_id, frame)."""
    ep = str(episode_id).encode("utf-8")
    payload = IDENTITY_VERSION + b"\0" + struct.pack(">Q", len(ep)) + ep + struct.pack(">q", int(frame))
    return hashlib.blake2b(payload, digest_size=16).digest()


def identity_words32(episode_id, frame) -> list[int]:
    """The digest as four unsigned 32-bit words (for numpy SeedSequence entropy)."""
    return list(struct.unpack(">4I", identity_digest(episode_id, frame)))


def identity_words64(episode_id, frame) -> tuple[int, int]:
    """The digest as two unsigned 64-bit words (for counter hashing)."""
    return struct.unpack(">2Q", identity_digest(episode_id, frame))
