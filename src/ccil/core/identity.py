"""Collision-resistant digests of transition identities, for deterministic per-sample randomness.

A transition is identified by (episode_id, frame). Seeding randomness from a
32-bit hash of the episode id lets distinct episodes share a stream (CRC32
collisions are easy to find, e.g. "4300bByhudVrUGx5" and "0BZ5Gzkl9ccfsEbx"), so
the identity is digested in full: BLAKE2b-128 over a versioned encoding of the
UTF-8 episode id (length-prefixed) and the frame (signed 64-bit).
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
