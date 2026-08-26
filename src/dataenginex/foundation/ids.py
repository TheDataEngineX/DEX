"""Typed identifiers.

Every domain object carries a prefixed, sortable ID. The prefix makes IDs
self-describing in logs, errors, and lineage graphs; the UUIDv7 body makes them
time-ordered so index locality holds without a separate created_at sort.

Spec §4.5 requires UUIDv7 for revision IDs. We use it uniformly rather than
mixing generators.
"""

from __future__ import annotations

import os
import threading
import time
from typing import NewType

__all__ = [
    "ApprovalId",
    "ArtifactId",
    "AttemptId",
    "InstallationId",
    "OperationId",
    "PolicyDecisionId",
    "PrincipalId",
    "ProjectId",
    "ResourceId",
    "RevisionId",
    "RunId",
    "WorkspaceId",
    "new_id",
    "uuid7",
]

InstallationId = NewType("InstallationId", str)
WorkspaceId = NewType("WorkspaceId", str)
ProjectId = NewType("ProjectId", str)
RevisionId = NewType("RevisionId", str)
ResourceId = NewType("ResourceId", str)
ArtifactId = NewType("ArtifactId", str)
RunId = NewType("RunId", str)
AttemptId = NewType("AttemptId", str)
PrincipalId = NewType("PrincipalId", str)
PolicyDecisionId = NewType("PolicyDecisionId", str)
OperationId = NewType("OperationId", str)
ApprovalId = NewType("ApprovalId", str)


_lock = threading.Lock()
_last_ms = -1
_counter = 0


def uuid7() -> str:
    """Return a UUIDv7 string: 48-bit millisecond timestamp, counter, then random.

    Python's stdlib ``uuid`` has no v7 generator as of 3.13, so the bits are laid
    out directly per RFC 9562 §5.7. Time-ordered IDs keep B-tree inserts
    append-mostly, which matters for the append-only event and attempt tables.

    Ordering has to hold *within* a millisecond too — a burst of attempts or
    lineage events easily lands in the same tick, and random tails alone are
    unordered. The 12-bit ``rand_a`` field carries a monotonic counter for that
    case (RFC 9562 §6.2 "replace leftmost random bits with increased clock
    precision"), which gives 4096 ordered IDs per millisecond per process.
    """
    global _last_ms, _counter
    with _lock:
        timestamp_ms = time.time_ns() // 1_000_000
        if timestamp_ms == _last_ms:
            _counter += 1
            if _counter > 0x0FFF:
                # Counter exhausted: borrow from the next millisecond rather than
                # emit an out-of-order ID.
                _last_ms += 1
                timestamp_ms = _last_ms
                _counter = 0
        else:
            if timestamp_ms < _last_ms:
                # Clock moved backwards (NTP step). Hold the previous instant.
                timestamp_ms = _last_ms
            _last_ms = timestamp_ms
            _counter = 0
        counter = _counter

    value = bytearray(timestamp_ms.to_bytes(6, "big") + os.urandom(10))
    # Octets 6-7: version 7 nibble + 12-bit counter. Octet 8 high bits: variant 0b10.
    value[6] = 0x70 | (counter >> 8)
    value[7] = counter & 0xFF
    value[8] = (value[8] & 0x3F) | 0x80
    h = value.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def new_id(prefix: str) -> str:
    """Return a prefixed UUIDv7, e.g. ``rev_018f4c2a-...``."""
    return f"{prefix}_{uuid7()}"


if __name__ == "__main__":  # pragma: no cover - self-check
    # Burst well past the 4096/ms counter to exercise the borrow path.
    ids = [uuid7() for _ in range(20_000)]
    assert len(set(ids)) == 20_000, "uuid7 collision"
    assert ids == sorted(ids), "uuid7 not monotonic within a millisecond"
    for candidate in ids[:50]:
        assert len(candidate) == 36, candidate
        assert candidate[14] == "7", f"version nibble wrong: {candidate}"
        assert candidate[19] in "89ab", f"variant bits wrong: {candidate}"

    # Concurrent generation must not collide.
    collected: list[list[str]] = []
    threads = [
        threading.Thread(target=lambda: collected.append([uuid7() for _ in range(2000)]))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [i for batch in collected for i in batch]
    assert len(set(flat)) == len(flat), "uuid7 collision across threads"

    assert new_id("rev").startswith("rev_")
    print("ids self-check ok")
