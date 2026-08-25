"""Shared resource-limit helpers for self-hosted, container-constrained deployments."""

from __future__ import annotations

import threading
from pathlib import Path

__all__ = ["duckdb_memory_limit", "note_duckdb_connection_closed", "note_duckdb_connection_opened"]

# ponytail: process-global counter (not per-container-cgroup) — fine because
# every DuckDB connection this process opens shares the same cgroup ceiling.
_lock = threading.Lock()
_active_connections = 0


def note_duckdb_connection_opened() -> None:
    """Record that a DuckDB connection is about to be opened.

    Call before ``duckdb.connect(...)`` so ``duckdb_memory_limit()`` sees an
    accurate count (including the connection being opened) when sizing it.
    """
    global _active_connections
    with _lock:
        _active_connections += 1


def note_duckdb_connection_closed() -> None:
    """Record that a previously-opened DuckDB connection has been closed."""
    global _active_connections
    with _lock:
        _active_connections = max(0, _active_connections - 1)


def duckdb_memory_limit() -> str:
    """Cap DuckDB's memory to a fraction of the container's cgroup limit.

    Without an explicit cap, a large operation (a hash join build side, or a
    big CSV-to-Parquet conversion) can grow past the container's actual RAM
    before DuckDB's own accounting decides to spill to temp_directory, taking
    the whole process down with it. On a resource-constrained self-hosted
    deployment the limit can't just be raised, so read the real cgroup
    ceiling (v2, then v1) and target a fraction of it, leaving headroom for
    the rest of the process. Falls back to a conservative fixed value
    outside a container.

    Each caller opens its own DuckDB connection and must bracket it with
    ``note_duckdb_connection_opened()``/``note_duckdb_connection_closed()``.
    When this is the only connection open (the common case — e.g.
    max_concurrent_pipelines=1) we can safely use 50% of container RAM. When
    more connections are open at the same time — e.g. several pipelines
    genuinely executing in parallel, or a pipeline's own DuckDB connection
    still open while its HttpConnector opens a second, nested one — the
    shared ~60%-of-RAM DuckDB budget is divided by however many connections
    are actually open, so the SUM of every connection's own cap stays within
    the container's real ceiling.

    Previously this was a flat "30% each if more than one is open" — correct
    for exactly 2 concurrent connections, but with N>2 (e.g. 4 pipelines
    genuinely running in parallel, which never happened before the pipeline
    executor was serialized to one worker) each still independently claimed
    30%, summing to well over 100% of the container's actual RAM and
    crashing the process (observed live: a 6.5GB-limited container OOMing
    inside DuckDB at "1.8 GiB/1.8 GiB used" — 30% of 6.5GB — while 4
    connections were open at once).
    """
    concurrent = max(1, _active_connections)
    fraction = min(0.50, 0.60 / concurrent)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit_bytes = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports a huge sentinel (close to 2**63) when unset.
        if 0 < limit_bytes < (1 << 60):
            return f"{max(1, round(limit_bytes * fraction / 1024**3))}GB"
    return "1GB"
