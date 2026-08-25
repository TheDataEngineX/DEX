"""duckdb_memory_limit() must divide the shared DuckDB budget by however many
connections are actually concurrently open, so the SUM of every open
connection's own cap stays within the container's real ceiling.

Previously it was a flat "30% each if more than one connection is open" —
correct only for exactly 2 concurrent connections. With N>2 (now reachable
since the pipeline executor runs real work in parallel instead of
serializing everything through one worker thread), each connection still
independently claimed 30%, so N connections summed to N*30% — e.g. 4
connections summing to 120% of the container's actual RAM, which is exactly
what crashed a live 6.5GB-limited container (DuckDB OOMing internally at
"1.8 GiB/1.8 GiB used", i.e. 30% of 6.5GB, with 4 connections open at once).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dataenginex.runtime import resource_limits as resources


@pytest.fixture(autouse=True)
def _reset_connection_count():
    resources._active_connections = 0
    yield
    resources._active_connections = 0


def _fake_cgroup_limit_gb(gb: float):
    """Patch Path.read_text so duckdb_memory_limit() sees a fixed cgroup limit."""
    limit_bytes = str(int(gb * 1024**3))

    def _read_text(self: Path, *a: object, **k: object) -> str:
        if str(self) == "/sys/fs/cgroup/memory.max":
            return limit_bytes
        raise OSError("no such path")

    return patch.object(Path, "read_text", _read_text)


def _parse_gb(s: str) -> float:
    assert s.endswith("GB")
    return float(s[:-2])


def test_single_connection_gets_half() -> None:
    resources.note_duckdb_connection_opened()
    with _fake_cgroup_limit_gb(10.0):
        assert _parse_gb(resources.duckdb_memory_limit()) == pytest.approx(5.0)


def test_two_connections_get_30_percent_each() -> None:
    resources.note_duckdb_connection_opened()
    resources.note_duckdb_connection_opened()
    with _fake_cgroup_limit_gb(10.0):
        assert _parse_gb(resources.duckdb_memory_limit()) == pytest.approx(3.0)


def test_sum_across_many_concurrent_connections_stays_within_container_limit() -> None:
    """The historical bug: N connections each independently claiming a flat
    fraction can sum to more than the container's actual RAM. For any N,
    N * per-connection-limit must stay comfortably within the total."""
    total_gb = 32.0  # realistic container size — whole-GB rounding matters less
    for n in (1, 2, 4, 8, 16):
        resources._active_connections = n
        with _fake_cgroup_limit_gb(total_gb):
            per_connection_gb = _parse_gb(resources.duckdb_memory_limit())
        # Rounds to whole GB, so small totals see some rounding overshoot —
        # 0.7 is still far below the historical bug's 100%+-of-RAM sum.
        assert per_connection_gb * n <= total_gb * 0.7, (
            f"n={n}: {per_connection_gb} * {n} = {per_connection_gb * n} "
            f"exceeds safe share of {total_gb}GB total"
        )


if __name__ == "__main__":
    test_single_connection_gets_half()
    test_two_connections_get_30_percent_each()
    test_sum_across_many_concurrent_connections_stays_within_container_limit()
    print("ok")
