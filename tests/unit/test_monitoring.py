"""Tests for runtime monitoring and health projection."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.runtime.monitoring import HealthMonitor
from dataenginex.runtime.state import ControlStore


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        yield s


class TestHealthMonitor:
    def test_worker_health_empty(self, store: ControlStore) -> None:
        monitor = HealthMonitor(store)
        result = monitor.worker_health()
        assert result == []

    def test_stream_health_empty(self, store: ControlStore) -> None:
        monitor = HealthMonitor(store)
        result = monitor.stream_health()
        assert result == []

    def test_resource_usage_empty(self, store: ControlStore) -> None:
        monitor = HealthMonitor(store)
        result = monitor.resource_usage()
        assert result == {"attempt_count": 0}

    def test_resource_usage_per_project(self, store: ControlStore) -> None:
        monitor = HealthMonitor(store)
        result = monitor.resource_usage(project_id="proj_test")
        assert result == {"attempt_count": 0}
