"""Tests for dataenginex.runtime.circuit_breaker."""

from __future__ import annotations

import time

from dataenginex.runtime.circuit_breaker import CircuitBreaker, State


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.state == State.CLOSED
        assert cb.allow() is True

    def test_records_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        assert cb._failure_count == 1
        assert cb.state == State.CLOSED

    def test_trips_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
        cb.record_failure()
        assert cb.state == State.CLOSED
        cb.record_failure()
        assert cb.state == State.OPEN
        assert cb.allow() is False

    def test_records_success_resets(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == State.CLOSED
        assert cb.allow() is True

    def test_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == State.OPEN
        time.sleep(0.02)
        assert cb.state == State.HALF_OPEN
        assert cb.allow() is True

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == State.HALF_OPEN
        cb.record_success()
        assert cb.state == State.CLOSED

    def test_half_open_failure_opens_again(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == State.HALF_OPEN
        cb.record_failure()
        assert cb.state == State.OPEN

    def test_default_params(self) -> None:
        cb = CircuitBreaker()
        assert cb._failure_threshold == 5
        assert cb._recovery_timeout == 30.0
