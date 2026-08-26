"""Simple circuit breaker — stdlib only."""

from __future__ import annotations

import time
from enum import Enum


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker that trips after consecutive failures.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        if breaker.allow():
            try:
                result = do_work()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = State.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> State:
        if (
            self._state == State.OPEN
            and time.monotonic() - self._last_failure_time >= self._recovery_timeout
        ):
            self._state = State.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        s = self.state
        return s in (State.CLOSED, State.HALF_OPEN)

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = State.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = State.OPEN
