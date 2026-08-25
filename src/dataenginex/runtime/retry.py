"""Generic retry decorator — stdlib only, no tenacity."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 60.0,
    jitter: float = 0.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Retry a function on transient failures with exponential backoff.

    Usage:
        @retry(max_attempts=3, exceptions=(httpx.HTTPStatusError,))
        def fetch(url): ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                    delay *= 1 + random.uniform(-jitter, jitter)  # noqa: S311
                    time.sleep(delay)
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator
