"""Tests for dataenginex.runtime.retry."""

from __future__ import annotations

from dataenginex.runtime.retry import retry


class TestRetry:
    def test_succeeds_first_try(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0, jitter=0)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_exception(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0, jitter=0, exceptions=(ValueError,))
        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("nope")
            return "ok"

        assert fail_twice() == "ok"
        assert call_count == 3

    def test_raises_after_max_attempts(self) -> None:
        @retry(max_attempts=2, backoff_base=0, jitter=0, exceptions=(ValueError,))
        def always_fail() -> None:
            raise ValueError("boom")

        try:
            always_fail()
            raise AssertionError("should have raised")
        except ValueError as exc:
            assert "boom" in str(exc)

    def test_only_catches_specified_exceptions(self) -> None:
        @retry(max_attempts=3, backoff_base=0, jitter=0, exceptions=(ValueError,))
        def raise_type_error() -> None:
            raise TypeError("wrong type")

        try:
            raise_type_error()
            raise AssertionError("should have raised")
        except TypeError:
            pass

    def test_preserves_function_metadata(self) -> None:
        @retry(max_attempts=1)
        def documented() -> str:
            """My docstring."""
            return "ok"

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "My docstring."

    def test_passes_args_and_kwargs(self) -> None:
        @retry(max_attempts=1, backoff_base=0, jitter=0)
        def add(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        assert add(1, 2) == 3
        assert add(1, 2, c=10) == 13
