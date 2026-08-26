"""Tests for the generic backend registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from dataenginex.runtime.registry import BackendRegistry


class BaseWidget(ABC):
    """Dummy ABC for testing."""

    @abstractmethod
    def do_work(self) -> str: ...


class TestBackendRegistry:
    """BackendRegistry discovers, registers, and instantiates backends."""

    def setup_method(self) -> None:
        self.registry: BackendRegistry[BaseWidget] = BackendRegistry("widget")

    def test_register_and_get(self) -> None:
        class FooWidget(BaseWidget):
            def do_work(self) -> str:
                return "foo"

        self.registry.register("foo", FooWidget)
        cls = self.registry.get("foo")
        assert cls is FooWidget

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="widget.*unknown"):
            self.registry.get("unknown")

    def test_list_registered(self) -> None:
        class A(BaseWidget):
            def do_work(self) -> str:
                return "a"

        class B(BaseWidget):
            def do_work(self) -> str:
                return "b"

        self.registry.register("a", A)
        self.registry.register("b", B)
        assert sorted(self.registry.list()) == ["a", "b"]

    def test_register_duplicate_raises(self) -> None:
        class W(BaseWidget):
            def do_work(self) -> str:
                return "w"

        self.registry.register("w", W)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register("w", W)

    def test_register_decorator(self) -> None:
        @self.registry.decorator("bar")
        class BarWidget(BaseWidget):
            def do_work(self) -> str:
                return "bar"

        assert self.registry.get("bar") is BarWidget

    def test_default_backend(self) -> None:
        class DefaultWidget(BaseWidget):
            def do_work(self) -> str:
                return "default"

        self.registry.register("default", DefaultWidget, is_default=True)
        assert self.registry.get_default() is DefaultWidget

    def test_no_default_raises(self) -> None:
        with pytest.raises(ValueError, match="no default"):
            self.registry.get_default()
