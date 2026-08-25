"""Tests for EngineRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from dataenginex.engines.base import (
    BaseEngine,
    EngineCapabilities,
    EngineConfig,
    LoadResult,
    MergeResult,
    QualityResult,
    SCD2Result,
)
from dataenginex.engines.registry import EngineRegistry


class MockEngine(BaseEngine):
    """Minimal engine for testing the registry."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name
        self._connected = False

    def connect(self, config: EngineConfig) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def extract(self, source_config: Any) -> Any:
        return None

    def transform(self, df: Any, steps: list[Any]) -> Any:
        return df

    def quality_check(self, df: Any, checks: Any) -> QualityResult:
        return QualityResult(passed=True)

    def load(self, df: Any, target_config: Any) -> LoadResult:
        return LoadResult(success=True)

    def merge(
        self, target: str, source: str, keys: list[str], strategy: str = "upsert"
    ) -> MergeResult:
        return MergeResult(success=True)

    def scd_type2(
        self, target: str, source: str, keys: list[str], valid_from: str = "valid_from"
    ) -> SCD2Result:
        return SCD2Result(success=True)

    def content_hash(self, df: Any) -> str:
        return "mock_hash"

    def read_table(self, table: str) -> Any:
        return None

    def write_table(self, df: Any, table: str, format: str = "parquet") -> None:
        pass

    def execute_sql(self, sql: str) -> Any:
        return None

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(name=self._name)


class TestEngineRegistry:
    def test_register_and_get(self) -> None:
        registry = EngineRegistry()
        engine = MockEngine("test")
        registry.register("test", engine)
        assert registry.get("test") is engine

    def test_register_duplicate_raises(self) -> None:
        registry = EngineRegistry()
        registry.register("a", MockEngine("a"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register("a", MockEngine("a2"))

    def test_get_unknown_raises(self) -> None:
        registry = EngineRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_engines(self) -> None:
        registry = EngineRegistry()
        registry.register("a", MockEngine("a"))
        registry.register("b", MockEngine("b"))
        assert registry.list_engines() == ["a", "b"]

    def test_is_registered(self) -> None:
        registry = EngineRegistry()
        registry.register("a", MockEngine("a"))
        assert registry.is_registered("a") is True
        assert registry.is_registered("b") is False

    def test_engine_connect_disconnect(self) -> None:
        registry = EngineRegistry()
        engine = MockEngine("test")
        registry.register("test", engine)

        config = EngineConfig(type="test", path=":memory:")
        engine.connect(config)
        assert engine._connected is True

        engine.disconnect()
        assert engine._connected is False

    def test_engine_capabilities(self) -> None:
        engine = MockEngine("test")
        caps = engine.capabilities()
        assert caps.name == "test"
