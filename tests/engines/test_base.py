"""Tests for engine abstraction — BaseEngine, EngineConfig, EngineCapabilities."""

from __future__ import annotations

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


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_minimal_config(self) -> None:
        config = EngineConfig(type="duckdb")
        assert config.type == "duckdb"
        assert config.path is None
        assert config.threads is None
        assert config.memory_limit is None
        assert config.master is None
        assert config.executor_memory is None
        assert config.executor_cores is None
        assert config.file_format == "parquet"
        assert config.warehouse is None
        assert config.catalog is None
        assert config.options == {}

    def test_duckdb_config(self) -> None:
        config = EngineConfig(
            type="duckdb",
            path="./test.duckdb",
            threads=4,
            memory_limit="8GB",
        )
        assert config.type == "duckdb"
        assert config.path == "./test.duckdb"
        assert config.threads == 4
        assert config.memory_limit == "8GB"

    def test_spark_config(self) -> None:
        config = EngineConfig(
            type="spark",
            master="local[*]",
            executor_memory="4g",
            executor_cores=2,
            file_format="iceberg",
            warehouse=".dex/lakehouse",
        )
        assert config.type == "spark"
        assert config.master == "local[*]"
        assert config.executor_memory == "4g"
        assert config.executor_cores == 2
        assert config.file_format == "iceberg"
        assert config.warehouse == ".dex/lakehouse"

    def test_config_is_frozen(self) -> None:
        config = EngineConfig(type="duckdb")
        with pytest.raises(AttributeError):
            config.type = "spark"  # type: ignore[misc]


class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_duckdb_capabilities(self) -> None:
        caps = EngineCapabilities(
            name="duckdb",
            streaming=False,
            distributed=False,
            auto_cdc=False,
            iceberg_read=True,
            iceberg_write=False,
            delta_read=True,
            delta_write=True,
        )
        assert caps.name == "duckdb"
        assert caps.streaming is False
        assert caps.distributed is False
        assert caps.auto_cdc is False
        assert caps.iceberg_read is True
        assert caps.iceberg_write is False
        assert caps.delta_read is True
        assert caps.delta_write is True
        assert caps.mllib is False
        assert caps.catalyst is False

    def test_spark_capabilities(self) -> None:
        caps = EngineCapabilities(
            name="spark",
            streaming=True,
            distributed=True,
            auto_cdc=True,
            iceberg_read=True,
            iceberg_write=True,
            delta_read=True,
            delta_write=True,
            mllib=True,
            catalyst=True,
            spark_connect=True,
        )
        assert caps.name == "spark"
        assert caps.streaming is True
        assert caps.distributed is True
        assert caps.auto_cdc is True
        assert caps.mllib is True
        assert caps.spark_connect is True


class TestQualityResult:
    def test_default(self) -> None:
        result = QualityResult(passed=True)
        assert result.passed is True
        assert result.completeness_score == 0.0
        assert result.uniqueness_score == 0.0
        assert result.custom_passed is True
        assert result.details == {}

    def test_with_scores(self) -> None:
        result = QualityResult(
            passed=False,
            completeness_score=0.85,
            uniqueness_score=0.92,
            custom_passed=False,
            details={"missing_cols": ["email"]},
        )
        assert result.passed is False
        assert result.completeness_score == 0.85
        assert result.details["missing_cols"] == ["email"]


class TestLoadResult:
    def test_default(self) -> None:
        result = LoadResult(success=True)
        assert result.success is True
        assert result.rows_output == 0
        assert result.format == "parquet"
        assert result.error is None


class TestMergeResult:
    def test_default(self) -> None:
        result = MergeResult(success=True)
        assert result.success is True
        assert result.rows_inserted == 0
        assert result.rows_updated == 0


class TestSCD2Result:
    def test_default(self) -> None:
        result = SCD2Result(success=True)
        assert result.success is True
        assert result.rows_inserted == 0
        assert result.rows_updated == 0
        assert result.rows_expired == 0


class TestBaseEngineABC:
    """Test that BaseEngine cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseEngine()  # type: ignore[abstract]

    def test_must_implement_all_methods(self) -> None:
        """A subclass missing methods cannot be instantiated."""

        class IncompleteEngine(BaseEngine):
            def connect(self, config: EngineConfig) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def capabilities(self) -> EngineCapabilities:
                return EngineCapabilities(name="incomplete")

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]
