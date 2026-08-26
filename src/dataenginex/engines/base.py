"""Engine abstraction — pluggable execution backends for data pipelines.

Phase 1: BaseEngine ABC + EngineCapabilities.
Phase 2: DuckDBEngine (full).
Phase 3: SparkEngine (full).

Usage::

    from dataenginex.engines import engine_registry

    engine = engine_registry.get("duckdb")
    engine.connect(config)
    df = engine.extract(source)
    df = engine.transform(df, steps)
    engine.load(df, target)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "EngineConfig",
    "QualityResult",
    "LoadResult",
    "MergeResult",
    "SCD2Result",
]


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for an engine instance."""

    type: str  # "duckdb", "spark"
    # DuckDB-specific
    path: str | None = None
    threads: int | None = None
    memory_limit: str | None = None
    # Spark-specific
    master: str | None = None  # "local[*]", "spark://host:port", "sc://host:port"
    executor_memory: str | None = None
    executor_cores: int | None = None
    file_format: str = "parquet"  # "parquet", "delta", "iceberg"
    warehouse: str | None = None
    catalog: str | None = None
    # Generic
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can do — lets scheduler refuse unsupported features."""

    name: str
    streaming: bool = False
    distributed: bool = False
    auto_cdc: bool = False
    iceberg_read: bool = False
    iceberg_write: bool = False
    delta_read: bool = False
    delta_write: bool = False
    mllib: bool = False
    catalyst: bool = False
    spark_connect: bool = False


@dataclass
class QualityResult:
    """Result of a quality check."""

    passed: bool
    completeness_score: float = 0.0
    uniqueness_score: float = 0.0
    custom_passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadResult:
    """Result of a load operation."""

    success: bool
    rows_output: int = 0
    location: str | None = None
    format: str = "parquet"
    error: str | None = None


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    rows_inserted: int = 0
    rows_updated: int = 0
    error: str | None = None


@dataclass
class SCD2Result:
    """Result of a SCD Type 2 operation."""

    success: bool
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_expired: int = 0
    error: str | None = None


class BaseEngine(abc.ABC):
    """Abstract engine interface — all execution backends implement this.

    The pipeline runner dispatches to engines via this interface,
    keeping the runner engine-agnostic.
    """

    @abc.abstractmethod
    def connect(self, config: EngineConfig) -> None:
        """Initialize the engine with the given configuration."""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Clean up engine resources."""
        ...

    @abc.abstractmethod
    def extract(self, source_config: Any) -> Any:
        """Extract data from a source into a DataFrame.

        Args:
            source_config: SourceConfig from dex.yaml.

        Returns:
            DataFrame (pandas for DuckDB, Spark DataFrame for Spark).
        """
        ...

    @abc.abstractmethod
    def transform(self, df: Any, steps: list[Any]) -> Any:
        """Apply transform steps to a DataFrame.

        Args:
            df: Input DataFrame.
            steps: List of TransformStepConfig.

        Returns:
            Transformed DataFrame.
        """
        ...

    @abc.abstractmethod
    def quality_check(self, df: Any, checks: Any) -> QualityResult:
        """Run quality checks on a DataFrame.

        Args:
            df: DataFrame to check.
            checks: QualityCheckConfig.

        Returns:
            QualityResult with pass/fail and scores.
        """
        ...

    @abc.abstractmethod
    def load(self, df: Any, target_config: Any) -> LoadResult:
        """Load a DataFrame to the target location.

        Args:
            df: DataFrame to load.
            target_config: Target configuration (layer, format, path).

        Returns:
            LoadResult with row count and status.
        """
        ...

    @abc.abstractmethod
    def merge(
        self, target: str, source: str, keys: list[str], strategy: str = "upsert"
    ) -> MergeResult:
        """Merge source into target table.

        Args:
            target: Target table name or path.
            source: Source table name or path.
            keys: Join keys for matching.
            strategy: "upsert", "insert_only", "delete_insert".

        Returns:
            MergeResult with insert/update counts.
        """
        ...

    @abc.abstractmethod
    def scd_type2(
        self, target: str, source: str, keys: list[str], valid_from: str = "valid_from"
    ) -> SCD2Result:
        """SCD Type 2 merge.

        Args:
            target: Target table name or path.
            source: Source table name or path.
            keys: Business keys for matching.
            valid_from: Column name for valid_from timestamp.

        Returns:
            SCD2Result with insert/update/expire counts.
        """
        ...

    @abc.abstractmethod
    def content_hash(self, df: Any) -> str:
        """Compute a content hash of the DataFrame.

        Used for change detection — skip pipeline if hash unchanged.
        """
        ...

    @abc.abstractmethod
    def read_table(self, table: str) -> Any:
        """Read a table into a DataFrame."""
        ...

    @abc.abstractmethod
    def write_table(self, df: Any, table: str, format: str = "parquet") -> None:
        """Write a DataFrame to a table."""
        ...

    @abc.abstractmethod
    def execute_sql(self, sql: str) -> Any:
        """Execute raw SQL and return results as a DataFrame."""
        ...

    @abc.abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        ...
