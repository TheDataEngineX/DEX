"""Spark adapter for DuckDB-backed quality gates.

Converts a PySpark DataFrame to a DuckDB in-memory relation via Arrow,
then delegates to :func:`check_quality`. PySpark is optional — calling
:func:`check_quality_spark` without it installed raises ``ImportError``.

Usage::

    from dataenginex.domains.analytics.quality.spark import check_quality_spark

    result = check_quality_spark(
        df,
        conn=conn,
        completeness=0.95,
        schema=[ColumnSpec("id", dtype="VARCHAR", nullable=False)],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb

from dataenginex.domains.analytics.quality.gates import ColumnSpec, QualityResult, check_quality

try:
    import pyspark  # noqa: F401

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False

if TYPE_CHECKING:
    import pyspark.sql

__all__ = ["check_quality_spark"]

_DEFAULT_TABLE = "_spark_tmp"


def check_quality_spark(
    df: pyspark.sql.DataFrame,
    *,
    conn: duckdb.DuckDBPyConnection,
    table_name: str = _DEFAULT_TABLE,
    completeness: float | None = None,
    uniqueness: list[str] | None = None,
    schema: list[ColumnSpec] | None = None,
    custom_sql: str | None = None,
) -> QualityResult:
    """Run quality checks against a PySpark DataFrame via DuckDB.

    Converts *df* to an Arrow table, registers it as a DuckDB view named
    *table_name*, delegates to :func:`check_quality`, then drops the view.

    Args:
        df: PySpark DataFrame to check.
        conn: Active DuckDB connection (caller-owned).
        table_name: Temporary view name registered in *conn*.
        completeness: Minimum non-null fraction (0.0–1.0).
        uniqueness: Columns that must be globally unique.
        schema: Expected column specs (existence, type, nullability).
        custom_sql: SQL expression that must return count > 0 to pass.

    Returns:
        :class:`QualityResult` with pass/fail and scores.

    Raises:
        ImportError: If PySpark is not installed.
    """
    if not _PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for check_quality_spark. Install it with: uv sync --group data"
        )

    arrow_table: Any = df.toArrow()
    conn.register(table_name, arrow_table)
    try:
        return check_quality(
            conn,
            table_name,
            completeness=completeness,
            uniqueness=uniqueness,
            schema=schema,
            custom_sql=custom_sql,
        )
    finally:
        conn.execute(f"DROP VIEW IF EXISTS {table_name}")
