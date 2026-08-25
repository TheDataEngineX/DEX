"""Data quality gates — completeness, uniqueness, schema, custom SQL checks.

Quality gates run after transforms and before loading to the target layer.
They use DuckDB SQL aggregations for speed (no row-by-row Python).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import duckdb
import structlog

logger = structlog.get_logger()

_SCHEMA_CHECK_FAILED = "quality check failed: schema"


@dataclass(frozen=True)
class ColumnSpec:
    """Expected schema for a single column.

    Attributes:
        name: Column name (case-sensitive).
        dtype: Expected DuckDB type string e.g. ``"VARCHAR"``, ``"INTEGER"``.
               ``None`` skips the type check.
        nullable: If ``False``, the column must contain no NULL values.
    """

    name: str
    dtype: str | None = None
    nullable: bool = True


@dataclass
class QualityResult:
    """Result of a quality gate check."""

    passed: bool
    completeness_score: float = 1.0
    uniqueness_score: float = 1.0
    custom_passed: bool = True
    schema_violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _check_completeness(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    threshold: float,
    total_rows: int,
    result: QualityResult,
) -> None:
    """Check non-null ratio across all columns."""
    cols = [row[0] for row in conn.execute(f"DESCRIBE {table}").fetchall()]
    total_nulls = 0
    for c in cols:
        qc = f'"{c}"'
        row = conn.execute(f"SELECT count(*) FROM {table} WHERE {qc} IS NULL").fetchone()
        total_nulls += int(row[0]) if row else 0
    total_cells = total_rows * len(cols)
    score = (total_cells - total_nulls) / total_cells if total_cells > 0 else 1.0
    result.completeness_score = score
    if score < threshold:
        result.passed = False
        logger.warning(
            "quality check failed: completeness",
            table=table,
            score=round(score, 4),
            threshold=threshold,
        )


def _check_uniqueness(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    total_rows: int,
    result: QualityResult,
) -> None:
    """Check that specified columns have no duplicates.

    Counts distinct *hashes* of the key columns rather than distinct composite
    tuples — for wide/text-heavy tables, DISTINCT over the raw columns forces
    DuckDB to materialize a hash table keyed on the full row content, which
    can exceed available memory. Hashing first collapses each row to a fixed
    8-byte key before the distinct count, at the cost of a statistically
    negligible collision risk.
    """
    key_cols = ", ".join(f'"{c}"' for c in columns)
    distinct_row = conn.execute(f"SELECT count(DISTINCT hash({key_cols})) FROM {table}").fetchone()
    distinct_count: int = int(distinct_row[0]) if distinct_row else 0
    score = distinct_count / total_rows if total_rows > 0 else 1.0
    result.uniqueness_score = score
    if distinct_count < total_rows:
        result.passed = False
        logger.warning(
            "quality check failed: uniqueness",
            table=table,
            columns=columns,
            distinct=distinct_count,
            total=total_rows,
        )


def _check_schema(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[ColumnSpec],
    result: QualityResult,
) -> None:
    """Validate column existence, type, and nullability against a ColumnSpec list."""
    # DESCRIBE returns (column_name, column_type, null, key, default, extra)
    actual: dict[str, str] = {
        row[0]: row[1].upper() for row in conn.execute(f"DESCRIBE {table}").fetchall()
    }
    for spec in columns:
        if spec.name not in actual:
            msg = f"column '{spec.name}' missing"
            result.schema_violations.append(msg)
            result.passed = False
            logger.warning(_SCHEMA_CHECK_FAILED, table=table, violation=msg)
            continue

        if spec.dtype is not None and actual[spec.name] != spec.dtype.upper():
            actual_type = actual[spec.name]
            expected_type = spec.dtype.upper()
            msg = f"column '{spec.name}' type {actual_type!r} != expected {expected_type!r}"
            result.schema_violations.append(msg)
            result.passed = False
            logger.warning(_SCHEMA_CHECK_FAILED, table=table, violation=msg)

        if not spec.nullable:
            null_row = conn.execute(
                f'SELECT count(*) FROM {table} WHERE "{spec.name}" IS NULL'
            ).fetchone()
            null_count = int(null_row[0]) if null_row else 0
            if null_count > 0:
                msg = f"column '{spec.name}' has {null_count} NULL value(s) but nullable=False"
                result.schema_violations.append(msg)
                result.passed = False
                logger.warning(_SCHEMA_CHECK_FAILED, table=table, violation=msg)


def _check_row_count(
    total_rows: int,
    min_rows: int,
    table: str,
    result: QualityResult,
) -> None:
    """Fail if the table has fewer rows than the minimum."""
    if total_rows < min_rows:
        result.passed = False
        result.details["row_count"] = total_rows
        result.details["row_count_min"] = min_rows
        logger.warning(
            "quality check failed: row_count_min",
            table=table,
            rows=total_rows,
            minimum=min_rows,
        )


def check_quality(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    *,
    completeness: float | None = None,
    uniqueness: list[str] | None = None,
    row_count_min: int | None = None,
    schema: list[ColumnSpec] | None = None,
    custom_sql: str | None = None,
) -> QualityResult:
    """Run quality checks against a DuckDB table.

    Args:
        conn: Active DuckDB connection.
        table: Table name to check.
        completeness: Minimum fraction of non-null values (0.0-1.0).
        uniqueness: Columns that must be unique (no duplicates).
        schema: Expected column specs (existence, type, nullability).
               Checked even on empty tables.
        custom_sql: SQL that must return count > 0 to pass.

    Returns:
        QualityResult with pass/fail, scores, and any schema violations.
    """
    result = QualityResult(passed=True)

    # Schema checks run before the empty-table short-circuit — column
    # existence and type are independent of row count.
    if schema is not None:
        _check_schema(conn, table, schema, result)

    count_row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    total_rows: int = int(count_row[0]) if count_row else 0

    if total_rows == 0:
        logger.info("quality check: empty table — passing vacuously", table=table)
        return result

    if row_count_min is not None:
        _check_row_count(total_rows, row_count_min, table, result)

    if completeness is not None:
        _check_completeness(conn, table, completeness, total_rows, result)

    if uniqueness is not None:
        _check_uniqueness(conn, table, uniqueness, total_rows, result)

    if custom_sql is not None:
        _check_custom_sql(conn, table, custom_sql, result)

    if result.passed:
        logger.info("quality check passed", table=table)

    return result


def _check_custom_sql(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    custom_sql: str,
    result: QualityResult,
) -> None:
    custom_row = conn.execute(custom_sql).fetchone()
    result.custom_passed = int(custom_row[0]) > 0 if custom_row else False
    if not result.custom_passed:
        result.passed = False
        logger.warning("quality check failed: custom SQL", table=table)
