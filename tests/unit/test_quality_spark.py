"""Tests for the PySpark quality-gate adapter.

Skipped automatically when PySpark or Java is unavailable
(requires ``uv sync --group data`` and a Java runtime).
"""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

from dataenginex.domains.analytics.quality.gates import ColumnSpec
from tests.conftest import requires_pyspark


@requires_pyspark
class TestCheckQualitySparkImportError:
    """Verify ImportError is raised when pyspark is not available (mocked)."""

    def test_raises_import_error_without_pyspark(self, monkeypatch: Any) -> None:
        import dataenginex.domains.analytics.quality.spark as spark_mod

        monkeypatch.setattr(spark_mod, "_PYSPARK_AVAILABLE", False)
        with pytest.raises(ImportError, match="PySpark is required"):
            spark_mod.check_quality_spark(
                None,  # type: ignore[arg-type]
                conn=duckdb.connect(":memory:"),
            )


@requires_pyspark
class TestCheckQualitySpark:
    def test_completeness_pass(self, spark: Any) -> None:
        from dataenginex.domains.analytics.quality.spark import check_quality_spark

        df = spark.createDataFrame([(1, "a"), (2, "b")], schema=["id", "name"])
        conn = duckdb.connect(":memory:")
        result = check_quality_spark(df, conn=conn, completeness=0.9)
        conn.close()
        assert result.passed is True

    def test_completeness_fail_with_nulls(self, spark: Any) -> None:
        from dataenginex.domains.analytics.quality.spark import check_quality_spark

        df = spark.createDataFrame([(1, "a"), (None, None)], schema=["id", "name"])
        conn = duckdb.connect(":memory:")
        result = check_quality_spark(df, conn=conn, completeness=0.9)
        conn.close()
        assert result.passed is False

    def test_uniqueness_pass(self, spark: Any) -> None:
        from dataenginex.domains.analytics.quality.spark import check_quality_spark

        df = spark.createDataFrame([(1,), (2,)], schema=["id"])
        conn = duckdb.connect(":memory:")
        result = check_quality_spark(df, conn=conn, uniqueness=["id"])
        conn.close()
        assert result.passed is True

    def test_uniqueness_fail_with_dupes(self, spark: Any) -> None:
        from dataenginex.domains.analytics.quality.spark import check_quality_spark

        df = spark.createDataFrame([(1,), (1,)], schema=["id"])
        conn = duckdb.connect(":memory:")
        result = check_quality_spark(df, conn=conn, uniqueness=["id"])
        conn.close()
        assert result.passed is False

    def test_schema_validation(self, spark: Any) -> None:
        from dataenginex.domains.analytics.quality.spark import check_quality_spark

        df = spark.createDataFrame([(1, "alice")], schema=["id", "name"])
        conn = duckdb.connect(":memory:")
        result = check_quality_spark(
            df,
            conn=conn,
            schema=[ColumnSpec("id"), ColumnSpec("name"), ColumnSpec("missing")],
        )
        conn.close()
        assert result.passed is False
        assert any("missing" in v for v in result.schema_violations)

    def test_view_is_dropped_after_check(self, spark: Any) -> None:
        from dataenginex.domains.analytics.quality.spark import check_quality_spark

        df = spark.createDataFrame([(1,)], schema=["id"])
        conn = duckdb.connect(":memory:")
        check_quality_spark(df, conn=conn, table_name="tmp_test_view")
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        conn.close()
        assert "tmp_test_view" not in tables
