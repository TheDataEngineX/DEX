"""Tests for data quality gates."""

from __future__ import annotations

import duckdb
import pytest

from dataenginex.domains.analytics.quality.gates import ColumnSpec, check_quality


@pytest.fixture()
def duckdb_conn():
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


class TestQualityGates:
    def test_completeness_pass(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id, 'a' AS name UNION ALL SELECT 2, 'b'")
        result = check_quality(duckdb_conn, "t", completeness=0.9)
        assert result.passed is True

    def test_completeness_fail_with_nulls(self, duckdb_conn) -> None:
        duckdb_conn.execute(
            "CREATE TABLE t AS SELECT 1 AS id, 'a' AS name UNION ALL SELECT NULL, NULL"
        )
        result = check_quality(duckdb_conn, "t", completeness=0.9)
        assert result.passed is False
        assert result.completeness_score < 0.9

    def test_uniqueness_pass(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id UNION ALL SELECT 2")
        result = check_quality(duckdb_conn, "t", uniqueness=["id"])
        assert result.passed is True

    def test_uniqueness_fail_with_dupes(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id UNION ALL SELECT 1")
        result = check_quality(duckdb_conn, "t", uniqueness=["id"])
        assert result.passed is False

    def test_custom_sql_check(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 10 AS value")
        sql = "SELECT count(*) FROM t WHERE value > 0"
        result = check_quality(duckdb_conn, "t", custom_sql=sql)
        assert result.passed is True

    def test_empty_table_passes_vacuously(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
        result = check_quality(duckdb_conn, "t", completeness=0.5)
        assert result.passed is True


class TestSchemaValidation:
    def test_schema_pass_matching_types(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id, 'alice' AS name")
        result = check_quality(
            duckdb_conn,
            "t",
            schema=[ColumnSpec("id", dtype="INTEGER"), ColumnSpec("name", dtype="VARCHAR")],
        )
        assert result.passed is True
        assert result.schema_violations == []

    def test_schema_fail_missing_column(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id")
        result = check_quality(duckdb_conn, "t", schema=[ColumnSpec("missing_col")])
        assert result.passed is False
        assert any("missing" in v for v in result.schema_violations)

    def test_schema_fail_wrong_type(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 'abc' AS id")
        result = check_quality(duckdb_conn, "t", schema=[ColumnSpec("id", dtype="INTEGER")])
        assert result.passed is False
        assert any("id" in v for v in result.schema_violations)

    def test_schema_fail_nullable_false_with_nulls(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id UNION ALL SELECT NULL")
        result = check_quality(duckdb_conn, "t", schema=[ColumnSpec("id", nullable=False)])
        assert result.passed is False
        assert any("nullable=False" in v for v in result.schema_violations)

    def test_schema_pass_nullable_false_no_nulls(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id UNION ALL SELECT 2")
        result = check_quality(duckdb_conn, "t", schema=[ColumnSpec("id", nullable=False)])
        assert result.passed is True

    def test_schema_checked_on_empty_table(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t (id INTEGER)")
        result = check_quality(duckdb_conn, "t", schema=[ColumnSpec("missing_col")])
        assert result.passed is False
        assert any("missing" in v for v in result.schema_violations)

    def test_schema_dtype_none_skips_type_check(self, duckdb_conn) -> None:
        duckdb_conn.execute("CREATE TABLE t AS SELECT 1 AS id")
        result = check_quality(duckdb_conn, "t", schema=[ColumnSpec("id")])
        assert result.passed is True
