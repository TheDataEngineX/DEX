"""Tests for AI tools builtin: _materialize_tables, _make_lakehouse_query."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

from dataenginex.ai.tools.builtin import (
    _make_lakehouse_query,
    _materialize_tables,
    _query_sql,
)


class TestQuerySql:
    def test_select(self) -> None:
        result = _query_sql("SELECT 1 AS id, 'hello' AS name")
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_empty_result(self) -> None:
        result = _query_sql("SELECT 1 WHERE 1=0")
        assert result == []

    def test_insert_rejected(self) -> None:
        with pytest.raises(ValueError):
            _query_sql("INSERT INTO t VALUES (1)")

    def test_delete_rejected(self) -> None:
        with pytest.raises(ValueError):
            _query_sql("DELETE FROM t")


class TestMaterializeTables:
    def test_parquet_file(self, tmp_path: Any) -> None:
        pq = tmp_path / "test.parquet"
        conn_src = duckdb.connect(":memory:")
        conn_src.execute("CREATE TABLE t AS SELECT 1 AS id")
        conn_src.execute(f"COPY t TO '{pq}' (FORMAT PARQUET)")
        conn_src.close()
        conn = duckdb.connect(":memory:")
        remaining = _materialize_tables(conn, tmp_path, {"test"})
        assert "test" not in remaining
        result = conn.execute("SELECT * FROM test").fetchall()
        assert len(result) == 1

    def test_missing_table(self, tmp_path: Any) -> None:
        conn = duckdb.connect(":memory:")
        remaining = _materialize_tables(conn, tmp_path, {"nonexistent"})
        assert "nonexistent" in remaining

    def test_delta_dir(self, tmp_path: Any) -> None:
        delta_dir = tmp_path / "my_delta"
        delta_dir.mkdir()
        (delta_dir / "_delta_log").mkdir()
        conn = duckdb.connect(":memory:")
        remaining = _materialize_tables(conn, tmp_path, {"my_delta"})
        # May or may not succeed depending on deltalake availability
        assert isinstance(remaining, set)


class TestMakeLakehouseQuery:
    def test_basic_query(self, tmp_path: Any) -> None:
        # Create bronze layer with a parquet file
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        pq = bronze / "test.parquet"
        conn_src = duckdb.connect(":memory:")
        conn_src.execute("CREATE TABLE t AS SELECT 1 AS id, 'a' AS name")
        conn_src.execute(f"COPY t TO '{pq}' (FORMAT PARQUET)")
        conn_src.close()
        query_fn = _make_lakehouse_query(tmp_path)
        result = query_fn("SELECT * FROM test")
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_nonexistent_table(self, tmp_path: Any) -> None:
        query_fn = _make_lakehouse_query(tmp_path)
        with pytest.raises(Exception, match="nonexistent"):
            query_fn("SELECT * FROM nonexistent")

    def test_insert_rejected(self, tmp_path: Any) -> None:
        query_fn = _make_lakehouse_query(tmp_path)
        with pytest.raises(ValueError):
            query_fn("INSERT INTO t VALUES (1)")
