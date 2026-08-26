"""Tests for PortableSQLAdapter (Phase 4)."""

from __future__ import annotations

from dataenginex.spark.sql.portable_sql import Dialect, PortableSQLAdapter


class TestPortableSQLPassthrough:
    def test_portable_returns_unchanged(self) -> None:
        adapter = PortableSQLAdapter(Dialect.PORTABLE)
        sql = "SELECT id, name FROM users WHERE id = 1"
        assert adapter.translate(sql) == sql


class TestDuckDBToSpark:
    def test_integer_type(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        assert "INT" in adapter.translate("SELECT CAST(x AS INTEGER)")

    def test_varchar_type(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        assert "STRING" in adapter.translate("SELECT CAST(x AS VARCHAR)")

    def test_current_date(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT CURRENT_DATE")
        assert "current_date()" in result

    def test_date_add(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT DATE_ADD(dt, INTERVAL 7 DAY)")
        assert "date_add(" in result

    def test_string_split(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT STRING_SPLIT(name, ',')")
        assert "split(" in result

    def test_list_contains(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT LIST_CONTAINS(arr, 'x')")
        assert "array_contains(" in result

    def test_list_len(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT LIST_LEN(arr)")
        assert "size(" in result

    def test_json_extract(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT JSON_EXTRACT(j, '$.key')")
        assert "get_json_object(" in result


class TestSparkToDuckDB:
    def test_int_type(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        assert "INTEGER" in adapter.translate("SELECT CAST(x AS INT)")

    def test_string_type(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        assert "VARCHAR" in adapter.translate("SELECT CAST(x AS STRING)")

    def test_current_date(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT current_date()")
        assert "CURRENT_DATE" in result

    def test_date_add(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT date_add(dt, 7)")
        assert "DATE_ADD(" in result

    def test_split(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT split(name, ',')")
        assert "STRING_SPLIT(" in result

    def test_array_contains(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT array_contains(arr, 'x')")
        assert "LIST_CONTAINS(" in result
