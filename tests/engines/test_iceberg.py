"""Tests for IcebergAdapter (Phase 5)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from dataenginex.spark.catalog.iceberg import IcebergAdapter


class TestIcebergAdapterConnect:
    def test_connect_in_memory(self) -> None:
        adapter = IcebergAdapter(
            warehouse="/tmp/test-warehouse",
            connection_config={"type": "in-memory"},
        )
        adapter.connect()
        assert adapter._catalog is not None
        adapter.disconnect()
        assert adapter._catalog is None


class TestIcebergAdapterTableOps:
    @pytest.fixture()
    def adapter(self, tmp_path: Any) -> IcebergAdapter:
        adapter = IcebergAdapter(
            warehouse=str(tmp_path),
            connection_config={"type": "in-memory"},
        )
        adapter.connect()
        return adapter

    def test_table_not_exists(self, adapter: IcebergAdapter) -> None:
        assert adapter.table_exists("default", "nonexistent") is False

    def test_list_tables_empty(self, adapter: IcebergAdapter) -> None:
        assert adapter.list_tables("default") == []

    def test_write_and_read(self, adapter: IcebergAdapter) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        adapter.write_table("default", "test_table", table, mode="append")
        assert adapter.table_exists("default", "test_table") is True

        result = adapter.read_table("default", "test_table")
        assert len(result) == 3
        assert result.column("id").to_pylist() == [1, 2, 3]

    def test_overwrite(self, adapter: IcebergAdapter) -> None:
        table1 = pa.table({"id": [1, 2]})
        adapter.write_table("default", "test_table", table1, mode="append")

        table2 = pa.table({"id": [10, 20, 30]})
        adapter.write_table("default", "test_table", table2, mode="overwrite")

        result = adapter.read_table("default", "test_table")
        assert len(result) == 3
        assert result.column("id").to_pylist() == [10, 20, 30]

    def test_drop_table(self, adapter: IcebergAdapter) -> None:
        table = pa.table({"id": [1]})
        adapter.write_table("default", "to_drop", table, mode="append")
        assert adapter.table_exists("default", "to_drop") is True

        adapter.drop_table("default", "to_drop")
        assert adapter.table_exists("default", "to_drop") is False

    def test_list_tables(self, adapter: IcebergAdapter) -> None:
        t1 = pa.table({"id": [1]})
        t2 = pa.table({"id": [2]})
        adapter.write_table("ns", "alpha", t1, mode="append")
        adapter.write_table("ns", "beta", t2, mode="append")

        tables = adapter.list_tables("ns")
        assert "alpha" in tables
        assert "beta" in tables


class TestIcebergAdapterDisconnected:
    def test_read_raises(self) -> None:
        adapter = IcebergAdapter()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.read_table("default", "t")

    def test_write_raises(self) -> None:
        adapter = IcebergAdapter()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.write_table("default", "t", pa.table({"id": [1]}))

    def test_drop_raises(self) -> None:
        adapter = IcebergAdapter()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.drop_table("default", "t")
