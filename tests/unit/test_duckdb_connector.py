"""DuckDB connector tests — conformance + DuckDB-specific tests."""

from __future__ import annotations

import pytest

from dataenginex.providers.connectors.duckdb import DuckDBConnector
from tests.conformance.test_connector import ConnectorConformanceTests


class TestDuckDBConnector(ConnectorConformanceTests):
    @pytest.fixture()
    def connector(self, tmp_path):  # type: ignore[override]
        return DuckDBConnector(database=str(tmp_path / "test.duckdb"))

    def test_in_memory_mode(self) -> None:
        conn = DuckDBConnector(database=":memory:")
        conn.connect()
        assert conn.health_check() is True
        conn.disconnect()

    def test_execute_raw_sql(self, tmp_path) -> None:
        conn = DuckDBConnector(database=str(tmp_path / "test.duckdb"))
        conn.connect()
        conn.write([{"x": 1}, {"x": 2}], table="nums")
        result = conn.execute("SELECT sum(x) as total FROM nums")
        assert result[0]["total"] == 3
        conn.disconnect()

    def test_read_not_connected_raises(self) -> None:
        conn = DuckDBConnector()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.read(table="anything")

    def test_write_empty_list_noop(self, tmp_path) -> None:
        conn = DuckDBConnector(database=str(tmp_path / "test.duckdb"))
        conn.connect()
        conn.write([], table="empty")
        conn.disconnect()

    def test_write_overwrites_existing_table(self, tmp_path) -> None:
        conn = DuckDBConnector(database=str(tmp_path / "test.duckdb"))
        conn.connect()
        conn.write([{"x": 1}], table="t")
        conn.write([{"x": 2}, {"x": 3}], table="t")
        rows = conn.read(table="t")
        assert len(rows) == 2, "second write must overwrite, not be silently dropped"
        assert {r["x"] for r in rows} == {2, 3}
        conn.disconnect()

    def test_write_pyarrow_table(self, tmp_path) -> None:
        import pyarrow as pa

        conn = DuckDBConnector(database=str(tmp_path / "test.duckdb"))
        conn.connect()
        tbl = pa.Table.from_pylist([{"y": 10}, {"y": 20}])
        conn.write(tbl, table="arrow_t")
        rows = conn.read(table="arrow_t")
        assert [r["y"] for r in rows] == [10, 20]
        conn.disconnect()

    def test_write_unsupported_type_raises(self, tmp_path) -> None:
        conn = DuckDBConnector(database=str(tmp_path / "test.duckdb"))
        conn.connect()
        with pytest.raises(TypeError, match="Unsupported data type"):
            conn.write("not a list", table="t")
        conn.disconnect()
