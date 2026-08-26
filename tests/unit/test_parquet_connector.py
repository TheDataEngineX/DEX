"""Tests for dataenginex.providers.connectors.parquet — ParquetConnector."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.providers.connectors.parquet import ParquetConnector


@pytest.fixture()
def parquet_file(tmp_path: Path) -> Path:
    """Write a tiny parquet file and return its path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pa.table({"id": [1, 2, 3], "name": ["alice", "bob", "carol"]})
    p = tmp_path / "test.parquet"
    pq.write_table(tbl, p)
    return p


@pytest.fixture()
def parquet_dir(tmp_path: Path, parquet_file: Path) -> Path:
    """Return a directory containing one parquet file."""
    import shutil

    d = tmp_path / "parquet_dir"
    d.mkdir()
    shutil.copy(parquet_file, d / "data.parquet")
    return d


class TestParquetConnectorConnect:
    def test_connect_creates_connection(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        c.connect()
        assert c._conn is not None
        c.disconnect()

    def test_disconnect_clears_connection(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        c.connect()
        c.disconnect()
        assert c._conn is None

    def test_disconnect_idempotent(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        c.connect()
        c.disconnect()
        c.disconnect()  # should not raise

    def test_health_check_connected(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        c.connect()
        assert c.health_check() is True
        c.disconnect()

    def test_health_check_disconnected(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        assert c.health_check() is False


class TestParquetConnectorRead:
    def test_read_file(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        c.connect()
        rows = c.read()
        assert len(rows) == 3
        assert rows[0]["name"] == "alice"
        c.disconnect()

    def test_read_not_connected_raises(self, parquet_file: Path) -> None:
        c = ParquetConnector(path=str(parquet_file))
        with pytest.raises(RuntimeError, match="Not connected"):
            c.read()

    def test_read_directory_by_table(self, parquet_dir: Path) -> None:
        c = ParquetConnector(path=str(parquet_dir))
        c.connect()
        rows = c.read(table="data")
        assert len(rows) == 3
        c.disconnect()

    def test_read_directory_by_table_with_ext(self, parquet_dir: Path) -> None:
        c = ParquetConnector(path=str(parquet_dir))
        c.connect()
        rows = c.read(table="data.parquet")
        assert len(rows) == 3
        c.disconnect()

    def test_read_missing_file_with_default(self, parquet_dir: Path) -> None:
        c = ParquetConnector(path=str(parquet_dir))
        c.connect()
        rows = c.read(table="nonexistent", default=[])
        assert rows == []
        c.disconnect()

    def test_read_missing_file_raises_without_default(self, parquet_dir: Path) -> None:
        c = ParquetConnector(path=str(parquet_dir))
        c.connect()
        with pytest.raises(FileNotFoundError):
            c.read(table="nonexistent")
        c.disconnect()

    def test_read_directory_no_table_raises(self, parquet_dir: Path) -> None:
        c = ParquetConnector(path=str(parquet_dir))
        c.connect()
        with pytest.raises(ValueError, match="No table"):
            c.read()
        c.disconnect()

    def test_read_with_default_file(self, parquet_dir: Path) -> None:
        c = ParquetConnector(path=str(parquet_dir), default_file="data.parquet")
        c.connect()
        rows = c.read()
        assert len(rows) == 3
        c.disconnect()


class TestParquetConnectorWrite:
    def test_write_list_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "output.parquet"
        c = ParquetConnector(path=str(out))
        c.connect()
        c.write([{"x": 1}, {"x": 2}])
        assert out.exists()
        c.disconnect()

    def test_write_not_connected_raises(self, tmp_path: Path) -> None:
        c = ParquetConnector(path=str(tmp_path / "out.parquet"))
        with pytest.raises(RuntimeError, match="Not connected"):
            c.write([{"x": 1}])

    def test_write_pyarrow_table(self, tmp_path: Path) -> None:
        import pyarrow as pa

        out = tmp_path / "arrow.parquet"
        c = ParquetConnector(path=str(out))
        c.connect()
        tbl = pa.table({"a": [1, 2]})
        c.write(tbl)
        assert out.exists()
        c.disconnect()

    def test_write_unsupported_type_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "bad.parquet"
        c = ParquetConnector(path=str(out))
        c.connect()
        with pytest.raises(TypeError):
            c.write("not a list or table")
        c.disconnect()

    def test_write_to_dir_uses_table_name(self, tmp_path: Path) -> None:
        d = tmp_path / "parquet_out"
        d.mkdir()
        c = ParquetConnector(path=str(d))
        c.connect()
        c.write([{"v": 42}], table="mydata.parquet")
        assert (d / "mydata.parquet").exists()
        c.disconnect()

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "roundtrip.parquet"
        c = ParquetConnector(path=str(out))
        c.connect()
        data = [{"id": 1, "val": "hello"}, {"id": 2, "val": "world"}]
        c.write(data)
        rows = c.read()
        assert len(rows) == 2
        assert rows[0]["val"] == "hello"
        c.disconnect()
