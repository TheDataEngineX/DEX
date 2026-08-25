"""Tests for SparkCatalogAdapter file-mode catalog scanning (§20.5)."""

from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake

from dataenginex.spark.catalog.adapter import SparkCatalogAdapter


def test_file_catalog_lists_real_delta_tables(tmp_path: Path) -> None:
    lakehouse = tmp_path / ".dex" / "lakehouse"
    table_dir = lakehouse / "gold" / "top_rated"
    table_dir.mkdir(parents=True)
    write_deltalake(str(table_dir), pa.table({"id": [1, 2], "name": ["a", "b"]}))

    adapter = SparkCatalogAdapter(
        catalog_type="file", connection_config={"lakehouse_path": str(lakehouse)}
    )
    adapter.connect()

    dbs = adapter.list_databases()
    assert any(db["name"] == "gold" for db in dbs)

    tables = adapter.list_tables("gold")
    assert any(t["name"] == "top_rated" for t in tables)


def test_file_catalog_empty_lakehouse_returns_empty(tmp_path: Path) -> None:
    adapter = SparkCatalogAdapter(
        catalog_type="file",
        connection_config={"lakehouse_path": str(tmp_path / "nonexistent")},
    )
    adapter.connect()
    assert adapter.list_databases() == []
