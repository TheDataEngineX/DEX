"""Tests for dataenginex.lakehouse — storage, catalog, partitioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.domains.data.catalog import CatalogEntry, DataCatalog
from dataenginex.providers.object_store.partitioning import DatePartitioner, HashPartitioner
from dataenginex.providers.object_store.storage import JsonStorage, ParquetStorage

# ============================================================================
# JsonStorage
# ============================================================================


class TestJsonStorage:
    def test_write_and_read(self, tmp_path: Path) -> None:
        store = JsonStorage(str(tmp_path))
        records = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
        assert store.write(records, "test/batch1") is True

        data = store.read("test/batch1")
        assert data is not None
        assert len(data) == 2
        assert data[0]["id"] == 1

    def test_write_single_dict(self, tmp_path: Path) -> None:
        store = JsonStorage(str(tmp_path))
        assert store.write({"x": 1}, "single") is True
        data = store.read("single")
        assert data is not None
        assert data[0]["x"] == 1

    def test_read_missing(self, tmp_path: Path) -> None:
        store = JsonStorage(str(tmp_path))
        assert store.read("nope") is None

    def test_delete(self, tmp_path: Path) -> None:
        store = JsonStorage(str(tmp_path))
        store.write([{"a": 1}], "del")
        assert store.delete("del") is True
        assert store.read("del") is None


# ============================================================================
# ParquetStorage (JSON fallback if pyarrow not installed)
# ============================================================================


class TestParquetStorage:
    def test_write_and_read_fallback(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        records = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        assert store.write(records, "parq/test") is True
        data = store.read("parq/test")
        assert data is not None
        assert len(data) == 2

    def test_delete_fallback(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        store.write([{"x": 1}], "del")
        assert store.delete("del") is True

    def test_empty_records(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        # Empty list should not write
        result = store.write([], "empty")
        # Either writes empty file (json fallback) or returns False (parquet)
        assert isinstance(result, bool)


# ============================================================================
# DataCatalog
# ============================================================================


class TestDataCatalog:
    def test_register_and_get(self) -> None:
        cat = DataCatalog()
        entry = CatalogEntry(
            name="jobs",
            layer="bronze",
            format="json",
            location="/data/bronze/jobs",
        )
        cat.register(entry)
        got = cat.get("jobs")
        assert got is not None
        assert got.layer == "bronze"

    def test_version_increments_on_update(self) -> None:
        cat = DataCatalog()
        cat.register(CatalogEntry(name="jobs", layer="bronze", format="json", location="a"))
        cat.register(CatalogEntry(name="jobs", layer="bronze", format="json", location="b"))
        got = cat.get("jobs")
        assert got is not None
        assert got.version == 2

    def test_search_by_layer(self) -> None:
        cat = DataCatalog()
        cat.register(CatalogEntry(name="a", layer="bronze", format="json", location="x"))
        cat.register(CatalogEntry(name="b", layer="silver", format="json", location="y"))
        results = cat.search(layer="bronze")
        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_by_tags(self) -> None:
        cat = DataCatalog()
        cat.register(
            CatalogEntry(
                name="a",
                layer="bronze",
                format="json",
                location="x",
                tags=["jobs", "raw"],
            )
        )
        cat.register(
            CatalogEntry(
                name="b",
                layer="bronze",
                format="json",
                location="y",
                tags=["users"],
            )
        )
        results = cat.search(tags=["jobs"])
        assert len(results) == 1

    def test_delete(self) -> None:
        cat = DataCatalog()
        cat.register(CatalogEntry(name="a", layer="bronze", format="json", location="x"))
        assert cat.delete("a") is True
        assert cat.get("a") is None

    def test_summary(self) -> None:
        cat = DataCatalog()
        cat.register(CatalogEntry(name="a", layer="bronze", format="json", location="x"))
        cat.register(CatalogEntry(name="b", layer="silver", format="parquet", location="y"))
        s = cat.summary()
        assert s["total_datasets"] == 2
        assert s["by_layer"]["bronze"] == 1

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        cat = DataCatalog(persist_path=path)
        cat.register(CatalogEntry(name="x", layer="bronze", format="json", location="a"))
        cat2 = DataCatalog(persist_path=path)
        assert cat2.get("x") is not None


# ============================================================================
# Partitioning
# ============================================================================


class TestDatePartitioner:
    def test_day_granularity(self) -> None:
        p = DatePartitioner("ts", "day")
        record = {"ts": "2024-03-15T10:00:00"}
        key = p.partition_key(record)
        assert "year=2024" in key
        assert "month=03" in key
        assert "day=15" in key

    def test_month_granularity(self) -> None:
        p = DatePartitioner("ts", "month")
        key = p.partition_key({"ts": "2024-03-15"})
        assert "day=" not in key
        assert "month=03" in key

    def test_year_granularity(self) -> None:
        p = DatePartitioner("ts", "year")
        key = p.partition_key({"ts": "2024-03-15"})
        assert "month=" not in key
        assert "year=2024" in key

    def test_partition_path(self) -> None:
        p = DatePartitioner("ts", "day")
        path = p.partition_path({"ts": "2024-01-01"}, "bronze/jobs")
        assert path.startswith("bronze/jobs/")

    def test_invalid_granularity(self) -> None:
        with pytest.raises(ValueError):
            DatePartitioner("ts", "hour")


class TestHashPartitioner:
    def test_partition_key(self) -> None:
        p = HashPartitioner(["id"], n_buckets=16)
        key = p.partition_key({"id": "abc"})
        assert key.startswith("bucket=")

    def test_deterministic(self) -> None:
        p = HashPartitioner(["id"], n_buckets=16)
        k1 = p.partition_key({"id": "abc"})
        k2 = p.partition_key({"id": "abc"})
        assert k1 == k2

    def test_distribution(self) -> None:
        p = HashPartitioner(["id"], n_buckets=4)
        buckets = {p.partition_key({"id": str(i)}) for i in range(100)}
        # Should use multiple buckets
        assert len(buckets) > 1

    def test_no_fields_raises(self) -> None:
        with pytest.raises(ValueError):
            HashPartitioner([], n_buckets=4)
