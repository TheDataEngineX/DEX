"""Extended tests for core/medallion_architecture.py — storage backends and data lineage."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.domains.data.medallion import (
    DataLayer,
    DataLineage,
    LayerConfiguration,
    LocalParquetStorage,
    MedallionArchitecture,
    StorageBackend,
    StorageFormat,
)

# ── StorageFormat ─────────────────────────────────────────────────────────────


class TestStorageFormat:
    def test_values(self) -> None:
        assert StorageFormat.PARQUET == "parquet"
        assert StorageFormat.DELTA == "delta"
        assert StorageFormat.ICEBERG == "iceberg"
        assert StorageFormat.BIGQUERY == "bigquery"


# ── DataLayer ─────────────────────────────────────────────────────────────────


class TestDataLayerExtended:
    def test_all_layers(self) -> None:
        assert {DataLayer.BRONZE, DataLayer.SILVER, DataLayer.GOLD} == {"bronze", "silver", "gold"}


# ── LayerConfiguration ────────────────────────────────────────────────────────


class TestLayerConfiguration:
    def _make(self, threshold: float = 0.5) -> LayerConfiguration:
        return LayerConfiguration(
            layer_name="bronze",
            description="raw",
            purpose="preserve",
            storage_format=StorageFormat.PARQUET,
            local_path="data/bronze",
            bigquery_dataset="bronze",
            retention_days=90,
            schema_validation=False,
            quality_threshold=threshold,
        )

    def test_valid_creation(self) -> None:
        lc = self._make()
        assert lc.layer_name == "bronze"
        assert lc.compression == "snappy"

    def test_threshold_zero(self) -> None:
        lc = self._make(0.0)
        assert lc.quality_threshold == 0.0

    def test_threshold_one(self) -> None:
        lc = self._make(1.0)
        assert lc.quality_threshold == 1.0

    def test_invalid_threshold_negative(self) -> None:
        with pytest.raises(ValueError):
            self._make(-0.1)

    def test_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValueError):
            self._make(1.1)

    def test_none_retention(self) -> None:
        lc = LayerConfiguration(
            layer_name="gold",
            description="",
            purpose="",
            storage_format=StorageFormat.PARQUET,
            local_path="data/gold",
            bigquery_dataset="gold",
            retention_days=None,
            schema_validation=True,
            quality_threshold=0.9,
        )
        assert lc.retention_days is None


# ── MedallionArchitecture ─────────────────────────────────────────────────────


class TestMedallionArchitectureExtended:
    def test_get_bronze_config(self) -> None:
        c = MedallionArchitecture.get_layer_config(DataLayer.BRONZE)
        assert c is not None
        assert c.quality_threshold == 0.0

    def test_get_silver_config(self) -> None:
        c = MedallionArchitecture.get_layer_config(DataLayer.SILVER)
        assert c is not None
        assert c.quality_threshold == 0.75

    def test_get_gold_config(self) -> None:
        c = MedallionArchitecture.get_layer_config(DataLayer.GOLD)
        assert c is not None
        assert c.quality_threshold == 0.90

    def test_get_all_layers(self) -> None:
        layers = MedallionArchitecture.get_all_layers()
        assert len(layers) == 3
        names = [lyr.layer_name for lyr in layers]
        assert "bronze" in names and "silver" in names and "gold" in names


# ── StorageBackend (ABC) ──────────────────────────────────────────────────────


class TestStorageBackendABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            StorageBackend()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class MinimalBackend(StorageBackend):
            def write(self, data, path, format):
                return True

            def read(self, path, format):
                return None

            def delete(self, path):
                return True

            def list_objects(self, prefix=""):
                return []

            def exists(self, path):
                return False

        b = MinimalBackend()
        assert b.write(None, "x", StorageFormat.PARQUET) is True
        assert b.read("x", StorageFormat.PARQUET) is None
        assert b.delete("x") is True
        assert b.list_objects() == []
        assert b.exists("x") is False


# ── LocalParquetStorage ───────────────────────────────────────────────────────


class TestLocalParquetStorage:
    def test_init(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        assert s.base_path == str(tmp_path)

    def test_write_and_exists(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        data = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        result = s.write(data, "test_table", StorageFormat.PARQUET)
        assert result is True
        assert s.exists("test_table")

    def test_read_after_write(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        data = [{"x": 10}]
        s.write(data, "mytable", StorageFormat.PARQUET)
        records = s.read("mytable", StorageFormat.PARQUET)
        assert records is not None
        assert len(records) == 1

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        result = s.read("nonexistent", StorageFormat.PARQUET)
        assert result is None

    def test_list_objects(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        s.write([{"v": 1}], "table_a", StorageFormat.PARQUET)
        s.write([{"v": 2}], "table_b", StorageFormat.PARQUET)
        objects = s.list_objects()
        assert len(objects) == 2

    def test_list_objects_with_prefix(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        s.write([{"v": 1}], "abc_table", StorageFormat.PARQUET)
        s.write([{"v": 2}], "xyz_table", StorageFormat.PARQUET)
        objects = s.list_objects(prefix="abc")
        assert all("abc" in o for o in objects)

    def test_list_objects_empty(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        assert s.list_objects() == []

    def test_delete_existing(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        s.write([{"v": 1}], "del_table", StorageFormat.PARQUET)
        result = s.delete("del_table")
        assert result is True
        assert not s.exists("del_table")

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        result = s.delete("ghost")
        assert result is False

    def test_exists_false_when_missing(self, tmp_path: Path) -> None:
        s = LocalParquetStorage(base_path=str(tmp_path))
        assert s.exists("missing") is False


# ── DataLineage ───────────────────────────────────────────────────────────────


class TestDataLineage:
    def test_init_empty(self) -> None:
        dl = DataLineage()
        assert dl.lineage == {}

    def test_record_bronze_ingestion(self) -> None:
        dl = DataLineage()
        lid = dl.record_bronze_ingestion("s3://bucket/file.csv", 1000, "2026-01-01")
        assert lid.startswith("bronze_")
        event = dl.get_lineage(lid)
        assert event is not None
        assert event["layer"] == "bronze"
        assert event["record_count"] == 1000

    def test_record_silver_transformation(self) -> None:
        dl = DataLineage()
        bronze_id = dl.record_bronze_ingestion("src", 100, "2026-01-01")
        silver_id = dl.record_silver_transformation(bronze_id, 95, 0.88)
        assert silver_id == f"{bronze_id}_silver"
        event = dl.get_lineage(silver_id)
        assert event["layer"] == "silver"
        assert event["parent"] == bronze_id
        assert pytest.approx(event["quality_score"]) == 0.88

    def test_record_gold_enrichment(self) -> None:
        dl = DataLineage()
        bronze_id = dl.record_bronze_ingestion("src", 100, "2026-01-01")
        silver_id = dl.record_silver_transformation(bronze_id, 95, 0.88)
        gold_id = dl.record_gold_enrichment(silver_id, 90, "all-MiniLM")
        assert gold_id == f"{silver_id}_gold"
        event = dl.get_lineage(gold_id)
        assert event["layer"] == "gold"
        assert event["embedding_model"] == "all-MiniLM"

    def test_get_lineage_missing_returns_none(self) -> None:
        dl = DataLineage()
        assert dl.get_lineage("ghost") is None

    def test_full_medallion_chain(self) -> None:
        dl = DataLineage()
        bid = dl.record_bronze_ingestion("kafka-topic", 5000, "2026-06-01")
        sid = dl.record_silver_transformation(bid, 4800, 0.92)
        gid = dl.record_gold_enrichment(sid, 4800, "text-embedding-ada-002")
        assert dl.get_lineage(bid)["status"] == "raw"
        assert dl.get_lineage(sid)["status"] == "cleaned"
        assert dl.get_lineage(gid)["status"] == "enriched"
