"""Tests for medallion architecture components."""

from __future__ import annotations

import pytest

from dataenginex.domains.data.medallion import (
    DataLayer,
    DataLineage,
    LayerConfiguration,
    LocalParquetStorage,
    MedallionArchitecture,
    StorageFormat,
)

pa = pytest.importorskip("pyarrow", reason="pyarrow required for storage tests")


# ---------------------------------------------------------------------------
# LayerConfiguration
# ---------------------------------------------------------------------------


class TestLayerConfiguration:
    def test_valid_construction(self) -> None:
        cfg = LayerConfiguration(
            layer_name="bronze",
            description="raw data",
            purpose="preserve",
            storage_format=StorageFormat.PARQUET,
            local_path="data/bronze",
            bigquery_dataset="ds_bronze",
            retention_days=30,
            schema_validation=False,
            quality_threshold=0.0,
        )
        assert cfg.layer_name == "bronze"
        assert cfg.compression == "snappy"

    def test_invalid_quality_threshold_high(self) -> None:
        with pytest.raises(ValueError, match="quality_threshold"):
            LayerConfiguration(
                layer_name="gold",
                description="d",
                purpose="p",
                storage_format=StorageFormat.PARQUET,
                local_path="data/gold",
                bigquery_dataset="ds_gold",
                retention_days=None,
                schema_validation=True,
                quality_threshold=1.5,
            )

    def test_invalid_quality_threshold_negative(self) -> None:
        with pytest.raises(ValueError, match="quality_threshold"):
            LayerConfiguration(
                layer_name="gold",
                description="d",
                purpose="p",
                storage_format=StorageFormat.PARQUET,
                local_path="data/gold",
                bigquery_dataset="ds_gold",
                retention_days=None,
                schema_validation=True,
                quality_threshold=-0.1,
            )


# ---------------------------------------------------------------------------
# MedallionArchitecture
# ---------------------------------------------------------------------------


class TestMedallionArchitecture:
    def test_get_bronze_config(self) -> None:
        cfg = MedallionArchitecture.get_layer_config(DataLayer.BRONZE)
        assert cfg is not None
        assert cfg.layer_name == "bronze"
        assert cfg.quality_threshold == 0.0

    def test_get_silver_config(self) -> None:
        cfg = MedallionArchitecture.get_layer_config(DataLayer.SILVER)
        assert cfg is not None
        assert cfg.quality_threshold == 0.75

    def test_get_gold_config(self) -> None:
        cfg = MedallionArchitecture.get_layer_config(DataLayer.GOLD)
        assert cfg is not None
        assert cfg.quality_threshold == 0.90

    def test_get_all_layers_returns_three(self) -> None:
        layers = MedallionArchitecture.get_all_layers()
        assert len(layers) == 3
        assert [layer.layer_name for layer in layers] == ["bronze", "silver", "gold"]


# ---------------------------------------------------------------------------
# LocalParquetStorage
# ---------------------------------------------------------------------------


class TestLocalParquetStorage:
    def test_write_and_read_roundtrip(self, tmp_path: object) -> None:
        storage = LocalParquetStorage(str(tmp_path))
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert storage.write(data, "test/data.parquet") is True
        result = storage.read("test/data.parquet")
        assert result is not None
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_write_wrong_format_raises(self) -> None:
        storage = LocalParquetStorage("data")
        with pytest.raises(ValueError, match="PARQUET"):
            storage.write([{"a": 1}], "test/path", StorageFormat.BIGQUERY)

    def test_read_missing_returns_none(self, tmp_path: object) -> None:
        storage = LocalParquetStorage(str(tmp_path))
        result = storage.read("nonexistent/path.parquet")
        assert result is None

    def test_delete_existing_file(self, tmp_path: object) -> None:
        storage = LocalParquetStorage(str(tmp_path))
        storage.write([{"a": 1}], "todelete.parquet")
        assert storage.delete("todelete.parquet") is True
        assert storage.read("todelete.parquet") is None

    def test_delete_nonexistent_returns_false(self, tmp_path: object) -> None:
        storage = LocalParquetStorage(str(tmp_path))
        assert storage.delete("no_such_file.parquet") is False


# ---------------------------------------------------------------------------
# DataLineage
# ---------------------------------------------------------------------------


class TestDataLineage:
    def test_record_bronze(self) -> None:
        dl = DataLineage()
        lid = dl.record_bronze_ingestion("linkedin", 100, "2025-01-01")
        assert lid.startswith("bronze_")
        info = dl.get_lineage(lid)
        assert info is not None
        assert info["record_count"] == 100

    def test_record_silver(self) -> None:
        dl = DataLineage()
        b_id = dl.record_bronze_ingestion("indeed", 50, "2025-01-01")
        s_id = dl.record_silver_transformation(b_id, 45, 0.82)
        info = dl.get_lineage(s_id)
        assert info is not None
        assert info["quality_score"] == 0.82
        assert info["parent"] == b_id

    def test_record_gold(self) -> None:
        dl = DataLineage()
        b_id = dl.record_bronze_ingestion("indeed", 50, "2025-01-01")
        g_id = dl.record_gold_enrichment(b_id, 40, "all-MiniLM-L6-v2")
        info = dl.get_lineage(g_id)
        assert info is not None
        assert info["embedding_model"] == "all-MiniLM-L6-v2"

    def test_get_missing_lineage(self) -> None:
        dl = DataLineage()
        assert dl.get_lineage("nonexistent") is None
