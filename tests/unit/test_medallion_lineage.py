"""Tests for medallion architecture: LayerConfiguration, DataLineage."""

from __future__ import annotations

import pytest

from dataenginex.core.medallion_architecture import (
    DataLayer,
    DataLineage,
    LayerConfiguration,
    MedallionArchitecture,
    StorageFormat,
)


class TestLayerConfiguration:
    def test_valid(self) -> None:
        cfg = LayerConfiguration(
            layer_name="bronze",
            description="Raw data",
            purpose="Store raw",
            storage_format=StorageFormat.PARQUET,
            local_path="data/bronze",
            bigquery_dataset="bronze",
            retention_days=90,
            schema_validation=False,
            quality_threshold=0.0,
        )
        assert cfg.layer_name == "bronze"
        assert cfg.compression == "snappy"

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="quality_threshold"):
            LayerConfiguration(
                layer_name="bad",
                description="bad",
                purpose="bad",
                storage_format=StorageFormat.PARQUET,
                local_path="bad",
                bigquery_dataset="bad",
                retention_days=None,
                schema_validation=False,
                quality_threshold=1.5,
            )


class TestMedallionArchitecture:
    def test_get_layer_bronze(self) -> None:
        cfg = MedallionArchitecture.get_layer_config(DataLayer.BRONZE)
        assert cfg is not None
        assert cfg.layer_name == "bronze"

    def test_get_layer_silver(self) -> None:
        cfg = MedallionArchitecture.get_layer_config(DataLayer.SILVER)
        assert cfg is not None
        assert cfg.layer_name == "silver"

    def test_get_layer_gold(self) -> None:
        cfg = MedallionArchitecture.get_layer_config(DataLayer.GOLD)
        assert cfg is not None
        assert cfg.layer_name == "gold"

    def test_get_all_layers(self) -> None:
        layers = MedallionArchitecture.get_all_layers()
        assert len(layers) == 3


class TestDataLineage:
    def test_record_bronze(self) -> None:
        dl = DataLineage()
        lid = dl.record_bronze_ingestion("linkedin", 100, "2026-01-01")
        assert lid.startswith("bronze_")
        info = dl.get_lineage(lid)
        assert info is not None
        assert info["layer"] == "bronze"
        assert info["record_count"] == 100

    def test_record_silver(self) -> None:
        dl = DataLineage()
        bid = dl.record_bronze_ingestion("s", 100, "ts")
        sid = dl.record_silver_transformation(bid, 90, 0.85)
        assert sid.endswith("_silver")
        info = dl.get_lineage(sid)
        assert info is not None
        assert info["layer"] == "silver"
        assert info["parent"] == bid

    def test_record_gold(self) -> None:
        dl = DataLineage()
        bid = dl.record_bronze_ingestion("s", 100, "ts")
        sid = dl.record_silver_transformation(bid, 90, 0.85)
        gid = dl.record_gold_enrichment(sid, 85, "text-embedding-ada")
        assert gid.endswith("_gold")
        info = dl.get_lineage(gid)
        assert info is not None
        assert info["layer"] == "gold"

    def test_get_lineage_missing(self) -> None:
        dl = DataLineage()
        assert dl.get_lineage("nonexistent") is None
