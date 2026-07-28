"""Tests for engine.py source and warehouse methods."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import MagicMock

import pytest

from dataenginex.engine import DexEngine


@pytest.fixture()
def dex_yaml(tmp_path: Path) -> Path:
    cfg = dedent("""\
        project:
          name: test-project
          version: "0.1.0"
        data:
          sources:
            csv_source:
              type: csv
              path: data/test.csv
          pipelines: {}
        ai:
          agents: {}
    """)
    p = tmp_path / "dex.yaml"
    p.write_text(cfg)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "test.csv"
    csv_file.write_text("id,name,value\n1,alice,10\n2,bob,20\n")
    return p


@pytest.fixture()
def engine(dex_yaml: Path) -> Generator[DexEngine]:
    eng = DexEngine(dex_yaml)
    eng._save_config = MagicMock()  # type: ignore[method-assign]
    yield eng
    eng.close()


class TestSourceReadFn:
    def test_csv(self, engine: DexEngine) -> None:
        assert engine._source_read_fn("csv_source") == "read_csv_auto"

    def test_missing(self, engine: DexEngine) -> None:
        assert engine._source_read_fn("nonexistent") is None


class TestSourceQueryPath:
    def test_csv(self, engine: DexEngine) -> None:
        result = engine._source_query_path("csv_source")
        assert result is not None
        assert result.endswith(".csv")

    def test_missing(self, engine: DexEngine) -> None:
        assert engine._source_query_path("nonexistent") is None


class TestSourcePath:
    def test_csv(self, engine: DexEngine) -> None:
        result = engine._source_path("csv_source")
        assert result is not None
        src, path = result
        assert path.exists()

    def test_missing(self, engine: DexEngine) -> None:
        assert engine._source_path("nonexistent") is None


class TestSourceRowCount:
    def test_csv(self, engine: DexEngine) -> None:
        count = engine.source_row_count("csv_source")
        assert count == 2

    def test_missing(self, engine: DexEngine) -> None:
        assert engine.source_row_count("nonexistent") is None


class TestSourceSchema:
    def test_csv(self, engine: DexEngine) -> None:
        schema = engine.source_schema("csv_source")
        assert schema is not None
        names = [c["column_name"] for c in schema]
        assert "id" in names
        assert "name" in names

    def test_missing(self, engine: DexEngine) -> None:
        assert engine.source_schema("nonexistent") is None


class TestSourceSample:
    def test_csv(self, engine: DexEngine) -> None:
        rows = engine.source_sample("csv_source", limit=1)
        assert rows is not None
        assert len(rows) == 1
        assert rows[0]["id"] == 1

    def test_missing(self, engine: DexEngine) -> None:
        assert engine.source_sample("nonexistent") is None


class TestSourceStats:
    def test_csv(self, engine: DexEngine) -> None:
        stats = engine.source_stats("csv_source")
        assert stats is not None
        assert stats["row_count"] == 2
        assert stats["size_bytes"] > 0
        assert stats["column_count"] is not None

    def test_missing(self, engine: DexEngine) -> None:
        assert engine.source_stats("nonexistent") is None


class TestWarehouseTableLocation:
    def test_parquet(self, engine: DexEngine, tmp_path: Any) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "silver"
        lakehouse.mkdir(parents=True)
        pq = lakehouse / "test.parquet"
        pq.write_bytes(b"fake")
        path, fmt = engine._lakehouse_table_location("test", "silver")
        assert fmt == "parquet"
        assert path.exists()

    def test_delta(self, engine: DexEngine, tmp_path: Any) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "gold"
        lakehouse.mkdir(parents=True)
        delta_dir = lakehouse / "test_delta"
        delta_dir.mkdir()
        (delta_dir / "_delta_log").mkdir()
        path, fmt = engine._lakehouse_table_location("test_delta", "gold")
        assert fmt == "delta"


class TestWarehouseLayers:
    def test_empty(self, engine: DexEngine) -> None:
        layers = engine.warehouse_layers()
        assert len(layers) == 3
        assert all(layer["table_count"] == 0 for layer in layers)


class TestWarehouseTableSchema:
    def test_missing_table(self, engine: DexEngine) -> None:
        assert engine.warehouse_table_schema("nonexistent", "silver") == []


class TestWarehouseTableStats:
    def test_missing_table(self, engine: DexEngine) -> None:
        assert engine.warehouse_table_stats("nonexistent", "silver") == {}


class TestWarehouseTableLineage:
    def test_empty(self, engine: DexEngine) -> None:
        result = engine.warehouse_table_lineage("any", "silver")
        assert result == {"upstream": [], "downstream": []}
