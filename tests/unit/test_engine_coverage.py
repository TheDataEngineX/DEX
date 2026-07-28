"""Tests for engine.py uncovered methods."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from textwrap import dedent

import duckdb
import pytest

from dataenginex.engine import DexEngine


@pytest.fixture()
def dex_yaml(tmp_path: Path) -> Path:
    cfg = dedent("""\
        project:
          name: test-project
          version: "0.1.0"
        data:
          sources: {}
          pipelines: {}
        ai:
          agents: {}
    """)
    p = tmp_path / "dex.yaml"
    p.write_text(cfg)
    return p


@pytest.fixture()
def engine(dex_yaml: Path) -> Generator[DexEngine]:
    eng = DexEngine(dex_yaml)
    yield eng
    eng.close()


class TestSaveConfig:
    def test_save_config(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="s1")
        engine._save_config()
        assert engine.config_path.exists()
        content = engine.config_path.read_text()
        assert "p1" in content


class TestWarehouseLayers:
    def test_with_tables(self, engine: DexEngine) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "bronze"
        lakehouse.mkdir(parents=True)
        (lakehouse / "t1.parquet").write_bytes(b"fake")
        (lakehouse / "t2.parquet").write_bytes(b"fake")
        layers = engine.warehouse_layers()
        assert len(layers) == 3
        bronze = next(ly for ly in layers if ly["name"] == "bronze")
        assert bronze["table_count"] == 2


class TestWarehouseTableSchema:
    def test_missing(self, engine: DexEngine) -> None:
        result = engine.warehouse_table_schema("nonexistent", "silver")
        assert result == []


class TestWarehouseTableStats:
    def test_with_parquet(self, engine: DexEngine) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "gold"
        lakehouse.mkdir(parents=True)
        pq = lakehouse / "stats_test.parquet"
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE t AS SELECT 1 AS id, 'x' AS val")
        conn.execute(f"COPY t TO '{pq}' (FORMAT PARQUET)")
        conn.close()
        stats = engine.warehouse_table_stats("stats_test", "gold")
        assert stats["size_bytes"] > 0
        assert stats["column_count"] == 2
        assert stats["format"] == "parquet"


class TestQualityCheckTable:
    def test_with_parquet(self, engine: DexEngine) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "silver"
        lakehouse.mkdir(parents=True)
        pq = lakehouse / "quality_test.parquet"
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE t AS SELECT 1 AS id, 'a' AS name")
        conn.execute(f"COPY t TO '{pq}' (FORMAT PARQUET)")
        conn.close()
        result = engine.quality_check_table("silver.quality_test")
        assert result is not None
        assert "score" in result
        assert "passed" in result


class TestSourceReadFn:
    def test_parquet_type(self, engine: DexEngine) -> None:
        engine.config.data.sources["psrc"] = type(
            "S",
            (),
            {
                "type": type("T", (), {"value": "parquet"})(),
                "path": "x.parquet",
                "url": None,
                "connection": {},
            },
        )()
        assert engine._source_read_fn("psrc") == "read_parquet"

    def test_json_type(self, engine: DexEngine) -> None:
        engine.config.data.sources["jsrc"] = type(
            "S",
            (),
            {
                "type": type("T", (), {"value": "json"})(),
                "path": "x.json",
                "url": None,
                "connection": {},
            },
        )()
        assert engine._source_read_fn("jsrc") == "read_json_auto"

    def test_jsonl_type(self, engine: DexEngine) -> None:
        engine.config.data.sources["jlsrc"] = type(
            "S",
            (),
            {
                "type": type("T", (), {"value": "jsonl"})(),
                "path": "x.jsonl",
                "url": None,
                "connection": {},
            },
        )()
        assert engine._source_read_fn("jlsrc") == "read_ndjson_auto"

    def test_unknown_type(self, engine: DexEngine) -> None:
        engine.config.data.sources["usrc"] = type(
            "S",
            (),
            {
                "type": type("T", (), {"value": "unknown"})(),
                "path": "x",
                "url": None,
                "connection": {},
            },
        )()
        assert engine._source_read_fn("usrc") is None


class TestSourceQueryPath:
    def test_glob_pattern(self, engine: DexEngine, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.csv").write_text("id\n1\n")
        (data_dir / "b.csv").write_text("id\n2\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["gsrc"] = SourceConfig(
            type="csv", path=str(data_dir / "*.csv"), url=None, connection={}
        )
        result = engine._source_query_path("gsrc")
        assert result is not None
        assert "*.csv" in result

    def test_directory_source(self, engine: DexEngine, tmp_path: Path) -> None:
        data_dir = tmp_path / "dir_data"
        data_dir.mkdir()
        (data_dir / "test.csv").write_text("id\n1\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["dsrc"] = SourceConfig(
            type="csv", path=str(data_dir), url=None, connection={}
        )
        result = engine._source_query_path("dsrc")
        assert result is not None
        assert "*.csv" in result

    def test_absolute_path(self, engine: DexEngine) -> None:
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["asrc"] = SourceConfig(
            type="csv", path="/tmp/test.csv", url=None, connection={}
        )
        result = engine._source_query_path("asrc")
        assert result is not None
        assert result.endswith(".csv")


class TestSourcePath:
    def test_with_url(self, engine: DexEngine) -> None:
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["usrc"] = SourceConfig(
            type="csv", path=None, url="http://example.com/data.csv", connection={}
        )
        result = engine._source_path("usrc")
        assert result is None


class TestSourceRowCount:
    def test_with_csv(self, engine: DexEngine, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,a\n2,b\n3,c\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["csrc"] = SourceConfig(
            type="csv", path=str(csv_path), url=None, connection={}
        )
        count = engine.source_row_count("csrc")
        assert count == 3


class TestSourceSample:
    def test_with_csv(self, engine: DexEngine, tmp_path: Path) -> None:
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text("id,name\n1,a\n2,b\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["ssrc"] = SourceConfig(
            type="csv", path=str(csv_path), url=None, connection={}
        )
        rows = engine.source_sample("ssrc", limit=1)
        assert rows is not None
        assert len(rows) == 1


class TestSourceSchema:
    def test_with_csv(self, engine: DexEngine, tmp_path: Path) -> None:
        csv_path = tmp_path / "schema.csv"
        csv_path.write_text("id,name\n1,a\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["schsrc"] = SourceConfig(
            type="csv", path=str(csv_path), url=None, connection={}
        )
        schema = engine.source_schema("schsrc")
        assert schema is not None
        names = [c["column_name"] for c in schema]
        assert "id" in names


class TestSourceStats:
    def test_with_csv(self, engine: DexEngine, tmp_path: Path) -> None:
        csv_path = tmp_path / "stats.csv"
        csv_path.write_text("id,name\n1,a\n2,b\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["stsrc"] = SourceConfig(
            type="csv", path=str(csv_path), url=None, connection={}
        )
        stats = engine.source_stats("stsrc")
        assert stats is not None
        assert stats["row_count"] == 2
        assert stats["size_bytes"] > 0

    def test_with_directory(self, engine: DexEngine, tmp_path: Path) -> None:
        data_dir = tmp_path / "dirstats"
        data_dir.mkdir()
        (data_dir / "a.csv").write_text("id\n1\n")
        from dataenginex.config.schema import SourceConfig

        engine.config.data.sources["ds"] = SourceConfig(
            type="csv", path=str(data_dir), url=None, connection={}
        )
        stats = engine.source_stats("ds")
        assert stats is not None
        assert stats["row_count"] is not None
