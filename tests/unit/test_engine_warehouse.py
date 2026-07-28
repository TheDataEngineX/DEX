"""Tests for remaining engine.py warehouse_tables and vector store coverage."""

from __future__ import annotations

import shutil
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


class TestWarehouseTablesParquet:
    def test_with_parquet(self, engine: DexEngine, tmp_path: Path) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "silver"
        lakehouse.mkdir(parents=True)
        pq = lakehouse / "test.parquet"
        src = tmp_path / "src.parquet"
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE t AS SELECT 1 AS id, 'hello' AS name")
        conn.execute(f"COPY t TO '{src}' (FORMAT PARQUET)")
        conn.close()
        shutil.copy2(src, pq)
        tables = engine.warehouse_tables("silver")
        assert len(tables) == 1
        assert tables[0]["name"] == "test"
        assert tables[0]["format"] == "parquet"
        assert tables[0]["row_count"] == 1

    def test_with_delta(self, engine: DexEngine) -> None:
        lakehouse = engine._dex_dir / "lakehouse" / "gold"
        lakehouse.mkdir(parents=True)
        delta_dir = lakehouse / "my_delta"
        delta_dir.mkdir()
        (delta_dir / "_delta_log").mkdir()
        tables = engine.warehouse_tables("gold")
        assert len(tables) == 1
        assert tables[0]["name"] == "my_delta"
        assert tables[0]["format"] == "delta"

    def test_empty_layer(self, engine: DexEngine) -> None:
        tables = engine.warehouse_tables("bronze")
        assert tables == []


class TestWarehouseTableLineage:
    def test_empty(self, engine: DexEngine) -> None:
        result = engine.warehouse_table_lineage("any_table", "silver")
        assert result == {"upstream": [], "downstream": []}


class TestLakehouseScanSql:
    def test_parquet(self, engine: DexEngine, tmp_path: Path) -> None:
        pq = tmp_path / "test.parquet"
        pq.write_bytes(b"fake")
        sql = engine._lakehouse_scan_sql(pq, "parquet")
        assert "read_parquet" in sql
        assert str(pq) in sql
