"""Tests for DexEngine — the primary library entry point."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from textwrap import dedent

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
            sample_csv:
              type: csv
              path: data/sample.csv
          pipelines:
            ingest:
              source: sample_csv
              destination: raw_sample
              steps: []
        ml:
          tracking:
            backend: local
          serving:
            engine: builtin
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


class TestDexEngineInit:
    def test_init_creates_dex_dir(self, dex_yaml: Path) -> None:
        DexEngine(dex_yaml)
        assert (dex_yaml.parent / ".dex").is_dir()

    def test_init_creates_store(self, dex_yaml: Path) -> None:
        DexEngine(dex_yaml)
        assert (dex_yaml.parent / ".dex" / "store.duckdb").exists()

    def test_project_dir(self, engine: DexEngine, dex_yaml: Path) -> None:
        assert engine.project_dir == dex_yaml.parent

    def test_config_loaded(self, engine: DexEngine) -> None:
        assert engine.config.project.name == "test-project"

    def test_store_is_accessible(self, engine: DexEngine) -> None:
        from dataenginex.store import DexStore

        assert isinstance(engine.store, DexStore)

    def test_catalog_is_accessible(self, engine: DexEngine) -> None:
        from dataenginex.domains.data.catalog import DataCatalog

        assert isinstance(engine.catalog, DataCatalog)

    def test_close(self, engine: DexEngine) -> None:
        engine.close()


class TestSourceCRUD:
    def test_add_source(self, engine: DexEngine) -> None:
        engine.add_source("new_src", "csv", path="data/new.csv")
        assert "new_src" in engine.config.data.sources

    def test_delete_source(self, engine: DexEngine) -> None:
        engine.add_source("to_delete", "csv", path="data/x.csv")
        engine.delete_source("to_delete")
        assert "to_delete" not in engine.config.data.sources

    def test_delete_nonexistent_source_is_safe(self, engine: DexEngine) -> None:
        engine.delete_source("ghost")


class TestAgentCRUD:
    def test_add_agent(self, engine: DexEngine) -> None:
        engine.add_agent("my_agent", system_prompt="You are a helpful assistant.")
        assert "my_agent" in engine.config.ai.agents

    def test_delete_agent(self, engine: DexEngine) -> None:
        engine.add_agent("temp_agent")
        engine.delete_agent("temp_agent")
        assert "temp_agent" not in engine.config.ai.agents

    def test_delete_nonexistent_agent_is_safe(self, engine: DexEngine) -> None:
        engine.delete_agent("ghost")


class TestWarehouseTables:
    def test_warehouse_tables_empty(self, engine: DexEngine) -> None:
        tables = engine.warehouse_tables("bronze")
        assert isinstance(tables, list)

    def test_warehouse_table_stats_missing(self, engine: DexEngine) -> None:
        result = engine.warehouse_table_stats("nonexistent", "bronze")
        assert isinstance(result, dict)

    def test_warehouse_table_lineage(self, engine: DexEngine) -> None:
        result = engine.warehouse_table_lineage("tbl", "bronze")
        assert "upstream" in result
        assert "downstream" in result


class TestSourceInspection:
    def test_source_row_count_missing_file(self, engine: DexEngine) -> None:
        result = engine.source_row_count("sample_csv")
        assert result is None or isinstance(result, int)

    def test_source_schema_missing_file(self, engine: DexEngine) -> None:
        result = engine.source_schema("sample_csv")
        assert result is None or isinstance(result, list)

    def test_source_stats_missing_source(self, engine: DexEngine) -> None:
        result = engine.source_stats("nonexistent_source")
        assert result is None

    def test_source_sample_missing_source(self, engine: DexEngine) -> None:
        result = engine.source_sample("nonexistent_source")
        assert result is None


class TestPipelineExecution:
    def test_run_missing_pipeline(self, engine: DexEngine) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            engine.run_pipeline("nonexistent")

    def test_run_pipeline_missing_source_file(self, engine: DexEngine) -> None:
        result = engine.run_pipeline("ingest")
        assert isinstance(result.success, bool)
