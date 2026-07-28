"""Tests for DexEngine properties and lightweight methods."""

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


class TestEngineProperties:
    def test_lineage_returns_store(self, engine: DexEngine) -> None:
        assert engine.lineage is engine.store

    def test_ai_long_memory(self, engine: DexEngine) -> None:
        assert engine.ai_long_memory is engine.ai_memory

    def test_model_registry_not_none(self, engine: DexEngine) -> None:
        assert engine.model_registry is not None

    def test_project_dir(self, engine: DexEngine, dex_yaml: Path) -> None:
        assert engine.project_dir == dex_yaml.parent


class TestPipelineStats:
    def test_returns_dict(self, engine: DexEngine) -> None:
        stats = engine.pipeline_stats()
        assert "total" in stats
        assert isinstance(stats["total"], int)

    def test_with_pipeline(self, engine: DexEngine) -> None:
        stats = engine.pipeline_stats()
        assert stats["total"] >= 1


class TestPipelineLastRun:
    def test_nonexistent_pipeline(self, engine: DexEngine) -> None:
        result = engine.pipeline_last_run("does_not_exist")
        assert result is None


class TestConfigDumpClean:
    def test_returns_dict(self, engine: DexEngine) -> None:
        result = engine._config_dump_clean()
        assert isinstance(result, dict)
        assert "project" in result


class TestHealth:
    def test_returns_healthy(self, engine: DexEngine) -> None:
        h = engine.health()
        assert h["status"] == "healthy"
        assert h["components"]["store"] is True


class TestQualityHistory:
    def test_empty_history(self, engine: DexEngine) -> None:
        result = engine.quality_history()
        assert isinstance(result, dict)


class TestDeleteModel:
    def test_delete_nonexistent(self, engine: DexEngine) -> None:
        engine.delete_model("ghost_model_xyz")
