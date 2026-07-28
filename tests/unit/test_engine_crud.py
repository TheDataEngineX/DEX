"""Tests for DexEngine CRUD and helper methods."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from textwrap import dedent
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
    eng._save_config = MagicMock()  # type: ignore[method-assign]
    yield eng
    eng.close()


class TestConfigDumpClean:
    def test_returns_dict(self, engine: DexEngine) -> None:
        result = engine._config_dump_clean()
        assert isinstance(result, dict)
        assert "project" in result


class TestHealth:
    def test_returns_healthy(self, engine: DexEngine) -> None:
        h = engine.health()
        assert h["status"] == "healthy"
        assert "components" in h
        assert "pipeline_runner" in h["components"]


class TestAddPipeline:
    def test_adds_pipeline(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        assert "p1" in engine.config.data.pipelines
        engine._save_config.assert_called()

    def test_add_pipeline_with_schedule(self, engine: DexEngine) -> None:
        engine.add_pipeline("p2", source="csv2", schedule="0 * * * *")
        assert engine.config.data.pipelines["p2"].schedule == "0 * * * *"


class TestDeletePipeline:
    def test_deletes_pipeline(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.delete_pipeline("p1")
        assert "p1" not in engine.config.data.pipelines

    def test_delete_nonexistent_silent(self, engine: DexEngine) -> None:
        engine.delete_pipeline("nonexistent")
        engine._save_config.assert_called()


class TestAddDeleteSource:
    def test_add_source(self, engine: DexEngine) -> None:
        engine.add_source("s1", type_="csv", path="/tmp/data.csv")
        assert "s1" in engine.config.data.sources

    def test_delete_source(self, engine: DexEngine) -> None:
        engine.add_source("s1", type_="csv", path="/tmp/data.csv")
        engine.delete_source("s1")
        assert "s1" not in engine.config.data.sources

    def test_delete_nonexistent_source(self, engine: DexEngine) -> None:
        engine.delete_source("nonexistent")
        engine._save_config.assert_called()


class TestAddDeleteAgent:
    def test_add_agent(self, engine: DexEngine) -> None:
        engine.add_agent("a1", runtime="builtin", system_prompt="hello")
        assert "a1" in engine.config.ai.agents

    def test_delete_agent(self, engine: DexEngine) -> None:
        engine.add_agent("a1")
        engine.delete_agent("a1")
        assert "a1" not in engine.config.ai.agents

    def test_delete_agent_no_ai(self, engine: DexEngine) -> None:
        # When config.ai is falsy (empty agents dict), delete_agent still works
        engine.delete_agent("a1")
        engine._save_config.assert_called()


class TestUpdatePipelineSchedule:
    def test_updates_schedule(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.update_pipeline_schedule("p1", "0 * * * *")
        assert engine.config.data.pipelines["p1"].schedule == "0 * * * *"

    def test_raises_on_missing(self, engine: DexEngine) -> None:
        with pytest.raises(KeyError, match="not found"):
            engine.update_pipeline_schedule("nonexistent", "0 * * * *")


class TestAddPipelineTransform:
    def test_adds_transform(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.add_pipeline_transform("p1", {"type": "filter", "config": {"expr": "id > 0"}})
        assert len(engine.config.data.pipelines["p1"].transforms) == 1

    def test_raises_on_missing_pipeline(self, engine: DexEngine) -> None:
        with pytest.raises(KeyError, match="not found"):
            engine.add_pipeline_transform("nonexistent", {"type": "filter", "config": {}})


class TestDeletePipelineTransform:
    def test_deletes_transform(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.add_pipeline_transform("p1", {"type": "filter", "config": {"expr": "id > 0"}})
        engine.delete_pipeline_transform("p1", 0)
        assert len(engine.config.data.pipelines["p1"].transforms) == 0

    def test_out_of_range_noop(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.delete_pipeline_transform("p1", 99)
        assert len(engine.config.data.pipelines["p1"].transforms) == 0

    def test_raises_on_missing_pipeline(self, engine: DexEngine) -> None:
        with pytest.raises(KeyError, match="not found"):
            engine.delete_pipeline_transform("nonexistent", 0)


class TestReorderPipelineTransform:
    def test_reorder(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.add_pipeline_transform("p1", {"type": "filter", "config": {"expr": "a"}})
        engine.add_pipeline_transform("p1", {"type": "cast", "config": {"columns": {}}})
        engine.reorder_pipeline_transform("p1", 0, 1)
        assert engine.config.data.pipelines["p1"].transforms[0].type == "cast"

    def test_raises_on_missing_pipeline(self, engine: DexEngine) -> None:
        with pytest.raises(KeyError, match="not found"):
            engine.reorder_pipeline_transform("nonexistent", 0, 1)

    def test_noop_when_out_of_bounds(self, engine: DexEngine) -> None:
        engine.add_pipeline("p1", source="csv1")
        engine.add_pipeline_transform("p1", {"type": "filter", "config": {"expr": "a"}})
        engine.reorder_pipeline_transform("p1", 0, 1)
        # No crash, transform stays at index 0
        assert engine.config.data.pipelines["p1"].transforms[0].type == "filter"


class TestDeleteModel:
    def test_delegates_to_store(self, engine: DexEngine) -> None:
        engine.store = MagicMock()
        engine.delete_model("mymodel")
        engine.store.delete_model.assert_called_once_with("mymodel")
