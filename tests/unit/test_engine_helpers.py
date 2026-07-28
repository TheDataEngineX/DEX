"""Tests for engine.py quality check, config dump, close, and remaining methods."""

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


class TestClose:
    def test_close(self, engine: DexEngine) -> None:
        engine.close()


class TestDuckDBRo:
    def test_context_manager(self, engine: DexEngine) -> None:
        with engine._duckdb_ro() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1


class TestConfigDumpClean:
    def test_returns_dict(self, engine: DexEngine) -> None:
        result = engine._config_dump_clean()
        assert isinstance(result, dict)
        assert "project" in result


class TestQualityCheckTable:
    def test_missing_table(self, engine: DexEngine) -> None:
        result = engine.quality_check_table("nonexistent.table")
        assert result is None

    def test_bad_format(self, engine: DexEngine) -> None:
        result = engine.quality_check_table("badformat")
        assert result is None


class TestQualityCheckAllTables:
    def test_empty(self, engine: DexEngine) -> None:
        result = engine.quality_check_all_tables()
        assert isinstance(result, dict)


class TestQualityHistory:
    def test_returns_dict(self, engine: DexEngine) -> None:
        result = engine.quality_history()
        assert isinstance(result, dict)
