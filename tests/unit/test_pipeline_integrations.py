"""Tests for optional pipeline sink integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb

from dataenginex.config.schema import DexConfig, PipelineConfig
from dataenginex.domains.data.pipeline.runner import PipelineRunner
from dataenginex.orm import EntityResolutionMatch, get_engine, get_session


def _config() -> DexConfig:
    return DexConfig.model_validate(
        {
            "project": {"name": "test"},
            "data": {
                "sources": {
                    "sink": {
                        "type": "test-sink",
                        "connection": {"spool_path": "spool/events.jsonl"},
                    }
                },
                "pipelines": {},
            },
        }
    )


def test_connector_paths_resolve_from_project_directory(tmp_path: Path) -> None:
    runner = PipelineRunner(_config(), data_dir=tmp_path / "lakehouse", project_dir=tmp_path)
    resolved = runner._resolve_connector_paths(  # noqa: SLF001
        {"spool_path": "spool/events.jsonl", "topic": "events"}
    )
    assert resolved["spool_path"] == str((tmp_path / "spool/events.jsonl").resolve())
    assert resolved["topic"] == "events"


def test_publish_output_uses_sink_and_does_not_change_loaded_data(tmp_path: Path) -> None:
    writes: list[dict[str, Any]] = []
    received_kwargs: dict[str, Any] = {}

    class _Sink:
        def __init__(self, **kwargs: Any) -> None:
            received_kwargs.update(kwargs)

        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def write(self, data: Any, *, table: str = "", **kwargs: Any) -> None:
            writes.extend(data)

    runner = PipelineRunner(_config(), data_dir=tmp_path / "lakehouse", project_dir=tmp_path)
    cfg = PipelineConfig(source="unused", destination="events", publish_to=["sink"])
    with (
        duckdb.connect(":memory:") as conn,
        patch(
            "dataenginex.domains.data.pipeline.runner.connector_registry.get",
            return_value=_Sink,
        ),
    ):
        conn.execute("CREATE TABLE output AS SELECT 1 AS id, 'ready' AS status")
        runner._publish_outputs(conn, cfg, "output", MagicMock())  # noqa: SLF001

    assert writes == [{"id": 1, "status": "ready"}]
    assert received_kwargs["spool_path"] == str((tmp_path / "spool/events.jsonl").resolve())


def test_entity_resolution_sink_upserts_match_rows(tmp_path: Path) -> None:
    runner = PipelineRunner(_config(), data_dir=tmp_path / "lakehouse", project_dir=tmp_path)
    cfg = PipelineConfig(
        source="unused",
        orm_sink={
            "model": "entity_resolution_match",
            "source_a_id": "imdb_id",
            "source_b_id": "tmdb_id",
            "confidence": "score",
            "db_path": ".dex/matches.db",
        },
    )
    with duckdb.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE matches AS SELECT 'tt1' AS imdb_id, 101 AS tmdb_id, 0.95 AS score"
        )
        runner._persist_entity_matches(conn, cfg, "matches", MagicMock())  # noqa: SLF001

    db_path = tmp_path / ".dex" / "matches.db"
    engine = get_engine(f"sqlite:///{db_path}")
    with get_session(engine) as session:
        row = session.get(EntityResolutionMatch, ("tt1", "101"))
        assert row is not None
        assert row.match_confidence == 0.95
    engine.dispose()
