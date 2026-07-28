"""Tests for warehouse lineage persistence, PostgresLineage, and remaining methods."""

from __future__ import annotations

import json
from typing import Any

from dataenginex.warehouse.lineage import LineageEvent, PersistentLineage


class TestPersistentLineageSaveLoad:
    def test_save_creates_file(self, tmp_path: Any) -> None:
        path = tmp_path / "lineage.json"
        pl = PersistentLineage(path)
        pl.record(operation="op", source="s", destination="d", input_count=1, output_count=1)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1

    def test_load_existing(self, tmp_path: Any) -> None:
        path = tmp_path / "lineage.json"
        ev = LineageEvent(
            operation="op",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
        )
        path.write_text(json.dumps([ev.to_dict()], default=str))
        pl = PersistentLineage(path)
        assert len(pl.all_events) == 1

    def test_empty_file(self, tmp_path: Any) -> None:
        path = tmp_path / "lineage.json"
        path.write_text("[]")
        pl = PersistentLineage(path)
        assert len(pl.all_events) == 0

    def test_corrupted_file(self, tmp_path: Any) -> None:
        path = tmp_path / "lineage.json"
        path.write_text("not json [[[")
        pl = PersistentLineage(path)
        assert len(pl.all_events) == 0

    def test_get_chain_empty(self) -> None:
        pl = PersistentLineage()
        chain = pl.get_chain("nonexistent")
        assert chain == []


class TestLineageEventToDict:
    def test_all_fields(self) -> None:
        ev = LineageEvent(
            operation="ingest",
            source="linkedin",
            destination="bronze",
            input_count=100,
            output_count=95,
            layer="bronze",
            parent_id="parent123",
            quality_score=0.85,
            pipeline_name="my_pipeline",
            metadata={"key": "value"},
        )
        d = ev.to_dict()
        assert d["operation"] == "ingest"
        assert d["source"] == "linkedin"
        assert d["destination"] == "bronze"
        assert d["input_count"] == 100
        assert d["output_count"] == 95
        assert d["layer"] == "bronze"
        assert d["parent_id"] == "parent123"
        assert d["quality_score"] == 0.85
        assert d["pipeline_name"] == "my_pipeline"
        assert d["metadata"] == {"key": "value"}
        assert "event_id" in d
        assert "timestamp" in d
