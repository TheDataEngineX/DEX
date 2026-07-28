"""Tests for warehouse lineage helpers: LineageEvent, PersistentLineage."""

from __future__ import annotations

from typing import Any

from dataenginex.warehouse.lineage import LineageEvent, PersistentLineage


class TestLineageEvent:
    def test_to_dict(self) -> None:
        ev = LineageEvent(
            operation="ingest",
            source="linkedin",
            destination="bronze",
            input_count=100,
            output_count=95,
            layer="bronze",
        )
        d = ev.to_dict()
        assert d["operation"] == "ingest"
        assert d["source"] == "linkedin"
        assert d["destination"] == "bronze"
        assert d["input_count"] == 100
        assert d["output_count"] == 95
        assert d["layer"] == "bronze"
        assert "timestamp" in d
        assert "event_id" in d

    def test_defaults(self) -> None:
        ev = LineageEvent(
            operation="op",
            source="s",
            destination="d",
            input_count=0,
            output_count=0,
        )
        assert ev.layer == ""
        assert ev.parent_id is None
        assert ev.quality_score is None
        assert ev.metadata == {}


class TestPersistentLineage:
    def test_record_and_get(self) -> None:
        pl = PersistentLineage()
        ev = pl.record(
            operation="ingest",
            source="s",
            destination="d",
            input_count=10,
            output_count=10,
            layer="bronze",
        )
        assert ev.event_id is not None
        fetched = pl.get_event(ev.event_id)
        assert fetched is not None
        assert fetched.operation == "ingest"

    def test_get_children(self) -> None:
        pl = PersistentLineage()
        parent = pl.record(
            operation="op",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="bronze",
        )
        child = pl.record(
            operation="op2",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="silver",
            parent_id=parent.event_id,
        )
        children = pl.get_children(parent.event_id)
        assert len(children) == 1
        assert children[0].event_id == child.event_id

    def test_get_chain(self) -> None:
        pl = PersistentLineage()
        e1 = pl.record(
            operation="op1",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="bronze",
        )
        e2 = pl.record(
            operation="op2",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="silver",
            parent_id=e1.event_id,
        )
        e3 = pl.record(
            operation="op3",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="gold",
            parent_id=e2.event_id,
        )
        chain = pl.get_chain(e3.event_id)
        assert len(chain) == 3
        assert chain[0].event_id == e1.event_id
        assert chain[-1].event_id == e3.event_id

    def test_get_by_layer(self) -> None:
        pl = PersistentLineage()
        pl.record(
            operation="op",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="bronze",
        )
        pl.record(
            operation="op",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="silver",
        )
        assert len(pl.get_by_layer("bronze")) == 1
        assert len(pl.get_by_layer("silver")) == 1
        assert len(pl.get_by_layer("gold")) == 0

    def test_get_by_pipeline(self) -> None:
        pl = PersistentLineage()
        pl.record(
            operation="op",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            pipeline_name="p1",
        )
        pl.record(
            operation="op",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            pipeline_name="p2",
        )
        assert len(pl.get_by_pipeline("p1")) == 1
        assert len(pl.get_by_pipeline("p2")) == 1

    def test_all_events(self) -> None:
        pl = PersistentLineage()
        pl.record(operation="op", source="s", destination="d", input_count=1, output_count=1)
        pl.record(operation="op", source="s", destination="d", input_count=1, output_count=1)
        assert len(pl.all_events) == 2

    def test_summary(self) -> None:
        pl = PersistentLineage()
        pl.record(
            operation="ingest",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="bronze",
        )
        pl.record(
            operation="transform",
            source="s",
            destination="d",
            input_count=1,
            output_count=1,
            layer="silver",
        )
        s = pl.summary()
        assert s["total_events"] == 2
        assert s["by_layer"]["bronze"] == 1
        assert s["by_layer"]["silver"] == 1
        assert s["by_operation"]["ingest"] == 1

    def test_persistence(self, tmp_path: Any) -> None:
        path = tmp_path / "lineage.json"
        pl = PersistentLineage(path)
        pl.record(operation="op", source="s", destination="d", input_count=1, output_count=1)
        # Reload from disk
        pl2 = PersistentLineage(path)
        assert len(pl2.all_events) == 1

    def test_get_event_missing(self) -> None:
        pl = PersistentLineage()
        assert pl.get_event("nonexistent") is None

    def test_get_chain_missing(self) -> None:
        pl = PersistentLineage()
        assert pl.get_chain("nonexistent") == []

    def test_load_corrupted(self, tmp_path: Any) -> None:
        path = tmp_path / "lineage.json"
        path.write_text("not valid json [[")
        pl = PersistentLineage(path)
        assert len(pl.all_events) == 0
