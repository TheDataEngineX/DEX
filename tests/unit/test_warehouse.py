"""Tests for dataenginex.warehouse — transforms and lineage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataenginex.warehouse.lineage import PersistentLineage
from dataenginex.warehouse.transforms import (
    AddTimestampTransform,
    CastTypesTransform,
    DropNullsTransform,
    FilterTransform,
    RenameFieldsTransform,
    TransformPipeline,
    TransformResult,
)

# ============================================================================
# Transforms
# ============================================================================


class TestRenameFieldsTransform:
    def test_renames(self) -> None:
        t = RenameFieldsTransform({"old": "new"})
        result = t.apply({"old": 1, "keep": 2})
        assert result == {"new": 1, "keep": 2}

    def test_missing_key(self) -> None:
        t = RenameFieldsTransform({"nothere": "new"})
        result = t.apply({"a": 1})
        assert result == {"a": 1}


class TestDropNullsTransform:
    def test_keeps_complete_records(self) -> None:
        t = DropNullsTransform(["a", "b"])
        assert t.apply({"a": 1, "b": 2}) is not None

    def test_drops_null_records(self) -> None:
        t = DropNullsTransform(["a"])
        assert t.apply({"a": None}) is None

    def test_drops_missing_records(self) -> None:
        t = DropNullsTransform(["a"])
        assert t.apply({"b": 1}) is None


class TestCastTypesTransform:
    def test_cast_int(self) -> None:
        t = CastTypesTransform({"val": "int"})
        assert t.apply({"val": "42"})["val"] == 42

    def test_cast_float(self) -> None:
        t = CastTypesTransform({"val": "float"})
        assert t.apply({"val": "3.14"})["val"] == pytest.approx(3.14)

    def test_cast_bool(self) -> None:
        t = CastTypesTransform({"val": "bool"})
        assert t.apply({"val": 1})["val"] is True

    def test_bad_cast_no_crash(self) -> None:
        t = CastTypesTransform({"val": "int"})
        result = t.apply({"val": "not_a_number"})
        assert result["val"] == "not_a_number"  # unchanged


class TestAddTimestampTransform:
    def test_adds_field(self) -> None:
        t = AddTimestampTransform("ts")
        result = t.apply({"a": 1})
        assert "ts" in result


class TestFilterTransform:
    def test_keeps_matching(self) -> None:
        t = FilterTransform("pos", lambda r: r.get("v", 0) > 0)
        assert t.apply({"v": 5}) is not None

    def test_drops_non_matching(self) -> None:
        t = FilterTransform("pos", lambda r: r.get("v", 0) > 0)
        assert t.apply({"v": -1}) is None


class TestTransformPipeline:
    def test_pipeline_runs(self) -> None:
        p = TransformPipeline("test")
        p.add(DropNullsTransform(["id"]))
        p.add(CastTypesTransform({"id": "int"}))

        records: list[dict[str, Any]] = [
            {"id": "1", "name": "a"},
            {"id": None, "name": "b"},
            {"id": "3", "name": "c"},
        ]
        result = p.run(records)
        assert result.input_count == 3
        assert result.output_count == 2
        assert result.dropped_count == 1
        assert result.records[0]["id"] == 1
        assert len(result.step_metrics) == 2

    def test_empty_pipeline(self) -> None:
        p = TransformPipeline("empty")
        result = p.run([{"a": 1}])
        assert result.output_count == 1

    def test_success_rate(self) -> None:
        result = TransformResult(input_count=10, output_count=8)
        assert result.success_rate == pytest.approx(0.8)


# ============================================================================
# Lineage
# ============================================================================


class TestPersistentLineage:
    def test_record_and_get(self) -> None:
        lin = PersistentLineage()
        ev = lin.record(operation="ingest", layer="bronze", source="linkedin", input_count=100)
        assert ev.operation == "ingest"
        got = lin.get_event(ev.event_id)
        assert got is not None
        assert got.event_id == ev.event_id

    def test_chain(self) -> None:
        lin = PersistentLineage()
        e1 = lin.record(operation="ingest", layer="bronze")
        e2 = lin.record(operation="transform", layer="silver", parent_id=e1.event_id)
        e3 = lin.record(operation="enrich", layer="gold", parent_id=e2.event_id)
        chain = lin.get_chain(e3.event_id)
        assert len(chain) == 3
        assert chain[0].event_id == e1.event_id

    def test_get_by_layer(self) -> None:
        lin = PersistentLineage()
        lin.record(operation="ingest", layer="bronze")
        lin.record(operation="transform", layer="silver")
        assert len(lin.get_by_layer("bronze")) == 1

    def test_summary(self) -> None:
        lin = PersistentLineage()
        lin.record(operation="ingest", layer="bronze")
        lin.record(operation="ingest", layer="bronze")
        s = lin.summary()
        assert s["total_events"] == 2
        assert s["by_layer"]["bronze"] == 2

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "lineage.json"
        lin = PersistentLineage(path)
        lin.record(operation="ingest", layer="bronze", source="test")
        # Reload
        lin2 = PersistentLineage(path)
        assert len(lin2.all_events) == 1

    def test_get_children(self) -> None:
        lin = PersistentLineage()
        e1 = lin.record(operation="ingest", layer="bronze")
        lin.record(operation="transform", layer="silver", parent_id=e1.event_id)
        lin.record(operation="transform", layer="silver", parent_id=e1.event_id)
        children = lin.get_children(e1.event_id)
        assert len(children) == 2

    def test_get_by_pipeline(self) -> None:
        lin = PersistentLineage()
        lin.record(operation="ingest", layer="bronze", pipeline_name="p1")
        lin.record(operation="ingest", layer="bronze", pipeline_name="p2")
        lin.record(operation="ingest", layer="bronze", pipeline_name="p1")
        assert len(lin.get_by_pipeline("p1")) == 2

    def test_all_events_property(self) -> None:
        lin = PersistentLineage()
        lin.record(operation="ingest", layer="bronze")
        events = lin.all_events
        assert len(events) == 1
        events.append("junk")
        assert len(lin.all_events) == 1

    def test_load_corrupted_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("NOT VALID JSON {{{")
        lin = PersistentLineage(path)
        assert len(lin.all_events) == 0

    def test_to_dict_event(self) -> None:
        from dataenginex.warehouse.lineage import LineageEvent

        ev = LineageEvent(operation="test", layer="bronze")
        d = ev.to_dict()
        assert d["operation"] == "test"
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)

    def test_event_default_fields(self) -> None:
        from dataenginex.warehouse.lineage import LineageEvent

        ev = LineageEvent()
        assert ev.event_id
        assert ev.parent_id is None
        assert ev.input_count == 0
        assert ev.quality_score is None


class TestPostgresLineageFallback:
    def test_fallback_to_json_when_asyncpg_missing(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        fallback = tmp_path / "fallback.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=fallback)
        assert pg._pg_ok is False
        ev = pg.record(operation="test", layer="bronze", source="s")
        assert ev.operation == "test"
        assert len(pg.all_events) == 1

    def test_fallback_get_event(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        ev = pg.record(operation="ingest", layer="bronze")
        assert pg.get_event(ev.event_id) is not None

    def test_fallback_get_children(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        e1 = pg.record(operation="ingest", layer="bronze")
        pg.record(operation="transform", layer="silver", parent_id=e1.event_id)
        assert len(pg.get_children(e1.event_id)) == 1

    def test_fallback_get_chain(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        e1 = pg.record(operation="ingest", layer="bronze")
        e2 = pg.record(operation="transform", layer="silver", parent_id=e1.event_id)
        chain = pg.get_chain(e2.event_id)
        assert len(chain) == 2

    def test_fallback_get_by_layer(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        pg.record(operation="ingest", layer="bronze")
        pg.record(operation="transform", layer="silver")
        assert len(pg.get_by_layer("bronze")) == 1

    def test_fallback_get_by_pipeline(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        pg.record(operation="ingest", layer="bronze", pipeline_name="p1")
        pg.record(operation="ingest", layer="bronze", pipeline_name="p2")
        assert len(pg.get_by_pipeline("p1")) == 1

    def test_fallback_all_events(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        pg.record(operation="ingest", layer="bronze")
        assert len(pg.all_events) == 1

    def test_fallback_summary(self, tmp_path: Path) -> None:
        import warnings

        from dataenginex.warehouse.lineage import PostgresLineage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pg = PostgresLineage(dsn="postgresql://invalid", fallback_path=tmp_path / "f.json")
        pg.record(operation="ingest", layer="bronze")
        s = pg.summary()
        assert s["total_events"] == 1
