"""Tests for the warehouse transforms.

Covers the Transform classes and TransformPipeline.
"""

from __future__ import annotations

import pytest

from dataenginex.domains.analytics.warehouse_transforms import (
    AddTimestampTransform,
    CastTypesTransform,
    DropNullsTransform,
    FilterTransform,
    RenameFieldsTransform,
    Transform,
    TransformPipeline,
    TransformResult,
)

# ── TransformResult ───────────────────────────────────────────────────────────


class TestTransformResult:
    def test_success_rate_all_pass(self) -> None:
        r = TransformResult(input_count=10, output_count=10)
        assert r.success_rate >= 1.0  # 10/10

    def test_success_rate_partial(self) -> None:
        import math

        r = TransformResult(input_count=10, output_count=7, dropped_count=3)
        assert math.isclose(r.success_rate, 0.7)

    def test_success_rate_zero_processed(self) -> None:
        r = TransformResult(input_count=0, output_count=0)
        assert not r.success_rate  # 0.0 is falsy

    def test_duration_and_records(self) -> None:
        import math

        r = TransformResult(input_count=5, output_count=4, dropped_count=1, duration_ms=12.5)
        assert r.dropped_count == 1
        assert math.isclose(r.duration_ms, 12.5)


# ── Transform (ABC) ───────────────────────────────────────────────────────────


class TestTransformABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Transform("t")  # type: ignore[abstract]

    def test_name_and_description(self) -> None:
        class MyT(Transform):
            def apply(self, r):
                return r

        t = MyT("my-transform", "does things")
        assert t.name == "my-transform"
        assert t.description == "does things"


# ── RenameFieldsTransform ─────────────────────────────────────────────────────


class TestRenameFieldsTransform:
    def test_renames_fields(self) -> None:
        t = RenameFieldsTransform({"old_name": "new_name"})
        result = t.apply({"old_name": "value", "other": 1})
        assert "new_name" in result
        assert "old_name" not in result
        assert result["other"] == 1

    def test_missing_field_ignored(self) -> None:
        t = RenameFieldsTransform({"missing": "renamed"})
        record = {"kept": "x"}
        result = t.apply(record)
        assert result == record

    def test_empty_mapping(self) -> None:
        t = RenameFieldsTransform({})
        record = {"a": 1}
        assert t.apply(record) == record


# ── DropNullsTransform ────────────────────────────────────────────────────────


class TestDropNullsTransform:
    def test_passes_complete_record(self) -> None:
        t = DropNullsTransform(required_fields=["id", "name"])
        assert t.apply({"id": 1, "name": "alice"}) is not None

    def test_drops_record_with_null(self) -> None:
        t = DropNullsTransform(required_fields=["id"])
        assert t.apply({"id": None}) is None

    def test_drops_record_with_missing_field(self) -> None:
        t = DropNullsTransform(required_fields=["id"])
        assert t.apply({"name": "x"}) is None

    def test_empty_required_fields(self) -> None:
        t = DropNullsTransform(required_fields=[])
        assert t.apply({"anything": None}) is not None


# ── CastTypesTransform ────────────────────────────────────────────────────────


class TestCastTypesTransform:
    def test_cast_to_int(self) -> None:
        t = CastTypesTransform(type_map={"age": "int"})
        result = t.apply({"age": "25"})
        assert result["age"] == 25
        assert isinstance(result["age"], int)

    def test_cast_to_float(self) -> None:
        t = CastTypesTransform(type_map={"score": "float"})
        result = t.apply({"score": "3.14"})
        assert isinstance(result["score"], float)

    def test_cast_to_str(self) -> None:
        t = CastTypesTransform(type_map={"code": "str"})
        result = t.apply({"code": 42})
        assert result["code"] == "42"

    def test_missing_field_unchanged(self) -> None:
        t = CastTypesTransform(type_map={"x": "int"})
        record = {"y": "hello"}
        result = t.apply(record)
        assert "x" not in result

    def test_cast_error_keeps_original(self) -> None:
        t = CastTypesTransform(type_map={"age": "int"})
        result = t.apply({"age": "not_an_int"})
        assert result["age"] == "not_an_int"


# ── AddTimestampTransform ─────────────────────────────────────────────────────


class TestAddTimestampTransform:
    def test_adds_timestamp_field(self) -> None:
        t = AddTimestampTransform()
        result = t.apply({"id": 1})
        assert "processed_at" in result

    def test_custom_field_name(self) -> None:
        t = AddTimestampTransform(field_name="ingested_at")
        result = t.apply({"id": 1})
        assert "ingested_at" in result

    def test_timestamp_is_string(self) -> None:
        t = AddTimestampTransform()
        result = t.apply({})
        assert isinstance(result["processed_at"], str)


# ── FilterTransform ───────────────────────────────────────────────────────────


class TestFilterTransform:
    def test_passes_when_predicate_true(self) -> None:
        t = FilterTransform("age_filter", predicate=lambda r: r.get("age", 0) >= 18)
        assert t.apply({"age": 25}) is not None

    def test_drops_when_predicate_false(self) -> None:
        t = FilterTransform("age_filter", predicate=lambda r: r.get("age", 0) >= 18)
        assert t.apply({"age": 10}) is None

    def test_custom_name(self) -> None:
        t = FilterTransform("my_filter", predicate=lambda _: True)
        assert t.name == "my_filter"


# ── TransformPipeline ─────────────────────────────────────────────────────────


class TestTransformPipeline:
    def test_empty_pipeline(self) -> None:
        p = TransformPipeline(name="empty")
        result = p.run([{"id": 1}])
        assert result.input_count == 1
        assert result.output_count == 1

    def test_single_transform(self) -> None:
        p = TransformPipeline(name="rename")
        p.add(RenameFieldsTransform({"old": "new"}))
        records = [{"old": "val"}, {"other": "x"}]
        result = p.run(records)
        assert result.input_count == 2

    def test_drop_nulls_removes_records(self) -> None:
        p = TransformPipeline(name="filter")
        p.add(DropNullsTransform(required_fields=["id"]))
        records = [{"id": 1}, {"id": None}, {"id": 3}]
        result = p.run(records)
        assert result.output_count == 2
        assert result.dropped_count == 1

    def test_chain_transforms(self) -> None:
        p = TransformPipeline(name="chain")
        p.add(RenameFieldsTransform({"raw_age": "age"}))
        p.add(CastTypesTransform({"age": "int"}))
        p.add(FilterTransform("adult", predicate=lambda r: r.get("age", 0) >= 18))
        records = [{"raw_age": "25"}, {"raw_age": "10"}, {"raw_age": "30"}]
        result = p.run(records)
        assert result.output_count == 2

    def test_empty_input(self) -> None:
        p = TransformPipeline(name="empty_input")
        p.add(DropNullsTransform(["id"]))
        result = p.run([])
        assert result.input_count == 0
        assert result.output_count == 0

    def test_fluent_add(self) -> None:
        p = TransformPipeline(name="fluent")
        ret = p.add(DropNullsTransform(["id"]))
        assert ret is p  # fluent API returns self

    def test_pipeline_name(self) -> None:
        p = TransformPipeline(name="my_pipeline")
        assert p.name == "my_pipeline"
