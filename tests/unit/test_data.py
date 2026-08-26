"""Tests for dataenginex.data — profiler, schema registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataenginex.domains.data.profiler import DataProfiler
from dataenginex.domains.data.registry import SchemaRegistry, SchemaVersion

# ============================================================================
# Profiler
# ============================================================================


class TestDataProfiler:
    def test_empty_records(self) -> None:
        report = DataProfiler().profile([], "empty")
        assert report.record_count == 0
        assert report.column_count == 0

    def test_numeric_column(self) -> None:
        records = [{"val": i} for i in range(10)]
        report = DataProfiler().profile(records, "nums")
        assert report.record_count == 10
        col = report.columns[0]
        assert col.dtype == "numeric"
        assert col.min_value == 0
        assert col.max_value == 9
        assert col.null_count == 0

    def test_string_column(self) -> None:
        records = [{"name": "alice"}, {"name": "bob"}, {"name": None}]
        report = DataProfiler().profile(records, "names")
        col = report.columns[0]
        assert col.dtype == "string"
        assert col.null_count == 1
        assert col.min_length is not None

    def test_mixed_column(self) -> None:
        records: list[dict[str, Any]] = [{"x": 1}, {"x": "two"}, {"x": True}]
        report = DataProfiler().profile(records, "mix")
        col = report.columns[0]
        assert col.dtype == "mixed"

    def test_completeness(self) -> None:
        records = [{"a": 1, "b": None}, {"a": 2, "b": None}]
        report = DataProfiler().profile(records, "nulls")
        assert report.completeness < 1.0

    def test_to_dict(self) -> None:
        records = [{"x": 1}]
        report = DataProfiler().profile(records, "d")
        d = report.to_dict()
        assert "columns" in d
        assert d["dataset_name"] == "d"


# ============================================================================
# Schema registry
# ============================================================================


class TestSchemaRegistry:
    def test_register_and_get_latest(self) -> None:
        reg = SchemaRegistry()
        v1 = SchemaVersion(name="jobs", version="1.0.0", fields={"id": "str"})
        reg.register(v1)
        latest = reg.get_latest("jobs")
        assert latest is not None
        assert latest.version == "1.0.0"

    def test_duplicate_version_rejected(self) -> None:
        reg = SchemaRegistry()
        v1 = SchemaVersion(name="jobs", version="1.0.0", fields={"id": "str"})
        reg.register(v1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(v1)

    def test_get_version(self) -> None:
        reg = SchemaRegistry()
        reg.register(SchemaVersion(name="jobs", version="1.0.0", fields={"id": "str"}))
        reg.register(
            SchemaVersion(
                name="jobs",
                version="2.0.0",
                fields={"id": "str", "title": "str"},
            )
        )
        v = reg.get_version("jobs", "1.0.0")
        assert v is not None
        assert v.version == "1.0.0"

    def test_list_schemas_and_versions(self) -> None:
        reg = SchemaRegistry()
        reg.register(SchemaVersion(name="A", version="1.0.0", fields={}))
        reg.register(SchemaVersion(name="A", version="2.0.0", fields={}))
        reg.register(SchemaVersion(name="B", version="1.0.0", fields={}))
        assert sorted(reg.list_schemas()) == ["A", "B"]
        assert reg.list_versions("A") == ["1.0.0", "2.0.0"]

    def test_validate_record(self) -> None:
        reg = SchemaRegistry()
        reg.register(
            SchemaVersion(
                name="jobs",
                version="1.0.0",
                fields={"id": "str", "title": "str"},
                required_fields=["id", "title"],
            )
        )
        ok, errors = reg.validate("jobs", {"id": "1", "title": "Dev"})
        assert ok
        ok2, errors2 = reg.validate("jobs", {"id": "1"})
        assert not ok2
        assert len(errors2) == 1

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "schemas.json"
        reg = SchemaRegistry(persist_path=path)
        reg.register(SchemaVersion(name="X", version="1.0.0", fields={"a": "int"}))
        # Reload
        reg2 = SchemaRegistry(persist_path=path)
        assert reg2.get_latest("X") is not None
