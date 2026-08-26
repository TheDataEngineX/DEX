"""Tests for data profiler."""

from __future__ import annotations

from dataenginex.domains.data.profiler import ColumnProfile, DataProfiler, ProfileReport


class TestColumnProfile:
    def test_null_rate_empty(self) -> None:
        col = ColumnProfile(name="x", dtype="null", total_count=0, null_count=0)
        assert col.null_rate == 0.0

    def test_null_rate(self) -> None:
        col = ColumnProfile(name="x", dtype="string", total_count=10, null_count=3)
        assert col.null_rate == 0.3

    def test_uniqueness_empty(self) -> None:
        col = ColumnProfile(name="x", dtype="null", total_count=5, null_count=5)
        assert col.uniqueness == 0.0

    def test_uniqueness(self) -> None:
        col = ColumnProfile(name="x", dtype="string", total_count=10, null_count=2, unique_count=6)
        assert col.uniqueness == 6 / 8


class TestProfileReport:
    def test_to_dict(self) -> None:
        col = ColumnProfile(
            name="age", dtype="numeric", total_count=3, null_count=0,
            unique_count=3, min_value=10.0, max_value=30.0, mean_value=20.0,
            median_value=20.0, stddev_value=10.0,
        )
        report = ProfileReport(
            dataset_name="people", record_count=3, column_count=1, columns=[col],
            duration_ms=1.5,
        )
        d = report.to_dict()
        assert d["dataset_name"] == "people"
        assert d["record_count"] == 3
        assert len(d["columns"]) == 1
        assert d["columns"][0]["name"] == "age"

    def test_completeness_empty(self) -> None:
        report = ProfileReport(dataset_name="empty", record_count=0, column_count=0, columns=[])
        assert report.completeness == 1.0

    def test_completeness(self) -> None:
        col = ColumnProfile(name="x", dtype="string", total_count=10, null_count=2)
        report = ProfileReport(
            dataset_name="test", record_count=10, column_count=1, columns=[col],
        )
        assert report.completeness == 0.8


class TestDataProfiler:
    def test_empty_records(self) -> None:
        profiler = DataProfiler()
        report = profiler.profile([], dataset_name="empty")
        assert report.record_count == 0
        assert report.column_count == 0
        assert report.columns == []

    def test_numeric_column(self) -> None:
        profiler = DataProfiler()
        records = [{"age": 25}, {"age": 30}, {"age": 35}]
        report = profiler.profile(records, dataset_name="ages")
        assert report.record_count == 3
        assert report.column_count == 1
        col = report.columns[0]
        assert col.dtype == "numeric"
        assert col.min_value == 25.0
        assert col.max_value == 35.0
        assert col.mean_value == 30.0

    def test_string_column(self) -> None:
        profiler = DataProfiler()
        records = [{"name": "alice"}, {"name": "bob"}, {"name": "charlie"}]
        report = profiler.profile(records, dataset_name="names")
        col = report.columns[0]
        assert col.dtype == "string"
        assert col.min_length == 3
        assert col.max_length == 7

    def test_boolean_column(self) -> None:
        profiler = DataProfiler()
        records = [{"flag": True}, {"flag": False}]
        report = profiler.profile(records)
        col = report.columns[0]
        assert col.dtype == "boolean"

    def test_mixed_column(self) -> None:
        profiler = DataProfiler()
        records = [{"val": 1}, {"val": "hello"}]
        report = profiler.profile(records)
        col = report.columns[0]
        assert col.dtype == "mixed"

    def test_null_column(self) -> None:
        profiler = DataProfiler()
        records = [{"val": None}, {"val": None}]
        report = profiler.profile(records)
        col = report.columns[0]
        assert col.dtype == "null"

    def test_multiple_columns(self) -> None:
        profiler = DataProfiler()
        records = [
            {"name": "alice", "age": 25, "active": True},
            {"name": "bob", "age": 30, "active": False},
        ]
        report = profiler.profile(records)
        assert report.column_count == 3

    def test_duration_ms_populated(self) -> None:
        profiler = DataProfiler()
        report = profiler.profile([{"x": 1}])
        assert report.duration_ms >= 0
