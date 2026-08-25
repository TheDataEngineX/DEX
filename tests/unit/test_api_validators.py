"""Tests for generic data quality validators."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dataenginex.domains.analytics.validators import DataQualityChecks, ValidationReport


class TestDataQualityChecks:
    def test_completeness_all_present(self) -> None:
        record = {"name": "Alice", "age": 30}
        ok, missing = DataQualityChecks.check_completeness(record, {"name", "age"})
        assert ok is True
        assert missing == []

    def test_completeness_missing_field(self) -> None:
        record = {"name": "Alice"}
        ok, missing = DataQualityChecks.check_completeness(record, {"name", "age"})
        assert ok is False
        assert "age" in missing

    def test_completeness_null_value_counts_as_missing(self) -> None:
        record = {"name": None}
        ok, missing = DataQualityChecks.check_completeness(record, {"name"})
        assert ok is False
        assert "name" in missing

    def test_completeness_empty_required_fields(self) -> None:
        ok, missing = DataQualityChecks.check_completeness({"a": 1}, set())
        assert ok is True
        assert missing == []

    def test_dates_consistent(self) -> None:
        now = datetime.now(tz=UTC)
        posted = now - timedelta(days=10)
        modified = now - timedelta(days=5)
        ok, issues = DataQualityChecks.check_consistency_dates(posted, modified)
        assert ok is True
        assert issues == []

    def test_posted_after_modified_is_inconsistent(self) -> None:
        now = datetime.now(tz=UTC)
        posted = now - timedelta(days=1)
        modified = now - timedelta(days=5)
        ok, issues = DataQualityChecks.check_consistency_dates(posted, modified)
        assert ok is False
        assert any("modified" in i for i in issues)

    def test_expiration_before_posted_is_inconsistent(self) -> None:
        now = datetime.now(tz=UTC)
        posted = now - timedelta(days=5)
        modified = now - timedelta(days=1)
        expired = now - timedelta(days=10)
        ok, issues = DataQualityChecks.check_consistency_dates(posted, modified, expired)
        assert ok is False
        assert any("Expiration" in i for i in issues)

    def test_posted_in_future_is_inconsistent(self) -> None:
        now = datetime.now(tz=UTC)
        posted = now + timedelta(days=1)
        modified = now + timedelta(days=2)
        ok, issues = DataQualityChecks.check_consistency_dates(posted, modified)
        assert ok is False
        assert any("future" in i for i in issues)


class TestValidationReport:
    def test_initial_state(self) -> None:
        report = ValidationReport()
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.invalid_records == 0
        assert report.errors == []
        assert report.warnings == []

    def test_add_error(self) -> None:
        report = ValidationReport()
        report.add_error("rec1", "missing_field", "Name is required")
        assert report.invalid_records == 1
        assert len(report.errors) == 1
        assert report.errors[0]["record_id"] == "rec1"

    def test_add_warning(self) -> None:
        report = ValidationReport()
        report.add_warning("rec2", "low_quality", "Score below threshold")
        assert len(report.warnings) == 1
        assert report.warnings[0]["type"] == "low_quality"

    def test_mark_valid(self) -> None:
        report = ValidationReport()
        report.mark_valid()
        report.mark_valid()
        assert report.valid_records == 2

    def test_finalize_with_mixed_records(self) -> None:
        report = ValidationReport()
        report.mark_valid()
        report.mark_valid()
        report.add_error("r1", "err", "bad")
        result = report.finalize()
        assert result["total_records"] == 3
        assert result["valid_records"] == 2
        assert result["invalid_records"] == 1
        assert abs(result["validity_percentage"] - 66.67) < 0.1

    def test_finalize_no_records_zero_pct(self) -> None:
        result = ValidationReport().finalize()
        assert result["total_records"] == 0
        assert result["validity_percentage"] == 0
