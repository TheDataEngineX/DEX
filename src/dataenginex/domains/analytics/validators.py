"""Generic data quality validators for DataEngineX framework.

Domain-specific validators should live in the application package.

This module retains only generic, reusable checks:
- ``DataQualityChecks`` — completeness and date consistency
- ``ValidationReport`` — aggregated validation results
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

__all__ = [
    "DataQualityChecks",
    "ValidationReport",
]


class DataQualityChecks:
    """Generic data quality checks — not tied to any domain schema."""

    @staticmethod
    def check_completeness(
        record: dict[str, Any],
        required_fields: set[str],
    ) -> tuple[bool, list[str]]:
        """Check that all required fields are present and non-null.

        Args:
            record: Data record to check.
            required_fields: Set of field names that must be present.

        Returns:
            Tuple of (is_complete, missing_fields).
        """
        missing = [
            field for field in required_fields if field not in record or record[field] is None
        ]
        return len(missing) == 0, missing

    @staticmethod
    def check_consistency_dates(
        posted_date: datetime,
        last_modified_date: datetime,
        expiration_date: datetime | None = None,
    ) -> tuple[bool, list[str]]:
        """Check temporal consistency of dates.

        Returns:
            Tuple of (is_consistent, issues).
        """
        issues: list[str] = []

        if posted_date > last_modified_date:
            issues.append("Posted date is after last modified date")

        if expiration_date and expiration_date < posted_date:
            issues.append("Expiration date is before posted date")

        if posted_date > datetime.now(tz=UTC):
            issues.append("Posted date is in the future")

        return len(issues) == 0, issues


class ValidationReport:
    """Generates validation reports for data quality assessment."""

    def __init__(self) -> None:
        self.total_records = 0
        self.valid_records = 0
        self.invalid_records = 0
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def add_error(self, record_id: str, error_type: str, message: str) -> None:
        """Record a validation error."""
        self.invalid_records += 1
        self.errors.append({"record_id": record_id, "type": error_type, "message": message})

    def add_warning(self, record_id: str, warning_type: str, message: str) -> None:
        """Record a validation warning."""
        self.warnings.append({"record_id": record_id, "type": warning_type, "message": message})

    def mark_valid(self) -> None:
        """Mark a record as valid."""
        self.valid_records += 1

    def finalize(self) -> dict[str, Any]:
        """Generate final validation report."""
        total = self.valid_records + self.invalid_records
        valid_pct = (self.valid_records / total * 100) if total > 0 else 0

        return {
            "total_records": total,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "validity_percentage": valid_pct,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors[:100],  # Top 100 errors
            "warnings": self.warnings[:100],  # Top 100 warnings
        }
