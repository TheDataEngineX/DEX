"""
Data profiler — automated dataset statistics and quality overview.

``DataProfiler`` accepts a list of records (dicts) and produces a
``ProfileReport`` summarising column types, null rates, cardinality,
numeric distributions, and string-length statistics.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "ColumnProfile",
    "DataProfiler",
    "ProfileReport",
]


@dataclass
class ColumnProfile:
    """Statistics for a single column / field."""

    name: str
    dtype: str  # "string", "numeric", "boolean", "null", "mixed"
    total_count: int = 0
    null_count: int = 0
    unique_count: int = 0

    # numeric stats (populated when dtype == "numeric")
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    median_value: float | None = None
    stddev_value: float | None = None

    # string stats (populated when dtype == "string")
    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None

    # top values
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    @property
    def null_rate(self) -> float:
        """Fraction of values that are ``None`` (0.0–1.0)."""
        return self.null_count / self.total_count if self.total_count else 0.0

    @property
    def uniqueness(self) -> float:
        """Ratio of unique non-null values to total non-null values."""
        non_null = self.total_count - self.null_count
        return self.unique_count / non_null if non_null else 0.0


@dataclass
class ProfileReport:
    """Aggregated profiling report for a dataset.

    Attributes:
        dataset_name: Name of the profiled dataset.
        record_count: Total number of records profiled.
        column_count: Number of columns/fields discovered.
        columns: Per-column profile statistics.
        profiled_at: Timestamp when profiling was performed.
        duration_ms: Time taken for profiling in milliseconds.
    """

    dataset_name: str
    record_count: int
    column_count: int
    columns: list[ColumnProfile]
    profiled_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile report to a plain dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "record_count": self.record_count,
            "column_count": self.column_count,
            "profiled_at": self.profiled_at.isoformat(),
            "duration_ms": self.duration_ms,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "total_count": c.total_count,
                    "null_count": c.null_count,
                    "null_rate": round(c.null_rate, 4),
                    "unique_count": c.unique_count,
                    "uniqueness": round(c.uniqueness, 4),
                    "min_value": c.min_value,
                    "max_value": c.max_value,
                    "mean_value": round(c.mean_value, 4) if c.mean_value is not None else None,
                    "median_value": c.median_value,
                    "stddev_value": (
                        round(c.stddev_value, 4) if c.stddev_value is not None else None
                    ),
                    "min_length": c.min_length,
                    "max_length": c.max_length,
                    "avg_length": round(c.avg_length, 2) if c.avg_length is not None else None,
                    "top_values": c.top_values[:5],
                }
                for c in self.columns
            ],
        }

    @property
    def completeness(self) -> float:
        """Overall dataset completeness (1 − avg null rate)."""
        if not self.columns:
            return 1.0
        avg_null = statistics.mean(c.null_rate for c in self.columns)
        return round(1.0 - avg_null, 4)


class DataProfiler:
    """Profile a list of dict records in pure Python (no pandas needed)."""

    def profile(
        self,
        records: list[dict[str, Any]],
        dataset_name: str = "unnamed",
    ) -> ProfileReport:
        """Profile a list of dict records and return column-level statistics."""
        import time

        start = time.perf_counter()

        if not records:
            return ProfileReport(
                dataset_name=dataset_name,
                record_count=0,
                column_count=0,
                columns=[],
            )

        # Discover all keys across all records
        all_keys: list[str] = []
        seen: set[str] = set()
        for rec in records:
            for k in rec:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        columns = [self._profile_column(key, records) for key in all_keys]
        duration = (time.perf_counter() - start) * 1000

        return ProfileReport(
            dataset_name=dataset_name,
            record_count=len(records),
            column_count=len(columns),
            columns=columns,
            duration_ms=round(duration, 2),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _profile_column(key: str, records: list[dict[str, Any]]) -> ColumnProfile:
        values = [r.get(key) for r in records]
        total = len(values)
        non_null = [v for v in values if v is not None]
        null_count = total - len(non_null)

        # Determine dominant type
        dtype = DataProfiler._infer_dtype(non_null)
        counter: Counter[Any] = Counter(str(v) for v in non_null)
        top_values = counter.most_common(5)
        unique_count = len(counter)

        col = ColumnProfile(
            name=key,
            dtype=dtype,
            total_count=total,
            null_count=null_count,
            unique_count=unique_count,
            top_values=top_values,
        )

        if dtype == "numeric":
            nums = [float(v) for v in non_null if DataProfiler._is_numeric(v)]
            if nums:
                col.min_value = min(nums)
                col.max_value = max(nums)
                col.mean_value = statistics.mean(nums)
                col.median_value = statistics.median(nums)
                col.stddev_value = statistics.stdev(nums) if len(nums) > 1 else 0.0

        elif dtype == "string":
            lengths = [len(str(v)) for v in non_null]
            if lengths:
                col.min_length = min(lengths)
                col.max_length = max(lengths)
                col.avg_length = statistics.mean(lengths)

        return col

    @staticmethod
    def _infer_dtype(values: list[Any]) -> str:
        if not values:
            return "null"
        types: set[str] = set()
        for v in values:
            if isinstance(v, bool):
                types.add("boolean")
            elif isinstance(v, (int, float)):
                types.add("numeric")
            elif isinstance(v, str):
                types.add("string")
            else:
                types.add("mixed")
        if len(types) == 1:
            return types.pop()
        return "mixed"

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
