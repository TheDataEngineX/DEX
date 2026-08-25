"""
Partitioning strategies for the DataEngineX lakehouse.

``PartitionStrategy`` is an ABC whose subclasses generate path segments
used by storage backends to organise data into predictable directory trees.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "DatePartitioner",
    "HashPartitioner",
    "PartitionStrategy",
]


class PartitionStrategy(ABC):
    """Base class for partitioning strategies."""

    @abstractmethod
    def partition_key(self, record: dict[str, Any]) -> str:
        """Return the partition path segment for *record*."""
        ...

    @abstractmethod
    def partition_path(self, record: dict[str, Any], base: str = "") -> str:
        """Return the full relative path (base + partition) for *record*."""
        ...


class DatePartitioner(PartitionStrategy):
    """Partition by a date field using ``year=…/month=…/day=…`` layout.

    Parameters
    ----------
    date_field:
        Name of the record field containing a date/datetime value.
    granularity:
        ``"day"`` (default), ``"month"``, or ``"year"``.
    """

    def __init__(self, date_field: str = "created_at", granularity: str = "day") -> None:
        self.date_field = date_field
        if granularity not in ("day", "month", "year"):
            raise ValueError(f"granularity must be day/month/year, got {granularity!r}")
        self.granularity = granularity

    def partition_key(self, record: dict[str, Any]) -> str:
        """Return a ``year=.../month=.../day=...`` key for *record*."""
        dt = self._extract_date(record)
        parts = [f"year={dt.year}"]
        if self.granularity in ("month", "day"):
            parts.append(f"month={dt.month:02d}")
        if self.granularity == "day":
            parts.append(f"day={dt.day:02d}")
        return "/".join(parts)

    def partition_path(self, record: dict[str, Any], base: str = "") -> str:
        """Return *base* joined with the date partition key."""
        key = self.partition_key(record)
        return f"{base}/{key}" if base else key

    def _extract_date(self, record: dict[str, Any]) -> datetime:
        value = record.get(self.date_field)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        # Fallback to now
        return datetime.now(tz=UTC)


class HashPartitioner(PartitionStrategy):
    """Partition by a hash of one or more fields, distributing across *n_buckets*.

    Parameters
    ----------
    fields:
        Record fields whose values are hashed.
    n_buckets:
        Number of hash buckets (directories).
    """

    def __init__(self, fields: list[str], n_buckets: int = 16) -> None:
        if not fields:
            raise ValueError("At least one field is required for hash partitioning")
        self.fields = fields
        self.n_buckets = max(1, n_buckets)

    def partition_key(self, record: dict[str, Any]) -> str:
        """Return a ``bucket=NNNN`` key based on hashed field values."""
        content = "|".join(str(record.get(f, "")) for f in self.fields)
        digest = hashlib.md5(content.encode()).hexdigest()  # noqa: S324
        bucket = int(digest, 16) % self.n_buckets
        return f"bucket={bucket:04d}"

    def partition_path(self, record: dict[str, Any], base: str = "") -> str:
        """Return *base* joined with the hash-bucket partition key."""
        key = self.partition_key(record)
        return f"{base}/{key}" if base else key
