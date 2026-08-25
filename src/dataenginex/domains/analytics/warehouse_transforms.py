"""
Transform framework — declarative, composable data transformations.

Each ``Transform`` is a named, reusable function that maps a record dict to a
new record dict.  ``TransformPipeline`` chains multiple transforms and collects
per-step metrics for observability.
"""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger()
__all__ = [
    "AddTimestampTransform",
    "CastTypesTransform",
    "DropNullsTransform",
    "FilterTransform",
    "RenameFieldsTransform",
    "Transform",
    "TransformPipeline",
    "TransformResult",
]


@dataclass
class TransformResult:
    """Outcome of running a pipeline over a batch of records.

    Attributes:
        input_count: Number of records fed into the pipeline.
        output_count: Number of records that passed all transform steps.
        dropped_count: Records dropped during transformation.
        error_count: Records that caused transform errors.
        duration_ms: Total pipeline execution time in milliseconds.
        records: Transformed output records.
        step_metrics: Per-step metrics (input/output/dropped/duration).
        completed_at: Timestamp of pipeline completion.
    """

    input_count: int
    output_count: int
    dropped_count: int = 0
    error_count: int = 0
    duration_ms: float = 0.0
    records: list[dict[str, Any]] = field(default_factory=list)
    step_metrics: list[dict[str, Any]] = field(default_factory=list)
    completed_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def success_rate(self) -> float:
        """Fraction of input records that made it to the output."""
        return self.output_count / self.input_count if self.input_count else 0.0


# ---------------------------------------------------------------------------
# Single transform
# ---------------------------------------------------------------------------


class Transform(ABC):
    """Base class for a single data transform step.

    Subclass and implement ``apply`` which receives one record and returns
    either a transformed record or *None* to drop it.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Transform *record* in place or return a new dict.

        Return *None* to drop the record from the pipeline.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


class RenameFieldsTransform(Transform):
    """Rename keys in a record according to a mapping."""

    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__(name="rename_fields", description="Rename record fields")
        self.mapping = mapping

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        """Rename keys in *record* according to the configured mapping."""
        out = dict(record)
        for old_key, new_key in self.mapping.items():
            if old_key in out:
                out[new_key] = out.pop(old_key)
        return out


class DropNullsTransform(Transform):
    """Drop records that have *None* in any of the specified fields."""

    def __init__(self, required_fields: list[str]) -> None:
        super().__init__(name="drop_nulls", description="Drop records with null required fields")
        self.required_fields = required_fields

    def apply(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Return ``None`` if any required field is null, else the record."""
        for f in self.required_fields:
            if record.get(f) is None:
                return None
        return record


class CastTypesTransform(Transform):
    """Cast fields to target types (``int``, ``float``, ``str``, ``bool``)."""

    _CASTERS: dict[str, type] = {"int": int, "float": float, "str": str, "bool": bool}

    def __init__(self, type_map: dict[str, str]) -> None:
        super().__init__(name="cast_types", description="Cast field types")
        self.type_map = type_map

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        """Cast specified fields to their target types."""
        out = dict(record)
        for field_name, target in self.type_map.items():
            if field_name in out and out[field_name] is not None:
                caster = self._CASTERS.get(target)
                if caster:
                    with contextlib.suppress(ValueError, TypeError):
                        out[field_name] = caster(out[field_name])
        return out


class AddTimestampTransform(Transform):
    """Add a processing timestamp field to every record."""

    def __init__(self, field_name: str = "processed_at") -> None:
        super().__init__(name="add_timestamp", description="Add processing timestamp")
        self.field_name = field_name

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        """Add an ISO-8601 processing timestamp to *record*."""
        out = dict(record)
        out[self.field_name] = datetime.now(tz=UTC).isoformat()
        return out


class FilterTransform(Transform):
    """Keep only records that match a predicate expression.

    ``predicate`` receives a record dict and returns ``True`` to keep it.
    """

    def __init__(self, name: str, predicate: Any) -> None:
        super().__init__(name=name, description="Filter records by predicate")
        self._predicate = predicate

    def apply(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Keep *record* if the predicate returns ``True``, else drop it."""
        if self._predicate(record):
            return record
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TransformPipeline:
    """Execute an ordered chain of ``Transform`` steps over a batch.

    Example::

        pipeline = TransformPipeline("bronze_to_silver")
        pipeline.add(DropNullsTransform(["event_id", "user_name"]))
        pipeline.add(CastTypesTransform({"amount": "float"}))
        result = pipeline.run(records)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[Transform] = []

    def add(self, transform: Transform) -> TransformPipeline:
        """Append a transform step to the pipeline."""
        self._steps.append(transform)
        return self  # fluent API

    def run(self, records: list[dict[str, Any]]) -> TransformResult:
        """Execute all steps in order and return the aggregated result."""
        start = time.perf_counter()
        current = list(records)
        step_metrics: list[dict[str, Any]] = []

        for step in self._steps:
            step_start = time.perf_counter()
            before_count = len(current)
            output: list[dict[str, Any]] = []

            for rec in current:
                try:
                    result = step.apply(rec)
                    if result is not None:
                        output.append(result)
                except Exception as exc:
                    logger.warning(
                        "Transform %s failed on record: %s",
                        step.name,
                        exc,
                    )

            step_duration = (time.perf_counter() - step_start) * 1000
            step_metrics.append(
                {
                    "step": step.name,
                    "input_count": before_count,
                    "output_count": len(output),
                    "dropped": before_count - len(output),
                    "duration_ms": round(step_duration, 2),
                }
            )
            current = output

        total_duration = (time.perf_counter() - start) * 1000
        return TransformResult(
            input_count=len(records),
            output_count=len(current),
            dropped_count=len(records) - len(current),
            duration_ms=round(total_duration, 2),
            records=current,
            step_metrics=step_metrics,
        )
