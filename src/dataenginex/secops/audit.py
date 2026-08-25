"""SecOps audit logging — structured log of PII detection and masking operations.

Every time PII is detected or masked, an ``AuditEvent`` is emitted via structlog
and appended to a bounded in-process buffer that the SecOps views read.

**This buffer is not the audit trail.** The durable, legally-meaningful record is
the control store's append-only ``audit_events`` table (§8.2, invariant 9),
written through ``ControlStore.emit_audit`` in the same transaction as the state
change it describes, and dispatched via the outbox.

That distinction is why the SQLite backend that used to live here is gone. It
opened a *fourth* database file with its own ``audit_events`` table, its own
schema, and — the real problem — its own ``DELETE`` statement for FIFO eviction.
An audit record you can silently evict is not an audit record, and two tables of
the same name in two databases meant "show me the audit trail" had two answers
that never agreed.

What remains is deliberately in-memory and deliberately bounded: a recent-events
view for the UI and for tests. Anything that must survive a restart is handed to
an :class:`AuditSink` — the control store in a wired system — which appends where
eviction is impossible.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger()

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "AuditOperation",
    "AuditSink",
]


class AuditOperation(StrEnum):
    """Type of SecOps operation being logged."""

    PII_SCAN = "pii_scan"
    PII_MASK = "pii_mask"
    PII_ACCESS = "pii_access"


@dataclass(frozen=True)
class AuditEvent:
    """A single audit log entry.

    Attributes:
        operation: Type of operation performed.
        dataset_name: Logical name of the dataset processed.
        pii_fields: Field names identified or masked.
        record_count: Number of records processed.
        actor: System component or user that triggered the operation.
        metadata: Extra context (strategy used, confidence scores, etc.).
        occurred_at: Timestamp of the event.
    """

    operation: AuditOperation
    dataset_name: str
    pii_fields: list[str]
    record_count: int
    actor: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event to a plain dictionary."""
        return {
            "operation": self.operation.value,
            "dataset_name": self.dataset_name,
            "pii_fields": self.pii_fields,
            "record_count": self.record_count,
            "actor": self.actor,
            "metadata": self.metadata,
            "occurred_at": self.occurred_at.isoformat(),
        }


@runtime_checkable
class AuditSink(Protocol):
    """Somewhere an audit event can be durably appended.

    Narrow on purpose. The guard does not need to read the audit trail, only to
    add to it, and a port that cannot express deletion is a port through which
    nothing can quietly delete.
    """

    def append_secops_event(self, event: AuditEvent) -> None: ...


class AuditLogger:
    """Recent SecOps audit events, in memory, plus an optional durable sink.

    Parameters
    ----------
    max_history:
        How many recent events to keep for display. Eviction here loses only
        the view, never the record — the record went to *sink*.
    sink:
        Durable destination, typically the control store. When ``None`` the
        events are observable in structlog and in this buffer only, which is
        the right shape for a script or a test that has no control plane.
    """

    def __init__(self, max_history: int = 1000, sink: AuditSink | None = None) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=max_history)
        self._sink = sink
        # The guard is called from request threads in Studio and from worker
        # threads in a pipeline; deque append is atomic but read-modify-write
        # over the whole buffer is not.
        self._lock = threading.Lock()

    def log(self, event: AuditEvent) -> None:
        """Record an audit event: durable sink first, then buffer and structlog.

        Sink first, because that is the write that matters. If it raises, the
        caller finds out rather than getting a success that only updated a
        buffer which vanishes on restart.
        """
        if self._sink is not None:
            self._sink.append_secops_event(event)
        with self._lock:
            self._events.append(event)
        logger.info(
            "secops audit event",
            operation=event.operation.value,
            dataset=event.dataset_name,
            pii_fields=event.pii_fields,
            record_count=event.record_count,
            actor=event.actor,
            **{k: v for k, v in event.metadata.items() if isinstance(v, (str, int, float, bool))},
        )

    def log_scan(
        self,
        dataset_name: str,
        pii_fields: list[str],
        record_count: int,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Convenience method to log a PII scan operation."""
        event = AuditEvent(
            operation=AuditOperation.PII_SCAN,
            dataset_name=dataset_name,
            pii_fields=pii_fields,
            record_count=record_count,
            actor=actor,
            metadata=metadata or {},
        )
        self.log(event)
        return event

    def log_mask(
        self,
        dataset_name: str,
        pii_fields: list[str],
        record_count: int,
        strategy: str,
        actor: str = "system",
    ) -> AuditEvent:
        """Convenience method to log a PII masking operation."""
        event = AuditEvent(
            operation=AuditOperation.PII_MASK,
            dataset_name=dataset_name,
            pii_fields=pii_fields,
            record_count=record_count,
            actor=actor,
            metadata={"strategy": strategy},
        )
        self.log(event)
        return event

    @property
    def events(self) -> list[AuditEvent]:
        """Recent audit events, oldest first."""
        with self._lock:
            return list(self._events)

    def events_for(self, dataset_name: str) -> list[AuditEvent]:
        """Recent events for a specific dataset."""
        with self._lock:
            return [e for e in self._events if e.dataset_name == dataset_name]

    def clear(self) -> None:
        """Drop the in-memory view.

        Clears what is displayed, not what was recorded — anything handed to the
        sink stays there, which is the point of having a sink.
        """
        with self._lock:
            self._events.clear()

    def close(self) -> None:
        """No resources to release. Kept so callers need not care which backend."""
        return

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
