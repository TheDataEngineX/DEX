"""Micro-batch stream processing (§7.11, §14.8).

A continuous workload does not finish, so almost nothing the batch runtime
assumes applies to it. It has no output to commit at the end, no exit code to
classify, and no natural point at which "the run succeeded" becomes true. What
it has instead is a position in a source that must survive a restart.

The guarantees here are deliberately modest, and saying so plainly is the
point (§14.8):

**At-least-once, not exactly-once.** A crash between processing a batch and
saving its cursor replays that batch. Pretending otherwise would require a
distributed transaction with every source we support, which we do not have.
Consumers deduplicate on a declared key, so a replay is harmless rather than
invisible.

**Durable cursor, saved after the work.** The order matters and is the whole
correctness argument: process, then checkpoint. Checkpointing first would turn
a crash into silent data loss, which is strictly worse than a duplicate —
duplicates are detectable and removable, gaps are neither.

**Bounded retries, then a dead letter.** One record that fails deterministically
would otherwise block the stream forever, because the cursor cannot advance
past it. After the retry budget it is parked in ``stream_dead_letters`` with
its cursor, and the stream continues.

**Backpressure by shrinking the batch.** When processing takes longer than the
interval, the stream is behind, and reading more per cycle makes it further
behind. The batch size halves until it recovers, rather than growing a queue
in memory.

Deferred until something concrete needs them (§14.8): distributed watermarks,
complex event-time windows, exactly-once. All three are large, and none has a
use case in the reference projects.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pydantic import Field

from dataenginex.foundation import ProjectId, RunId, new_id, utcnow
from dataenginex.foundation.projects import FrozenModel
from dataenginex.runtime.state import ControlStore

__all__ = [
    "DeadLetter",
    "StreamCheckpoint",
    "StreamConfig",
    "StreamError",
    "StreamProgress",
    "StreamRecord",
    "StreamRunner",
]


class StreamError(RuntimeError):
    """Stream operation that could not be completed."""


class StreamRecord(FrozenModel):
    """One record read from a source.

    ``event_time`` is when the thing described actually happened;
    ``ingested_at`` is when we saw it. Keeping both is not bookkeeping (§8.8) —
    a device with a drifted clock, or a reconnect that replays an hour of
    buffered messages, produces records whose two timestamps disagree by a lot,
    and aggregating by the wrong one files them into the wrong window.

    ``cursor`` is the source position *after* this record, so checkpointing it
    means "everything through here is done".
    """

    key: str
    payload: dict[str, Any]
    cursor: dict[str, Any]
    event_time: datetime | None = None
    ingested_at: datetime = Field(default_factory=utcnow)


class StreamCheckpoint(FrozenModel):
    """A durable source position (§8.7)."""

    stream_name: str
    cursor: dict[str, Any]
    created_at: datetime


class StreamConfig(FrozenModel):
    """Stream tuning, from the ``kind: Stream`` manifest (§7.11)."""

    stream_name: str
    # How often the cursor is written. Longer means less write amplification
    # and more replay after a crash; this is the knob that trades one against
    # the other, and there is no correct default for every source.
    checkpoint_interval_seconds: float = 30.0
    # Micro-batch ceiling, not a guarantee — backpressure lowers the working
    # size below it.
    batch_size: int = 500
    min_batch_size: int = 1
    # Per-record retry budget before the dead letter.
    max_attempts: int = 3
    # The field or fields that identify a duplicate. Without one, an
    # at-least-once stream double-counts on every restart, so a stream that
    # declares no key gets no dedup and must be idempotent by nature.
    dedup_key: tuple[str, ...] = ()
    # How many recent keys to remember. Bounded on purpose: an unbounded set
    # is a memory leak with a long fuse on a stream that never ends.
    dedup_window: int = 10_000


@dataclass
class StreamProgress:
    """What happened over one or more cycles.

    Mutable and plain: a running tally the runner updates in place, not a
    value that crosses a boundary.
    """

    batches: int = 0
    records_read: int = 0
    records_processed: int = 0
    duplicates_skipped: int = 0
    dead_lettered: int = 0
    checkpoints: int = 0
    current_batch_size: int = 0
    last_cursor: dict[str, Any] | None = None


class DeadLetter(FrozenModel):
    """A record the stream gave up on, kept for replay."""

    dead_letter_id: str
    stream_name: str
    cursor: dict[str, Any]
    payload: dict[str, Any]
    error: str
    attempts: int
    event_time: datetime | None
    created_at: datetime


class _DedupWindow:
    """Bounded set of recently seen keys, in insertion order.

    A plain ``dict`` rather than an ``OrderedDict``: insertion order has been
    part of the language since 3.7, and this needs nothing extra.
    """

    __slots__ = ("_capacity", "_seen")

    def __init__(self, capacity: int) -> None:
        self._capacity = max(capacity, 0)
        self._seen: dict[str, None] = {}

    def seen(self, key: str) -> bool:
        """True when this key was already processed. Records it if not."""
        if self._capacity == 0:
            return False
        if key in self._seen:
            return True
        self._seen[key] = None
        while len(self._seen) > self._capacity:
            # Oldest first. A key that falls out of the window can be
            # reprocessed, which is exactly the at-least-once promise: the
            # window narrows the duplicate rate, it does not eliminate it.
            del self._seen[next(iter(self._seen))]
        return False


class StreamRunner:
    """Runs one continuous workload against a source (§7.11).

    The runner owns the loop, the cursor, and the failure policy. It does not
    own the source or the processing — those are callables, so a stream over
    MQTT and a stream over a file differ by one argument rather than by a
    subclass.
    """

    def __init__(
        self,
        store: ControlStore,
        config: StreamConfig,
        *,
        project_id: ProjectId,
        run_id: RunId | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.project_id = project_id
        self.run_id = run_id
        self.progress = StreamProgress(current_batch_size=config.batch_size)
        self._dedup = _DedupWindow(config.dedup_window if config.dedup_key else 0)
        self._last_checkpoint: datetime | None = None
        self._pending_cursor: dict[str, Any] | None = None

    # --- cursor (§8.7) ------------------------------------------------------

    def load_checkpoint(self) -> StreamCheckpoint | None:
        """The position this stream left off at, or None on a first run.

        This is what makes a restart resume rather than start over — the most
        visible difference between a stream that survives a restart and one
        that merely runs again.
        """
        row = self.store.query_one(
            "SELECT stream_name, cursor_json, created_at FROM checkpoint_records "
            "WHERE project_id = ? AND stream_name = ?",
            (self.project_id, self.config.stream_name),
        )
        if row is None:
            return None
        return StreamCheckpoint(
            stream_name=row["stream_name"],
            cursor=json.loads(row["cursor_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def save_checkpoint(self, cursor: dict[str, Any]) -> None:
        """Record the position, replacing the previous one.

        One row per stream, upserted. History is not kept: a stale cursor is
        not evidence of anything, and resuming from an old one would replay
        work that already committed.
        """
        if self.run_id is None:
            # ``checkpoint_records.run_id`` references ``runs``, so there is no
            # honest value to write here. Raising beats inserting a placeholder
            # that would satisfy the column and make the provenance question
            # "which run advanced this cursor?" permanently unanswerable.
            raise StreamError(
                f"stream {self.config.stream_name!r} cannot checkpoint without a run"
            )

        now = utcnow()
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO checkpoint_records "
                "(checkpoint_id, project_id, run_id, stream_name, cursor_json, created_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(project_id, stream_name) DO UPDATE SET "
                "cursor_json = excluded.cursor_json, created_at = excluded.created_at, "
                "run_id = excluded.run_id",
                (
                    new_id("ckpt"),
                    self.project_id,
                    self.run_id,
                    self.config.stream_name,
                    json.dumps(cursor, default=str),
                    now.isoformat(),
                ),
            )
        self._last_checkpoint = now
        self._pending_cursor = None
        self.progress.checkpoints += 1

    def _checkpoint_due(self, now: datetime) -> bool:
        if self._last_checkpoint is None:
            return True
        return now - self._last_checkpoint >= timedelta(
            seconds=self.config.checkpoint_interval_seconds
        )

    # --- dead letters (§14.8) -----------------------------------------------

    def dead_letter(self, record: StreamRecord, error: str, attempts: int) -> DeadLetter:
        """Park a record the stream could not process.

        The cursor is stored with it. Without that, replaying a dead letter
        later means guessing where it came from.
        """
        entry = DeadLetter(
            dead_letter_id=new_id("dlq"),
            stream_name=self.config.stream_name,
            cursor=record.cursor,
            payload=record.payload,
            error=error,
            attempts=attempts,
            event_time=record.event_time,
            created_at=utcnow(),
        )
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO stream_dead_letters (dead_letter_id, project_id, stream_name, "
                "run_id, cursor_json, payload_json, error, attempts, event_time, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    entry.dead_letter_id,
                    self.project_id,
                    entry.stream_name,
                    self.run_id,
                    json.dumps(entry.cursor, default=str),
                    json.dumps(entry.payload, default=str),
                    entry.error,
                    entry.attempts,
                    entry.event_time.isoformat() if entry.event_time else None,
                    entry.created_at.isoformat(),
                ),
            )
        self.progress.dead_lettered += 1
        return entry

    def dead_letters(self) -> list[DeadLetter]:
        """Everything parked for this stream, oldest first."""
        rows = self.store.query(
            "SELECT * FROM stream_dead_letters WHERE project_id = ? AND stream_name = ? "
            "ORDER BY created_at",
            (self.project_id, self.config.stream_name),
        )
        return [
            DeadLetter(
                dead_letter_id=row["dead_letter_id"],
                stream_name=row["stream_name"],
                cursor=json.loads(row["cursor_json"]),
                payload=json.loads(row["payload_json"]),
                error=row["error"],
                attempts=row["attempts"],
                event_time=(
                    datetime.fromisoformat(row["event_time"]) if row["event_time"] else None
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    # --- dedup (§14.8) ------------------------------------------------------

    def _dedup_key(self, record: StreamRecord) -> str | None:
        """The declared key for this record, or None when dedup is off.

        A record missing one of the key fields is *not* treated as a
        duplicate. Guessing would silently drop data whose shape changed,
        which is the failure nobody notices until a month of numbers is wrong.
        """
        if not self.config.dedup_key:
            return None
        try:
            return "\x1f".join(str(record.payload[name]) for name in self.config.dedup_key)
        except KeyError:
            return None

    # --- the loop (§7.11) ---------------------------------------------------

    def run_batch(
        self,
        read: Callable[[int], Sequence[StreamRecord]],
        process: Callable[[StreamRecord], None],
        *,
        now: datetime | None = None,
    ) -> int:
        """Read one micro-batch, process it, then checkpoint. Returns the count.

        The order is the correctness argument. Processing precedes
        checkpointing, so a crash in between replays the batch — at-least-once.
        Checkpointing first would lose it, which is worse: a duplicate can be
        removed by a dedup key, a gap cannot be recovered from anything.
        """
        now = now or utcnow()
        batch = list(read(self.progress.current_batch_size))
        self.progress.batches += 1
        self.progress.records_read += len(batch)
        if not batch:
            # Idle. Still checkpoint if one is due: a stream that catches up
            # and then goes quiet should not keep replaying its last batch for
            # as long as it stays quiet.
            if self._pending_cursor is not None and self._checkpoint_due(now):
                self.save_checkpoint(self._pending_cursor)
            return 0

        for record in batch:
            key = self._dedup_key(record)
            if key is not None and self._dedup.seen(key):
                self.progress.duplicates_skipped += 1
                # A duplicate is still progress: the cursor moves past it, or
                # a restart would read it again forever.
                self._pending_cursor = record.cursor
                continue

            if self._process_with_retries(record, process):
                self.progress.records_processed += 1
            # Whether it succeeded or went to the dead letter, the stream is
            # done with it and the cursor advances. That is what stops one
            # poison record from blocking everything behind it.
            self._pending_cursor = record.cursor

        self.progress.last_cursor = self._pending_cursor
        if self._pending_cursor is not None and self._checkpoint_due(now):
            self.save_checkpoint(self._pending_cursor)
        return len(batch)

    def _process_with_retries(
        self, record: StreamRecord, process: Callable[[StreamRecord], None]
    ) -> bool:
        """Process one record within its retry budget. False if dead-lettered."""
        last_error = ""
        for _attempt in range(self.config.max_attempts):
            try:
                process(record)
            except Exception as exc:  # noqa: BLE001 - a handler may raise anything
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            return True
        self.dead_letter(record, last_error, self.config.max_attempts)
        return False

    # --- backpressure (§14.8) -----------------------------------------------

    def apply_backpressure(self, elapsed_seconds: float, interval_seconds: float) -> int:
        """Resize the batch from how long the last cycle took.

        Falling behind means the batch is too big for this machine, and
        reading more per cycle makes it worse. Halving sheds load quickly;
        recovery adds a quarter at a time, because jumping straight back to the
        ceiling just oscillates.
        """
        size = self.progress.current_batch_size
        if elapsed_seconds > interval_seconds:
            size = max(self.config.min_batch_size, size // 2)
        elif elapsed_seconds < interval_seconds / 2:
            size = min(self.config.batch_size, max(size + 1, size + size // 4))
        self.progress.current_batch_size = size
        return size
