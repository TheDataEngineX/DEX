"""Micro-batch stream semantics (§7.11, §14.8).

The tests worth having here are the ones about failure. A stream that reads
records and writes them somewhere is easy and proves nothing; what the design
claims is that it survives a restart without losing or double-counting, that
one bad record cannot stall it forever, and that it sheds load instead of
falling further behind. Those are the tests below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from dataenginex.foundation import (
    ProjectId,
    RunId,
    ServiceState,
    can_service_transition,
    is_service_terminal,
    utcnow,
)
from dataenginex.runtime.state import ControlStore
from dataenginex.runtime.streams import StreamConfig, StreamRecord, StreamRunner

PROJECT = ProjectId("proj_stream")
RUN = RunId("run_stream")
TS = "2026-08-03T00:00:00+00:00"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        with s.transaction() as tx:
            tx.execute(
                "INSERT INTO installations (installation_id, name, created_at) "
                "VALUES ('inst_1', 'test', ?)",
                (TS,),
            )
            tx.execute(
                "INSERT INTO workspaces (workspace_id, installation_id, name, "
                "created_at) VALUES ('ws_1', 'inst_1', 'default', ?)",
                (TS,),
            )
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                "VALUES (?, 'ws_1', 'stream', ?)",
                (PROJECT, TS),
            )
            tx.execute(
                "INSERT INTO project_revisions (revision_id, project_id, content_hash, "
                "created_by, created_at, manifest_schema_version, status) "
                "VALUES ('rev_1', ?, 'sha256:x', 'prin_test', ?, 'dex/v1alpha1', 'published')",
                (PROJECT, TS),
            )
            # A checkpoint references the run that advanced the cursor, so a
            # stream test needs a real run rather than a bare project.
            for run in (RUN, RunId("run_after_restart")):
                tx.execute(
                    "INSERT INTO runs (run_id, project_id, revision_id, workload_name, "
                    "kind, state, trigger_type, requested_by, created_at) "
                    "VALUES (?, ?, 'rev_1', 'ingest_telemetry', 'stream', 'running', "
                    "'manual', 'prin_test', ?)",
                    (run, PROJECT, TS),
                )
        yield s


def make_config(**overrides: Any) -> StreamConfig:
    base: dict[str, Any] = {
        "stream_name": "telemetry",
        "checkpoint_interval_seconds": 30.0,
        "batch_size": 10,
        "dedup_key": ("device_id", "reading_at"),
    }
    return StreamConfig(**(base | overrides))


def reading(device: str, at: str, *, offset: int, value: float = 1.0) -> StreamRecord:
    """One sensor reading, with its source position and its event time."""
    return StreamRecord(
        key=f"{device}:{at}",
        payload={"device_id": device, "reading_at": at, "value": value},
        cursor={"offset": offset},
        event_time=datetime.fromisoformat(at),
    )


def reader(records: Sequence[StreamRecord]) -> Callable[[int], list[StreamRecord]]:
    """A source that hands out records once, then reports itself empty."""
    remaining = list(records)

    def read(limit: int) -> list[StreamRecord]:
        batch = remaining[:limit]
        del remaining[: len(batch)]
        return batch

    return read


def noop(record: StreamRecord) -> None:
    return None


# --- cursor durability (§8.7) ----------------------------------------------


def test_a_stream_with_no_history_starts_from_nothing(store: ControlStore) -> None:
    runner = StreamRunner(store, make_config(), project_id=PROJECT, run_id=RUN)
    assert runner.load_checkpoint() is None


def test_the_cursor_survives_a_restart(store: ControlStore) -> None:
    """The claim §7.11 makes: a restarted stream resumes where it stopped.

    Two runners over one store is what a restart looks like from the control
    store's side — new process, same project, same stream name.
    """
    config = make_config()
    first = StreamRunner(store, config, project_id=PROJECT, run_id=RUN)
    first.run_batch(reader([reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)]), noop)

    resumed = StreamRunner(
        store, config, project_id=PROJECT, run_id=RunId("run_after_restart")
    )
    checkpoint = resumed.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.cursor == {"offset": 1}


def test_the_cursor_is_saved_after_processing_not_before(store: ControlStore) -> None:
    """At-least-once, stated as a test.

    The processor raises on every record, so every one is dead-lettered. If
    the checkpoint were written before the work, a crash would skip records
    entirely; the surviving evidence is that nothing was counted as processed.
    """
    runner = StreamRunner(store, make_config(max_attempts=1), project_id=PROJECT, run_id=RUN)

    def explode(record: StreamRecord) -> None:
        raise ValueError("handler is broken")

    runner.run_batch(reader([reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)]), explode)

    assert runner.progress.records_processed == 0
    assert runner.progress.dead_lettered == 1


def test_one_checkpoint_row_per_stream(store: ControlStore) -> None:
    """A stale cursor is not evidence of anything, so checkpoints replace
    rather than accumulate. Keeping history would also risk resuming from an
    old position and replaying committed work."""
    runner = StreamRunner(
        store, make_config(checkpoint_interval_seconds=0), project_id=PROJECT, run_id=RUN
    )
    for offset in (1, 2, 3):
        runner.run_batch(
            reader([reading("dev-1", f"2026-08-03T10:0{offset}:00+00:00", offset=offset)]),
            noop,
        )

    rows = store.query(
        "SELECT cursor_json FROM checkpoint_records WHERE project_id = ?", (PROJECT,)
    )
    assert len(rows) == 1
    checkpoint = runner.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.cursor == {"offset": 3}


def test_the_checkpoint_interval_is_respected(store: ControlStore) -> None:
    """Checkpointing every record would be correct and far too expensive. The
    interval trades write amplification against how much gets replayed after a
    crash."""
    runner = StreamRunner(
        store, make_config(checkpoint_interval_seconds=60), project_id=PROJECT, run_id=RUN
    )
    start = utcnow()

    runner.run_batch(
        reader([reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)]), noop, now=start
    )
    assert runner.progress.checkpoints == 1  # the first is always due

    runner.run_batch(
        reader([reading("dev-2", "2026-08-03T10:00:01+00:00", offset=2)]),
        noop,
        now=start + timedelta(seconds=5),
    )
    assert runner.progress.checkpoints == 1, "checkpointed before the interval elapsed"

    runner.run_batch(
        reader([reading("dev-3", "2026-08-03T10:00:02+00:00", offset=3)]),
        noop,
        now=start + timedelta(seconds=90),
    )
    assert runner.progress.checkpoints == 2


def test_an_idle_stream_still_checkpoints_pending_progress(store: ControlStore) -> None:
    """A stream that catches up and goes quiet should not sit on an unsaved
    cursor for as long as the quiet lasts, replaying that batch on any restart
    in between."""
    runner = StreamRunner(
        store, make_config(checkpoint_interval_seconds=60), project_id=PROJECT, run_id=RUN
    )
    start = utcnow()
    runner.run_batch(
        reader([reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)]), noop, now=start
    )
    runner.run_batch(
        reader([reading("dev-2", "2026-08-03T10:00:01+00:00", offset=2)]),
        noop,
        now=start + timedelta(seconds=1),
    )
    assert runner.progress.checkpoints == 1

    # Nothing to read, but the interval has now passed.
    runner.run_batch(reader([]), noop, now=start + timedelta(seconds=120))
    assert runner.progress.checkpoints == 2
    checkpoint = runner.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.cursor == {"offset": 2}


# --- deduplication (§14.8) --------------------------------------------------


def test_a_replayed_record_is_processed_once(store: ControlStore) -> None:
    """The point of the dedup key. A reconnecting device replays what it
    already sent; without the key those readings would be counted twice, and
    at-least-once delivery would mean at-least-once *counting*."""
    seen: list[str] = []
    runner = StreamRunner(store, make_config(), project_id=PROJECT, run_id=RUN)
    duplicate = reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)

    runner.run_batch(reader([duplicate, duplicate]), lambda r: seen.append(r.key))

    assert seen == ["dev-1:2026-08-03T10:00:00+00:00"]
    assert runner.progress.duplicates_skipped == 1


def test_a_duplicate_still_advances_the_cursor(store: ControlStore) -> None:
    """Skipping the work is not skipping the position. If the cursor stalled
    on a duplicate, every restart would read it again forever."""
    runner = StreamRunner(store, make_config(), project_id=PROJECT, run_id=RUN)
    record = reading("dev-1", "2026-08-03T10:00:00+00:00", offset=7)

    runner.run_batch(reader([record, record]), noop)

    checkpoint = runner.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.cursor == {"offset": 7}


def test_without_a_dedup_key_nothing_is_deduplicated(store: ControlStore) -> None:
    """A stream that declares no key must be idempotent by nature. Inventing
    one would be a guess about which fields identify a record."""
    seen: list[str] = []
    runner = StreamRunner(store, make_config(dedup_key=()), project_id=PROJECT, run_id=RUN)
    record = reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)

    runner.run_batch(reader([record, record]), lambda r: seen.append(r.key))

    assert len(seen) == 2
    assert runner.progress.duplicates_skipped == 0


def test_a_record_missing_a_key_field_is_not_dropped(store: ControlStore) -> None:
    """Treating an unkeyable record as a duplicate would silently discard data
    whose shape changed — the failure nobody notices until a month of numbers
    is wrong."""
    seen: list[str] = []
    runner = StreamRunner(store, make_config(), project_id=PROJECT, run_id=RUN)
    malformed = StreamRecord(key="odd", payload={"value": 1.0}, cursor={"offset": 1})

    runner.run_batch(reader([malformed, malformed]), lambda r: seen.append(r.key))

    assert len(seen) == 2
    assert runner.progress.duplicates_skipped == 0


def test_the_dedup_window_is_bounded(store: ControlStore) -> None:
    """An unbounded set of seen keys is a memory leak with a long fuse on a
    stream that never ends. Falling out of the window means a record can be
    reprocessed, which is the at-least-once promise being honest."""
    seen: list[str] = []
    runner = StreamRunner(
        store, make_config(dedup_window=2, batch_size=100), project_id=PROJECT, run_id=RUN
    )
    first = reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)
    records = [
        first,
        reading("dev-2", "2026-08-03T10:00:01+00:00", offset=2),
        reading("dev-3", "2026-08-03T10:00:02+00:00", offset=3),
        first,  # evicted by now
    ]

    runner.run_batch(reader(records), lambda r: seen.append(r.key))

    assert len(seen) == 4
    assert runner.progress.duplicates_skipped == 0


# --- dead letters (§14.8) ---------------------------------------------------


def test_a_poison_record_does_not_block_the_stream(store: ControlStore) -> None:
    """The failure this exists to prevent. One record that fails
    deterministically would stall the stream forever, because the cursor
    cannot advance past it."""
    processed: list[str] = []
    runner = StreamRunner(store, make_config(max_attempts=2), project_id=PROJECT, run_id=RUN)

    def process(record: StreamRecord) -> None:
        if record.payload["device_id"] == "poison":
            raise ValueError("cannot parse")
        processed.append(record.key)

    runner.run_batch(
        reader(
            [
                reading("poison", "2026-08-03T10:00:00+00:00", offset=1),
                reading("dev-2", "2026-08-03T10:00:01+00:00", offset=2),
            ]
        ),
        process,
    )

    assert processed == ["dev-2:2026-08-03T10:00:01+00:00"]
    assert runner.progress.dead_lettered == 1
    checkpoint = runner.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.cursor == {"offset": 2}


def test_a_dead_letter_keeps_what_is_needed_to_replay_it(store: ControlStore) -> None:
    """Storing the payload without the cursor would leave a record nobody can
    place back in the source."""
    runner = StreamRunner(store, make_config(max_attempts=1), project_id=PROJECT, run_id=RUN)

    def explode(record: StreamRecord) -> None:
        raise ValueError("cannot parse")

    runner.run_batch(reader([reading("dev-1", "2026-08-03T10:00:00+00:00", offset=4)]), explode)

    parked = runner.dead_letters()
    assert len(parked) == 1
    assert parked[0].cursor == {"offset": 4}
    assert parked[0].payload["device_id"] == "dev-1"
    assert parked[0].attempts == 1
    assert "cannot parse" in parked[0].error
    # Event time is kept separately from when we gave up (§8.8).
    assert parked[0].event_time == datetime.fromisoformat("2026-08-03T10:00:00+00:00")
    assert parked[0].event_time != parked[0].created_at


def test_retries_are_exhausted_before_giving_up(store: ControlStore) -> None:
    """A transient failure should not cost a record. Retrying until the budget
    runs out is the difference between a blip and data loss."""
    attempts: list[int] = []
    runner = StreamRunner(store, make_config(max_attempts=3), project_id=PROJECT, run_id=RUN)

    def flaky(record: StreamRecord) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("source is slow")

    runner.run_batch(reader([reading("dev-1", "2026-08-03T10:00:00+00:00", offset=1)]), flaky)

    assert len(attempts) == 3
    assert runner.progress.records_processed == 1
    assert runner.dead_letters() == []


# --- backpressure (§14.8) ---------------------------------------------------


def test_a_slow_cycle_shrinks_the_batch(store: ControlStore) -> None:
    """Falling behind means the batch is too big for this machine. Reading
    more per cycle would make it worse."""
    runner = StreamRunner(store, make_config(batch_size=100), project_id=PROJECT, run_id=RUN)
    assert runner.apply_backpressure(elapsed_seconds=20, interval_seconds=10) == 50
    assert runner.apply_backpressure(elapsed_seconds=20, interval_seconds=10) == 25


def test_recovery_is_gradual(store: ControlStore) -> None:
    """Jumping straight back to the ceiling after one fast cycle just
    oscillates between overloaded and idle."""
    runner = StreamRunner(store, make_config(batch_size=100), project_id=PROJECT, run_id=RUN)
    runner.apply_backpressure(elapsed_seconds=20, interval_seconds=10)  # -> 50

    recovered = runner.apply_backpressure(elapsed_seconds=1, interval_seconds=10)
    assert 50 < recovered < 100


def test_the_batch_never_shrinks_to_nothing(store: ControlStore) -> None:
    """A batch size of zero is a stream that has stopped, which is not what
    backpressure is for."""
    runner = StreamRunner(
        store, make_config(batch_size=4, min_batch_size=1), project_id=PROJECT, run_id=RUN
    )
    for _ in range(10):
        runner.apply_backpressure(elapsed_seconds=99, interval_seconds=1)
    assert runner.progress.current_batch_size == 1


def test_the_batch_never_exceeds_the_declared_ceiling(store: ControlStore) -> None:
    """The configured size is a limit the project set, not a starting point to
    grow past."""
    runner = StreamRunner(store, make_config(batch_size=10), project_id=PROJECT, run_id=RUN)
    for _ in range(20):
        runner.apply_backpressure(elapsed_seconds=0.1, interval_seconds=10)
    assert runner.progress.current_batch_size == 10


# --- service lifecycle (§7.4) -----------------------------------------------


def test_a_service_is_not_healthy_until_it_has_started() -> None:
    """"Started" and "working" are different claims. Only a health check can
    make the second one, so STARTING is a state rather than a formality."""
    assert can_service_transition(ServiceState.STARTING, ServiceState.HEALTHY)
    assert can_service_transition(ServiceState.STARTING, ServiceState.DEGRADED)


def test_a_degraded_service_is_not_a_failed_one() -> None:
    """A stream whose source is briefly unreachable is degraded, not broken.
    It can recover on its own, and forcing it through a restart would redo
    work that did not need redoing."""
    assert can_service_transition(ServiceState.HEALTHY, ServiceState.DEGRADED)
    assert can_service_transition(ServiceState.DEGRADED, ServiceState.HEALTHY)
    assert not is_service_terminal(ServiceState.DEGRADED)


def test_a_restart_is_not_a_recovery() -> None:
    """RESTARTING cannot jump to HEALTHY. Claiming health without observing it
    is how a crash loop reports itself as fine."""
    assert not can_service_transition(ServiceState.RESTARTING, ServiceState.HEALTHY)
    assert can_service_transition(ServiceState.RESTARTING, ServiceState.STARTING)


def test_a_paused_service_resumes_through_starting() -> None:
    """Same reason as a restart: whatever made it pausable may have changed
    while it was paused."""
    assert can_service_transition(ServiceState.HEALTHY, ServiceState.PAUSED)
    assert can_service_transition(ServiceState.PAUSED, ServiceState.STARTING)
    assert not can_service_transition(ServiceState.PAUSED, ServiceState.HEALTHY)


def test_stopped_is_the_end() -> None:
    """A stopped service that should run again is a new run, so its cursor and
    history stay attributable to the run that produced them."""
    assert is_service_terminal(ServiceState.STOPPED)
    for state in ServiceState:
        assert not can_service_transition(ServiceState.STOPPED, state)


def test_every_service_state_can_be_stopped() -> None:
    """An operator has to be able to stop a service from wherever it is. A
    state that cannot be stopped is a service that has to be killed, and a
    killed service leaves its cursor wherever it happened to be."""
    for state in ServiceState:
        if state is ServiceState.STOPPED:
            continue
        assert can_service_transition(state, ServiceState.STOPPED), state


def test_every_service_state_has_transition_rules() -> None:
    """A state missing from the table would raise KeyError the first time a
    real service reached it — at runtime, in production, on the path that
    handles failure."""
    for state in ServiceState:
        can_service_transition(state, ServiceState.STOPPED)
