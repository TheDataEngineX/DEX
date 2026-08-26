"""Schedule service — cron-driven run requests (§7.5).

The ``schedules`` table has always carried ``next_fire_at`` and an index on
``(enabled, next_fire_at)``. Nothing read it. Cron lived in the Studio process
instead, where a tick both decided a pipeline was due *and* executed it inline —
the arrangement §17 Phase 1 rules out with *"no workload runs in the Studio
process"*.

This service only decides. A due schedule becomes a ``request_run``, which
authorizes, pins the active revision, and enqueues; a worker picks the run up.
There is no execution path here at all, so a Studio that never starts a daemon
still gets its scheduled runs, and a control plane that dies mid-tick loses
nothing that was not already durable.

**Firing is a claim, not a read.** ``next_fire_at`` advances in the same
transaction that selected the row, conditional on it not having moved. Two
control planes racing on one schedule therefore produce one run, not two — the
alternative is at-least-once cron, which for a billing job means charging twice.
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import CroniterBadCronError, croniter

from dataenginex.application.runs import RunService
from dataenginex.application.services import ApplicationError, Service
from dataenginex.foundation import (
    FrozenModel,
    PrincipalId,
    ProjectId,
    RunId,
    new_id,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

__all__ = ["ScheduleFired", "ScheduleService", "ScheduleView"]

# The principal a cron fire acts as. A run with an empty actor is unattributable,
# and §4.15 requires every audited action to name one. "The scheduler did it" is
# a weaker answer than a named human, but a far better one than a blank column.
_SCHEDULER_PRINCIPAL = PrincipalId("prin_scheduler")


class ScheduleView(FrozenModel):
    """A schedule and the workload it fires."""

    schedule_id: str
    project_id: ProjectId
    workload_id: str
    workload_name: str
    cron: str
    timezone: str = "UTC"
    enabled: bool = True
    next_fire_at: datetime | None = None
    last_fired_at: datetime | None = None


class ScheduleFired(FrozenModel):
    """One schedule that fired, and what it produced.

    ``run_id`` is ``None`` when the fire was claimed but the run was refused —
    policy denial, or no published revision. The schedule still advanced,
    because a schedule that retries a denied run every tick becomes a denial
    loop that fills the audit trail without ever succeeding.
    """

    schedule_id: str
    workload_name: str
    run_id: RunId | None = None
    error: str | None = None


class ScheduleService(Service):
    """Decides which schedules are due and turns them into run requests."""

    def __init__(self, store: ControlStore, *, runs: RunService | None = None) -> None:
        super().__init__(store)
        self.runs = runs or RunService(store)

    # --- commands -----------------------------------------------------------

    def create(
        self,
        project_id: ProjectId,
        workload_name: str,
        *,
        cron: str,
        timezone: str = "UTC",
        enabled: bool = True,
        now: datetime | None = None,
    ) -> ScheduleView:
        """Attach a cron schedule to a workload of the published revision."""
        moment = now or utcnow()
        revision = self.active_revision(project_id)
        row = self.require_row(
            "SELECT workload_id, name FROM workload_definitions "
            "WHERE revision_id = ? AND name = ?",
            (revision, workload_name),
            subject=f"no workload {workload_name!r} in the published revision",
        )
        next_fire = _next_fire(cron, moment, _zone(timezone))

        schedule_id = new_id("sched")
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO schedules (schedule_id, project_id, workload_id, cron, "
                "timezone, enabled, next_fire_at, last_fired_at) VALUES (?,?,?,?,?,?,?,NULL)",
                (
                    schedule_id,
                    project_id,
                    row["workload_id"],
                    cron,
                    timezone,
                    int(enabled),
                    next_fire.isoformat(),
                ),
            )
        return ScheduleView(
            schedule_id=schedule_id,
            project_id=project_id,
            workload_id=str(row["workload_id"]),
            workload_name=str(row["name"]),
            cron=cron,
            timezone=timezone,
            enabled=enabled,
            next_fire_at=next_fire,
        )

    def set_enabled(self, schedule_id: str, *, enabled: bool, now: datetime | None = None) -> None:
        """Pause or resume a schedule.

        Resuming re-bases ``next_fire_at`` to the next occurrence from now
        rather than replaying what was missed. A catch-up stampede is how a
        paused nightly job takes the warehouse down on the morning someone
        turns it back on.
        """
        # Resolved before either branch. Disabling used to UPDATE blind, so
        # pausing a schedule that had been deleted — or a typo'd id — reported
        # success and changed nothing, which is the worst possible answer to
        # "is this thing off?".
        schedule = self.get(schedule_id)

        if not enabled:
            with self.store.transaction() as tx:
                tx.execute(
                    "UPDATE schedules SET enabled = 0 WHERE schedule_id = ?", (schedule_id,)
                )
            return

        upcoming = _next_fire(schedule.cron, now or utcnow(), _zone(schedule.timezone))
        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE schedules SET enabled = 1, next_fire_at = ? WHERE schedule_id = ?",
                (upcoming.isoformat(), schedule_id),
            )

    def delete(self, schedule_id: str) -> None:
        """Remove a schedule, or raise if there is nothing to remove.

        Same reason as ``set_enabled``: a DELETE that matches no row is not a
        successful delete, it is a caller holding an id that means nothing.
        """
        self.get(schedule_id)
        with self.store.transaction() as tx:
            tx.execute("DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,))

    def tick(self, *, now: datetime | None = None, limit: int = 50) -> list[ScheduleFired]:
        """Fire every schedule that is due, and report what happened.

        The whole daemon is this method plus a sleep. Keeping it a plain call
        with an injectable clock is what makes cron testable without waiting on
        wall time, and lets the CLI expose a single-shot tick.
        """
        moment = now or utcnow()
        fired: list[ScheduleFired] = []
        for schedule in self._due(moment, limit):
            if not self._claim(schedule, moment):
                # Another control plane took this fire. Not an error.
                continue
            fired.append(self._request(schedule))
        return fired

    # --- queries ------------------------------------------------------------

    def list_for_project(self, project_id: ProjectId) -> list[ScheduleView]:
        rows = self.store.query(
            "SELECT s.*, w.name AS workload_name FROM schedules s "
            "JOIN workload_definitions w ON w.workload_id = s.workload_id "
            "WHERE s.project_id = ? ORDER BY w.name",
            (project_id,),
        )
        return [_row_to_view(row) for row in rows]

    def get(self, schedule_id: str) -> ScheduleView:
        row = self.require_row(
            "SELECT s.*, w.name AS workload_name FROM schedules s "
            "JOIN workload_definitions w ON w.workload_id = s.workload_id "
            "WHERE s.schedule_id = ?",
            (schedule_id,),
            subject=f"no schedule {schedule_id}",
        )
        return _row_to_view(row)

    # --- internals ----------------------------------------------------------

    def _due(self, now: datetime, limit: int) -> list[ScheduleView]:
        """Schedules whose next fire has passed.

        A NULL ``next_fire_at`` is excluded rather than read as "due now": an
        unset column means the schedule was never planned, and treating that as
        a firing turns a bookkeeping gap into an unexpected run.
        """
        rows = self.store.query(
            "SELECT s.*, w.name AS workload_name FROM schedules s "
            "JOIN workload_definitions w ON w.workload_id = s.workload_id "
            "WHERE s.enabled = 1 AND s.next_fire_at IS NOT NULL AND s.next_fire_at <= ? "
            "ORDER BY s.next_fire_at LIMIT ?",
            (now.isoformat(), limit),
        )
        return [_row_to_view(row) for row in rows]

    def _claim(self, schedule: ScheduleView, now: datetime) -> bool:
        """Advance ``next_fire_at``, but only if nobody else already has.

        The ``WHERE next_fire_at = ?`` guard is the whole concurrency story: this
        transaction updates the row it read, or it updates nothing. Whoever
        loses the race sees zero rows changed and skips the fire.
        """
        if schedule.next_fire_at is None:  # pragma: no cover - excluded by _due
            return False
        try:
            upcoming = _next_fire(schedule.cron, now, _zone(schedule.timezone))
        except ApplicationError:
            # A malformed cron would otherwise be re-read every tick forever.
            # Disabling stops the loop and leaves the bad expression visible for
            # someone to fix, rather than silently dropping their schedule.
            self.set_enabled(schedule.schedule_id, enabled=False)
            return False

        with self.store.transaction() as tx:
            cursor = tx.execute(
                "UPDATE schedules SET next_fire_at = ?, last_fired_at = ? "
                "WHERE schedule_id = ? AND next_fire_at = ?",
                (
                    upcoming.isoformat(),
                    now.isoformat(),
                    schedule.schedule_id,
                    schedule.next_fire_at.isoformat(),
                ),
            )
            return cursor.rowcount > 0

    def _request(self, schedule: ScheduleView) -> ScheduleFired:
        """Turn a claimed fire into a run request.

        The idempotency key pins the run to this occurrence, so a control plane
        that dies between claiming and enqueueing produces the same run on
        recovery rather than a second one (§13.4).
        """
        occurrence = schedule.next_fire_at.isoformat() if schedule.next_fire_at else "unplanned"
        try:
            accepted = self.runs.request_run(
                schedule.project_id,
                schedule.workload_name,
                principal_id=_SCHEDULER_PRINCIPAL,
                idempotency_key=f"sched:{schedule.schedule_id}:{occurrence}",
                trigger_type="schedule",
            )
        except ApplicationError as exc:
            # The schedule has already advanced. A denied or unpublishable
            # workload must not hold the cron loop hostage.
            return ScheduleFired(
                schedule_id=schedule.schedule_id,
                workload_name=schedule.workload_name,
                error=str(exc),
            )
        return ScheduleFired(
            schedule_id=schedule.schedule_id,
            workload_name=schedule.workload_name,
            run_id=accepted.run_id,
        )


def _zone(name: str) -> tzinfo:
    """Resolve a timezone, falling back to UTC.

    A named zone is the point of the column: ``0 6 * * *`` for a European team
    means 06:00 local across a DST boundary, not 06:00 UTC.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def _next_fire(cron: str, after: datetime, tz: tzinfo) -> datetime:
    """The next occurrence of ``cron`` strictly after ``after``, in UTC.

    Evaluated in the schedule's own zone, then converted. Doing the arithmetic
    in UTC and labelling it local is what makes a daily job drift by an hour
    twice a year.
    """
    local = after.astimezone(tz)
    try:
        upcoming: datetime = croniter(cron, local).get_next(datetime)
    except (CroniterBadCronError, ValueError, KeyError) as exc:
        raise ApplicationError(f"invalid cron expression {cron!r}: {exc}") from exc
    if upcoming.tzinfo is None:  # pragma: no cover - croniter preserves tzinfo
        upcoming = upcoming.replace(tzinfo=tz)
    return upcoming.astimezone(UTC)


def _parse_ts(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _row_to_view(row: object) -> ScheduleView:
    data = dict(row)  # type: ignore[call-overload]
    return ScheduleView(
        schedule_id=str(data["schedule_id"]),
        project_id=ProjectId(data["project_id"]),
        workload_id=str(data["workload_id"]),
        workload_name=str(data["workload_name"]),
        cron=str(data["cron"]),
        timezone=str(data["timezone"] or "UTC"),
        enabled=bool(data["enabled"]),
        next_fire_at=_parse_ts(data["next_fire_at"]),
        last_fired_at=_parse_ts(data["last_fired_at"]),
    )
