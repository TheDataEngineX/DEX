"""``dex runtime`` — the control-plane daemon (§5.6, §7.5).

This is what replaced Studio's background scheduler. That loop ran inside the
web process and executed pipelines on its own threads, so cron only fired while
someone had the UI running, and a workload could take the request handler down
with it. §17 Phase 1 rules that out: *"No workload executes in Studio process."*

The daemon does three things on a tick, none of which is running a workload:

1. Fire schedules that are due, which creates queued runs.
2. Reclaim leases whose worker stopped heartbeating, and requeue what may still
   be retried (§7.6).
3. Sleep until the next tick.

Workers pick the queued runs up. Killing this process stops *scheduling*, not
execution — the separation that lets the control plane restart without losing
in-flight work (§14.5).
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from types import FrameType

import click
import structlog

from dataenginex.bootstrap import build_lite_gateway, open_control_store
from dataenginex.bootstrap.settings import Settings
from dataenginex.foundation import PrincipalId, ProjectId
from dataenginex.interfaces.embedded import EmbeddedGateway
from dataenginex.interfaces.gateway import Command
from dataenginex.runtime.queue import DurableQueue, Scheduler

log = structlog.get_logger().bind(src="runtime")

# The daemon acts as itself. A scheduled run is not attributable to a human, and
# §4.15 wants a named actor rather than a blank one.
_DAEMON_PRINCIPAL = PrincipalId("prin_scheduler")

_STATE_DIR_HELP = "Where the control store lives (default: $DEX_STATE_DIR or .dex)."

__all__ = ["runtime"]


@click.group()
def runtime() -> None:
    """Run and inspect the control plane."""


@runtime.command()
@click.option("--state-dir", type=click.Path(path_type=Path), default=None, help=_STATE_DIR_HELP)
@click.option("--interval", default=10.0, show_default=True, help="Seconds between ticks.")
@click.option("--limit", default=50, show_default=True, help="Max schedules fired per tick.")
@click.option("--once", is_flag=True, help="Run a single tick and exit.")
def serve(state_dir: Path | None, interval: float, limit: int, once: bool) -> None:
    """Fire due schedules and reclaim lost leases, forever (§7.5, §7.6).

    ``--once`` exists for cron-driven deployments and for tests: the same code
    path, one iteration. A daemon whose single-shot mode took a different route
    would be a daemon nobody could test.
    """
    store = open_control_store(Settings.from_env(state_dir=state_dir))
    gateway = build_lite_gateway(store)
    scheduler = Scheduler(store, DurableQueue(store))

    stopping = False

    def _stop(signum: int, _frame: FrameType | None) -> None:
        # Finish the tick in progress rather than dying inside it. A SIGTERM
        # between claiming a schedule and enqueueing its run would otherwise
        # leave the schedule advanced with nothing queued.
        nonlocal stopping
        stopping = True
        log.info("stop requested, finishing current tick", signal=signum)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log.info("control plane started", interval=interval)
    try:
        while not stopping:
            _tick(gateway, scheduler, limit=limit)
            if once:
                break
            # Sleep in short slices so a signal is noticed promptly rather than
            # after a full interval.
            slept = 0.0
            while slept < interval and not stopping:
                time.sleep(min(0.5, interval - slept))
                slept += 0.5
    finally:
        store.close()
        log.info("control plane stopped")


def _tick(gateway: EmbeddedGateway, scheduler: Scheduler, *, limit: int) -> None:
    """One pass. Never raises — a daemon that dies on a bad tick is not one.

    Each half is guarded separately so a project with a broken schedule cannot
    stop lease reclaim, which is what keeps a crashed worker's run recoverable.
    """
    try:
        result = gateway.tick_schedules(Command(principal_id=_DAEMON_PRINCIPAL), limit=limit)
        if result.message:
            log.debug("schedules ticked", detail=result.message)
    except Exception as exc:  # noqa: BLE001 — the loop must outlive one bad project
        log.error("schedule tick failed", error=str(exc))

    try:
        reclaimed, requeued = scheduler.reclaim_and_retry()
        if reclaimed:
            log.info("leases reclaimed", reclaimed=reclaimed, requeued=requeued)
    except Exception as exc:  # noqa: BLE001
        log.error("lease reclaim failed", error=str(exc))


@runtime.command(name="tick")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None, help=_STATE_DIR_HELP)
@click.option("--limit", default=50, show_default=True)
def tick_once(state_dir: Path | None, limit: int) -> None:
    """Fire due schedules once and report what happened.

    For an operator asking "why has this not run?" — it answers with the
    schedules that fired and the ones that were refused, rather than making them
    read the daemon's logs.
    """
    store = open_control_store(Settings.from_env(state_dir=state_dir))
    try:
        fired = build_lite_gateway(store).schedules.tick(limit=limit)
        if not fired:
            click.echo("Nothing due.")
            return
        for item in fired:
            if item.error:
                click.echo(f"  x {item.workload_name}: {item.error}", err=True)
            else:
                click.echo(f"  - {item.workload_name} -> {item.run_id}")
        click.echo(f"\n{len(fired)} schedule(s) fired.")
    finally:
        store.close()


@runtime.command(name="schedules")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None, help=_STATE_DIR_HELP)
@click.option("--project", required=True, help="Project id.")
def list_schedules(state_dir: Path | None, project: str) -> None:
    """Show every schedule for a project and when it next fires."""
    store = open_control_store(Settings.from_env(state_dir=state_dir))
    try:
        rows = build_lite_gateway(store).schedules.list_for_project(ProjectId(project))
        if not rows:
            click.echo("No schedules.")
            return
        width = max(len(r.workload_name) for r in rows) + 2
        for row in rows:
            state = "enabled" if row.enabled else "paused"
            nxt = row.next_fire_at.isoformat() if row.next_fire_at else "(unplanned)"
            click.echo(f"  {row.workload_name:<{width}}{row.cron:<16}{state:<10}{nxt}")
    finally:
        store.close()
