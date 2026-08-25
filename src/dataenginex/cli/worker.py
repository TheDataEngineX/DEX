"""``dex worker`` — the process that actually runs workloads (§5.6, §7.7).

The control plane decides *what* should run; a worker is what runs it. Doc line
492 names the role directly: ``dex worker start --pool batch``. Until this
existed the queue accepted runs and nothing ever picked them up — Studio had
correctly stopped executing work, and nothing had taken over.

The loop is deliberately dull:

1. Ask the scheduler for work this pool may take (§7.5 decides fairness,
   priority, quiet hours, and admission — the worker does not).
2. Heartbeat the lease while executing, so a crash is detectable (§7.6).
3. Execute through an execution backend, never inline logic of its own.
4. Complete with the commit token, which the control plane may refuse.

Point 4 is the one that matters under failure. If this worker stalls long enough
for its lease to expire, the control plane reclaims the attempt and may start a
new one; when the stalled process wakes and finishes, its token no longer
matches and its result is rejected. That is what stops a late completion from
overwriting a newer attempt's output (§14.3).
"""

from __future__ import annotations

import os
import signal
import socket
import threading
from pathlib import Path
from types import FrameType

import click
import structlog

from dataenginex.bootstrap import open_control_store
from dataenginex.bootstrap.lite import build_lite_backend
from dataenginex.bootstrap.settings import Settings
from dataenginex.domains.execution.backends import BackendError, InProcessBackend
from dataenginex.domains.governance.lineage import LineageService
from dataenginex.foundation import ExecutionPlan, WorkloadKind
from dataenginex.runtime.planning.planner import plan_attempt
from dataenginex.runtime.queue import ClaimedWork, DurableQueue, Scheduler
from dataenginex.runtime.state import ControlStore

log = structlog.get_logger().bind(src="worker")

__all__ = ["worker"]

_STATE_DIR_HELP = "Where the control store lives (default: $DEX_STATE_DIR or .dex)."

# Which workload kinds each pool will accept. The names come from §7.7's logical
# pools. A pool is a compatibility statement, not a promise of a dedicated
# process — a small installation runs one worker serving everything.
_POOLS: dict[str, tuple[WorkloadKind, ...]] = {
    "batch": (WorkloadKind.BATCH,),
    "interactive": (WorkloadKind.INTERACTIVE,),
    "stream": (WorkloadKind.SPARK_STREAM,),
    "service": (WorkloadKind.SERVICE,),
    "all": (),
}


@click.group()
def worker() -> None:
    """Run a worker process."""


@worker.command()
@click.option("--pool", type=click.Choice(sorted(_POOLS)), default="batch", show_default=True)
@click.option("--state-dir", help=_STATE_DIR_HELP)
@click.option("--worker-id", default="", help="Stable ID. Defaults to host:pid.")
@click.option("--poll", default=1.0, show_default=True, help="Seconds between empty polls.")
@click.option("--once", is_flag=True, help="Take at most one unit of work, then exit.")
def start(pool: str, state_dir: str | None, worker_id: str, poll: float, once: bool) -> None:
    """Claim queued runs and execute them until stopped."""
    settings = Settings.from_env(state_dir=Path(state_dir) if state_dir else None)
    store = open_control_store(settings)
    queue = DurableQueue(store)
    scheduler = Scheduler(store, queue)
    # The store, so interactive runs have somewhere to put their results. A
    # worker without it can still run batch work but refuses previews.
    backend = build_lite_backend(store)

    # Host and pid, so an operator reading the workers table can find the
    # process. A random ID would be reclaimable but not diagnosable.
    identity = worker_id or f"{socket.gethostname()}:{os.getpid()}"
    queue.register_worker(identity, pool=pool, hostname=socket.gethostname(), pid=os.getpid())
    kinds = _POOLS[pool]

    stopping = threading.Event()

    def _stop(_signum: int, _frame: FrameType | None) -> None:
        # Finish the unit in flight rather than dying inside it. Dropping work
        # mid-execution would leave the lease to expire on its own, delaying
        # the retry by the full lease duration for no reason.
        stopping.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log.info("worker started", worker_id=identity, pool=pool, kinds=[k.value for k in kinds])
    try:
        while not stopping.is_set():
            claimed = scheduler.dispatch(identity, kinds=kinds)
            if claimed is None:
                if once:
                    break
                stopping.wait(poll)
                continue

            _execute(store, queue, backend, claimed, worker_id=identity)
            if once:
                break
    finally:
        store.close()
        log.info("worker stopped", worker_id=identity)


def _execute(
    store: ControlStore,
    queue: DurableQueue,
    backend: InProcessBackend,
    claimed: ClaimedWork,
    *,
    worker_id: str,
) -> None:
    """Run one claimed unit, heartbeating until it finishes.

    The heartbeat runs on its own thread because execution blocks. Without it a
    workload that legitimately takes longer than one lease period would be
    reclaimed as dead and run twice — the failure mode that makes at-least-once
    delivery feel like a bug rather than a guarantee.
    """
    done = threading.Event()
    lost = threading.Event()

    def _beat() -> None:
        while not done.wait(_heartbeat_interval(queue)):
            if not queue.heartbeat(claimed.attempt_id, worker_id):
                # The lease is no longer ours: the control plane already
                # decided this attempt was lost. Stop claiming to hold it.
                lost.set()
                return

    beater = threading.Thread(target=_beat, name="worker-heartbeat", daemon=True)
    beater.start()

    succeeded = False
    error: str | None = None
    plan: ExecutionPlan | None = None
    try:
        log.info(
            "executing",
            run_id=claimed.run_id,
            attempt=claimed.attempt_number,
            kind=claimed.workload_kind.value,
        )
        plan = _run_plan(store, backend, claimed)
        succeeded = True
    except Exception as exc:  # noqa: BLE001 — a workload failing must not kill the worker
        error = str(exc)
        log.warning("execution failed", run_id=claimed.run_id, error=error)
    finally:
        done.set()
        beater.join(timeout=5)

    if lost.is_set():
        # Reporting a result we are no longer authorized to report would be
        # exactly the overwrite the commit token exists to prevent.
        log.warning("lease lost during execution; result discarded", run_id=claimed.run_id)
        return

    accepted = queue.complete(
        claimed.attempt_id,
        claimed.commit_token,
        succeeded=succeeded,
        error=error,
        error_class=None if succeeded else "execution_error",
    )
    if not accepted:
        log.warning("completion fenced — a newer attempt owns this run", run_id=claimed.run_id)
        return

    if plan is not None:
        # Only after the commit was accepted. A fenced attempt's output is
        # discarded, and lineage claiming it produced something would describe
        # a result no reader can see.
        _record_lineage(store, plan, claimed)


def _heartbeat_interval(queue: DurableQueue) -> float:
    """Beat at a third of the lease, so two may be missed before expiry."""
    return max(1.0, queue.lease_duration.total_seconds() / 3)


def _run_plan(
    store: ControlStore, backend: InProcessBackend, claimed: ClaimedWork
) -> ExecutionPlan:
    """Resolve the attempt into a plan, execute it, and return the plan.

    The plan comes back because the caller records lineage from it, and only
    once the completion has been accepted.

    The plan is built from the *published revision* the run pinned, never from
    whatever ``dex.yaml`` says now — editing a project after queueing a run must
    not change what that run does (§17 Phase 1).
    """
    plan, context = plan_attempt(store, claimed.attempt_id)
    result = backend.execute(plan, context)
    if not result.succeeded:
        raise BackendError(result.error or "execution failed without an error message")
    return plan


def _record_lineage(store: ControlStore, plan: ExecutionPlan, claimed: ClaimedWork) -> None:
    """Link what this run read to what it wrote (§8.5).

    ``LineageService`` had no caller, so ``lineage_edges`` was never written and
    every lineage view rendered an empty graph — the same defect the old design
    had, where ``parent_id`` was never set. Recording here, after a successful
    execution, is what makes the table mean anything.

    Failure to record does not fail the run. The work is done and committed, and
    raising here would turn a gap in the graph into lost output.
    """
    consumed = [name for op in plan.operations for name in op.bound_inputs]
    produced = [name for op in plan.operations for name in op.bound_outputs]
    if not consumed and not produced:
        return

    try:
        LineageService(store).record_run(
            run_id=claimed.run_id,
            project_id=plan.project_id,
            revision_id=plan.revision_id,
            consumed=consumed,
            produced=produced,
            attributes={
                "attempt_id": plan.attempt_id,
                "workload": plan.parameters.get("workload", ""),
            },
        )
    except Exception:
        log.exception("failed to record lineage", run_id=claimed.run_id)


@worker.command(name="list")
@click.option("--state-dir", help=_STATE_DIR_HELP)
def list_workers(state_dir: str | None) -> None:
    """Show registered workers and when they were last seen."""
    settings = Settings.from_env(state_dir=Path(state_dir) if state_dir else None)
    store = open_control_store(settings)
    try:
        rows = store.query(
            "SELECT worker_id, pool, state, last_heartbeat_at FROM workers ORDER BY worker_id"
        )
        if not rows:
            click.echo("no workers registered")
            return
        for row in rows:
            click.echo(
                f"  {row['worker_id']:<28} pool={row['pool']:<12} "
                f"state={row['state']:<8} last_seen={row['last_heartbeat_at']}"
            )
    finally:
        store.close()
