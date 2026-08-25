"""Storing what an interactive run produced (§7.3, §13.8).

The counterpart to ``handlers``' ``ResultSink``. A handler produces rows; this
writes them where the asking process can read them back.

Why persist at all, rather than returning rows to the caller directly: the
worker and the web process are different processes, and in a distributed profile
different machines. §13.8 settles it — "persisted state remains queryable through
ordinary APIs" — so the result goes to the control store and the browser reads it
back, which also survives the user reloading the page mid-query.

Why not an artifact: §7.3 says interactive results "may be ephemeral until
explicitly saved". An artifact is content-addressed and retained on purpose;
these expire, and a store that promises permanence is the wrong home for
something designed to be thrown away.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from dataenginex.foundation import ExecutionContext, ProjectId, RunId, utcnow
from dataenginex.runtime.state import ControlStore

__all__ = [
    "DEFAULT_TTL",
    "MAX_PAYLOAD_BYTES",
    "ResultNotStorable",
    "ResultTooLarge",
    "purge_expired_results",
    "store_interactive_result",
]

# How long a preview stays readable. Long enough to survive a slow page load or
# a user switching tabs, short enough that the table does not become a cache
# nobody prunes.
DEFAULT_TTL = timedelta(minutes=15)

# Hard ceiling on one stored result. The row cap alone is not enough — a
# thousand rows of wide text is a different size from a thousand rows of
# integers, and the control store is not a data store.
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


class ResultNotStorable(RuntimeError):
    """The result cannot be attached to a run.

    Distinct from the run failing: the work was done, and only the recording of
    it is impossible. Surfaced rather than dropped, since a caller waiting on a
    result would otherwise wait for one that will never arrive.
    """


class ResultTooLarge(ResultNotStorable):
    """The result exceeded what the control store will hold.

    Raised rather than truncated, because a silently shortened result is
    indistinguishable from a complete one at the point where it is displayed.
    """


def store_interactive_result(
    store: ControlStore,
    context: ExecutionContext,
    payload: dict[str, Any],
    row_count: int,
    truncated: bool,
    *,
    ttl: timedelta = DEFAULT_TTL,
) -> None:
    """Record an interactive run's output, replacing any previous one.

    The run id comes from the context's capability token, which is the only
    identifier a handler is given — and the right one, since it is scoped to
    exactly this attempt.

    Upsert rather than insert: a retried attempt of the same run produces a
    second result for one run_id, and the newest is the one that counts.
    """
    run_id = context.capability.run_id
    if run_id is None:
        # The token was minted without a run, so this handler is executing
        # outside an attempt. Storing under a fabricated id would attach the
        # result to a run that does not exist.
        raise ResultNotStorable(
            "interactive result has no run to belong to; the capability token carries no run_id"
        )

    encoded = json.dumps(payload)
    if len(encoded.encode()) > MAX_PAYLOAD_BYTES:
        raise ResultTooLarge(
            f"interactive result is larger than {MAX_PAYLOAD_BYTES // (1024 * 1024)}MiB; "
            "narrow the query or lower the row limit"
        )

    now = utcnow()
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO interactive_results (run_id, project_id, payload_json, "
            "row_count, truncated, created_at, expires_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id) DO UPDATE SET payload_json = excluded.payload_json, "
            "row_count = excluded.row_count, truncated = excluded.truncated, "
            "created_at = excluded.created_at, expires_at = excluded.expires_at",
            (
                RunId(run_id),
                ProjectId(context.capability.project_id),
                encoded,
                row_count,
                int(truncated),
                now.isoformat(),
                (now + ttl).isoformat(),
            ),
        )


def purge_expired_results(store: ControlStore) -> int:
    """Delete results past their expiry, returning how many went.

    Housekeeping, not correctness: reads already filter on ``expires_at``, so an
    installation that never calls this serves correct results and merely keeps a
    larger table.
    """
    with store.transaction() as tx:
        cursor = tx.execute(
            "DELETE FROM interactive_results WHERE expires_at <= ?", (utcnow().isoformat(),)
        )
        return int(cursor.rowcount or 0)
