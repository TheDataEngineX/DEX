"""Runtime monitoring and health projection (§7.2).

Provides worker health, stream health, and resource usage tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dataenginex.foundation.projects import utcnow
from dataenginex.runtime.state import ControlStore

__all__ = ["HealthMonitor"]


class HealthMonitor:
    """Monitors worker and stream health (§7.2)."""

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    def worker_health(self) -> list[dict[str, Any]]:
        """Get health status of all active workers."""
        rows = self.store.query(
            "SELECT worker_id, last_heartbeat_at "
            "FROM workers WHERE state = 'running'"
        )
        now = utcnow()
        result = []
        for row in rows:
            last_heartbeat = row["last_heartbeat_at"]
            heartbeat = (
                datetime.fromisoformat(last_heartbeat)
                if last_heartbeat
                else None
            )
            is_healthy = heartbeat and (now - heartbeat).total_seconds() < 300
            result.append({
                "worker_id": row["worker_id"],
                "healthy": is_healthy,
                "last_heartbeat": row["last_heartbeat_at"],
            })
        return result

    def stream_health(self) -> list[dict[str, Any]]:
        """Get health status of all active streams."""
        rows = self.store.query(
            "SELECT run_id, workload_name, state, started_at "
            "FROM runs WHERE kind = 'spark_stream' AND state = 'running'"
        )
        return [
            {
                "run_id": r["run_id"],
                "name": r["workload_name"],
                "state": r["state"],
                "started_at": r["started_at"],
            }
            for r in rows
        ]

    def resource_usage(self, project_id: str | None = None) -> dict[str, Any]:
        """Get aggregate resource usage, optionally per-project.

        Resource data is stored in ``observed_resources_json`` as a JSON blob.
        This method returns a summary based on attempt counts.
        """
        if project_id:
            rows = self.store.query(
                "SELECT COUNT(*) as attempt_count "
                "FROM attempts WHERE project_id = ? AND state = 'succeeded'",
                (project_id,),
            )
        else:
            rows = self.store.query(
                "SELECT COUNT(*) as attempt_count "
                "FROM attempts WHERE state = 'succeeded'",
                (),
            )
        row: dict[str, Any] = dict(rows[0]) if rows else {}
        return {
            "attempt_count": row.get("attempt_count") or 0,
        }
