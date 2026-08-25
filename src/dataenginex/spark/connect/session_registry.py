"""Spark session registry (§20.4).

Keys sessions by installation/project/workload constraints. Compatible
sessions may be reused for efficiency, but session reuse must not broaden
catalog, credential, or policy scope.
"""

from __future__ import annotations

from typing import Any

from dataenginex.foundation.ids import ProjectId, RunId

__all__ = ["SparkSessionRegistry"]


class SparkSessionRegistry:
    """Project-scoped Spark session registry (§20.4).

    Session reuse is an optimization, never an authorization shortcut.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def register(
        self,
        project_id: ProjectId,
        run_id: RunId,
        config: dict[str, Any],
    ) -> str:
        """Register a new session for a project/run."""
        session_key = f"{project_id}:{run_id}"
        self._sessions[session_key] = {
            "project_id": project_id,
            "run_id": run_id,
            "config": config,
        }
        return session_key

    def get(self, project_id: ProjectId, run_id: RunId) -> dict[str, Any] | None:
        """Get session config for a project/run."""
        key = f"{project_id}:{run_id}"
        return self._sessions.get(key)

    def unregister(self, project_id: ProjectId, run_id: RunId) -> None:
        """Remove a session from the registry."""
        key = f"{project_id}:{run_id}"
        self._sessions.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all registered sessions."""
        return list(self._sessions.values())
