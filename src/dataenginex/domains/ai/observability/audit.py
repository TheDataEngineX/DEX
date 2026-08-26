"""Audit logging — track every agent action for compliance and debugging."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AuditEntry(BaseModel):
    """A single audit log entry."""

    agent_name: str
    action: str
    input: str
    output: str
    timestamp: float
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (compatible with dex-studio expectations)."""
        return self.model_dump()


class AuditLog:
    """In-memory audit log for agent actions."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def log(self, entry: AuditEntry) -> None:
        """Record an audit entry."""
        self._entries.append(entry)

    @property
    def all_events(self) -> list[AuditEntry]:
        """All audit entries, newest first."""
        return list(reversed(self._entries))

    def get_entries(self, agent_name: str | None = None, limit: int = 100) -> list[AuditEntry]:
        """Get audit entries, optionally filtered by agent name."""
        if agent_name is None:
            return self._entries[-limit:]
        filtered = [e for e in self._entries if e.agent_name == agent_name]
        return filtered[-limit:]
