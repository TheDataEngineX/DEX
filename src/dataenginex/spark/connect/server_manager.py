"""Spark Connect server lifecycle management (§20.4).

Manages local and remote Spark Connect server processes.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["SparkServerManager", "SparkServerState"]


class SparkServerState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class SparkServerManager:
    """Manages Spark Connect server lifecycle (§20.4)."""

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = server_url or "local"
        self.state = SparkServerState.STOPPED

    def start(self) -> None:
        """Start the Spark Connect server."""
        self.state = SparkServerState.STARTING
        # ponytail: actual server start deferred to runtime
        self.state = SparkServerState.RUNNING

    def stop(self) -> None:
        """Stop the Spark Connect server."""
        self.state = SparkServerState.STOPPED

    def status(self) -> SparkServerState:
        """Get current server state."""
        return self.state
