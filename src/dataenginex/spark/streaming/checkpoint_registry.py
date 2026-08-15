"""Streaming checkpoint registry (§20.7).

Manages checkpoint state for streaming queries.
"""

from __future__ import annotations

__all__ = ["CheckpointRegistry"]


class CheckpointRegistry:
    """Manages checkpoint state for streaming queries (§20.7)."""

    def __init__(self, base_path: str | None = None) -> None:
        self.base_path = base_path or "/tmp/checkpoints"
        self._checkpoints: dict[str, str] = {}

    def get_checkpoint_path(self, query_id: str) -> str:
        return f"{self.base_path}/{query_id}"

    def register(self, query_id: str, path: str) -> None:
        self._checkpoints[query_id] = path

    def exists(self, query_id: str) -> bool:
        return query_id in self._checkpoints
