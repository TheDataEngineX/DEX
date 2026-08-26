"""Agent state checkpointing — save and restore agent state between runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Checkpoint(BaseModel):
    """Snapshot of agent state at a point in time."""

    agent_name: str
    state: dict[str, Any]
    timestamp: float
    iteration: int


class CheckpointManager:
    """Manages agent checkpoints — in-memory dict storage."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint, keyed by agent name."""
        self._checkpoints[checkpoint.agent_name] = checkpoint

    def load(self, agent_name: str) -> Checkpoint | None:
        """Load the latest checkpoint for an agent, or None if not found."""
        return self._checkpoints.get(agent_name)
