"""Agent memory system — short-term, long-term, and episodic memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry."""

    content: str
    role: str  # "user", "assistant", "system", "tool"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class BaseMemory(ABC):
    """Abstract base class for agent memory."""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search memory by semantic similarity."""

    @abstractmethod
    def recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get the most recent entries."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""


class ShortTermMemory(BaseMemory):
    """Session/conversation memory — lives in-process, lost on restart."""

    def __init__(self, max_entries: int = 100) -> None:
        self._entries: list[MemoryEntry] = []
        self._max = max_entries

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries.pop(0)

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return [e for e in self._entries if query.lower() in e.content.lower()][:top_k]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        return self._entries[-n:]

    def clear(self) -> None:
        self._entries.clear()
