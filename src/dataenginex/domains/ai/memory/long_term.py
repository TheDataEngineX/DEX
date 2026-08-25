"""Long-term memory — keyword-searchable persistent memory store."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from dataenginex import _json
from dataenginex.domains.ai.memory.base import BaseMemory, MemoryEntry


class LongTermMemory(BaseMemory):
    """Persistent memory with keyword search — no external vector DB required.

    Data is stored as a flat list and scored by keyword overlap.
    Call :meth:`persist` to write to disk and :meth:`load_from_file` to restore.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        if not entry.timestamp:
            entry.timestamp = time.time()
        self._entries.append(entry)

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        query_lower = query.lower()
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self._entries:
            score = sum(1 for word in query_lower.split() if word in entry.content.lower())
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        return self._entries[-n:]

    def clear(self) -> None:
        self._entries.clear()

    def persist(self, path: str) -> None:
        """Persist all memory entries to a JSON file at *path*."""
        data = [asdict(e) for e in self._entries]
        Path(path).write_text(_json.dumps(data, indent=2), encoding="utf-8")

    def load_from_file(self, path: str) -> None:
        """Replace in-memory entries with those from a JSON file at *path*."""
        raw: list[dict[str, object]] = _json.loads(Path(path).read_text(encoding="utf-8"))
        self._entries = [MemoryEntry(**item) for item in raw]  # type: ignore[arg-type]
