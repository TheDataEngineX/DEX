"""Lineage contract for the warehouse pipeline.

The recording protocol. The concrete stores that used to live here — a JSON
file and a Postgres table — are gone: pipeline lineage is written through
``DexStore`` (see ``engine.py``, ``lineage=self.store``), and provenance
proper is the control store's job (§8.5).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger()
__all__ = [
    "LineageBackend",
]


@runtime_checkable
class LineageBackend(Protocol):
    """Structural interface for lineage recording.

    Any object with a ``record(**kwargs)`` method satisfies this Protocol;
    ``dataenginex.store.DexStore`` is what the pipeline actually uses.
    """

    def record(self, **kwargs: Any) -> Any: ...
