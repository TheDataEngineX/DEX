"""Retrieval vocabulary — what a document, a hit, and a store *are* (§5.3).

These are the types the AI domain reasons in. Agents, tools, lexical search,
and the retrievers all pass ``Document`` and ``SearchResult`` around, and none
of them should need Qdrant or sentence-transformers installed to name the type
they are handling.

``VectorStoreBackend`` is the contract those concrete stores implement from
``providers/vector/``. Keeping it here is what makes the backend swappable: the
domain depends on the abstraction, the mechanism depends on the domain, and
nothing points the other way.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Document", "SearchResult", "VectorStoreBackend"]


# ======================================================================
# Data models
# ======================================================================


@dataclass(frozen=True)
class Document:
    """A text document with optional metadata and embedding."""

    id: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            # Content-hash id, not random: re-ingesting the same text upserts
            # (overwrites) the existing entry instead of piling up a new
            # random-id duplicate. InMemoryBackend has no eviction/cap, so a
            # random id here means unbounded growth across repeated scheduled
            # pipeline runs that re-ingest the same rows.
            # object.__setattr__ required: instance is frozen, but this is
            # the one-time id derivation during construction, not a later mutation.
            object.__setattr__(
                self, "id", hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
            )


@dataclass(frozen=True)
class SearchResult:
    """Single search hit from a vector store query."""

    document: Document
    score: float


# ======================================================================
# Abstract backend
# ======================================================================


class VectorStoreBackend(abc.ABC):
    """Abstract vector-store backend.

    All backends store fixed-dimension vectors keyed by string ID and
    support nearest-neighbour queries by cosine similarity.
    """

    @abc.abstractmethod
    def upsert(self, documents: list[Document]) -> int:
        """Insert or update documents. Returns count upserted."""

    @abc.abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return top-k nearest documents by cosine similarity."""

    @abc.abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete documents by id. Returns count deleted."""

    @abc.abstractmethod
    def count(self) -> int:
        """Number of documents in the store."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Delete all documents."""

    @abc.abstractmethod
    def get(self, doc_id: str) -> Document | None:
        """Retrieve a single document by ID."""
