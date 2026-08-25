"""Graph-augmented retriever — dual-level (entity + semantic) RAG.

Inspired by LightRAG (EMNLP 2025). Retrieves at two complementary levels
and fuses the rankings via reciprocal rank fusion (RRF):

1. **Entity-level** (high-level, high recall) — pull docs mentioning the
   named entities of the query.
2. **Semantic-level** (low-level, high precision) — dense vector similarity
   against the embedded query.

Entity extraction defaults to a lightweight rule-based extractor
(capitalised tokens and bigrams). Supply ``entity_extractor`` for LLM-driven
NER or domain-specific extraction.

Example::

    from dataenginex.domains.ai.retrieval.graph import GraphRetriever

    retriever = GraphRetriever(
        vector_store=store,
        embed_fn=embed_fn,
        documents=docs,
    )
    hits = retriever.retrieve("How does Snowflake handle schema drift?", top_k=5)
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import structlog

from dataenginex.domains.ai.retrieval import retriever_registry
from dataenginex.domains.ai.retrieval.builtin import _BM25, _rrf
from dataenginex.foundation.plugin_contracts import BaseRetriever

logger = structlog.get_logger()

__all__ = ["GraphRetriever", "default_extract_entities"]


_ENTITY_REGEX = re.compile(r"\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,3}\b")
_STOPWORDS: frozenset[str] = frozenset(
    {"The", "This", "That", "A", "An", "Is", "Are", "Was", "Were", "I", "We", "You", "It", "If"},
)


def default_extract_entities(text: str) -> list[str]:
    """Rule-based entity extractor — capitalised tokens and multi-word phrases."""
    return [m for m in _ENTITY_REGEX.findall(text) if m not in _STOPWORDS]


@retriever_registry.decorator("graph")
class GraphRetriever(BaseRetriever):
    """Dual-level (entity + semantic) retriever.

    Args:
        vector_store: :class:`BaseVectorStore` for dense search. When ``None``
            the semantic level falls back to BM25.
        embed_fn: Callable mapping text → embedding vector.
        entity_extractor: Entity extractor. Defaults to
            :func:`default_extract_entities`.
        documents: Initial documents to index (each dict must have ``"text"``).
    """

    def __init__(
        self,
        vector_store: Any = None,
        embed_fn: Any = None,
        entity_extractor: Callable[[str], list[str]] | None = None,
        documents: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> None:
        self._store = vector_store
        self._embed_fn = embed_fn
        self._extract = entity_extractor or default_extract_entities
        self._docs: list[dict[str, Any]] = []
        self._entity_index: dict[str, set[int]] = {}
        self._bm25 = _BM25()
        if documents:
            self.index(documents)

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Index documents for dual-level retrieval."""
        self._docs = documents
        self._entity_index = {}
        for idx, doc in enumerate(documents):
            for ent in self._extract(doc.get("text", "")):
                self._entity_index.setdefault(ent.lower(), set()).add(idx)
        self._bm25.index(documents)
        if self._store is not None and self._embed_fn is not None:
            for doc in documents:
                embedding = self._embed_fn(doc.get("text", ""))
                self._store.add(
                    ids=[doc.get("id", str(id(doc)))],
                    embeddings=[embedding],
                    documents=[doc.get("text", "")],
                    metadata=[doc.get("metadata", {})],
                )
        logger.info(
            "graph retriever indexed",
            docs=len(documents),
            entities=len(self._entity_index),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k docs by fusing entity-level and semantic-level rankings."""
        pool = max(top_k * 2, top_k)
        entity_ranks = self._entity_level(query, pool)
        semantic_ranks = self._semantic_level(query, pool)
        fused = _rrf(entity_ranks, semantic_ranks)[:top_k]
        return [
            {**self._docs[idx], "score": score, "method": "graph"}
            for idx, score in fused
            if 0 <= idx < len(self._docs)
        ]

    def _entity_level(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """High-level ranking: docs containing query entities, weighted by overlap."""
        query_entities = [e.lower() for e in self._extract(query)]
        if not query_entities:
            return self._bm25.score(query, top_k)
        counts: Counter[int] = Counter()
        for ent in query_entities:
            for doc_idx in self._entity_index.get(ent, ()):
                counts[doc_idx] += 1
        if not counts:
            return self._bm25.score(query, top_k)
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, float(c)) for idx, c in ranked]

    def _semantic_level(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Low-level ranking: dense vector similarity (falls back to BM25)."""
        if self._store is None or self._embed_fn is None:
            return self._bm25.score(query, top_k)
        embedding = self._embed_fn(query)
        dense = self._store.search(embedding, top_k=top_k)
        mapped: list[tuple[int, float]] = []
        for r in dense:
            doc_text = r.get("document", r.get("text", ""))
            score = float(r.get("score", 0.0))
            for i, doc in enumerate(self._docs):
                if doc.get("text", "") == doc_text:
                    mapped.append((i, score))
                    break
        return mapped

    @property
    def entities(self) -> dict[str, set[int]]:
        """Expose the entity → doc-indices index (read-only view)."""
        return self._entity_index
