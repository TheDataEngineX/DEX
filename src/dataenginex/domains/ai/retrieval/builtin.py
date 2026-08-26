"""Built-in retriever — dense, sparse (BM25), and hybrid strategies.

Uses the existing VectorStoreBackend for dense search and a lightweight
in-process BM25 scorer for sparse search. Hybrid combines both via
reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import structlog

from dataenginex.domains.ai.retrieval import retriever_registry
from dataenginex.foundation.plugin_contracts import BaseRetriever

logger = structlog.get_logger()


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25:
    """Minimal in-memory BM25 scorer."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[dict[str, Any]] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_dl: float = 0.0

    def index(self, docs: list[dict[str, Any]]) -> None:
        """Index documents (each must have 'text' key)."""
        self._docs = docs
        self._tf = []
        self._df = Counter()
        total_len = 0
        for doc in docs:
            tokens = _tokenize(doc.get("text", ""))
            tf = Counter(tokens)
            self._tf.append(tf)
            total_len += len(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._avg_dl = total_len / max(len(docs), 1)

    def score(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending score."""
        terms = _tokenize(query)
        n = len(self._docs)
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            dl = sum(tf.values())
            for t in terms:
                if t not in tf:
                    continue
                idf = math.log((n - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1)
                tf_norm = (tf[t] * (self.k1 + 1)) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                )
                s += idf * tf_norm
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _rrf(
    *rankings: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal rank fusion across multiple rankings."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    result = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return result


@retriever_registry.decorator("builtin", is_default=True)
class BuiltinRetriever(BaseRetriever):
    """Dense + sparse + hybrid retriever.

    Args:
        strategy: ``"dense"``, ``"sparse"``, or ``"hybrid"`` (default).
        vector_store: A VectorStoreBackend for dense search.
        embed_fn: Callable that maps text → embedding vector.
        documents: Initial documents to index.
        lexical_backend: Optional ``LexicalSearchBackend`` (e.g.
            ``ElasticsearchBackend``) for real lexical search in hybrid mode.
            When configured, hybrid fuses dense + lexical results. If the
            backend's health check fails or ``search()`` errors, hybrid
            silently degrades to dense (vector-only) results — an ES outage
            never fails retrieval.
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        vector_store: Any = None,
        embed_fn: Any = None,
        documents: list[dict[str, Any]] | None = None,
        lexical_backend: Any = None,
        **kwargs: Any,
    ) -> None:
        self._strategy = strategy
        self._store = vector_store
        self._embed_fn = embed_fn
        self._lexical = lexical_backend
        self._bm25 = _BM25()
        self._docs: list[dict[str, Any]] = []
        if documents:
            self.index(documents)

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Index documents for retrieval."""
        self._docs = documents
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
        if self._lexical is not None:
            from dataenginex.domains.ai.retrieval.types import Document

            self._lexical.index(
                [
                    Document(
                        id=str(doc.get("id", "")),
                        text=doc.get("text", ""),
                        metadata=doc.get("metadata", {}),
                    )
                    for doc in documents
                ]
            )
        logger.info("documents indexed", count=len(documents), strategy=self._strategy)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k relevant documents for query."""
        strategy = kwargs.get("strategy", self._strategy)

        if strategy == "sparse":
            return self._sparse(query, top_k)
        if strategy == "dense":
            return self._dense(query, top_k)
        # hybrid (default)
        return self._hybrid(query, top_k)

    def _sparse(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scored = self._bm25.score(query, top_k)
        return [{**self._docs[idx], "score": score, "method": "bm25"} for idx, score in scored]

    def _dense(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._store is None or self._embed_fn is None:
            logger.warning("dense search unavailable — no vector store or embed_fn")
            return []
        embedding = self._embed_fn(query)
        results = self._store.search(embedding, top_k=top_k)
        return [{**r, "method": "dense"} for r in results]

    def _hybrid(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._store is not None and self._embed_fn is not None:
            embedding = self._embed_fn(query)
            dense_raw = self._store.search(embedding, top_k=top_k * 2)
            dense_results = self._map_dense_to_indices(dense_raw)
        else:
            dense_results = []

        if self._lexical is not None:
            lexical_results = self._lexical_indices(query, top_k * 2)
            if lexical_results is not None:
                fused = _rrf(dense_results, lexical_results)[:top_k]
            else:
                # ponytail: ES down/erroring — degrade to vector-only rather
                # than fail hybrid retrieval outright.
                fused = dense_results[:top_k]
            return [{**self._docs[idx], "score": score, "method": "hybrid"} for idx, score in fused]

        sparse_results = self._bm25.score(query, top_k * 2)
        fused = _rrf(sparse_results, dense_results)[:top_k]
        return [{**self._docs[idx], "score": score, "method": "hybrid"} for idx, score in fused]

    def _lexical_indices(self, query: str, top_k: int) -> list[tuple[int, float]] | None:
        """Query the lexical backend, mapping hits back to doc indices.

        Returns None (never raises) on any failure so the caller can fall
        back to vector-only results.
        """
        try:
            if not self._lexical.health_check():
                logger.warning("lexical backend health check failed — using vector-only results")
                return None
            hits = self._lexical.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("lexical search errored — using vector-only results", error=str(exc))
            return None

        if not hits:
            return None

        by_id = {str(doc.get("id", "")): i for i, doc in enumerate(self._docs)}
        mapped: list[tuple[int, float]] = []
        for hit in hits:
            idx = by_id.get(hit.document.id)
            if idx is not None:
                mapped.append((idx, hit.score))
        return mapped

    def _map_dense_to_indices(
        self,
        dense_results: list[dict[str, Any]],
    ) -> list[tuple[int, float]]:
        """Map dense search results back to document indices."""
        mapped: list[tuple[int, float]] = []
        for r in dense_results:
            doc_text = r.get("document", r.get("text", ""))
            score = r.get("score", 0.0)
            for i, doc in enumerate(self._docs):
                if doc.get("text", "") == doc_text:
                    mapped.append((i, float(score)))
                    break
        return mapped
