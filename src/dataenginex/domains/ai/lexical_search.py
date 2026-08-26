"""Lexical (keyword/BM25) search abstraction — sibling to ``VectorStoreBackend``.

Provides a pluggable lexical-search backend, mirroring the shape of
:class:`~dataenginex.providers.vector.vectorstore.VectorStoreBackend` (``index``/``search``)
so the two can be combined by a hybrid retriever.

- **ElasticsearchBackend** — Elasticsearch-backed BM25 lexical search.

Reliability contract: every backend method must never raise out of this
module. Connection/timeout errors are caught, logged, and degrade to an
empty result (``search``) or a skipped no-op (``index``) so a lexical
backend outage never crashes the caller.
"""

from __future__ import annotations

import abc
from typing import Any

import structlog

from dataenginex.domains.ai.retrieval.types import Document, SearchResult

logger = structlog.get_logger()

__all__ = ["ElasticsearchBackend", "LexicalSearchBackend"]


class LexicalSearchBackend(abc.ABC):
    """Abstract lexical (keyword) search backend.

    Mirrors :class:`~dataenginex.providers.vector.vectorstore.VectorStoreBackend`'s shape
    where sensible, reusing the same ``Document``/``SearchResult`` types.
    """

    @abc.abstractmethod
    def index(self, documents: list[Document]) -> int:
        """Index documents for keyword search. Returns count indexed."""

    @abc.abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return top-k documents matching ``query`` by lexical relevance (e.g. BM25)."""

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and usable."""


# ======================================================================
# Elasticsearch backend
# ======================================================================


class ElasticsearchBackend(LexicalSearchBackend):
    """Elasticsearch-backed lexical search.

    Never raises: connection/timeout errors are caught and logged, with
    ``index`` skipping the batch and ``search`` returning ``[]`` so a
    hybrid retriever can fall back to vector-only results.

    Args:
        hosts: Elasticsearch node URLs, e.g. ``["http://elasticsearch:9200"]``.
        index_name: Index to read/write documents to.
        timeout_seconds: Per-request timeout applied to every ES call.
    """

    def __init__(
        self,
        hosts: list[str],
        index_name: str,
        timeout_seconds: float = 5.0,
        username: str = "",
        password: str = "",
    ) -> None:
        self.index_name = index_name
        self.timeout_seconds = timeout_seconds
        from elasticsearch import Elasticsearch

        auth = (username, password) if username and password else None
        self._client: Any = Elasticsearch(
            hosts,
            request_timeout=timeout_seconds,
            basic_auth=auth,
        )

    def index(self, documents: list[Document]) -> int:
        """Bulk-index documents. Logs and skips (returns 0) on any ES error."""
        if not documents:
            return 0
        try:
            from elasticsearch.helpers import bulk

            actions = [
                {
                    "_index": self.index_name,
                    "_id": doc.id,
                    "_source": {"text": doc.text, **doc.metadata},
                }
                for doc in documents
            ]
            success, _errors = bulk(self._client, actions, request_timeout=self.timeout_seconds)
            logger.info("elasticsearch indexed", count=success, index=self.index_name)
            return int(success)
        except Exception as exc:
            logger.warning(
                "elasticsearch index failed — skipping batch",
                error=str(exc),
                index=self.index_name,
            )
            return 0

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 match query. Returns [] (never raises) if ES is unavailable."""
        try:
            resp = self._client.search(
                index=self.index_name,
                query={"match": {"text": query}},
                size=top_k,
                request_timeout=self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "elasticsearch search failed — returning empty results",
                error=str(exc),
                index=self.index_name,
            )
            return []

        hits = resp.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = dict(hit.get("_source", {}))
            text = str(source.pop("text", ""))
            doc = Document(id=str(hit.get("_id", "")), text=text, metadata=source)
            results.append(SearchResult(document=doc, score=float(hit.get("_score", 0.0))))
        return results

    def health_check(self) -> bool:
        """Return True if the ES cluster responds to ping."""
        try:
            return bool(self._client.ping())
        except Exception as exc:
            logger.warning("elasticsearch health check failed", error=str(exc))
            return False
