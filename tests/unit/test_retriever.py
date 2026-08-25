"""Tests for the built-in retriever (BM25, dense, hybrid)."""

from __future__ import annotations

from typing import Any

from dataenginex.domains.ai.retrieval.builtin import _BM25, BuiltinRetriever, _rrf
from dataenginex.providers.vector.vectorstore import Document, SearchResult

SAMPLE_DOCS = [
    {"id": "1", "text": "Python is a programming language"},
    {"id": "2", "text": "DuckDB is an in-process SQL OLAP database"},
    {"id": "3", "text": "Machine learning uses Python and data"},
    {"id": "4", "text": "SQL queries can filter and transform data"},
    {"id": "5", "text": "FastAPI is a modern Python web framework"},
]


class TestBM25:
    def test_basic_scoring(self) -> None:
        bm25 = _BM25()
        bm25.index(SAMPLE_DOCS)
        results = bm25.score("python programming")
        assert len(results) > 0
        # Doc 1 should score highest for "python programming"
        assert results[0][0] == 0

    def test_empty_query(self) -> None:
        bm25 = _BM25()
        bm25.index(SAMPLE_DOCS)
        results = bm25.score("")
        assert results == []

    def test_no_match(self) -> None:
        bm25 = _BM25()
        bm25.index(SAMPLE_DOCS)
        results = bm25.score("xyzzyplugh")
        assert results == []

    def test_top_k_limit(self) -> None:
        bm25 = _BM25()
        bm25.index(SAMPLE_DOCS)
        results = bm25.score("data", top_k=2)
        assert len(results) <= 2


class TestRRF:
    def test_basic_fusion(self) -> None:
        r1 = [(0, 1.0), (1, 0.5), (2, 0.3)]
        r2 = [(2, 1.0), (0, 0.5), (3, 0.3)]
        fused = _rrf(r1, r2)
        assert len(fused) == 4
        # Both r1 and r2 have doc 0, so it should rank high
        doc_ids = [idx for idx, _ in fused]
        assert 0 in doc_ids[:2]


class TestBuiltinRetriever:
    def test_sparse_retrieval(self) -> None:
        retriever = BuiltinRetriever(strategy="sparse", documents=SAMPLE_DOCS)
        results = retriever.retrieve("python", top_k=3)
        assert len(results) > 0
        assert all(r["method"] == "bm25" for r in results)

    def test_hybrid_without_vector_store(self) -> None:
        retriever = BuiltinRetriever(strategy="hybrid", documents=SAMPLE_DOCS)
        results = retriever.retrieve("SQL database", top_k=3)
        assert len(results) > 0
        assert all(r["method"] == "hybrid" for r in results)

    def test_dense_without_store_returns_empty(self) -> None:
        retriever = BuiltinRetriever(strategy="dense", documents=SAMPLE_DOCS)
        results = retriever.retrieve("python", top_k=3)
        assert results == []

    def test_strategy_override(self) -> None:
        retriever = BuiltinRetriever(strategy="hybrid", documents=SAMPLE_DOCS)
        results = retriever.retrieve("python", top_k=3, strategy="sparse")
        assert all(r["method"] == "bm25" for r in results)

    def test_empty_documents(self) -> None:
        retriever = BuiltinRetriever(strategy="sparse")
        results = retriever.retrieve("python", top_k=3)
        assert results == []


class _StubVectorStore:
    """Minimal BaseVectorStore stub — always returns doc id '2' as the top hit."""

    def add(self, **kwargs: Any) -> None:
        pass

    def search(self, _embedding: list[float], top_k: int = 10, **_kwargs: Any) -> list[dict]:
        return [{"document": SAMPLE_DOCS[1]["text"], "score": 0.9}][:top_k]


class _RaisingLexicalBackend:
    """LexicalSearchBackend whose search() raises — simulates an ES outage."""

    def index(self, documents: list[Document]) -> int:
        return len(documents)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        raise ConnectionError("elasticsearch unreachable")

    def health_check(self) -> bool:
        return True


class _UnhealthyLexicalBackend:
    """LexicalSearchBackend that's configured but reports itself unhealthy."""

    def index(self, documents: list[Document]) -> int:
        return len(documents)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:  # pragma: no cover
        raise AssertionError("search() must not be called when health_check() fails")

    def health_check(self) -> bool:
        return False


class TestHybridLexicalFallback:
    """Hybrid retrieval must degrade to vector-only results, never raise, on ES failure."""

    def test_hybrid_degrades_to_vector_only_on_lexical_search_error(self) -> None:
        retriever = BuiltinRetriever(
            strategy="hybrid",
            documents=SAMPLE_DOCS,
            vector_store=_StubVectorStore(),
            embed_fn=lambda _text: [0.1, 0.2, 0.3],
            lexical_backend=_RaisingLexicalBackend(),
        )
        results = retriever.retrieve("python", top_k=3)
        assert len(results) > 0
        assert all(r["method"] == "hybrid" for r in results)
        assert results[0]["id"] == SAMPLE_DOCS[1]["id"]

    def test_hybrid_degrades_to_vector_only_on_unhealthy_lexical_backend(self) -> None:
        retriever = BuiltinRetriever(
            strategy="hybrid",
            documents=SAMPLE_DOCS,
            vector_store=_StubVectorStore(),
            embed_fn=lambda _text: [0.1, 0.2, 0.3],
            lexical_backend=_UnhealthyLexicalBackend(),
        )
        results = retriever.retrieve("python", top_k=3)
        assert len(results) > 0
        assert results[0]["id"] == SAMPLE_DOCS[1]["id"]
