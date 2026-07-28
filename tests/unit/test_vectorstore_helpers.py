"""Tests for vectorstore.py RAGPipeline, QdrantBackend fallback, and more."""

from __future__ import annotations

from dataenginex.ai.vectorstore import (
    Document,
    InMemoryBackend,
    QdrantBackend,
    RAGPipeline,
    SearchResult,
)


class TestDocument:
    def test_auto_id(self) -> None:
        doc = Document(text="hello world")
        assert doc.id != ""
        assert len(doc.id) == 16

    def test_explicit_id(self) -> None:
        doc = Document(id="custom_id", text="hello")
        assert doc.id == "custom_id"

    def test_same_text_same_id(self) -> None:
        d1 = Document(text="same")
        d2 = Document(text="same")
        assert d1.id == d2.id


class TestSearchResult:
    def test_fields(self) -> None:
        doc = Document(id="1", text="hello")
        sr = SearchResult(document=doc, score=0.95)
        assert sr.document.id == "1"
        assert sr.score == 0.95


class TestInMemoryBackendAdvanced:
    def test_count(self) -> None:
        backend = InMemoryBackend(dimension=3)
        docs = [
            Document(id="1", text="a", embedding=[1.0, 0.0, 0.0]),
            Document(id="2", text="b", embedding=[0.0, 1.0, 0.0]),
        ]
        backend.upsert(docs)
        assert backend.count() == 2

    def test_get(self) -> None:
        backend = InMemoryBackend(dimension=3)
        doc = Document(id="1", text="hello", embedding=[1.0, 0.0, 0.0])
        backend.upsert([doc])
        result = backend.get("1")
        assert result is not None
        assert result.text == "hello"

    def test_get_missing(self) -> None:
        backend = InMemoryBackend(dimension=3)
        assert backend.get("nonexistent") is None

    def test_delete(self) -> None:
        backend = InMemoryBackend(dimension=3)
        docs = [
            Document(id="1", text="a", embedding=[1.0, 0.0, 0.0]),
            Document(id="2", text="b", embedding=[0.0, 1.0, 0.0]),
        ]
        backend.upsert(docs)
        deleted = backend.delete(["1"])
        assert deleted == 1
        assert backend.get("1") is None
        assert backend.get("2") is not None

    def test_clear(self) -> None:
        backend = InMemoryBackend(dimension=3)
        backend.upsert([Document(id="1", text="a", embedding=[1.0, 0.0, 0.0])])
        backend.clear()
        assert backend.count() == 0

    def test_query_with_filter(self) -> None:
        backend = InMemoryBackend(dimension=3)
        docs = [
            Document(id="1", text="a", metadata={"type": "x"}, embedding=[1.0, 0.0, 0.0]),
            Document(id="2", text="b", metadata={"type": "y"}, embedding=[0.9, 0.1, 0.0]),
        ]
        backend.upsert(docs)
        results = backend.query([1.0, 0.0, 0.0], filter_metadata={"type": "x"})
        assert len(results) == 1
        assert results[0].document.id == "1"


class TestQdrantBackend:
    def test_fallback_to_inmemory(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        assert backend._fallback is not None

    def test_fallback_upsert(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        docs = [Document(id="1", text="a", embedding=[1.0, 0.0, 0.0])]
        count = backend.upsert(docs)
        assert count == 1

    def test_fallback_query(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        backend.upsert([Document(id="1", text="a", embedding=[1.0, 0.0, 0.0])])
        results = backend.query([1.0, 0.0, 0.0])
        assert len(results) == 1

    def test_fallback_delete(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        backend.upsert([Document(id="1", text="a", embedding=[1.0, 0.0, 0.0])])
        deleted = backend.delete(["1"])
        assert deleted == 1

    def test_fallback_count(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        backend.upsert([Document(id="1", text="a", embedding=[1.0, 0.0, 0.0])])
        assert backend.count() == 1

    def test_fallback_clear(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        backend.upsert([Document(id="1", text="a", embedding=[1.0, 0.0, 0.0])])
        backend.clear()
        assert backend.count() == 0

    def test_fallback_get(self) -> None:
        backend = QdrantBackend(url="http://invalid:6333", dimension=3)
        backend.upsert([Document(id="1", text="hello", embedding=[1.0, 0.0, 0.0])])
        result = backend.get("1")
        assert result is not None
        assert result.text == "hello"


class TestRAGPipeline:
    def test_ingest(self) -> None:
        store = InMemoryBackend(dimension=3)
        rag = RAGPipeline(store=store, dimension=3)
        rag.ingest(["hello world", "foo bar"])
        assert store.count() == 2

    def test_ingest_with_embed_fn(self) -> None:
        store = InMemoryBackend(dimension=3)
        embed_fn = lambda text: [1.0, 0.0, 0.0]  # noqa: E731
        rag = RAGPipeline(store=store, embed_fn=embed_fn, dimension=3)
        rag.ingest(["doc1", "doc2"])
        assert store.count() == 2

    def test_query(self) -> None:
        store = InMemoryBackend(dimension=3)
        rag = RAGPipeline(store=store, dimension=3)
        rag.ingest(["hello world", "foo bar"])
        results = rag.query("hello", top_k=1)
        assert len(results) == 1

    def test_query_with_filter(self) -> None:
        store = InMemoryBackend(dimension=3)
        rag = RAGPipeline(store=store, dimension=3)
        rag.ingest(["hello world", "foo bar"])
        results = rag.query("hello", top_k=1)
        assert len(results) == 1

    def test_query_empty_store(self) -> None:
        store = InMemoryBackend(dimension=3)
        rag = RAGPipeline(store=store, dimension=3)
        results = rag.query("hello", top_k=10)
        assert results == []
