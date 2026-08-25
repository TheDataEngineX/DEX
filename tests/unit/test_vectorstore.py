"""Tests for dataenginex.ml.vectorstore — vector DB adapter & RAG pipeline."""

from __future__ import annotations

import pytest

from dataenginex.providers.vector.vectorstore import (
    Document,
    InMemoryBackend,
    QdrantBackend,
    RAGPipeline,
)

# ============================================================================
# InMemoryBackend
# ============================================================================


class TestInMemoryBackend:
    """Test the in-memory brute-force vector store."""

    def test_upsert_and_count(self) -> None:
        backend = InMemoryBackend(dimension=3)
        docs = [
            Document(id="a", text="hello", embedding=[1.0, 0.0, 0.0]),
            Document(id="b", text="world", embedding=[0.0, 1.0, 0.0]),
        ]
        count = backend.upsert(docs)
        assert count == 2
        assert backend.count() == 2

    def test_query_returns_nearest(self) -> None:
        backend = InMemoryBackend(dimension=3)
        backend.upsert(
            [
                Document(id="x", text="python", embedding=[1.0, 0.0, 0.0]),
                Document(id="y", text="java", embedding=[0.0, 1.0, 0.0]),
                Document(id="z", text="rust", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        results = backend.query([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].document.id == "x"
        assert results[0].score > 0.99

    def test_query_with_metadata_filter(self) -> None:
        backend = InMemoryBackend(dimension=2)
        backend.upsert(
            [
                Document(id="a", text="one", embedding=[1.0, 0.0], metadata={"lang": "en"}),
                Document(id="b", text="two", embedding=[0.9, 0.1], metadata={"lang": "fr"}),
            ]
        )
        results = backend.query([1.0, 0.0], top_k=10, filter_metadata={"lang": "fr"})
        assert len(results) == 1
        assert results[0].document.id == "b"

    def test_delete(self) -> None:
        backend = InMemoryBackend(dimension=2)
        backend.upsert([Document(id="a", text="x", embedding=[1.0, 0.0])])
        assert backend.delete(["a"]) == 1
        assert backend.count() == 0
        assert backend.delete(["nonexistent"]) == 0

    def test_clear(self) -> None:
        backend = InMemoryBackend(dimension=2)
        backend.upsert(
            [
                Document(id="a", text="x", embedding=[1.0, 0.0]),
                Document(id="b", text="y", embedding=[0.0, 1.0]),
            ]
        )
        backend.clear()
        assert backend.count() == 0

    def test_get(self) -> None:
        backend = InMemoryBackend(dimension=2)
        backend.upsert([Document(id="a", text="hello", embedding=[1.0, 0.0])])
        doc = backend.get("a")
        assert doc is not None
        assert doc.text == "hello"
        assert backend.get("nonexistent") is None

    def test_dimension_mismatch_skipped(self) -> None:
        backend = InMemoryBackend(dimension=3)
        docs = [Document(id="bad", text="x", embedding=[1.0, 0.0])]
        count = backend.upsert(docs)
        assert count == 0

    def test_upsert_overwrites(self) -> None:
        backend = InMemoryBackend(dimension=2)
        backend.upsert([Document(id="a", text="v1", embedding=[1.0, 0.0])])
        backend.upsert([Document(id="a", text="v2", embedding=[0.0, 1.0])])
        assert backend.count() == 1
        assert backend.get("a").text == "v2"


# ============================================================================
# QdrantBackend (fallback mode — no Qdrant server in unit tests)
# ============================================================================


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestQdrantBackendFallback:
    """QdrantBackend falls back to InMemoryBackend when Qdrant is unreachable."""

    def test_fallback_works(self) -> None:
        backend = QdrantBackend(
            url="http://localhost:16333", collection="test_fallback", dimension=3
        )
        docs = [Document(id="a", text="hello", embedding=[1.0, 0.0, 0.0])]
        backend.upsert(docs)
        assert backend.count() >= 1

    def test_fallback_query(self) -> None:
        backend = QdrantBackend(url="http://localhost:16333", dimension=3)
        assert backend._fallback is not None  # must be in fallback mode
        backend.upsert(
            [
                Document(id="a", text="python", embedding=[1.0, 0.0, 0.0]),
                Document(id="b", text="java", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        results = backend.query([1.0, 0.0, 0.0], top_k=1)
        assert len(results) >= 1


# ============================================================================
# RAGPipeline
# ============================================================================


class TestRAGPipeline:
    """Test the RAG pipeline orchestrator."""

    def test_ingest_and_query(self) -> None:
        rag = RAGPipeline(dimension=32)
        count = rag.ingest(["Python is great", "Java is also great", "Rust is fast"])
        assert count == 3
        assert rag.store.count() == 3

        results = rag.query("Python", top_k=1)
        assert len(results) >= 1

    def test_ingest_with_metadata(self) -> None:
        rag = RAGPipeline(dimension=32)
        rag.ingest(
            texts=["doc1", "doc2"],
            metadata=[{"source": "web"}, {"source": "file"}],
        )
        assert rag.store.count() == 2

    def test_ingest_with_custom_ids(self) -> None:
        rag = RAGPipeline(dimension=32)
        rag.ingest(texts=["hello"], ids=["my_id"])
        doc = rag.store.get("my_id")
        assert doc is not None
        assert doc.text == "hello"

    def test_build_context(self) -> None:
        rag = RAGPipeline(dimension=32)
        rag.ingest(["Python is a language", "Data engineering is fun"])
        context = rag.build_context("What is Python?", top_k=1)
        assert len(context) > 0

    def test_build_context_respects_max_chars(self) -> None:
        rag = RAGPipeline(dimension=32)
        rag.ingest(["A" * 100, "B" * 100])
        context = rag.build_context("query", top_k=10, max_context_chars=50)
        assert len(context) <= 150  # some overhead from ID prefix

    def test_custom_embed_fn(self) -> None:
        def dummy_embed(text: str) -> list[float]:
            # Deterministic fixed-dimension embedding
            vals = [float(ord(c) % 10) for c in text[:16]]
            return (vals + [0.0] * 16)[:16]

        rag = RAGPipeline(embed_fn=dummy_embed, dimension=16)
        rag.ingest(["test text here!"])
        assert rag.store.count() == 1
