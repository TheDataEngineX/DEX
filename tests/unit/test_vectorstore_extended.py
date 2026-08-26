"""Extended tests for the vector store provider.

Covers InMemoryBackend, RAGPipeline, and Document.
"""

from __future__ import annotations

import pytest

from dataenginex.providers.vector.vectorstore import (
    Document,
    InMemoryBackend,
    RAGPipeline,
    SearchResult,
    VectorStoreBackend,
)

# ── Document ──────────────────────────────────────────────────────────────────


class TestDocument:
    def test_default_id_generated(self) -> None:
        d = Document(text="hello")
        assert len(d.id) > 0

    def test_ids_deterministic_by_content(self) -> None:
        """Ids are content-hash derived, not random — re-ingesting the same
        text must upsert (same id), and distinct texts must not collide."""
        assert Document(text="hello").id == Document(text="hello").id
        ids = {Document(text=f"doc-{i}").id for i in range(50)}
        assert len(ids) == 50

    def test_explicit_id_not_overridden(self) -> None:
        d = Document(id="custom-id", text="hello")
        assert d.id == "custom-id"

    def test_metadata_default_empty(self) -> None:
        d = Document(text="x")
        assert d.metadata == {}

    def test_embedding_default_empty(self) -> None:
        d = Document(text="x")
        assert d.embedding == []

    def test_custom_id(self) -> None:
        d = Document(id="my-id", text="hello")
        assert d.id == "my-id"


# ── SearchResult ──────────────────────────────────────────────────────────────


class TestSearchResult:
    def test_fields(self) -> None:
        doc = Document(text="test")
        sr = SearchResult(document=doc, score=0.95)
        assert sr.document is doc
        assert sr.score == 0.95


# ── VectorStoreBackend (ABC) ──────────────────────────────────────────────────


class TestVectorStoreBackendABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VectorStoreBackend()  # type: ignore[abstract]


# ── InMemoryBackend ───────────────────────────────────────────────────────────


def _vec(n: int, dim: int = 4) -> list[float]:
    """Generate a simple normalised vector."""
    import math

    v = [float(i % (n + 1) + 1) for i in range(dim)]
    mag = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / mag for x in v]


class TestInMemoryBackendUpsert:
    def test_upsert_returns_count(self) -> None:
        b = InMemoryBackend(dimension=4)
        docs = [
            Document(id="a", text="x", embedding=_vec(1)),
            Document(id="b", text="y", embedding=_vec(2)),
        ]
        assert b.upsert(docs) == 2

    def test_upsert_empty(self) -> None:
        b = InMemoryBackend(dimension=4)
        assert b.upsert([]) == 0

    def test_upsert_updates_existing(self) -> None:
        b = InMemoryBackend(dimension=4)
        d1 = Document(id="x", text="old", embedding=_vec(1))
        d2 = Document(id="x", text="new", embedding=_vec(2))
        b.upsert([d1])
        b.upsert([d2])
        assert b.count() == 1
        assert b.get("x").text == "new"  # type: ignore[union-attr]

    def test_upsert_wrong_dimension_skipped(self) -> None:
        b = InMemoryBackend(dimension=4)
        bad = Document(id="bad", text="x", embedding=[1.0, 2.0])  # wrong dim
        assert b.upsert([bad]) == 0
        assert b.count() == 0

    def test_upsert_no_embedding_allowed(self) -> None:
        b = InMemoryBackend(dimension=4)
        doc = Document(id="no-embed", text="x")  # empty embedding
        assert b.upsert([doc]) == 1


class TestInMemoryBackendQuery:
    def _populated(self, dim: int = 4) -> InMemoryBackend:
        b = InMemoryBackend(dimension=dim)
        docs = [Document(id=str(i), text=f"doc {i}", embedding=_vec(i, dim)) for i in range(5)]
        b.upsert(docs)
        return b

    def test_query_returns_results(self) -> None:
        b = self._populated()
        results = b.query(_vec(0), top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_sorted_by_score_desc(self) -> None:
        b = self._populated()
        results = b.query(_vec(0), top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_top_k_respected(self) -> None:
        b = self._populated()
        assert len(b.query(_vec(0), top_k=2)) == 2

    def test_query_empty_store(self) -> None:
        b = InMemoryBackend(dimension=4)
        assert b.query(_vec(0)) == []

    def test_query_skips_docs_without_embedding(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert([Document(id="no-embed", text="x")])
        results = b.query(_vec(0))
        assert results == []

    def test_query_metadata_filter(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert(
            [
                Document(id="a", text="match", embedding=_vec(1), metadata={"type": "good"}),
                Document(id="b", text="skip", embedding=_vec(2), metadata={"type": "bad"}),
            ]
        )
        results = b.query(_vec(1), filter_metadata={"type": "good"})
        assert all(r.document.metadata["type"] == "good" for r in results)

    def test_query_metadata_filter_no_match(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert([Document(id="a", embedding=_vec(1), metadata={"type": "bad"})])
        results = b.query(_vec(1), filter_metadata={"type": "good"})
        assert results == []


class TestInMemoryBackendCRUD:
    def test_count_empty(self) -> None:
        assert InMemoryBackend().count() == 0

    def test_count_after_upsert(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert([Document(id="x", embedding=_vec(1))])
        assert b.count() == 1

    def test_get_existing(self) -> None:
        b = InMemoryBackend(dimension=4)
        doc = Document(id="myid", text="hello")
        b.upsert([doc])
        assert b.get("myid") is doc

    def test_get_missing_returns_none(self) -> None:
        b = InMemoryBackend()
        assert b.get("nope") is None

    def test_delete_existing(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert([Document(id="del")])
        assert b.delete(["del"]) == 1
        assert b.count() == 0

    def test_delete_missing_returns_zero(self) -> None:
        b = InMemoryBackend()
        assert b.delete(["ghost"]) == 0

    def test_delete_partial(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert([Document(id="a"), Document(id="b")])
        assert b.delete(["a", "ghost"]) == 1
        assert b.count() == 1

    def test_clear(self) -> None:
        b = InMemoryBackend(dimension=4)
        b.upsert([Document(id="a"), Document(id="b")])
        b.clear()
        assert b.count() == 0


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = _vec(1)
        score = InMemoryBackend._cosine(v, v)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        score = InMemoryBackend._cosine(a, b)
        assert abs(score) < 1e-6

    def test_zero_vector_no_crash(self) -> None:
        score = InMemoryBackend._cosine([0.0, 0.0], [1.0, 0.0])
        assert isinstance(score, float)


class TestMatchesFilter:
    def test_match_all_keys(self) -> None:
        assert InMemoryBackend._matches_filter({"a": 1, "b": 2}, {"a": 1}) is True

    def test_mismatch(self) -> None:
        assert InMemoryBackend._matches_filter({"a": 1}, {"a": 2}) is False

    def test_missing_key(self) -> None:
        assert InMemoryBackend._matches_filter({}, {"a": 1}) is False


# ── RAGPipeline ───────────────────────────────────────────────────────────────


class TestRAGPipeline:
    def _pipeline(self, dim: int = 4) -> RAGPipeline:
        store = InMemoryBackend(dimension=dim)
        embed_fn = lambda text: _vec(hash(text) % 100, dim)  # noqa: E731
        return RAGPipeline(store=store, embed_fn=embed_fn, dimension=dim)

    def test_ingest_strings(self) -> None:
        rag = self._pipeline()
        count = rag.ingest(["doc one", "doc two"])
        assert count == 2

    def test_ingest_with_metadata(self) -> None:
        rag = self._pipeline()
        count = rag.ingest(["hello"], metadata=[{"source": "test"}])
        assert count == 1

    def test_ingest_with_ids(self) -> None:
        rag = self._pipeline()
        rag.ingest(["text"], ids=["my-id"])
        assert rag.store.get("my-id") is not None

    def test_query_returns_results(self) -> None:
        rag = self._pipeline()
        rag.ingest(["alpha beta", "gamma delta"])
        results = rag.query("alpha", top_k=2)
        assert len(results) <= 2

    def test_query_empty_store(self) -> None:
        rag = self._pipeline()
        assert rag.query("anything") == []

    def test_store_attribute(self) -> None:
        rag = self._pipeline()
        assert isinstance(rag.store, InMemoryBackend)

    def test_ingest_uses_hash_embed_fallback(self) -> None:
        store = InMemoryBackend(dimension=4)
        rag = RAGPipeline(store=store, embed_fn=None, dimension=4)
        count = rag.ingest(["text using hash embed"])
        assert count == 1

    def test_build_context_returns_str(self) -> None:
        rag = self._pipeline()
        rag.ingest(["context document one", "context document two"])
        ctx = rag.build_context("query", top_k=2)
        assert isinstance(ctx, str)

    def test_build_context_empty_store(self) -> None:
        rag = self._pipeline()
        ctx = rag.build_context("query")
        assert ctx == ""

    def test_build_context_respects_max_chars(self) -> None:
        rag = self._pipeline()
        long_text = "word " * 1000
        rag.ingest([long_text, long_text])
        ctx = rag.build_context("query", max_context_chars=50)
        assert len(ctx) <= 60  # small margin for ID prefix
