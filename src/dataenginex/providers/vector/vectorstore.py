"""RAG-ready Vector Database Adapter.

Provides a pluggable vector-store abstraction with concrete backends:

- **InMemoryBackend** — brute-force cosine similarity (testing / small datasets)
- **QdrantBackend** — Qdrant persistent vector store (production, already deployed in K8s)

All backends implement :class:`VectorStoreBackend` so they can be
swapped transparently.  A :class:`RAGPipeline` orchestrator combines
a ``VectorStoreBackend`` with an ``EmbeddingProvider`` and an optional
LLM to build a full retrieve-augment-generate pipeline.

Example::

    from dataenginex.providers.vector.vectorstore import QdrantBackend, RAGPipeline

    backend = QdrantBackend(url="http://localhost:6333", collection="dex_docs")
    rag = RAGPipeline(store=backend)
    rag.ingest(["doc1 text", "doc2 text"])
    results = rag.query("How do I deploy to K8s?", top_k=3)
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any

import structlog

from dataenginex.domains.ai.retrieval.types import Document, SearchResult, VectorStoreBackend

if TYPE_CHECKING:
    from dataenginex.domains.ai.llm import LLMProvider, LLMResponse

logger = structlog.get_logger()

__all__ = [
    "Document",
    "InMemoryBackend",
    "QdrantBackend",
    "RAGPipeline",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "VectorStoreBackend",
]


# ======================================================================
# In-memory backend
# ======================================================================


class InMemoryBackend(VectorStoreBackend):
    """Brute-force in-memory vector store (testing & prototyping).

    Stores all documents in a dict.  Queries iterate over all stored
    vectors and compute cosine similarity.

    Args:
        dimension: Expected embedding dimension (for validation).
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._docs: dict[str, Document] = {}

    def upsert(self, documents: list[Document]) -> int:
        """Insert or update documents."""
        count = 0
        for doc in documents:
            if doc.embedding and len(doc.embedding) != self.dimension:
                logger.warning(
                    "embedding dimension mismatch",
                    doc_id=doc.id,
                    expected=self.dimension,
                    got=len(doc.embedding),
                )
                continue
            self._docs[doc.id] = doc
            count += 1
        logger.info("in-memory upserted", count=count, total=len(self._docs))
        return count

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return top-k nearest documents by cosine similarity."""
        scored: list[SearchResult] = []
        for doc in self._docs.values():
            if not doc.embedding:
                continue
            if filter_metadata and not self._matches_filter(doc.metadata, filter_metadata):
                continue
            sim = self._cosine(embedding, doc.embedding)
            scored.append(SearchResult(document=doc, score=sim))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete(self, ids: list[str]) -> int:
        removed = 0
        for doc_id in ids:
            if doc_id in self._docs:
                del self._docs[doc_id]
                removed += 1
        return removed

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        ma = math.sqrt(sum(x * x for x in a)) or 1.0
        mb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (ma * mb)

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        return all(metadata.get(k) == v for k, v in filter_metadata.items())


# ======================================================================
# Qdrant backend (production — replaces ChromaDB)
# ======================================================================


class QdrantBackend(VectorStoreBackend):
    """Qdrant-backed vector store for production workloads.

    Qdrant is already deployed in K8s (``qdrant:6333``).  Falls back to
    :class:`InMemoryBackend` when ``qdrant-client`` is not installed so
    the package remains importable without the optional dependency.

    Args:
        url: Qdrant server URL (default ``http://localhost:6333``).
        collection: Collection name (default ``dex_documents``).
        dimension: Embedding dimension (default 384 for all-MiniLM-L6-v2).
        api_key: Optional Qdrant API key for cloud deployments.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "dex_documents",
        dimension: int = 384,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.dimension = dimension
        self._client: Any = None
        self._fallback: InMemoryBackend | None = None

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                VectorParams,
            )

            self._client = QdrantClient(url=url, api_key=api_key)
            existing = [c.name for c in self._client.get_collections().collections]
            if collection not in existing:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
            logger.info("qdrant backend ready", url=url, collection=collection)
        except ImportError:
            logger.warning(
                "qdrant-client not installed — falling back to InMemoryBackend. "
                "Install via: pip install 'dataenginex[qdrant]'"
            )
            self._fallback = InMemoryBackend(dimension=dimension)
        except Exception as exc:
            logger.warning("qdrant connection failed — falling back", error=str(exc))
            self._fallback = InMemoryBackend(dimension=dimension)

    def upsert(self, documents: list[Document]) -> int:
        if self._fallback:
            return self._fallback.upsert(documents)

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=doc.id,
                vector=doc.embedding or [0.0] * self.dimension,
                payload={"text": doc.text, **doc.metadata},
            )
            for doc in documents
            if doc.embedding
        ]
        if not points:
            return 0
        self._client.upsert(collection_name=self.collection, points=points)
        logger.info("qdrant upserted", count=len(points))
        return len(points)

    def _hit_to_result(self, hit: Any) -> SearchResult:
        payload: dict[str, Any] = dict(hit.payload or {})
        text = str(payload.pop("text", ""))
        return SearchResult(
            document=Document(id=str(hit.id), text=text, metadata=payload),
            score=float(hit.score),
        )

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._fallback:
            return self._fallback.query(embedding, top_k, filter_metadata)

        qdrant_filter: Any = None
        if filter_metadata:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
            )

            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_metadata.items()
                ]
            )

        hits = self._client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [self._hit_to_result(h) for h in hits]

    def delete(self, ids: list[str]) -> int:
        if self._fallback:
            return self._fallback.delete(ids)
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    def count(self) -> int:
        if self._fallback:
            return self._fallback.count()
        return int(self._client.count(collection_name=self.collection).count)

    def clear(self) -> None:
        if self._fallback:
            self._fallback.clear()
            return
        from qdrant_client.models import Distance, VectorParams

        self._client.delete_collection(self.collection)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

    def get(self, doc_id: str) -> Document | None:
        if self._fallback:
            return self._fallback.get(doc_id)
        results = self._client.retrieve(
            collection_name=self.collection, ids=[doc_id], with_payload=True
        )
        if not results:
            return None
        hit = results[0]
        payload: dict[str, Any] = dict(hit.payload or {})
        text = str(payload.pop("text", ""))
        return Document(id=str(hit.id), text=text, metadata=payload)


# ======================================================================
# Sentence-transformer embedding provider
# ======================================================================


class SentenceTransformerEmbedder:
    """Callable embedding provider backed by ``sentence-transformers``.

    Install the optional dependency group::

        uv add 'dataenginex[ml]'

    Then pass an instance as ``embed_fn`` to :class:`RAGPipeline`::

        from dataenginex.providers.vector.vectorstore import (
            RAGPipeline,
            SentenceTransformerEmbedder,
        )

        embedder = SentenceTransformerEmbedder()          # all-MiniLM-L6-v2
        rag = RAGPipeline(embed_fn=embedder, dimension=384)

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``"all-MiniLM-L6-v2"`` (384-dim, fast, good quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            logger.info("sentence transformer embedder initialised", model=model_name)
        except ImportError as exc:
            msg = "sentence-transformers is required: uv add 'dataenginex[ml]'"
            raise ImportError(msg) from exc

    def __call__(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()  # type: ignore[no-any-return]


# ======================================================================
# RAG pipeline orchestrator
# ======================================================================


class RAGPipeline:
    """Retrieve-Augment-Generate pipeline orchestrator.

    Combines a vector-store backend with an embedding provider to
    support document ingestion and semantic retrieval.  When an LLM
    adapter is attached, the ``generate`` method augments the prompt
    with retrieved context.

    Args:
        store: Vector-store backend to use.
        embed_fn: Callable that maps text → embedding vector.
            If ``None``, uses a simple hash-based fallback.
        dimension: Embedding dimension.
    """

    def __init__(
        self,
        store: VectorStoreBackend | None = None,
        embed_fn: Any | None = None,
        dimension: int = 384,
    ) -> None:
        self.dimension = dimension
        self.store = store or InMemoryBackend(dimension=dimension)
        self._embed_fn = embed_fn or self._hash_embed

    def ingest(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Embed and store a batch of texts.

        Args:
            texts: Raw text documents.
            metadata: Optional per-document metadata.
            ids: Optional document IDs (auto-generated if omitted).

        Returns:
            Number of documents stored.
        """
        meta = metadata or [{} for _ in texts]
        # Content-hash ids by default (same scheme as Document.__post_init__)
        # so re-ingesting identical text upserts instead of duplicating —
        # see Document's docstring for why a random id here matters.
        doc_ids = ids or [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts]

        docs: list[Document] = []
        for doc_id, text, m in zip(doc_ids, texts, meta, strict=True):
            embedding = self._embed_fn(text)
            docs.append(Document(id=doc_id, text=text, metadata=m, embedding=embedding))

        count = self.store.upsert(docs)
        logger.info("rag ingest complete", texts=len(texts), stored=count)
        return count

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve top-k relevant documents for *question*."""
        q_embed = self._embed_fn(question)
        results = self.store.query(q_embed, top_k=top_k, filter_metadata=filter_metadata)
        logger.info("rag query complete", top_k=top_k, results=len(results))
        return results

    def build_context(
        self,
        question: str,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ) -> str:
        """Build an LLM context string from retrieved documents.

        Args:
            question: User question.
            top_k: Number of documents to retrieve.
            max_context_chars: Maximum context length in characters.

        Returns:
            Formatted context string for LLM prompting.
        """
        results = self.query(question, top_k=top_k)
        parts: list[str] = []
        total = 0
        for r in results:
            chunk = f"[{r.document.id}] {r.document.text}"
            if total + len(chunk) > max_context_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    def answer(
        self,
        question: str,
        llm: LLMProvider,
        top_k: int = 5,
        max_context_chars: int = 4000,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Full RAG loop: retrieve → augment → generate.

        Combines :meth:`build_context` with
        :meth:`~dataenginex.ml.llm.LLMProvider.generate_with_context`
        into a single call.

        Args:
            question: User question.
            llm: Any :class:`~dataenginex.ml.llm.LLMProvider` instance.
            top_k: Documents to retrieve.
            max_context_chars: Context length cap in characters.
            system_prompt: Optional system-prompt override for the LLM.

        Returns:
            :class:`~dataenginex.ml.llm.LLMResponse` from the provider.
        """
        context = self.build_context(question, top_k=top_k, max_context_chars=max_context_chars)
        logger.info("rag answer complete", question_len=len(question), context_len=len(context))
        return llm.generate_with_context(question, context, system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Fallback embedding
    # ------------------------------------------------------------------

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic hash-based embedding for testing."""
        import hashlib

        h = hashlib.sha256(text.encode()).hexdigest()
        vec = [int(h[i : i + 2], 16) / 255.0 for i in range(0, min(len(h), self.dimension * 2), 2)]
        vec = (vec + [0.0] * self.dimension)[: self.dimension]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
