"""Tests for the GraphRetriever (dual-level entity + semantic RAG)."""

from __future__ import annotations

from typing import Any

from dataenginex.domains.ai.retrieval import retriever_registry
from dataenginex.domains.ai.retrieval.graph import GraphRetriever, default_extract_entities

_DOCS: list[dict[str, Any]] = [
    {"id": "1", "text": "Snowflake handles schema drift via ALTER TABLE."},
    {"id": "2", "text": "BigQuery uses schema auto-detection for CSV loads."},
    {"id": "3", "text": "DuckDB supports schema evolution on Parquet scans."},
    {"id": "4", "text": "Apache Iceberg formalises schema evolution semantics."},
]


class TestEntityExtractor:
    def test_picks_up_proper_nouns(self) -> None:
        ents = default_extract_entities("Snowflake and BigQuery both support schema drift.")
        assert "Snowflake" in ents
        assert "BigQuery" in ents

    def test_filters_stopwords(self) -> None:
        ents = default_extract_entities("The system is up.")
        assert "The" not in ents


class TestGraphRetriever:
    def test_registered(self) -> None:
        assert retriever_registry.get("graph") is GraphRetriever

    def test_index_populates_entity_map(self) -> None:
        r = GraphRetriever(documents=_DOCS)
        assert "snowflake" in r.entities
        assert "bigquery" in r.entities
        assert 0 in r.entities["snowflake"]

    def test_retrieve_matches_entity_level(self) -> None:
        r = GraphRetriever(documents=_DOCS)
        hits = r.retrieve("Tell me about Snowflake", top_k=2)
        assert hits, "should return at least one hit"
        assert hits[0]["id"] == "1"
        assert hits[0]["method"] == "graph"

    def test_retrieve_falls_back_when_no_entities(self) -> None:
        r = GraphRetriever(documents=_DOCS)
        hits = r.retrieve("how does schema drift work", top_k=3)
        assert len(hits) > 0

    def test_custom_entity_extractor(self) -> None:
        def extract(text: str) -> list[str]:
            return ["Iceberg"] if "Iceberg" in text else []

        r = GraphRetriever(entity_extractor=extract, documents=_DOCS)
        hits = r.retrieve("Iceberg tables", top_k=1)
        assert hits[0]["id"] == "4"

    def test_semantic_fallback_without_vector_store(self) -> None:
        r = GraphRetriever(documents=_DOCS)
        # No vector store → semantic level falls back to BM25
        hits = r.retrieve("Parquet scans DuckDB", top_k=1)
        assert hits[0]["id"] == "3"

    def test_semantic_with_vector_store(self) -> None:
        class StubStore:
            def __init__(self) -> None:
                self.added: list[str] = []

            def add(self, **kwargs: Any) -> None:
                self.added.extend(kwargs.get("documents", []))

            def search(
                self,
                _embedding: list[float],
                top_k: int = 10,
                **_kwargs: Any,
            ) -> list[dict[str, Any]]:
                return [{"document": _DOCS[3]["text"], "score": 0.95}][:top_k]

        store = StubStore()
        r = GraphRetriever(
            vector_store=store,
            embed_fn=lambda _t: [0.0, 0.1, 0.2],
            documents=_DOCS,
        )
        assert len(store.added) == len(_DOCS)
        hits = r.retrieve("unrelated query", top_k=1)
        assert hits[0]["id"] == "4"
