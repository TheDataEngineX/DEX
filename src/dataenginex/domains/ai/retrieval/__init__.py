"""Retriever registry.

Built-in retriever supports dense, sparse (BM25), and hybrid strategies.
Graph retriever adds LightRAG-style dual-level (entity + semantic) retrieval.
"""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseRetriever
from dataenginex.runtime.registry import BackendRegistry

retriever_registry: BackendRegistry[BaseRetriever] = BackendRegistry("retriever")

__all__ = ["retriever_registry"]
