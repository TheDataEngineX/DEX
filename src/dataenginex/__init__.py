"""
DataEngineX — unified Data + ML + AI library.

Quick start::

    from dataenginex import DexEngine

    engine = DexEngine("path/to/dex.yaml")
    engine.run_pipeline("ingest")
    engine.close()

Subpackage imports::

    from dataenginex.config import load_config
    from dataenginex.ai import LLMProvider, get_llm_provider, RAGPipeline
    from dataenginex.ml import ModelRegistry, DriftDetector, SklearnTrainer
    from dataenginex.lakehouse import DataCatalog
    from dataenginex.warehouse import PersistentLineage
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dataenginex")
except PackageNotFoundError:
    __version__ = "0.5.1"

# Primary entry point
# AI (LLM / agents / RAG)
from dataenginex.ai import (
    InMemoryBackend,
    LLMProvider,
    MockProvider,
    OllamaProvider,
    RAGPipeline,
    VectorStoreBackend,
    get_llm_provider,
)

# Core
from dataenginex.core import DataLayer, MedallionArchitecture, QualityGate

# Data
from dataenginex.data import DataProfiler, SchemaRegistry
from dataenginex.engine import DexBackend, DexEngine

# Lakehouse
from dataenginex.lakehouse import DataCatalog, ParquetStorage, StorageBackend

# ML (classical)
from dataenginex.ml import (
    DriftDetector,
    ModelRegistry,
    SklearnTrainer,
)

# Persistence
from dataenginex.store import DexStore
from dataenginex.warehouse import PersistentLineage, TransformPipeline

__all__ = [
    "__version__",
    # Entry point
    "DexEngine",
    "DexBackend",
    "DexStore",
    # Core
    "DataLayer",
    "MedallionArchitecture",
    "QualityGate",
    # Data
    "DataProfiler",
    "SchemaRegistry",
    # Lakehouse
    "DataCatalog",
    "ParquetStorage",
    "StorageBackend",
    # ML
    "DriftDetector",
    "ModelRegistry",
    "SklearnTrainer",
    # AI
    "get_llm_provider",
    "InMemoryBackend",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "RAGPipeline",
    "VectorStoreBackend",
    # Warehouse
    "PersistentLineage",
    "TransformPipeline",
]
