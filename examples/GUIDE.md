# DataEngineX Examples

Runnable examples demonstrating key features of the `dataenginex` library.

## Quick Start

```bash
# Install (from repo root)
uv sync

# Core examples (no extra deps)
uv run python examples/01_hello_pipeline.py
uv run python examples/03_quality_gate.py
uv run python examples/04_ml_training.py
uv run python examples/05_rag_demo.py
uv run python examples/06_llm_quickstart.py

# Build a FastAPI app on top (fastapi + uvicorn are NOT dataenginex deps)
uv run --with fastapi --with uvicorn python examples/02_api_quickstart.py

# PySpark examples (require Java + PySpark)
uv sync --group data
uv run python examples/07_api_ingestion.py
uv run python examples/08_spark_ml.py
uv run python examples/09_feature_engineering.py
uv run python examples/10_model_analysis.py
```

______________________________________________________________________

## Examples

| # | File | Description |
| - | ---- | ----------- |
| 1 | `01_hello_pipeline.py` | Minimal pipeline: profiler + medallion config |
| 2 | `02_api_quickstart.py` | Build a FastAPI app on top of DexEngine |
| 3 | `03_quality_gate.py` | Quality checks with `QualityGate` |
| 4 | `04_ml_training.py` | Train, register, and evaluate a model |
| 5 | `05_rag_demo.py` | RAG pipeline: ingest docs, query vector store, generate with LLM |
| 6 | `06_llm_quickstart.py` | LLM providers: mock, Ollama, OpenAI-compatible |
| 7 | `07_api_ingestion.py` | HTTP API ingestion with Bronze → Silver → Gold medallion pipeline |
| 8 | `08_spark_ml.py` | PySpark feature engineering + RandomForest via ModelRegistry |
| 9 | `09_feature_engineering.py` | Time, lag, rolling window, and interaction feature construction |
| 10 | `10_model_analysis.py` | Drift detection (PSI) + prediction error analysis |

## Template Projects

End-to-end templates showing multiple features together.

- **ShopMetrics** (`ecommerce/`) — Synthetic e-commerce. Customer churn ML, product RAG, medallion architecture.
  `uv run python examples/ecommerce/run_all.py`

## Running Tests

```bash
uv run poe check-all    # lint + typecheck + tests (1476 passed, 85% coverage)
uv run poe test-cov     # tests with coverage report
```

## Prerequisites

- Python ≥ 3.13
- `uv` installed (`pip install uv`)
- For core examples (01–06): `uv sync`
- For example 02 (FastAPI app): `uv run --with fastapi --with uvicorn`
- For PySpark examples (07–10): `uv sync --group data` + Java runtime (`JAVA_HOME` set)
