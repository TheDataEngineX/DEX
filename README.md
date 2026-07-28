# dataenginex

[![CI](https://github.com/TheDataEngineX/dataenginex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TheDataEngineX/dataenginex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dataenginex)](https://pypi.org/project/dataenginex/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The Python library that powers [DataEngineX Studio](https://github.com/TheDataEngineX/dex-studio) — an open-source, self-hosted, local-first Data + ML + AI workbench for individuals and small teams. **Use the library directly when you want code; install DataEngineX Studio when you want a UI.**

______________________________________________________________________

## Install

```bash
pip install dataenginex                  # lean base — DuckDB, structlog, Pydantic, Click, pyarrow
```

```python
from dataenginex.engine import DexEngine

engine = DexEngine("dex.yaml")           # loads config, inits SQLite store, wires backends
engine.run_pipeline("clean_users")       # execute a pipeline
models = engine.model_registry.list_models()
response = engine.agents["assistant"].chat("summarise the latest run")
```

Optional integrations install only what you need:

| Extra | What you get | Example use case |
| --- | --- | --- |
| `[postgres]` | `asyncpg` | Persist lineage events to Postgres |
| `[qdrant]` | `qdrant-client` | Production vector store (falls back to in-memory) |
| `[queue]` | `arq` (+ redis transitively) | Async background jobs |
| `[cloud]` | `boto3`, `google-cloud-storage`, `google-cloud-bigquery` | S3 / GCS / BigQuery sources & sinks |
| `[ml]` | `scikit-learn`, `sentence-transformers` | Train classical ML models, generate embeddings |
| `[tracking]` | `mlflow` | Experiment tracking via MLflow |
| `[data]` | `pyspark` | PySpark connector + dbt CLI connector (run dbt models as pipeline steps) |
| `[delta]` | `deltalake` | Delta Lake connector |
| `[pytorch]` | `torch` | PyTorch ML models |

> **LiteLLM** must be installed separately due to a `python-dotenv` pin conflict:
> ```bash
> pip install 'litellm>=1.83.3' --no-deps
> ```

______________________________________________________________________

## What it does

`dex.yaml` describes a project. `DexEngine` reads it and wires the pieces:

| Subsystem | Built-in default | Optional backend |
| --- | --- | --- |
| Data engine | DuckDB | PySpark (`[data]`) |
| Storage | Local parquet + DuckDB | S3, GCS, BigQuery (`[cloud]`) |
| Lineage | JSON / SQLite | Postgres (`[postgres]`) |
| Scheduler | croniter | — |
| ML training | scikit-learn wrapper (`[ml]`) | — |
| ML tracking | JSON-based | MLflow (`[tracking]`) |
| LLM providers | Ollama, OpenAI, Anthropic | Any OpenAI-compatible URL |
| Vector store | in-memory | Qdrant (`[qdrant]`) |
| Retrieval | BM25 + dense + hybrid + graph | — |
| Persistence | SQLite, WAL mode (`.dex/store.duckdb`) | — |
| Logging | structlog | — |
| Privacy | PrivacyGuard — PII detection, masking strategies, outbound call audit | — |
| Connectors | CSV, Parquet, DuckDB, REST, HTTP, Kafka | PySpark (`[data]`), dbt CLI (`[data]`), Delta (`[delta]`), PostgreSQL (`[postgres]`) |

**Local-first by default.** A fresh `pip install dataenginex` requires no external services — DuckDB is embedded; nothing reaches the network unless you explicitly configure it (or call a hosted LLM).

______________________________________________________________________

## Project structure

```text
src/dataenginex/
├── cli/                # `dex` CLI: validate, version, init
├── config/             # dex.yaml schema, loader
├── core/               # Base ABCs, registry, exceptions
├── engine.py           # DexEngine — entry point
├── store.py            # DexStore — SQLite (WAL) persistence
├── api/                # Pydantic response models (no HTTP server)
├── data/               # Connectors, pipeline runner, transforms, quality
├── ml/                 # Classical ML: training, registry, serving, drift
├── ai/                 # LLM providers, agents, RAG, vector store, memory
├── orchestration/      # Scheduler, background workers
├── middleware/         # structlog, Prometheus metrics
├── lakehouse/          # Storage backends, catalog, partitioning
├── warehouse/          # Transforms, lineage
├── secops/             # PII detection, masking, audit
└── plugins/            # Entry-point discovery
```

______________________________________________________________________

## Development

```bash
git clone https://github.com/TheDataEngineX/dataenginex && cd dataenginex
uv sync
uv run poe check-all          # lint + typecheck + tests
uv run poe test-cov           # tests + coverage
uv run poe lint-fix           # auto-fix lint issues
dex validate dex.yaml         # validate a config file
dex version                   # show version + environment
```

See [docs/development.md](docs/development.md) for the full setup.

______________________________________________________________________

## Want the full workbench?

`dataenginex` is the library. The web UI is [dex-studio](https://github.com/TheDataEngineX/dex-studio) — it imports `dataenginex` directly, no HTTP hop.

______________________________________________________________________

## Ecosystem

| Repo | Purpose |
| --- | --- |
| [dataenginex](https://github.com/TheDataEngineX/dataenginex) | This library (PyPI) |
| [dex-studio](https://github.com/TheDataEngineX/dex-studio) | Web UI — FastAPI + Jinja2 + HTMX |
| [infradex](https://github.com/TheDataEngineX/infradex) | Kubernetes deployment via ArgoCD |

______________________________________________________________________

## Documentation

| Guide | Description |
| --- | --- |
| [Quickstart](docs/quickstart.md) | Get running in 5 minutes |
| [Architecture](docs/architecture.md) | System design and patterns |
| [Development](docs/development.md) | Local setup and workflow |
| [API Reference](docs/api-reference/index.md) | Module-by-module reference |
| [CHANGELOG](CHANGELOG.md) | Release history |

______________________________________________________________________

**License:** MIT • **Python:** 3.13+ • **Status:** Pre-1.0 • **Version:** 0.5.2