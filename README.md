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
├── ai/                 # LLM providers, routing, agents, retrieval, tools, memory, workflows
├── api/                # Pydantic response models, GraphQL schema (no HTTP server)
├── application/        # Domain services (governance, projects, runs, resources)
├── bootstrap/          # Engine wiring and startup
├── cli/                # `dex` CLI: validate, version, init
├── config/             # dex.yaml schema, loader, env-var resolution
├── core/               # Core abstractions and shared utilities
├── data/               # Connectors (CSV, Parquet, DuckDB, REST, HTTP, Kafka), transforms
├── domains/            # Domain logic (ai, analytics, data, execution, governance, ml, plugins, security)
├── duckdb/             # DuckDB connection and helpers
├── engine.py           # DexEngine — entry point
├── engines/            # Engine implementations and registry (DuckDB, Spark, base)
├── foundation/         # Base ABCs, plugin contracts, errors, events, resources
├── interfaces/         # Gateway, embedded, external service interfaces
├── lakehouse/          # Bronze/Silver/Gold lakehouse storage and catalog
├── middleware/         # structlog, Prometheus metrics
├── ml/                 # ML training, registry, serving, drift, features
├── orchestration/      # Scheduler, background workers, queue backends
├── orm/                # SQLAlchemy ORM models
├── plugins/            # Entry-point plugin discovery
├── providers/          # Backend implementations (connectors, vector, tracking)
├── runtime/            # Compiler, registry, ML runtime, sandbox
├── secops/             # PII detection, masking, audit
├── spark/              # PySpark connector (SQL, streaming, ML, catalog, lineage)
├── store.py            # DexStore — DuckDB (WAL) persistence
└── warehouse/          # Warehouse operations
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
| [CHANGELOG](CHANGELOG.md) | Release history |
| [Release Notes](src/dataenginex/RELEASE_NOTES.md) | Package release details |

______________________________________________________________________

**License:** MIT • **Python:** 3.13+ • **Status:** Pre-1.0 • **Version:** 0.7.0