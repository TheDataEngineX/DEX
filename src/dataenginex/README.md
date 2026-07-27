# dataenginex

The Python library that powers [DataEngineX Studio](https://github.com/TheDataEngineX/dex-studio) — an open-source, self-hosted, local-first Data + ML + AI workbench. Use the library directly when you want code; install DataEngineX Studio when you want a UI.

> **Pre-1.0 status.** `0.5.1` is honest about that. See the [CHANGELOG](https://github.com/TheDataEngineX/dataenginex/blob/main/CHANGELOG.md) for details.

## Install

```bash
pip install dataenginex                    # lean base — DuckDB, structlog, Pydantic, Click, pyarrow
```

Optional integrations — install only what you need:

```bash
pip install 'dataenginex[postgres]'        # asyncpg-backed lineage, persistence
pip install 'dataenginex[qdrant]'          # Qdrant vector store backend
pip install 'dataenginex[queue]'           # ARQ async job queue (pulls redis)
pip install 'dataenginex[cloud]'           # S3, GCS, BigQuery storage backends
pip install 'dataenginex[delta]'           # Delta Lake connector
pip install 'dataenginex[pytorch]'         # PyTorch ML models
```

> **LiteLLM:** install separately — it pins `python-dotenv==1.0.1` which conflicts with our `>=1.2.2`:
>
> ```bash
> pip install 'litellm>=1.83.3' --no-deps
> ```

## Quick start

```python
from pathlib import Path
from dataenginex.engine import DexEngine

# Load config and initialize all backends
engine = DexEngine(Path("dex.yaml"))

# Data — run pipelines defined in dex.yaml
engine.run_pipeline("clean_users")

# ML — train, register, predict
models = engine.model_registry.list_models()
result = engine.model_registry.predict("churn_model", features)

# AI — chat with an agent over your data
response = engine.agents["assistant"].chat("summarise the latest pipeline run")

# Persistence — query SQLite-backed history
runs = engine.store.list_pipeline_runs(limit=10)
```

Smaller surfaces — use only what you need:

```python
from dataenginex.config import load_config
cfg = load_config("dex.yaml")

from dataenginex.core.interfaces import BaseConnector
from dataenginex.core.registry import BackendRegistry

from dataenginex.ml import ModelRegistry
from dataenginex.ai.llm import get_llm_provider
from dataenginex.ai.vectorstore import VectorStoreBackend
```

## Submodules

| Module | Description |
| --- | --- |
| `dataenginex.engine` | `DexEngine` — single entry point; loads config, inits store, wires backends |
| `dataenginex.store` | `DexStore` — SQLite (WAL) persistence (`.dex/store.duckdb`) |
| `dataenginex.config` | `dex.yaml` schema, loader, env-var resolution |
| `dataenginex.core` | Exceptions, `Base*` ABCs, `BackendRegistry` |
| `dataenginex.cli` | `dex` CLI (`validate`, `version`, `init`) |
| `dataenginex.data` | Connectors (CSV, Parquet, DuckDB, HTTP, …), pipeline runner, schema registry |
| `dataenginex.ml` | Classical ML — training, model registry, serving, drift |
| `dataenginex.ai` | LLM providers, agents, RAG, vector store, memory, observability |
| `dataenginex.orchestration` | Scheduler, background workers |
| `dataenginex.middleware` | structlog config, Prometheus metrics |
| `dataenginex.lakehouse` | Storage backends, catalog, partitioning |
| `dataenginex.warehouse` | Transforms, lineage tracking |
| `dataenginex.secops` | PII detection, masking, audit logging |
| `dataenginex.api` | Pydantic response models (no HTTP server bundled) |
| `dataenginex.plugins` | Entry-point plugin discovery |

## Want the UI?

`dataenginex` is the engine. The web UI lives in a separate repo: [github.com/TheDataEngineX/dex-studio](https://github.com/TheDataEngineX/dex-studio). It imports `dataenginex` directly — no separate API server.

## Links

- Source: [github.com/TheDataEngineX/dataenginex](https://github.com/TheDataEngineX/dataenginex)
- Docs: [docs.thedataenginex.org](https://docs.thedataenginex.org)
- Issues: [github.com/TheDataEngineX/dataenginex/issues](https://github.com/TheDataEngineX/dataenginex/issues)
- License: MIT
