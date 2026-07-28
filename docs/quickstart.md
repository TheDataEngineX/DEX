# Quickstart

Get a DataEngineX pipeline running in under five minutes.

```mermaid
flowchart LR
    Install["pip install dataenginex"] --> Config["Create dex.yaml"]
    Config --> Validate["dex validate dex.yaml"]
    Validate --> Engine["Instantiate DexEngine"]
    Engine --> Pipeline["engine.run_pipeline()"]
```

## 1. Install

```bash
pip install dataenginex
# or from source:
git clone https://github.com/TheDataEngineX/dataenginex && cd dataenginex
uv sync
```

## 2. Create a config file

`dex.yaml`:

```yaml
project:
  name: my-first-pipeline
  version: "0.1.0"

data:
  sources:
    raw_users:
      type: csv
      path: data/users.csv
  pipelines:
    clean_users:
      source: raw_users
      transforms:
        - type: filter
          condition: "age > 0"
      destination: silver.users
```

## 3. Validate

```bash
dex validate dex.yaml
```

## 4. Use DexEngine in your application

```python
from dataenginex.engine import DexEngine

engine = DexEngine("dex.yaml")

# Run a pipeline
result = engine.run_pipeline("clean_users")

# Query sources
schema = engine.source_schema("raw_users")
sample = engine.source_sample("raw_users", limit=5)

# Inspect warehouse
layers = engine.warehouse_layers()
tables = engine.warehouse_tables("silver")

# Check history
runs = engine.store.get_pipeline_runs(None)[:10]
```

## 5. Run Examples

Each example is a standalone script:

```bash
uv run python examples/01_hello_pipeline.py   # Minimal pipeline
uv run python examples/03_quality_gate.py     # Quality checks
uv run python examples/04_ml_training.py      # ML training + registry
uv run python examples/05_rag_demo.py         # RAG pipeline
uv run python examples/06_llm_quickstart.py   # LLM providers

# Build your own FastAPI app on top (see example 02)
uv run --with fastapi --with uvicorn python examples/02_api_quickstart.py
```

## 6. Run Tests

```bash
uv run poe test-cov  # Full test suite
uv run poe lint       # Lint with Ruff
uv run poe typecheck  # mypy strict
uv run poe check-all  # All of the above
```

**Coverage Note:** The test suite achieves 85% code coverage (1476 tests passed, 41 skipped). Connectors requiring external dependencies (HTTP, REST, SSE, Delta Lake, MLflow, ML registry, PySpark, Parquet, CSV, PostgreSQL, Qdrant, etc.) are excluded from coverage calculations as they require optional package installation and cloud credential setup.

## Spark and dbt sources

Register a PySpark or dbt source with the `connection` dict:

```python
# PySpark source
engine.add_source(
    "ratings",
    "spark",
    connection={"master": "local[*]", "format": "parquet"},
)

# dbt source (runs `dbt run --select my_model`, reads the DuckDB target)
engine.add_source(
    "dbt_revenue",
    "dbt",
    connection={"project_dir": "/path/to/dbt_project", "model": "revenue", "target": "dev"},
)
```

Install PySpark separately (`pip install pyspark`). dbt-core is CLI-only; install it globally.

## PrivacyGuard

Enable outbound-call interception and PII masking in `dex.yaml`:

```yaml
secops:
  guard:
    enabled: true
    block_on_detect: false
    strategies:
      email: hash
      ssn: redact
  audit:
    enabled: true
    db_path: ".dex/audit.duckdb"
```

Access the audit log programmatically:

```python
events = engine.secops_audit.events   # list of AuditEvent
```

## Next Steps

- [Architecture](architecture.md) — medallion layers, backend registry, DexEngine
- [Development Guide](development.md) — editor config and workflow
- [API Reference](api-reference/index.md) — auto-generated module docs
- `examples/` directory — full list of runnable examples

______________________________________________________________________

## DataEngineX Studio

DataEngineX Studio is the optional web UI that loads a `dex.yaml` and provides a single
control plane for Data / ML / AI / System. It uses `dataenginex` directly as a library
— no separate server process needed.

See [github.com/TheDataEngineX/dex-studio](https://github.com/TheDataEngineX/dex-studio).
