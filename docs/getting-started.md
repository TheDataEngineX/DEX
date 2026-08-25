# Getting Started

Install DataEngineX and run your first pipeline.

## Installation

```bash
pip install dataenginex
```

## Create a config file

Create `dex.yaml`:

```yaml
project:
  name: my-first-pipeline

data:
  sources:
    raw_users:
      type: csv
      connection:
        path: data/
        default_file: users.csv
  pipelines:
    clean_users:
      source: raw_users
      target:
        layer: silver
```

## Validate and run

```bash
dex validate dex.yaml
```

```python
from dataenginex.engine import DexEngine

engine = DexEngine("dex.yaml")
result = engine.run_pipeline("clean_users")
print(result.rows_output)
```

## Optional extras

```bash
pip install "dataenginex[cloud]"     # S3, GCS, BigQuery
pip install "dataenginex[qdrant]"    # Qdrant vector store
pip install "dataenginex[delta]"     # Delta Lake connector
pip install "dataenginex[pytorch]"   # PyTorch ML
```

Kafka, RabbitMQ, Elasticsearch, GraphQL, and SQLAlchemy are in the base install.

## CLI commands

```bash
dex validate dex.yaml     # Validate config file
dex version               # Show version + environment
dex run <pipeline>        # Queue a pipeline run (requires worker)
dex worker start          # Start a worker to process queued runs
```
