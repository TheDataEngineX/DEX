# Observability

DataEngineX provides structured logging, Prometheus metrics, and distributed tracing out of the box.

## Logging

All modules use [structlog](https://www.structlog.org/) for structured, machine-readable logs:

```python
import structlog

logger = structlog.get_logger()
logger.info("pipeline started", pipeline="clean_users", rows=1000)
```

Log output is JSON by default. Set `DEX_LOG_FORMAT=console` for human-readable output.

## Metrics

Prometheus metrics are exposed at the engine level:

```python
from dataenginex.engine import DexEngine

engine = DexEngine("dex.yaml")
# Metrics available via engine.metrics or Prometheus endpoint
```

Key metrics:
- `dex_pipeline_runs_total` — total pipeline runs
- `dex_pipeline_duration_seconds` — pipeline execution time
- `dex_rows_processed_total` — rows read/written
- `dex_engine_errors_total` — error count by type

## Tracing

Distributed tracing is integrated via structlog context:

```python
logger = structlog.get_logger()
logger.bind(trace_id="abc123", span_id="def456")
```

Correlation IDs propagate across pipeline stages, engine calls, and CLI invocations.

## Run history

Every pipeline run is recorded in `.dex/store.duckdb`:

```python
from dataenginex.engine import DexEngine

engine = DexEngine("dex.yaml")
runs = engine.store.get_pipeline_runs(pipeline="clean_users")
```

Run records include: status, start/end time, rows input/output, error messages, and lineage.
