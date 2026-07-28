# Release Notes

## [0.5.2] — 2026-07-28

### Added

- 20 new test modules: engine CRUD, ML training/registry/serving, lineage persistence, AI tools, vector store, LLM providers, circuit breaker, retry, JSON helpers
- Test coverage raised from ~64% to 85% (1476 passed, 41 skipped, 7843 statements)
- Coverage threshold updated from 80% to 85% in `poe_tasks.toml`

### Changed

- Version bumped to 0.5.2
- `sentence-transformers` 5.x mock fix in `test_ml_finetune.py`
- Documentation updated across all guides to reflect current architecture and test stats

---

## [0.5.1] — 2026-07-27

### Changed

- Version bumped to 0.5.1
- Minor dependency updates and bug fixes

---

## [0.5.0] — 2026-07-07

### Added

- `core/project_plugins.py` — project-local plugin loader for dex.yaml `plugins:` block
- `data/connectors/kafka.py` — KafkaConnector with produce + consume support (core dep)
- `orchestration/queue/rabbitmq.py` — RabbitMQ queue backend (core dep)
- `ai/retrieval/graph.py` — graph-based retrieval for structured knowledge graphs
- `ai/lexical_search.py` — LexicalSearchBackend ABC + ElasticsearchBackend (core dep)
- `api/graphql.py` — GraphQL schema-mount mechanism via strawberry-graphql (core dep)
- `data/transforms/sql.py` — ExplodeTransform for unnesting nested JSON/struct columns
- `orm/` — SQLAlchemy ORM models and session management (core dep)
- `ai/workflows/` — workflow DAG, condition nodes, human-in-the-loop support
- `ai/runtime/` — checkpointing, code executor, sandboxed execution
- `ai/memory/` — episodic and long-term memory backends
- `ai/tools/builtin.py` — built-in agent tools (sql_query, web_search, file_read, python_exec, vector_search)

### Changed

- Version bumped to 0.5.0
- Config schema extended: `plugins:`, `ai.workflows:`, `ai.memory:`, `ai.runtime:`, `ai.tools:` blocks
- Promoted to core deps: confluent-kafka, pika, elasticsearch, strawberry-graphql, sqlalchemy
- `data/connectors/` — KafkaConnector added alongside existing CSV/Parquet/DuckDB/REST/SSE/HTTP
- `orchestration/` — queue subpackage with RabbitMQ backend
- `ai/` — lexical search, graph retrieval, workflow orchestration, memory, tools, runtime submodules

### Removed

- Legacy `worker.py` module — background worker functionality superseded by orchestration/scheduler + queue backends
- `dataenginex.observability` — metrics/logging moved to `dataenginex.middleware`; tracing moved to ai observability

______________________________________________________________________

## [0.4.2] — 2026-06-23

### Added

- Example scripts refreshed: `08_spark_ml.py`, `09_feature_engineering.py`, `10_model_analysis.py` — demonstrate PySpark ML, feature transforms, and drift detection
- Documentation cleanup across all docs

### Removed

- Legacy connector layer (`DataConnector`, `DataSource`, `SourceType`, `RestConnector`, `FileConnector`, `ConnectorStatus`, `FetchResult`) removed from `dataenginex.data.connectors`. Use `RestApiConnector`, `HttpConnector`, or `SseConnector` via the connector registry instead.

### Changed

- Version bumped to 0.4.2

### Verification checklist

1. `uv run poe lint` — Ruff checks clean
1. `uv run poe typecheck` — mypy strict, 0 errors
1. `uv run poe test` — all tests pass

______________________________________________________________________

## [0.4.1] - 2026-06-12

### Added

- `dataenginex._json` — drop-in `orjson`-backed JSON shim (`dumps`, `loads`, `JSONResponse`) replacing stdlib `json` across the library for ~3–5× serialization throughput
- `DeltaConnector` — native Delta Lake read/write via `deltalake` (new `delta` optional extra: `pip install "dataenginex[delta]"`)
- `ml.features.builtin` — built-in feature transformers: `StandardScalerTransform`, `MinMaxScalerTransform`, `OneHotEncoderTransform`, `PolynomialFeaturesTransform`
- `core.interfaces` — new `Closeable` and `AsyncCloseable` protocols for uniform resource lifecycle
- `orjson>=3.11.0` and `zstandard>=0.25.0` promoted to core runtime dependencies

### Changed

- **Lakehouse storage** (`lakehouse/storage.py`) — full rewrite: unified `LakehouseStorage` with pluggable backends (local, S3, GCS), Zstandard compression throughout, columnar partition pruning
- **Lakehouse catalog** (`lakehouse/catalog.py`) — catalog entries now carry partition stats and schema fingerprints; `register` / `resolve` API stabilised
- **ML registry** (`ml/registry.py`) — artifact versioning with aliasing (`promote_alias`), stage transitions (`development` → `staging` → `production`), and metadata search
- **ML training** (`ml/training.py`) — `TrainingJob` lifecycle management, early-stopping callbacks, cross-validation harness, experiment comparison utilities
- **Pagination** (`api/pagination.py`) — cursor-based and page-number strategies unified under `PaginationResult`; `paginate_query` helper works with any iterable
- **Store** (`store.py`) — async-safe DuckDB connection pool, `get_pipeline_runs` and `list_model_artifacts` now return typed dataclasses
- **SecOps audit** (`secops/audit.py`) — structured audit events with severity levels, retention policy enforcement, export to JSONL
- **AI runtime** (`ai/runtime/executor.py`) — tool call concurrency limit, timeout per tool, structured error envelopes
- **Config loader** (`config/loader.py`) — environment variable interpolation (`${VAR}`) and `include:` directive for config composition
- `zstandard` used for pipeline run history compression reducing on-disk footprint by ~60%

### Fixed

- `mypy --strict` passes cleanly across all modules after strict type annotation pass
- `DeltaConnector` and `LakehouseStorage` excluded from coverage thresholds (require live filesystems); coverage gate unchanged for all other modules

______________________________________________________________________

## [0.4.0] - 2026-02-21

> **Scope reset from 1.x.** Versions 1.0.0–1.1.2 were prematurely tagged stable. Resetting to `0.4.0` to honestly reflect pre-1.0 maturity. See [ADR-0007](https://github.com/TheDataEngineX/docs/blob/main/adr/0007-local-first-scope-reset.md) for rationale. The 1.x versions on PyPI are yanked but remain installable by exact pin (`pip install 'dataenginex==1.1.2'`); plain `pip install dataenginex` now resolves to `0.4.0`.

### Added

- Stable `__all__` exports in every subpackage `__init__.py`
- `from __future__ import annotations` in all public modules
- Comprehensive module-level docstrings with usage examples
- New public API exports: `ComponentHealth`, `AuthMiddleware`, `AuthUser`,
  `create_token`, `decode_token`, `BadRequestError`, `NotFoundError`,
  `PaginationMeta`, `RateLimiter`, `RateLimitMiddleware`,
  `ConnectorStatus`, `FetchResult`, `ColumnProfile`, `get_logger`, `get_tracer`

### Changed

- Reorganized `__all__` in all subpackages for logical grouping
- Updated package version to 0.4.0

______________________________________________________________________

## [0.3.5] - 2026-02-13

### Added

- Production hardening: structured logging, Prometheus/OTel, health probes
- Data connectors: `RestConnector`, `FileConnector` with async interface
- Schema registry with versioned schema management
- Data profiler with automated dataset statistics
- Lakehouse catalog, partitioning, and storage backends
- ML framework: trainer, model registry, drift detection, serving
- Warehouse transforms and persistent lineage tracking
- JWT authentication middleware
- Rate limiting middleware
- Cursor-based pagination utilities
- Versioned API router (`/api/v1/`)

______________________________________________________________________

[0.3.5]: https://github.com/TheDataEngineX/dataenginex/releases/tag/v0.3.5
[0.4.0]: https://github.com/TheDataEngineX/dataenginex/compare/v0.3.5...v0.4.0
[0.4.1]: https://github.com/TheDataEngineX/dataenginex/compare/v0.4.0...v0.4.1
[0.4.2]: https://github.com/TheDataEngineX/dataenginex/compare/v0.4.1...v0.4.2
[0.5.0]: https://github.com/TheDataEngineX/dataenginex/compare/v0.4.2...v0.5.0
[0.5.1]: https://github.com/TheDataEngineX/dataenginex/compare/v0.5.0...v0.5.1
[0.5.2]: https://github.com/TheDataEngineX/dataenginex/compare/v0.5.1...v0.5.2
