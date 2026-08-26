# Architecture

DataEngineX is a unified Data + ML + AI library. Config-driven, self-hosted, local-first. Pure Python — no bundled HTTP server.

## Module map

```
src/dataenginex/
├── engine.py              # DexEngine — main entry point
├── store.py               # DexStore — SQLite/DuckDB persistence
├── config/                # YAML config loader + validator
├── cli/                   # Click CLI (dex command)
├── domains/               # Business logic domains
│   ├── ai/                # LLM, agents, embeddings, retrieval
│   ├── analytics/         # SQL transforms, warehouse transforms
│   ├── data/              # Profiler, medallion, catalog, pipeline
│   ├── execution/         # Sandbox, resource limits
│   ├── governance/        # Lineage, artifacts, audit trail
│   ├── ml/                # Training, serving, features, tracking
│   ├── plugins/           # Plugin discovery + loader
│   └── security/          # Governance, approval, policy
├── providers/             # Connectors + vector backends
├── runtime/               # Compiler, queue, scheduler, monitoring
├── foundation/            # Constants, errors, operations, invariants
├── interfaces/            # Abstract base classes
├── bootstrap/             # Settings, gateway factory
├── application/           # Services (runs, schedules, projects)
├── spark/                 # Spark engine, SQL, catalog, streaming, ML
├── duckdb/                # DuckDB engine, catalog, portable SQL
├── engines/               # Engine registry + built-in engines
├── plugins/               # Plugin registry + discovery
├── secops/                # PII guard, audit, masking, policies
├── orm/                   # SQLAlchemy ORM models
├── api/                   # REST/GraphQL error types
└── middleware/            # Logging config, Prometheus metrics
```

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Library, not platform** | Import and call — no server process required |
| **Config-driven** | `dex.yaml` declares sources, pipelines, transforms, ML, AI |
| **Pluggable backends** | Swap DuckDB for Spark, SQLite for Postgres, local for cloud |
| **Local-first** | Works offline; cloud is optional |
| **Immutable config** | Config is read once at startup; changes require restart |
| **Multi-process safe** | WAL-mode SQLite handles concurrent readers |
| **Security primitive** | PrivacyGuard + sandboxing built in, not bolted on |
| **Observability native** | Prometheus metrics, structlog, distributed tracing |

## Data flow

```
User → DexEngine → Config Loader → PipelineRunner
                                        ↓
                              Connector (CSV/Parquet/DB/...)
                                        ↓
                              Transform chain (filter, cast, derive, ...)
                                        ↓
                              Engine (DuckDB or Spark)
                                        ↓
                              Load → Lakehouse (bronze/silver/gold)
                                        ↓
                              DexStore (run record, lineage, audit)
```

## Version matrix

| Component | Version | Python |
|-----------|---------|--------|
| dataenginex | 0.7.0 | 3.13+ |
| dex-studio | 0.5.2 | 3.13+ |
