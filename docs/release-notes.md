# Release Notes

## v0.7.0 (2026-08-12)

Architecture restructure. The flat `core/`, `data/`, `ml/` layout was replaced by a layered architecture:

- `domains/` — business logic (ai, analytics, data, execution, governance, ml, plugins, security)
- `providers/` — connectors and vector backends
- `runtime/` — compiler, queue, scheduler, monitoring
- `foundation/` — constants, errors, operations, invariants
- `interfaces/` — abstract base classes
- `bootstrap/` — settings, gateway factory
- `application/` — services (runs, schedules, projects)
- `spark/` — Spark engine, SQL, catalog, streaming, ML
- `duckdb/` — DuckDB engine, catalog, portable SQL
- `orm/` — SQLAlchemy ORM models

**Breaking changes:**
- `orchestration.secops` router renamed to `orchestration.governance`
- Monorepo structure (`packages/`, `apps/`) removed — standalone repo layout restored
- `WorkloadKind.STREAM` renamed to `WorkloadKind.SPARK_STREAM`

## v0.6.0 (2026-08-01)

Monorepo restructuring (ADR-0001). Single uv workspace with `packages/` and `apps/` members. Reverted in v0.7.0.

## v0.5.2 (2026-07-28)

20 new test modules. Coverage raised from 64% to 85% (1476 passed). Coverage threshold raised to 85%.

## v0.5.1 (2026-07-14)

Bug fixes and stability improvements.

## v0.5.0 (2026-07-07)

Project plugins, Kafka connector, RabbitMQ backend, graph retrieval, lexical search, GraphQL, ExplodeTransform, ORM, AI workflows/runtime/memory/tools.

## v0.4.0 (2026-06-15)

ML training, serving, feature store, experiment tracking. Delta Lake connector.

## v0.3.4 (2026-05-20)

Initial public release. Core pipeline engine, DuckDB/Spark engines, config system, CLI, PrivacyGuard.
