# dataenginex Release Notes

See [CHANGELOG.md](../../CHANGELOG.md) for the complete release history.

This document tracks published package releases for `dataenginex` only.
Only include changes that modify files under `src/dataenginex/**`.

## v0.7.0 - 2026-08-12

- Released package version `0.7.0`.
- Tag: `v0.7.0`
- Release title: `Release v0.7.0 — Domain Architecture`
- Changes in this release:
  - Architecture restructure: legacy flat modules (`core/`, `data/`, `ml/`, `ai/`, `warehouse/`, `lakehouse/`) replaced by domain-driven layout (`domains/`, `providers/`, `runtime/`, `foundation/`, `interfaces/`).
  - New `application/` layer for domain services.
  - New `bootstrap/`, `orm/`, `spark/`, `duckdb/` modules.

## v0.6.0 - 2026-08-01

- Released package version `0.6.0`.
- Tag: `v0.6.0`
- Release title: `Release v0.6.0 — Monorepo`
- Changes in this release:
  - Monorepo restructuring (ADR-0001): single uv workspace with `packages/` and `apps/` layout.

## v0.5.2 - 2026-07-28

- Released package version `0.5.2`.
- Tag: `v0.5.2`
- Release title: `Release v0.5.2 — Test Coverage`
- Changes in this release:
  - 20 new test modules covering engine CRUD, ML training/registry/serving, lineage persistence, AI tools, vector store, LLM providers, circuit breaker, retry, JSON helpers.
  - Test coverage raised from ~64% to 85% (1476 passed, 41 skipped).
  - Documentation updated across all guides.

## v0.5.1 - 2026-07-27

- Released package version `0.5.1`.
- Tag: `v0.5.1`
- Release title: `Release v0.5.1`
- Changes in this release:
  - Minor dependency updates and bug fixes.

## v0.5.0 - 2026-07-07

- Released package version `0.5.0`.
- Tag: `v0.5.0`
- Release title: `Release v0.5.0 — AI & Integration`
- Changes in this release:
  - Project-local plugin loader for dex.yaml `plugins:` block.
  - KafkaConnector with produce + consume support (core dep).
  - RabbitMQ queue backend (core dep).
  - Graph-based retrieval, lexical search with Elasticsearch backend.
  - GraphQL schema-mount via strawberry-graphql (core dep).
  - ExplodeTransform for unnesting nested JSON/struct columns.
  - SQLAlchemy ORM models and session management (core dep).
  - AI workflows (DAG, condition nodes, human-in-the-loop), runtime (checkpointing, sandbox), memory (episodic, long-term), built-in tools.
  - Config schema extended: `plugins:`, `ai.workflows:`, `ai.memory:`, `ai.runtime:`, `ai.tools:` blocks.
  - Promoted to core deps: confluent-kafka, pika, elasticsearch, strawberry-graphql, sqlalchemy.

## v0.4.2 - 2026-06-23

- Released package version `0.4.2`.
- Tag: `v0.4.2`
- Release title: `Release v0.4.2`
- Changes in this release:
  - Example scripts refreshed for PySpark ML, feature transforms, and drift detection.
  - Documentation trimmed and corrected.

## v0.4.1 - 2026-06-12

- Released package version `0.4.1`.
- Tag: `v0.4.1`
- Release title: `Release v0.4.1 — Lakehouse & ML Hardening`
- Changes in this release:
  - `orjson`-backed JSON shim for ~3-5x serialization throughput.
  - DeltaConnector — native Delta Lake read/write via `deltalake`.
  - Built-in feature transformers (StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures).
  - LakehouseStorage rewrite with pluggable backends (local, S3, GCS), Zstandard compression.
  - ML registry artifact versioning with aliasing and stage transitions.
  - ML training lifecycle management, early-stopping, cross-validation.
  - `mypy --strict` passes cleanly across all modules.

## v0.4.0 - 2026-02-21

- Released package version `0.4.0`.
- Tag: `v0.4.0`
- Release title: `Release v0.4.0 — Scope Reset`
- Changes in this release:
  - Scope reset from 1.x (prematurely tagged stable) to 0.4.0.
  - Stable `__all__` exports in every subpackage.
  - Comprehensive module-level docstrings with usage examples.
  - New public API exports: ComponentHealth, AuthMiddleware, PaginationMeta, RateLimiter, etc.

## v0.3.4 - 2026-02-20

- Released package version `0.3.4`.
- Tag: `v0.3.4`
- Release title: `Release v0.3.4`
- Changes in this release:
  - Repo hygiene updates after package-layout migration.
  - Canonicalized package/app path references across docs and workflow guidance.
  - Updated CI/package validation and project metadata/docs to align with current structure.

## v0.3.3 - 2026-02-16

- Released package version `0.3.3`.
- Tag: `v0.3.3`
- Release title: `Release v0.3.3`
- Changes in this release (`src/dataenginex` only):
  - Significant package expansion and refactor across API, core, data, lakehouse, middleware, ML, and warehouse modules.
  - Added/expanded API modules including auth, pagination, rate limiting, and v1 router wiring.
  - Consolidated middleware layout (logging, metrics, tracing) and moved logging config under middleware.
  - Package diff from `v0.2.0` to `v0.3.3`: 39 files changed, 4213 insertions, 247 deletions.

## v0.2.0 - 2026-02-12

- Released package version `0.2.0`.
- Tag: `v0.2.0`
- Release title: `Release v0.2.0 - Production Hardening`
- Changes in this release (`src/dataenginex` only):
  - Established core package structure and initial module organization.
  - Added API readiness/health behavior and improved probe handling.
  - Added structured request logging with request ID tracking.
  - Added Prometheus metrics and OpenTelemetry tracing support in package middleware.
  - Added validation and error-handling improvements in API/core paths.
