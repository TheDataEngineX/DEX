# CLAUDE.md — DataEngineX (dataenginex)

Brief answers only. No explanations unless asked.
Goal is to save Claude code tokens for lower cost without losing quality.

> Repo-specific context. Workspace-level rules, coding standards, and git conventions are in `../CLAUDE.md`.

## Project Overview

**DataEngineX** — unified Data + ML + AI library. Config-driven, self-hosted, local-first. Pure Python — no bundled HTTP server.

| Directory | Purpose |
|-----------|---------|
| `src/dataenginex/` | Core library — engine, domains, providers, foundation, CLI |
| `schemas/` | Published JSON Schema for manifests |
| `tests/` | Unit, integration, and architecture tests |
| `docs/` | Design specs and integration plans |

**Stack:** Python 3.13+ · DuckDB · SQLite (WAL, persistence store) · structlog · Pydantic · Click · pyarrow · croniter · httpx · prometheus-client · uv · Ruff · mypy strict · pytest

**Version:** `uv run poe version`

**Domain:** thedataenginex.org | **Org:** github.com/TheDataEngineX

______________________________________________________________________

## Build & Run Commands

```bash
# Quality
uv run poe lint           # Ruff lint
uv run poe lint-fix       # Ruff lint + auto-fix
uv run poe typecheck      # mypy --strict
uv run poe check-all      # lint + typecheck + test

# Test
uv run poe test           # Core library tests
uv run poe test-cov       # tests + coverage

# CLI
dex validate dex.yaml     # Validate config file
dex version               # Show version + environment

# Deps
uv sync                   # Sync dependencies
uv lock                   # Regenerate lockfile
uv run poe security       # Audit deps for vulnerabilities
```

## Optional Extras

```bash
pip install "dataenginex[cloud]"        # S3, GCS, BigQuery connectors
pip install "dataenginex[qdrant]"       # Qdrant vector store
pip install "dataenginex[delta]"        # Delta Lake connector
pip install "dataenginex[pytorch]"      # PyTorch ML
pip install 'litellm>=1.83.3' --no-deps # LLM routing (separate: pins python-dotenv)

# Kafka, RabbitMQ, Elasticsearch, GraphQL, SQLAlchemy are in the base install (no extra needed).
```

______________________________________________________________________

## Validation

After any code change run: `uv run poe check-all` (lint + typecheck + test).
Tests passing ≠ app working — run `dex validate dex.yaml` to verify config.

## TMDB Data-Intelligence Re-Architecture (2026-07-06)

**Core vs custom boundary:** moviedex-specific logic (TMDB connector, Kafka topics, RabbitMQ handlers, ES mappings, GraphQL resolvers) stays in the moviedex project's own `plugins/` dir — never added to dataenginex. Only genuinely reusable capability (new connector types, transform types, generic backends/ABCs) goes in core.
