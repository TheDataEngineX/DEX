# Development Setup

Prerequisites, workflow, and troubleshooting for contributing to DataEngineX.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Java 17+ (for Spark features, optional)

## Setup

```bash
git clone https://github.com/TheDataEngineX/dataenginex
cd dataenginex
uv sync
```

## Build commands

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

## Project structure

```
dataenginex/
├── src/dataenginex/       # Source code
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── architecture/      # Architecture tests
├── schemas/               # JSON Schema for manifests
├── docs/                  # Design specs
├── pyproject.toml         # Project config
└── poe_tasks.toml         # Task runner config
```

## Validation

After any code change run: `uv run poe check-all` (lint + typecheck + test).

Tests passing ≠ app working — run `dex validate dex.yaml` to verify config.

## Troubleshooting

**PySpark tests skipped:** Install Java 17+ and set `JAVA_HOME`.

**DuckDB errors:** Ensure no other process holds a lock on `.dex/store.duckdb`.

**Import errors after restructure:** Run `uv sync` to update the virtualenv.
