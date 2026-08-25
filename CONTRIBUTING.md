# Contributing to DataEngineX

Thank you for your interest in contributing to DataEngineX!

## Quick Start

1. Fork and create a feature branch from `main`
1. Run local checks: `uv run poe check-all`
1. Open a PR

## Development Setup

```bash
git clone https://github.com/TheDataEngineX/dataenginex && cd dataenginex
uv sync
uv run poe check-all          # lint + typecheck + tests
```

## Code Style

- Python 3.13+, `from __future__ import annotations` in all public modules
- Ruff lint + mypy strict
- structlog for logging
- Pydantic for data validation

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
