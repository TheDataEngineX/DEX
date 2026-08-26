# Contributing

How to contribute to DataEngineX.

## Getting started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Make your changes
4. Run checks: `uv run poe check-all`
5. Commit and push
6. Open a PR

## Code standards

- **Python 3.13+** — use modern syntax (type unions, `match`, etc.)
- **mypy strict** — all code must pass `uv run poe typecheck`
- **Ruff** — zero warnings: `uv run poe lint`
- **Tests** — new code needs tests; coverage must stay above 85%
- **Docstrings** — public APIs need docstrings; internal code doesn't

## Commit messages

Use conventional commits:

```
feat: add Kafka batch connector
fix: handle empty CSV files in profiler
docs: update architecture overview
refactor: simplify pipeline runner init
test: add integration test for Spark streaming
```

## Pull requests

- Keep PRs focused — one change per PR
- Include a clear description of what changed and why
- Link related issues
- Ensure CI passes before requesting review

## Architecture decisions

Significant design changes require an ADR (Architecture Decision Record) in `docs/superpowers/plans/`:

```markdown
# ADR-NNNN: Title

## Status
Proposed | Accepted | Deprecated | Superseded by ADR-XXXX

## Context
What is the issue?

## Decision
What did we decide?

## Consequences
What are the trade-offs?
```

## Reporting issues

Open an issue on GitHub with:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version, OS, DataEngineX version
