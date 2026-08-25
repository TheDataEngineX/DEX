# CI/CD

DataEngineX uses GitHub Actions for continuous integration and PyPI releases.

## CI pipeline

Every push and PR triggers:

1. **Lint** — Ruff linting with zero warnings
2. **Typecheck** — mypy strict mode
3. **Test** — pytest with coverage threshold (85%+)
4. **Security** — dependency audit

## Release automation

Releases are automated via `uv run poe release`:

1. Version bump in `pyproject.toml`
2. Git tag created
3. Push to PyPI via trusted publishing
4. GitHub Release created with changelog

```bash
uv run poe release major   # 0.7.0 → 1.0.0
uv run poe release minor   # 0.7.0 → 0.8.0
uv run poe release patch   # 0.7.0 → 0.7.1
```

## Publishing to PyPI

```bash
uv run poe publish
```

Requires `PYPI_TOKEN` environment variable or trusted publisher configured in PyPI.

## Documentation

Docs are built and deployed via the website repo's sync workflow:

```bash
# Triggered automatically on release, or manually:
# .github/workflows/sync.yml copies docs/ → website/docs-store/
```
