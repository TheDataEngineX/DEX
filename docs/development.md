# Development Setup Guide

**Version**: `0.5.1` | see `pyproject.toml`

## Prerequisites

### System Dependencies

| Package | Required | Purpose |
| --------------------- | ----------- | ----------------------------------------------- |
| Git | Yes | Version control |
| curl | Yes | Downloading tools |
| Python 3.13+ | Yes | Runtime (managed by uv) |
| build-essential / gcc | Yes | Native extension compilation |
| Java 17+ JRE | Yes\* | PySpark tests (`openjdk-17-jre-headless`) |
| uv | Yes | Python package & env manager |

> Optional: `trivy` for local security scans, `actionlint` for GitHub Actions workflow linting.

\* PySpark tests are auto-skipped when Java is unavailable.

**One-command install** (Ubuntu/Debian, Fedora, Arch, macOS):

```bash
uv run poe setup                    # install deps and pre-commit hooks
```

This installs all Python dependencies and configures pre-commit hooks.

### Cloud Credentials (Optional)

- AWS / GCP credentials only needed for cloud storage adapters (stage/prod)
- Local development runs entirely on path-based storage

## Quick Start

```bash
# 1. Clone repo and create feature branch
git clone https://github.com/TheDataEngineX/dataenginex.git
cd dataenginex
git checkout -b my-feature-branch main

# 2. Install Python deps & pre-commit hooks
uv run poe setup

# 3. Verify setup
uv run poe check-all
uv run poe test-cov  # Coverage check: requires 79%+ (currently ~79%)
```

All tests and linting should pass. You're ready to develop!

## Project Structure

```
dataenginex/
├── src/dataenginex/        # Core framework package
├── examples/               # Runnable example scripts (01–10)
├── tests/                  # Test suite
├── docs/                   # Documentation
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml          # Project config
└── poe_tasks.toml          # Task definitions
```

## Development Workflow

### Branch & Commit

```bash
# 1. Create a feature branch from main
git checkout -b my-feature-branch main

# 2. Make changes to src/
# Add tests in tests/

# 3. Format & validate
uv run poe lint
uv run poe typecheck
uv run poe test

# 4. Commit (pre-commit hooks run automatically)
git commit -m "feat(#XXX): description"

# 5. Push & create PR
git push origin my-feature-branch
```

**PR Requirements:**

- Link to issue: `Closes #XXX`
- All checks pass (CI/CD ~3-5 min)
- 1 approval required
- Merge to `main` when ready

### Version Management

DataEngineX has a single version source:

- **dataenginex version**: root `pyproject.toml` — managed automatically by the `release.yml` workflow

```bash
# Releases are automated via release.yml.
# Push a v{X.Y.Z} tag to main; release.yml builds, publishes to PyPI, and creates a GitHub Release.
gh run list --workflow=release.yml          # monitor release workflow
```

On `main`, pushing a `v{X.Y.Z}` tag triggers:

- `release.yml` → builds the package, publishes to PyPI, creates GitHub Release

## Local Data Setup

### Path-Based (Local Dev)

```bash
mkdir -p ~/data/dex/{bronze,silver,gold}
```

### Optional Dependencies for Full Coverage

To achieve the ~79% code coverage, optional dependencies must be installed:

```bash
# Required for full test coverage:
pip install "dataenginex[cloud]"  # AWS/GCP/BigQuery connectors
pip install "dataenginex[delta]"  # Delta Lake connector
pip install "dataenginex[postgres]"  # PostgreSQL connector
pip install "dataenginex[qdrant]"  # Qdrant vector store
pip install "dataenginex[pytorch]"  # PyTorch ML
pip install "dataenginex[notebook]"  # Pandas
touch uv.lock  # Update lockfile
uv sync --reinstall
```

These connectors are excluded from coverage calculations when not installed:

- HTTP REST, SSE, CSV, Parquet, Spark, Delta Lake, dbt, PostgreSQL, BigQuery, Qdrant, MLflow, PyTorch, Pandas

### Optional Cloud Warehouse Adapter (Example: BigQuery)

Use this only when validating the cloud warehouse path; local development can run entirely on path-based storage.

```bash
export GCP_PROJECT=your-dex-project
bq mk --dataset dex_bronze
bq mk --dataset dex_silver
bq mk --dataset dex_gold
```

## Running Pipelines & Tests

### Example Scripts

```bash
# Medallion pipeline demo
uv run python examples/07_api_ingestion.py

# PySpark ML (requires Java 17+)
uv run python examples/08_spark_ml.py

# Feature engineering
uv run python examples/09_feature_engineering.py

# Model analysis + drift detection
uv run python examples/10_model_analysis.py
```

### Testing

```bash
# Run all tests with coverage (requires 80%)
uv run poe test-cov

# Run unit tests only
uv run poe test-unit

# Check code quality
uv run poe check-all
```

### Coverage Strategy

**Current Status**: ~79% (meets 79% threshold)

**What was omitted from coverage**: Optional dependency files to keep CI fast and focused:

```python
omit = [
    "*/src/dataenginex/data/connectors/delta.py",           # Requires `pip install "dataenginex[delta]"`
    "*/src/dataenginex/lakehouse/storage.py",               # Requires cloud storage extras
    "*/src/dataenginex/worker.py",                         # Worker module (if separate)
    "*/src/dataenginex/data/quality/spark.py",             # Requires PySpark
    "*/src/dataenginex/data/connectors/http.py",            # Requires network dependencies
    "*/src/dataenginex/data/connectors/rest.py",            # Requires REST client
    "*/src/dataenginex/data/connectors/sse.py",             # Requires SSE support
    "*/src/dataenginex/ml/mlflow_registry.py",              # Requires MLflow
]
```

**How to install all optional dependencies**: \`\`\`bash
pip install "dataenginex[cloud]" "dataenginex[delta]" "dataenginex[postgres]" \
"dataenginex[qdrant]" "dataenginex[queue]" "dataenginex[pytorch]" "dataenginex[notebook]" \
"dataenginex[ml]" "dataenginex[tracking]" "dataenginex[data]"
uv sync --reinstall

````

Expected coverage after installing all extras: 95%+

**Coverage Testing**: Run `uv run poe test-cov` to see current status and missing lines.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Pre-commit hooks fail | `uv run poe lint-fix` then retry |
| Tests fail locally but pass in CI | Check Python version (3.13+), run `uv sync --reinstall` |
| Import errors | Run `uv sync --reinstall` and restart the shell |
| PySpark examples fail | Check Java 17+ is installed (`java -version`) |

## Common Commands

```bash
uv run poe setup              # One-step setup (all deps + pre-commit hooks)
uv run poe check-all          # Run lint + typecheck + tests in sequence
uv run poe lint               # Ruff lint check
uv run poe lint-fix           # Auto-fix lint + format
uv run poe typecheck          # mypy strict type checking
uv run poe test               # Run all tests
uv run poe test-cov           # Tests with coverage report
uv run poe security           # pip-audit vulnerability scan
uv run poe pre-commit         # Run all pre-commit hooks
uv run poe clean              # Remove caches and build artifacts
````

## Resources & Support

- **Code Style**: See [contributing.md](./contributing.md)
- **Architecture**: See [architecture.md](./architecture.md)
- **Issues**: [GitHub Issues](https://github.com/TheDataEngineX/dataenginex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/orgs/TheDataEngineX/discussions)
