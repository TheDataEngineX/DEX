# DataEngineX Documentation

**The Python library (PyPI) — engine, config, CLI, pipelines, ML, AI, PrivacyGuard.**

Documentation for the core `dataenginex` library. For the web UI, see [dex-studio](https://github.com/TheDataEngineX/dex-studio).

## Quick start

```bash
pip install dataenginex
```

or

```bash
uv add dataenginex
```

See [Getting Started](getting-started.md) for a full walkthrough.

## Guides

- [Getting Started](getting-started.md) — Install and run your first pipeline
- [Architecture](architecture.md) — Core patterns, module map, design decisions
- [Development Setup](development.md) — Prerequisites, workflow, troubleshooting
- [Release Notes](release-notes.md) — Full version history
- [Observability](observability.md) — Metrics, logging, tracing
- [CI/CD](ci-cd.md) — CI pipeline and PyPI release automation
- [Contributing](contributing.md) — How to contribute to the project

## API Reference

- [DexEngine](api-reference/engine.md) — Main entry point
- [DexStore](api-reference/store.md) — Persistence layer
- [Config](api-reference/config.md) — YAML config loading
- [PipelineRunner](api-reference/pipeline-runner.md) — Pipeline execution
- [CLI Commands](api-reference/cli.md) — `dex` command group
- [SecOps](api-reference/secops.md) — PII, masking, audit, PrivacyGuard
- [Training](api-reference/training.md) — ML model training
- [Workloads](api-reference/workloads.md) — WorkloadKind, RunState
- [Engine Registry](api-reference/engine-registry.md) — Engine registration
