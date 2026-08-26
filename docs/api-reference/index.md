# API Reference

Public API reference for the `dataenginex` library.

## Core

| Module | Description |
|--------|-------------|
| [DexEngine](engine.md) | Main entry point — config, pipelines, warehouse, ML, AI |
| [DexStore](store.md) | SQLite-backed persistence — runs, lineage, models, audit |
| [Config](config.md) | YAML config loading and validation |

## Pipelines

| Module | Description |
|--------|-------------|
| [PipelineRunner](pipeline-runner.md) | Execute data pipelines — extract, transform, load |

## CLI

| Module | Description |
|--------|-------------|
| [CLI Commands](cli.md) | `dex` command group — validate, run, worker, secops |

## Security

| Module | Description |
|--------|-------------|
| [SecOps](secops.md) | PII detection, masking, audit logging, PrivacyGuard |

## ML

| Module | Description |
|--------|-------------|
| [Training](training.md) | Model training — sklearn, PyTorch, sentence transformers |

## Foundation

| Module | Description |
|--------|-------------|
| [Workloads](workloads.md) | WorkloadKind, RunState, workload definitions |

## Engines

| Module | Description |
|--------|-------------|
| [Engine Registry](engine-registry.md) | Register and retrieve execution engines |
