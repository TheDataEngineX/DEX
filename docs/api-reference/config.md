# Config

YAML config loading and validation for `dex.yaml`.

```python
from dataenginex.config import load_config, validate_config

config = load_config("dex.yaml")
validate_config(config)
```

## Functions

### `load_config`

```python
load_config(path: str | Path) -> DexConfig
```

Load and parse a `dex.yaml` file. Resolves `${ENV_VAR}` references. Raises `ConfigError` if the file is missing or invalid.

### `validate_config`

```python
validate_config(config: DexConfig) -> None
```

Validate config against the schema. Raises `ConfigError` with details on invalid fields.

### `resolve_env_vars`

```python
resolve_env_vars(config: dict) -> dict
```

Resolve `${ENV_VAR}` and `${ENV_VAR:-default}` references in config values.

## DexConfig Schema

The `DexConfig` Pydantic model defines the full config structure:

```yaml
project:
  name: string
  version: string

data:
  sources:
    <name>:
      type: csv | parquet | postgres | spark | dbt | ...
      connection:
        path: string
        default_file: string
  pipelines:
    <name>:
      source: string
      engine: duckdb | spark | auto
      destination: string
      schedule: string  # cron expression
      transforms:
        - type: filter | cast | derive | sql | ...
          condition: string
      quality:
        row_count_min: int

ml:
  models:
    <name>:
      algorithm: string
      features: list[string]

ai:
  default_model: string
  embedding_model: string
  agents:
    <name>:
      runtime: builtin | langchain
      system_prompt: string

secops:
  audit:
    enabled: bool
    db_path: string
  guard:
    enabled: bool
```
