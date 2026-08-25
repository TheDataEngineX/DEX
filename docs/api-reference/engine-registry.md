# Engine Registry

Register and retrieve execution engines (DuckDB, Spark, etc.).

```python
from dataenginex.engines.registry import engine_registry

# Check available engines
engine_registry.list_engines()  # ['duckdb', 'spark']

# Get an engine
engine = engine_registry.get("duckdb")
```

## EngineRegistry

```python
class EngineRegistry:
    def __init__(self) -> None
    def register(self, name: str, engine: BaseEngine) -> None
    def get(self, name: str) -> BaseEngine
    def list_engines(self) -> list[str]
    def is_registered(self, name: str) -> bool
```

| Method | Description |
|--------|-------------|
| `register(name, engine)` | Register an engine. Raises `ValueError` on duplicate. |
| `get(name)` | Get engine by name. Raises `KeyError` if not found. |
| `list_engines()` | List registered engine names. |
| `is_registered(name)` | Check if an engine is registered. |

## Singleton

```python
from dataenginex.engines.registry import engine_registry
```

The module-level `engine_registry` is a pre-populated singleton with DuckDB and Spark engines registered at import time.

## Built-in Engines

| Engine | Module | Capabilities |
|--------|--------|-------------|
| `duckdb` | `dataenginex.engines.duckdb_engine` | Local, embedded, SQL, Delta read/write |
| `spark` | `dataenginex.engines.spark_engine` | Distributed, streaming, ML, Catalyst |
