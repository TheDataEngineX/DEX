# DexEngine

Main entry point for DataEngineX. Initializes all subsystems from a `dex.yaml` config file.

```python
from dataenginex.engine import DexEngine

engine = DexEngine("dex.yaml")
```

## Constructor

```python
DexEngine(config_path: str | Path) -> None
```

Loads config, creates `.dex/` directory, initializes DuckDB store, plugin registry, ML subsystems, vector store, pipeline runner, AI layer, PrivacyGuard, and Spark subsystems.

## Properties

| Property | Return type | Description |
|----------|-------------|-------------|
| `model_registry` | `Any` | Model registry from serving engine |
| `lineage` | `Any` | DexStore instance (lineage backend) |
| `ai_long_memory` | `Any` | Short-term memory for AI agents |

## Pipeline Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `run_pipeline` | `(name: str, *, progress_cb=None, checkpoint_cb=None) -> Any` | Run a named pipeline |
| `pipeline_stats` | `() -> dict[str, int]` | Row counts per pipeline |
| `pipeline_last_run` | `(name: str) -> Any \| None` | Last run record for a pipeline |
| `update_pipeline_schedule` | `(name: str, schedule: str \| None) -> None` | Update cron schedule |
| `add_pipeline` | `(name, source, schedule="", destination="", engine="duckdb") -> None` | Add a pipeline |
| `add_pipeline_transform` | `(pipeline: str, step: dict) -> None` | Append a transform step |
| `delete_pipeline_transform` | `(pipeline: str, index: int) -> None` | Remove a transform by index |
| `reorder_pipeline_transform` | `(pipeline: str, index: int, direction: int) -> None` | Move a transform step |
| `preview_flow` | `(name: str) -> dict[str, Any]` | Preview pipeline output |
| `delete_pipeline` | `(name: str) -> None` | Delete a pipeline |

## Source Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_source` | `(name, type_, path="", url="", connection=None) -> None` | Register a data source |
| `delete_source` | `(name: str) -> None` | Remove a data source |
| `source_row_count` | `(source_name: str) -> int \| None` | Row count for a source |
| `source_schema` | `(source_name: str) -> list[dict] \| None` | Column schema for a source |
| `source_sample` | `(source_name: str, limit=10, offset=0) -> list[dict] \| None` | Sample rows from a source |
| `source_stats` | `(source_name: str) -> dict \| None` | Statistics for a source |

## Warehouse Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `warehouse_layers` | `() -> list[dict]` | List available layers (bronze/silver/gold) |
| `warehouse_tables` | `(layer: str) -> list[dict]` | Tables in a layer |
| `warehouse_table_schema` | `(table_name, layer) -> list[dict]` | Column schema for a table |
| `warehouse_table_stats` | `(table_name, layer) -> dict` | Row count, size, etc. |
| `warehouse_table_lineage` | `(table_name, layer) -> dict` | Lineage for a table |

## Quality Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `quality_check_table` | `(table_name: str) -> dict \| None` | Quality results for a table |
| `quality_check_all_tables` | `() -> dict` | Quality results for all tables |
| `quality_history` | `() -> dict` | Historical quality results |

## AI / Agent Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_agent` | `(name, runtime="builtin", system_prompt="") -> None` | Create an AI agent |
| `delete_agent` | `(name: str) -> None` | Remove an AI agent |
| `delete_model` | `(name: str) -> None` | Remove a registered model |
| `trigger_ai_index_refresh` | `() -> None` | Refresh AI search indices |

## Lifecycle

| Method | Signature | Description |
|--------|-----------|-------------|
| `health` | `() -> dict[str, Any]` | Engine health status |
| `close` | `() -> None` | Release resources |
