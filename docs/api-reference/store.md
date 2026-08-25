# DexStore

SQLite-backed persistence for all project metadata. Thread-safe, multi-process-safe (WAL mode).

```python
from pathlib import Path
from dataenginex.store import DexStore

store = DexStore(Path(".dex/store.duckdb"))
```

## Constructor

```python
DexStore(db_path: Path) -> None
```

## Pipeline Run Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `record_pipeline_run` | `(pipeline_name, success, rows_input=0, rows_output=0, steps_completed=0, duration_ms=0.0, error=None, skipped=False) -> PipelineRunRecord` | Record a pipeline run |
| `get_pipeline_runs` | `(pipeline_name: str \| None = None) -> list[PipelineRunRecord]` | Get run history |
| `get_last_pipeline_run` | `(pipeline_name: str) -> PipelineRunRecord \| None` | Last run for a pipeline |

## Lineage Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `record` | `(**kwargs) -> LineageEvent` | Record a lineage event |
| `get_lineage_event` | `(event_id: str) -> LineageEvent \| None` | Get event by ID |
| `get_lineage_children` | `(parent_id: str) -> list[LineageEvent]` | Child events |
| `get_lineage_by_pipeline` | `(pipeline_name: str) -> list[LineageEvent]` | Events for a pipeline |
| `get_lineage_by_layer` | `(layer: str) -> list[LineageEvent]` | Events for a layer |
| `lineage_summary` | `() -> dict` | Aggregate lineage stats |

## Model Registry Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_model` | `(artifact: ModelArtifact) -> ModelArtifact` | Register a model |
| `get_model` | `(name, version) -> ModelArtifact \| None` | Get specific version |
| `get_latest_model` | `(name: str) -> ModelArtifact \| None` | Latest version |
| `get_production_model` | `(name: str) -> ModelArtifact \| None` | Production-promoted model |
| `list_model_names` | `() -> list[str]` | All registered model names |
| `list_model_versions` | `(name: str) -> list[str]` | Versions for a model |
| `promote_model` | `(name, version, stage) -> ModelArtifact` | Promote to stage |
| `delete_model` | `(name, version=None) -> None` | Delete model(s) |

## Quality Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `record_quality_run` | `(results: dict) -> str` | Record quality check results |
| `get_quality_history` | `(limit=50) -> dict` | Historical quality data |

## Audit Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `log_audit` | `(action, resource="", resource_type="", actor="user", status="success", details=None, ip_address="") -> AuditEvent` | Log an audit event |
| `get_audit_events` | `(action=None, resource=None, actor=None, limit=100) -> list[AuditEvent]` | Query audit events |

## Memory Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `save_memory` | `(entry: MemoryEntry) -> None` | Save a memory entry |
| `get_recent_memory` | `(n=100) -> list[MemoryEntry]` | Recent memory entries |
| `search_memory` | `(query, top_k=5) -> list[MemoryEntry]` | Semantic search |

## Episode Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_episode` | `(episode: Episode) -> None` | Record an episode |
| `recall_episodes` | `(task, top_k=5) -> list[Episode]` | Recall relevant episodes |

## Catalog Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_catalog` | `(entry: CatalogEntry) -> CatalogEntry` | Register a catalog entry |
| `get_catalog` | `(name: str) -> CatalogEntry \| None` | Get by name |
| `search_catalog` | `(layer=None, name_contains=None) -> list[CatalogEntry]` | Search catalog |
| `all_catalog` | `() -> list[CatalogEntry]` | All entries |
| `delete_catalog` | `(name: str) -> None` | Delete entry |

## Data Classes

| Class | Description |
|-------|-------------|
| `PipelineRunRecord` | Pipeline run record |
| `LineageEvent` | Lineage event |
| `AuditEvent` | Audit log event |
| `MemoryEntry` | AI memory entry |
| `Episode` | AI episode |
| `CatalogEntry` | Data catalog entry |
| `ModelArtifact` | Registered model |
