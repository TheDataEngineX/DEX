# PipelineRunner

Execute data pipelines defined in `dex.yaml`. Flow: Config → Extract → Register views → Transform chain → Quality gate → Load.

```python
from pathlib import Path
from dataenginex.config import load_config
from dataenginex.domains.data.pipeline.runner import PipelineRunner

config = load_config("dex.yaml")
runner = PipelineRunner(config, data_dir=Path(".dex/lakehouse"))
result = runner.run("clean_users")
```

## Constructor

```python
PipelineRunner(
    config: DexConfig,
    data_dir: Path | None = None,
    project_dir: Path | None = None,
    lineage: LineageBackend | None = None,
    feature_store: Any = None,
    vector_store: Any = None,
    embed_fn: Any = None,
    lexical_backend: Any = None,
    lexical_backends: dict[str, Any] | None = None,
    engine_ref: Any = None,
) -> None
```

## Methods

### `run`

```python
run(
    pipeline_name: str,
    *,
    dry_run: bool = False,
    progress_cb: Callable[[str, int, int], None] | None = None,
    checkpoint_cb: Callable[[str], None] | None = None,
) -> PipelineResult
```

Execute a named pipeline. Returns `PipelineResult` with success status, row counts, and error details.

### `run_all`

```python
run_all() -> dict[str, PipelineResult]
```

Execute all pipelines in dependency order.

### `preview`

```python
preview(pipeline_name: str, sample: int = 200_000) -> dict[str, Any]
```

Preview pipeline output without writing to the lakehouse.

## PipelineResult

```python
@dataclass(frozen=True)
class PipelineResult:
    pipeline: str
    success: bool
    rows_input: int
    rows_output: int
    steps_completed: int
    dry_run: bool
    error: str | None
    skipped: bool
```
