"""DexEngine — primary entry point for the dataenginex library.

Usage::

    from dataenginex import DexEngine

    engine = DexEngine("path/to/dex.yaml")
    result = engine.run_pipeline("ingest")
    engine.close()

The engine initialises all subsystems (data, ML, AI) from the config file
and persists all state to the SQLite-backed ``.dex/store.duckdb`` file
alongside ``dex.yaml``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import duckdb
import structlog
import yaml

from dataenginex.config import load_config, validate_config
from dataenginex.config.schema import DexConfig
from dataenginex.domains.data.catalog import CatalogEntry as LakehouseCatalogEntry
from dataenginex.domains.data.catalog import DataCatalog
from dataenginex.foundation.ids import ProjectId
from dataenginex.plugins.registry import PluginRegistry, discover
from dataenginex.store import DexStore

logger = structlog.get_logger()

__all__ = ["DexBackend", "DexEngine"]


# ---------------------------------------------------------------------------
# Protocol — structural interface consumed by state classes / tests
# ---------------------------------------------------------------------------


@runtime_checkable
class DexBackend(Protocol):
    """Structural interface satisfied by DexEngine.

    Depend on this Protocol rather than DexEngine to keep state classes
    and tests decoupled from the concrete implementation.
    """

    config: Any
    store: Any
    agents: dict[str, Any]
    tracker: Any
    feature_store: Any
    serving_engine: Any
    llm: Any
    ai_memory: Any
    ai_metrics: Any
    ai_episodic: Any
    secops_audit: Any
    privacy_guard: Any
    plugins: Any
    catalog: Any
    project_dir: Any

    @property
    def ai_long_memory(self) -> Any: ...
    @property
    def model_registry(self) -> Any: ...
    @property
    def lineage(self) -> Any: ...

    def run_pipeline(
        self,
        name: str,
        *,
        progress_cb: Any = None,
        checkpoint_cb: Any = None,
    ) -> Any: ...
    def pipeline_stats(self) -> dict[str, int]: ...
    def pipeline_last_run(self, name: str) -> Any | None: ...
    def update_pipeline_schedule(self, name: str, schedule: str | None) -> None: ...

    def warehouse_layers(self) -> list[dict[str, Any]]: ...
    def warehouse_tables(self, layer: str) -> list[dict[str, Any]]: ...
    def warehouse_table_schema(self, table_name: str, layer: str) -> list[dict[str, Any]]: ...
    def warehouse_table_stats(self, table_name: str, layer: str) -> dict[str, Any]: ...
    def warehouse_table_lineage(self, table_name: str, layer: str) -> dict[str, Any]: ...

    def source_row_count(self, source_name: str) -> int | None: ...
    def source_schema(self, source_name: str) -> list[dict[str, Any]] | None: ...
    def source_sample(
        self, source_name: str, limit: int = 10, offset: int = 0
    ) -> list[dict[str, Any]] | None: ...
    def source_stats(self, source_name: str) -> dict[str, Any] | None: ...

    def quality_check_table(self, table_name: str) -> dict[str, Any] | None: ...
    def quality_check_all_tables(self) -> dict[str, Any]: ...
    def quality_history(self) -> dict[str, Any]: ...

    def add_pipeline(self, name: str, source: str, schedule: str, destination: str) -> None: ...
    def add_pipeline_transform(self, pipeline: str, step: dict[str, Any]) -> None: ...
    def delete_pipeline_transform(self, pipeline: str, index: int) -> None: ...
    def reorder_pipeline_transform(self, pipeline: str, index: int, direction: int) -> None: ...
    def preview_flow(self, name: str) -> dict[str, Any]: ...
    def delete_pipeline(self, name: str) -> None: ...
    def add_source(
        self,
        name: str,
        type_: str,
        path: str = "",
        url: str = "",
        connection: dict[str, Any] | None = None,
    ) -> None: ...
    def delete_source(self, name: str) -> None: ...
    def add_agent(self, name: str, runtime: str, system_prompt: str) -> None: ...
    def delete_agent(self, name: str) -> None: ...
    def delete_model(self, name: str) -> None: ...

    def health(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# DexEngine
# ---------------------------------------------------------------------------


class DexEngine:
    """Local dataenginex engine — direct library access, no HTTP.

    Initialises all subsystems from a ``dex.yaml`` config file.
    All persistent state is stored in the SQLite-backed ``.dex/store.duckdb``
    file (see :class:`~dataenginex.store.DexStore`) next to the config file.

    Args:
        config_path: Path to a ``dex.yaml`` file.
    """

    _SOURCE_EXT_MAP: dict[str, str] = {
        "csv": "*.csv",
        "parquet": "*.parquet",
        "json": "*.json",
        "jsonl": "*.jsonl",
    }

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            msg = f"Config file not found: {self.config_path}"
            raise FileNotFoundError(msg)

        self.config: DexConfig = load_config(self.config_path)
        validate_config(self.config)

        self.project_dir = self.config_path.parent
        self._dex_dir = self.project_dir / ".dex"
        self._dex_dir.mkdir(parents=True, exist_ok=True)

        # Single persistent store — replaces all JSON files
        self.store = DexStore(self._dex_dir / "store.duckdb")

        # Lakehouse catalog — shares the same SQLite store; eliminates split-brain
        self.catalog = DataCatalog(store=self.store)

        # Plugin discovery must precede subsystem construction so project-local
        # connector/transform registrations are available to the pipeline runner.
        self.plugins = PluginRegistry()
        self._load_plugins()
        from dataenginex.domains.plugins.loader import load_project_plugins

        load_project_plugins(self.project_dir)

        # ML backends — init before pipeline runner so feature_store is available
        self.tracker: Any = self._init_ml_tracker()
        self.feature_store: Any = self._init_ml_feature_store()
        self.serving_engine: Any = self._init_ml_serving()

        # Vector store — init before pipeline runner and AI so both can share it
        self._vector_store: Any = None
        self._embed_fn: Any = None
        self._vector_store, self._embed_fn = self._init_vector_store()
        self._lexical_backends: dict[str, Any] = self._init_lexical_backends()
        self._lexical_backend: Any = self._lexical_backends.get("movies")
        if self._lexical_backend is None and self._lexical_backends:
            self._lexical_backend = next(iter(self._lexical_backends.values()))

        # Data pipeline runner — has access to feature store + vector store
        from dataenginex.domains.data.pipeline.runner import PipelineRunner

        self.pipeline_runner = PipelineRunner(
            self.config,
            data_dir=self._dex_dir / "lakehouse",
            project_dir=self.project_dir,
            lineage=self.store,
            feature_store=self.feature_store,
            vector_store=self._vector_store,
            embed_fn=self._embed_fn,
            lexical_backend=self._lexical_backend,
            lexical_backends=self._lexical_backends,
            engine_ref=self,
        )

        # AI backends — ingest existing lakehouse into vector store, init agents
        self.llm: Any = None
        self.agents: dict[str, Any] = {}
        self._init_ai()

        # AI layer — memory, routing, sandbox
        self.ai_memory: Any = None
        self.ai_episodic: Any = None
        self.ai_metrics: Any = None
        self.sandbox: Any = None
        self.model_router: Any = None
        # SecOps AuditLogger (secops.audit config)
        self.secops_audit: Any = None
        # PrivacyGuard must initialise even if the AI layer fails — it's a
        # security primitive consumed independently (e.g. by dex-studio's
        # /privacy/* UI). Initialise before _init_ai_layer so the guard is
        # always available, then wire it into the router inside the AI init.
        self._init_privacy_guard()
        self._init_ai_layer()

        # Spark subsystem — cluster, SQL, catalog, streaming, ML, lineage
        self.spark_client: Any = None
        self.spark_sql: Any = None
        self.spark_catalog: Any = None
        self.spark_streaming: Any = None
        self.spark_ml: Any = None
        self.spark_metrics: Any = None
        self.spark_lineage: Any = None
        self.spark_datasource_registry: Any = None
        self._init_spark()

        logger.info(
            "DexEngine ready",
            project=self.config.project.name,
            config=str(self.config_path),
        )

    # -------------------------------------------------------------------------
    # Convenience properties — expose subsystem attributes used by Studio
    # -------------------------------------------------------------------------

    @property
    def model_registry(self) -> Any:
        if self.serving_engine is not None and hasattr(self.serving_engine, "_registry"):
            return self.serving_engine._registry
        from dataenginex.domains.ml.registry import ModelRegistry

        return ModelRegistry(store=self.store)

    @property
    def lineage(self) -> Any:
        return self.store

    @property
    def ai_long_memory(self) -> Any:
        return self.ai_memory

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    def run_pipeline(
        self,
        name: str,
        *,
        progress_cb: Any = None,
        checkpoint_cb: Any = None,
    ) -> Any:
        import time

        from dataenginex.domains.data.pipeline.runner import PipelineResult

        start = time.monotonic()
        result: PipelineResult = self.pipeline_runner.run(
            name,
            progress_cb=progress_cb,
            checkpoint_cb=checkpoint_cb,
        )
        duration_ms = (time.monotonic() - start) * 1000

        self.store.record_pipeline_run(
            pipeline_name=name,
            success=result.success,
            rows_input=result.rows_input,
            rows_output=result.rows_output,
            steps_completed=result.steps_completed,
            duration_ms=duration_ms,
            error=result.error,
            skipped=result.skipped,
        )
        self.store.log_audit(
            action="pipeline_run",
            resource=name,
            resource_type="pipeline",
            status="success" if result.success else "failure",
            details={
                "rows_input": result.rows_input,
                "rows_output": result.rows_output,
                "duration_ms": round(duration_ms, 2),
                "error": result.error,
            },
        )

        # Register output table in catalog if pipeline has a destination
        pipeline_cfg = self.config.data.pipelines.get(name)
        if pipeline_cfg and pipeline_cfg.destination and result.success:
            target_layer = (pipeline_cfg.target or {}).get("layer", "silver")
            target_format = (pipeline_cfg.target or {}).get("format", "parquet")
            table_path = (
                self._dex_dir
                / "lakehouse"
                / target_layer
                / (
                    f"{pipeline_cfg.destination}.parquet"
                    if target_format == "parquet"
                    else pipeline_cfg.destination
                )
            )
            if table_path.exists():
                self.catalog.register(
                    LakehouseCatalogEntry(
                        name=pipeline_cfg.destination,
                        layer=target_layer,
                        format=target_format,
                        location=str(table_path),
                        record_count=result.rows_output,
                    )
                )

        return result

    def pipeline_stats(self) -> dict[str, int]:
        pipelines = self.config.data.pipelines or {}
        total = len(pipelines)
        scheduled = sum(1 for p in pipelines.values() if p.schedule)
        failed = sum(
            1
            for name in pipelines
            if (run := self.store.get_last_pipeline_run(name)) and not run.success
        )
        return {"total": total, "scheduled": scheduled, "failed": failed, "running": 0}

    def pipeline_last_run(self, name: str) -> Any | None:
        return self.store.get_last_pipeline_run(name)

    def update_pipeline_schedule(self, name: str, schedule: str | None) -> None:
        if name not in self.config.data.pipelines:
            msg = f"Pipeline '{name}' not found"
            raise KeyError(msg)
        self.config.data.pipelines[name] = self.config.data.pipelines[name].model_copy(
            update={"schedule": schedule}
        )
        self._save_config()

    # -------------------------------------------------------------------------
    # CRUD — pipelines / sources / agents / models
    # -------------------------------------------------------------------------

    def add_pipeline(
        self, name: str, source: str, schedule: str = "", destination: str = "",
        engine: Literal["duckdb", "spark", "auto"] = "duckdb",
    ) -> None:
        from dataenginex.config.schema import PipelineConfig

        self.config.data.pipelines[name] = PipelineConfig(
            source=source, schedule=schedule or None, destination=destination or None, engine=engine
        )
        self._save_config()

    def add_pipeline_transform(self, pipeline: str, step: dict[str, Any]) -> None:
        """Append a transform step to a pipeline and persist to dex.yaml."""
        from dataenginex.config.schema import TransformStepConfig

        if pipeline not in self.config.data.pipelines:
            msg = f"Pipeline '{pipeline}' not found"
            raise KeyError(msg)
        self.config.data.pipelines[pipeline].transforms.append(TransformStepConfig(**step))
        self._save_config()

    def delete_pipeline_transform(self, pipeline: str, index: int) -> None:
        """Remove the transform step at *index* from a pipeline and persist."""
        if pipeline not in self.config.data.pipelines:
            msg = f"Pipeline '{pipeline}' not found"
            raise KeyError(msg)
        steps = self.config.data.pipelines[pipeline].transforms
        if 0 <= index < len(steps):
            steps.pop(index)
            self._save_config()

    def reorder_pipeline_transform(self, pipeline: str, index: int, direction: int) -> None:
        """Move a transform step up (-1) or down (+1) and persist."""
        if pipeline not in self.config.data.pipelines:
            msg = f"Pipeline '{pipeline}' not found"
            raise KeyError(msg)
        steps = self.config.data.pipelines[pipeline].transforms
        j = index + direction
        if 0 <= index < len(steps) and 0 <= j < len(steps):
            steps[index], steps[j] = steps[j], steps[index]
            self._save_config()

    def preview_flow(self, name: str) -> dict[str, Any]:
        """Per-stage row counts for the pipeline flow canvas (sampled, scaled)."""
        return self.pipeline_runner.preview(name)

    def delete_pipeline(self, name: str) -> None:
        self.config.data.pipelines.pop(name, None)
        self._save_config()

    def add_source(
        self,
        name: str,
        type_: str,
        path: str = "",
        url: str = "",
        connection: dict[str, Any] | None = None,
    ) -> None:
        from dataenginex.config.schema import SourceConfig

        self.config.data.sources[name] = SourceConfig(
            type=type_,
            path=path or None,
            url=url or None,
            connection=connection or {},
        )
        self._save_config()

    def delete_source(self, name: str) -> None:
        self.config.data.sources.pop(name, None)
        self._save_config()

    def add_agent(self, name: str, runtime: str = "builtin", system_prompt: str = "") -> None:
        from dataenginex.config.schema import AgentConfig

        self.config.ai.agents[name] = AgentConfig(runtime=runtime, system_prompt=system_prompt)
        self._save_config()

    def delete_agent(self, name: str) -> None:
        if self.config.ai:
            self.config.ai.agents.pop(name, None)
        self._save_config()

    def delete_model(self, name: str) -> None:
        self.store.delete_model(name)

    # -------------------------------------------------------------------------
    # Warehouse — reads from .dex/lakehouse/, registers in catalog
    # -------------------------------------------------------------------------

    def warehouse_layers(self) -> list[dict[str, Any]]:
        lakehouse = self._dex_dir / "lakehouse"
        layers: list[dict[str, Any]] = []
        for layer_name in ("bronze", "silver", "gold"):
            layer_path = lakehouse / layer_name
            pq = self._SOURCE_EXT_MAP["parquet"]
            if layer_path.exists():
                delta_count = sum(
                    1
                    for path in layer_path.iterdir()
                    if path.is_dir() and (path / "_delta_log").exists()
                )
                table_count = len(list(layer_path.glob(pq))) + delta_count
            else:
                table_count = 0
            layers.append({"name": layer_name, "table_count": table_count})
        return layers

    def warehouse_tables(self, layer: str) -> list[dict[str, Any]]:
        import datetime

        layer_path = self._dex_dir / "lakehouse" / layer
        if not layer_path.exists():
            return []
        tables: list[dict[str, Any]] = []
        parquet_tables = sorted(layer_path.glob(self._SOURCE_EXT_MAP["parquet"]))
        delta_tables = sorted(
            path
            for path in layer_path.iterdir()
            if path.is_dir() and (path / "_delta_log").exists()
        )
        for f in [*parquet_tables, *delta_tables]:
            try:
                stat = f.stat()
                size_bytes = (
                    stat.st_size
                    if f.is_file()
                    else sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
                )
                size = (
                    f"{size_bytes / 1024:.1f} KB"
                    if size_bytes < 1_048_576
                    else f"{size_bytes / 1_048_576:.1f} MB"
                )
                updated_at = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%b %d %H:%M")
                row_count: int | None = None
                table_format = "parquet" if f.is_file() else "delta"
                with contextlib.suppress(Exception):
                    if table_format == "parquet":
                        with self._duckdb_ro() as conn:
                            row = conn.execute(
                                f"SELECT COUNT(*) FROM read_parquet('{f}')"
                            ).fetchone()
                            row_count = int(row[0]) if row else None
                    else:
                        row_count = self._delta_row_count(f)

                # Register/refresh in catalog
                self.catalog.register(
                    LakehouseCatalogEntry(
                        name=f.stem,
                        layer=layer,
                        format=table_format,
                        location=str(f),
                        record_count=row_count or 0,
                    )
                )
                tables.append(
                    {
                        "name": f.stem,
                        "path": str(f),
                        "size_bytes": size_bytes,
                        "size": size,
                        "row_count": row_count,
                        "updated_at": updated_at,
                        "format": table_format,
                    }
                )
            except OSError:
                continue
        return tables

    def _lakehouse_table_location(self, table_name: str, layer: str) -> tuple[Path, str]:
        layer_path = self._dex_dir / "lakehouse" / layer
        parquet_path = layer_path / f"{table_name}.parquet"
        if parquet_path.exists():
            return parquet_path, "parquet"
        return layer_path / table_name, "delta"

    def _lakehouse_scan_sql(self, table_path: Path, table_format: str) -> str:
        if table_format == "parquet":
            safe_path = str(table_path).replace("'", "''")
            return f"read_parquet('{safe_path}')"
        from dataenginex.providers.object_store.storage import DeltaStorage

        return DeltaStorage(base_path=str(table_path.parent)).parquet_scan_sql(table_path.name)

    def _delta_row_count(self, table_path: Path) -> int | None:
        """Row count for a Delta table from transaction-log stats, no data scan.

        ``SELECT COUNT(*)`` over ``delta_scan()`` reads every row of every
        file — fine for small tables, but a full scan of a 100M+-row table
        (e.g. IMDB's bronze_principals) on every catalog listing is what was
        OOM-crashing the container repeatedly. Delta's add-file actions
        already carry ``num_records`` per file; summing that is O(files),
        not O(rows).
        """
        try:
            from deltalake import DeltaTable  # noqa: PLC0415

            actions = DeltaTable(str(table_path)).get_add_actions(flatten=True)
            if "num_records" not in actions.column_names:
                raise ValueError("no num_records stats")  # noqa: TRY301
            return int(sum(actions.column("num_records").to_pylist()))
        except Exception:
            with self._duckdb_ro() as conn:
                scan = self._lakehouse_scan_sql(table_path, "delta")
                row = conn.execute(f"SELECT COUNT(*) FROM {scan}").fetchone()
                return int(row[0]) if row else None

    def warehouse_table_schema(self, table_name: str, layer: str) -> list[dict[str, Any]]:
        table_path, table_format = self._lakehouse_table_location(table_name, layer)
        if not table_path.exists():
            return []
        try:
            with self._duckdb_ro() as conn:
                scan = self._lakehouse_scan_sql(table_path, table_format)
                conn.execute(f"CREATE VIEW IF NOT EXISTS _wts AS SELECT * FROM {scan}")
                cols = conn.execute("DESCRIBE _wts").fetchall()
                conn.execute("DROP VIEW IF EXISTS _wts")
                return [{"name": r[0], "dtype": r[1], "nullable": r[2] == "YES"} for r in cols]
        except Exception as exc:
            logger.warning("schema fetch failed", table=table_name, layer=layer, error=str(exc))
            return []

    def warehouse_table_stats(self, table_name: str, layer: str) -> dict[str, Any]:
        table_path, table_format = self._lakehouse_table_location(table_name, layer)
        if not table_path.exists():
            return {}
        size = (
            table_path.stat().st_size
            if table_path.is_file()
            else sum(p.stat().st_size for p in table_path.rglob("*") if p.is_file())
        )
        schema = self.warehouse_table_schema(table_name, layer)
        row_count = None
        with contextlib.suppress(Exception), self._duckdb_ro() as conn:
            scan = self._lakehouse_scan_sql(table_path, table_format)
            result = conn.execute(f"SELECT COUNT(*) FROM {scan}").fetchone()
            row_count = result[0] if result else None
        return {
            "size_bytes": size,
            "column_count": len(schema),
            "row_count": row_count,
            "format": table_format,
        }

    def warehouse_table_lineage(self, table_name: str, layer: str) -> dict[str, Any]:
        upstream: list[dict[str, Any]] = []
        downstream: list[dict[str, Any]] = []
        for ev in self.store.all_events:
            if ev.destination == table_name and ev.layer == layer and ev.parent_id:
                parent = self.store.get_lineage_event(ev.parent_id)
                if parent:
                    upstream.append(
                        {
                            "name": parent.destination,
                            "layer": parent.layer,
                            "event_id": parent.event_id,
                        }
                    )
            if ev.source == table_name and ev.layer == layer:
                for child in self.store.get_lineage_children(ev.event_id):
                    downstream.append(
                        {
                            "name": child.destination,
                            "layer": child.layer,
                            "event_id": child.event_id,
                        }
                    )
        return {"upstream": upstream, "downstream": downstream}

    # -------------------------------------------------------------------------
    # Sources
    # -------------------------------------------------------------------------

    def _source_read_fn(self, source_name: str) -> str | None:
        src = self.config.data.sources.get(source_name)
        if src is None:
            return None
        type_str = src.type.value if hasattr(src.type, "value") else str(src.type)
        _map = {
            "csv": "read_csv_auto",
            "parquet": "read_parquet",
            "json": "read_json_auto",
            "jsonl": "read_ndjson_auto",
        }
        return _map.get(type_str)

    def _source_path(self, source_name: str) -> tuple[Any, Path] | None:
        """Return (src_config, base_path) — base_path may be a dir or file."""
        src = self.config.data.sources.get(source_name)
        if src is None:
            return None
        raw = getattr(src, "path", None) or getattr(src, "uri", None)
        if not raw:
            return None
        p = Path(str(raw).split("*")[0].rstrip("/"))  # strip glob suffix for base
        if not p.is_absolute():
            p = self.project_dir / p
        return src, p.resolve()

    def _source_query_path(self, source_name: str) -> str | None:
        """Return a DuckDB-queryable path string (glob for directories/wildcards)."""
        src = self.config.data.sources.get(source_name)
        if src is None:
            return None
        raw = getattr(src, "path", None) or getattr(src, "uri", None)
        if not raw:
            return None
        raw_str = str(raw)
        # explicit glob pattern — resolve the base portion, keep glob suffix
        if "*" in raw_str:
            parts = raw_str.split("*", 1)
            base = Path(parts[0]) if Path(parts[0]).is_absolute() else self.project_dir / parts[0]
            return str(base.resolve()) + "*" + parts[1]
        p = Path(raw_str)
        if not p.is_absolute():
            p = self.project_dir / p
        resolved = p.resolve()
        # directory → build glob from type extension
        if resolved.is_dir():
            type_str = src.type.value if hasattr(src.type, "value") else str(src.type)
            glob_pat = self._SOURCE_EXT_MAP.get(type_str, "*.*")
            return str(resolved / glob_pat)
        return str(resolved)

    def source_row_count(self, source_name: str) -> int | None:
        qpath = self._source_query_path(source_name)
        if qpath is None:
            return None
        read_fn = self._source_read_fn(source_name)
        if read_fn is None:
            return None
        with contextlib.suppress(Exception), duckdb.connect(":memory:") as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {read_fn}('{qpath}')").fetchone()
            return row[0] if row else 0
        return None

    def source_schema(self, source_name: str) -> list[dict[str, Any]] | None:
        qpath = self._source_query_path(source_name)
        if qpath is None:
            return None
        read_fn = self._source_read_fn(source_name)
        if read_fn is None:
            return None
        with contextlib.suppress(Exception), duckdb.connect(":memory:") as conn:
            rows = conn.execute(f"DESCRIBE SELECT * FROM {read_fn}('{qpath}') LIMIT 1").fetchall()
            return [
                {"column_name": r[0], "column_type": r[1], "nullable": r[3] != "NO"} for r in rows
            ]
        return None

    def source_sample(
        self, source_name: str, limit: int = 10, offset: int = 0
    ) -> list[dict[str, Any]] | None:
        qpath = self._source_query_path(source_name)
        if qpath is None:
            return None
        read_fn = self._source_read_fn(source_name)
        if read_fn is None:
            return None
        with contextlib.suppress(Exception), duckdb.connect(":memory:") as conn:
            cursor = conn.execute(
                f"SELECT * FROM {read_fn}('{qpath}') LIMIT {limit} OFFSET {offset}"
            )
            col_names = [d[0] for d in cursor.description] if cursor.description else []
            return [dict(zip(col_names, row, strict=True)) for row in cursor.fetchall()]
        return None

    def source_stats(self, source_name: str) -> dict[str, Any] | None:
        result = self._source_path(source_name)
        if result is None:
            return None
        src, base_path = result
        # for directories/globs, sum sizes of all matched files
        if base_path.is_dir():
            type_str = src.type.value if hasattr(src.type, "value") else str(src.type)
            glob_pat = self._SOURCE_EXT_MAP.get(type_str, "*.*")
            size_bytes = sum(f.stat().st_size for f in base_path.glob(glob_pat) if f.is_file())
        else:
            size_bytes = base_path.stat().st_size if base_path.exists() else 0
        schema = self.source_schema(source_name)
        return {
            "row_count": self.source_row_count(source_name),
            "column_count": len(schema) if schema else None,
            "size_bytes": size_bytes,
            "path": str(base_path),
            "connector_type": getattr(src, "type", "unknown"),
        }

    # -------------------------------------------------------------------------
    # Quality
    # -------------------------------------------------------------------------

    def quality_check_table(self, table_name: str) -> dict[str, Any] | None:
        from dataenginex.domains.analytics.quality.gates import ColumnSpec, check_quality

        layer, _, tbl = table_name.partition(".")
        if not layer or not tbl:
            return None
        table_path, table_format = self._lakehouse_table_location(tbl, layer)
        if not table_path.exists():
            return None
        try:
            with self._duckdb_ro() as conn:
                scan = self._lakehouse_scan_sql(table_path, table_format)
                conn.execute(f"CREATE VIEW IF NOT EXISTS _qc AS SELECT * FROM {scan}")
                col_result = conn.execute("DESCRIBE _qc").fetchall()
                column_names = [r[0] for r in col_result]
                column_specs = [
                    ColumnSpec(name=r[0], dtype=r[1], nullable=True) for r in col_result
                ]
                result = check_quality(
                    conn,
                    "_qc",
                    completeness=0.0,
                    uniqueness=column_names,
                    schema=column_specs,
                )
                conn.execute("DROP VIEW IF EXISTS _qc")
                custom_weight = 1.0 if result.custom_passed else 0.0
                score = (
                    result.completeness_score * 0.4
                    + result.uniqueness_score * 0.3
                    + custom_weight * 0.3
                )
                return {
                    "score": score,
                    "completeness": result.completeness_score,
                    "uniqueness": result.uniqueness_score,
                    "custom_passed": result.custom_passed,
                    "schema_violations": result.schema_violations,
                    "passed": result.passed,
                    "details": result.details,
                }
        except Exception as exc:
            logger.warning("quality check failed", table=table_name, error=str(exc))
            return None

    def quality_check_all_tables(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for layer in ("bronze", "silver", "gold"):
            for table in self.warehouse_tables(layer):
                full_name = f"{layer}.{table['name']}"
                try:
                    results[full_name] = self.quality_check_table(full_name)
                except Exception:
                    logger.exception("quality check failed", table=full_name)
                    results[full_name] = None
        # Log summary
        passed = sum(1 for r in results.values() if r and r.get("passed"))
        failed = sum(1 for r in results.values() if r and not r.get("passed"))
        errored = sum(1 for r in results.values() if r is None)
        logger.info(
            "quality check all tables complete",
            total=len(results),
            passed=passed,
            failed=failed,
            errored=errored,
        )
        self.store.record_quality_run(results)
        return results

    def quality_history(self) -> dict[str, Any]:
        return self.store.get_quality_history()

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "project": self.config.project.name,
            "components": {
                "pipeline_runner": self.pipeline_runner is not None,
                "store": self.store is not None,
                "catalog": self.catalog is not None,
                "tracker": self.tracker is not None,
                "feature_store": self.feature_store is not None,
                "serving_engine": self.serving_engine is not None,
                "llm": self.llm is not None,
                "agents": len(self.agents),
                "ai_memory": self.ai_memory is not None,
                "plugins": self.plugins.count,
            },
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.feature_store and hasattr(self.feature_store, "close"):
                self.feature_store.close()
        self.store.close()
        logger.info("DexEngine shutdown", project=self.config.project.name)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @contextlib.contextmanager
    def _duckdb_ro(self) -> Iterator[Any]:
        """Yield a fresh in-memory DuckDB connection for read-only parquet queries.

        Keeps read traffic off the shared store connection so pipeline runs
        executing in a thread-pool executor don't race with web-request reads.
        """
        conn = duckdb.connect(":memory:")
        try:
            yield conn
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def _save_config(self) -> None:
        """Write config atomically: temp file → os.replace.

        Strips None/empty values so the file stays readable. Writes a .bak
        alongside the config before every save so the user can recover.
        """
        import os

        dump = self._config_dump_clean()
        tmp = self.config_path.with_suffix(".yaml.tmp")
        bak = self.config_path.with_suffix(".yaml.bak")
        try:
            with tmp.open("w") as f:
                yaml.dump(dump, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            # Backup current config before overwriting
            if self.config_path.exists():
                import shutil

                shutil.copy2(self.config_path, bak)
            os.replace(tmp, self.config_path)  # atomic on POSIX and Windows
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _config_dump_clean(self) -> dict[str, Any]:
        """Return config as dict with None/empty-dict values removed recursively."""
        from typing import cast as _cast

        def _strip(obj: Any) -> Any:
            if isinstance(obj, dict):
                cleaned = {k: _strip(v) for k, v in obj.items() if v is not None}
                return {k: v for k, v in cleaned.items() if v != {} and v != []}
            if isinstance(obj, list):
                return [_strip(i) for i in obj]
            return obj

        return _cast(dict[str, Any], _strip(self.config.model_dump()))

    def _init_ml_tracker(self) -> Any:
        try:
            from typing import cast as _cast

            import dataenginex.domains.ml.tracking.builtin  # noqa: F401
            from dataenginex.domains.ml.tracking import tracker_registry

            cls: Any = _cast(Any, tracker_registry.get(self.config.ml.tracking.backend))
            logger.info("ml tracker initialised", backend=self.config.ml.tracking.backend)
            return cls(storage_dir=str(self._dex_dir / "tracking"))
        except Exception as exc:
            logger.warning(
                "ml tracker init failed",
                backend=self.config.ml.tracking.backend,
                error=str(exc),
            )
            return None

    def _init_ml_feature_store(self) -> Any:
        try:
            import dataenginex.domains.ml.features.builtin  # noqa: F401
            from dataenginex.domains.ml.features import feature_store_registry

            cls = feature_store_registry.get(self.config.ml.features.backend)
            logger.info("ml feature store initialised", backend=self.config.ml.features.backend)
            return cls(**self.config.ml.features.options)
        except Exception as exc:
            logger.warning(
                "ml feature store init failed",
                backend=self.config.ml.features.backend,
                error=str(exc),
            )
            return None

    def _init_ml_serving(self) -> Any:
        try:
            from typing import cast

            import dataenginex.domains.ml.serving_engine.builtin  # noqa: F401

            # Use store-backed model registry
            from dataenginex.domains.ml.registry import ModelRegistry
            from dataenginex.domains.ml.serving_engine import serving_registry

            model_registry = ModelRegistry(store=self.store)
            cls: Any = cast(Any, serving_registry.get(self.config.ml.serving.engine))
            logger.info("ml serving engine initialised", engine=self.config.ml.serving.engine)
            return cls(model_registry=model_registry, model_dir=str(self._dex_dir / "models"))
        except Exception as exc:
            logger.warning(
                "ml serving engine init failed",
                engine=self.config.ml.serving.engine,
                error=str(exc),
            )
            return None

    def _init_vector_store(self) -> tuple[Any, Any]:
        """Initialise in-memory vector store + embedding function.

        Returns (vector_store, embed_fn). Both may be None on failure.
        Falls back to hash-based embedding when sentence-transformers is absent.
        """
        vector_store: Any = None
        embed_fn: Any = None
        try:
            from dataenginex.providers.vector.vectorstore import InMemoryBackend

            vector_store = InMemoryBackend(dimension=384)
            with contextlib.suppress(ImportError):
                from dataenginex.providers.vector.vectorstore import SentenceTransformerEmbedder

                embed_fn = SentenceTransformerEmbedder()
                logger.info("sentence transformer embedder ready")
            logger.info("vector store initialised", backend="in-memory")
        except Exception as exc:
            logger.warning("vector store init failed", error=str(exc))
        return vector_store, embed_fn

    def _init_lexical_backends(self) -> dict[str, Any]:
        lexical = self.config.ai.retrieval.options.get("lexical", {})
        if not isinstance(lexical, dict) or lexical.get("backend") != "elasticsearch":
            logger.debug(
                "lexical search disabled — elasticsearch not configured "
                "in dex.yaml (ai.retrieval.options.lexical.backend)"
            )
            return {}
        try:
            from dataenginex.domains.ai.lexical_search import ElasticsearchBackend

            indices = lexical.get("indices", {})
            hosts = lexical.get("hosts", ["http://localhost:9200"])
            configured = (
                indices
                if isinstance(indices, dict) and indices
                else {"movies": {"index_name": "dex_documents"}}
            )
            backends = {
                name: ElasticsearchBackend(
                    hosts=list(hosts),
                    index_name=str(
                        cfg.get("index_name", f"dex_{name}")
                        if isinstance(cfg, dict)
                        else f"dex_{name}"
                    ),
                    timeout_seconds=float(lexical.get("timeout_seconds", 5.0)),
                    username=str(lexical.get("username", "")),
                    password=str(lexical.get("password", "")),
                )
                for name, cfg in configured.items()
            }
            logger.info("lexical backends initialised", indices=list(backends.keys()), hosts=hosts)
            return backends
        except Exception as exc:
            logger.warning(
                "lexical backend init failed; using vector-only retrieval",
                error=str(exc),
                hosts=lexical.get("hosts"),
            )
            return {}

    def trigger_ai_index_refresh(self) -> None:
        """Re-index gold-layer movie data into the lexical (Elasticsearch) backend.

        Called by the scheduler after a pipeline configured with
        ``trigger_ai_index_refresh: true`` in ``on_pipeline_complete``
        completes. Never raises: a missing backend, missing source table, or
        ES outage all degrade to a no-op, matching ElasticsearchBackend's own
        reliability contract — a lexical-index refresh failure must never
        break the pipeline run that triggered it.
        """
        backend = self._lexical_backends.get("movies")
        if backend is None:
            return
        try:
            from dataenginex.domains.ai.tools import tool_registry
            from dataenginex.providers.vector.vectorstore import Document

            rows = tool_registry.call(
                "query",
                sql=(
                    "SELECT movie_id, title, overview, genres, release_year, "
                    "imdb_rating FROM gold_tmdb_enriched_movies"
                ),
            )
            if not isinstance(rows, list) or not rows:
                return
            docs = [
                Document(
                    id=str(row["movie_id"]),
                    text=" ".join(str(v) for v in (row.get("title"), row.get("overview")) if v),
                    metadata={
                        "title": row.get("title") or "",
                        "genres": row.get("genres") or "",
                        "release_year": row.get("release_year"),
                        "imdb_rating": row.get("imdb_rating"),
                    },
                )
                for row in rows
                if row.get("movie_id") is not None
            ]
            backend.index(docs)
        except Exception as exc:
            logger.warning("ai index refresh failed", error=str(exc))

    def _ingest_lakehouse_to_vector_store(self) -> None:
        """Embed and index gold/silver tables into the vector store on startup."""
        if self._vector_store is None:
            return
        lakehouse = self._dex_dir / "lakehouse"
        try:
            from dataenginex.providers.vector.vectorstore import Document, RAGPipeline

            rag = RAGPipeline(
                store=self._vector_store,
                embed_fn=self._embed_fn,
                dimension=384,
            )
            ingested = 0
            for layer in ("silver", "gold"):
                layer_dir = lakehouse / layer
                if not layer_dir.exists():
                    continue
                for pf in sorted(layer_dir.glob("*.parquet")):
                    with contextlib.suppress(Exception):
                        import duckdb

                        conn = duckdb.connect(":memory:")
                        safe = str(pf).replace("'", "''")
                        rows = conn.execute(
                            f"SELECT * FROM read_parquet('{safe}') LIMIT 5000"
                        ).fetchall()
                        desc = conn.execute(
                            f"DESCRIBE SELECT * FROM read_parquet('{safe}')"
                        ).fetchall()
                        conn.close()
                        cols = [d[0] for d in desc]
                        records = [dict(zip(cols, row, strict=True)) for row in rows]
                        ingested += self._index_lakehouse_records(
                            rag, Document, records, pf.stem, layer
                        )
                for delta_dir in sorted(path for path in layer_dir.iterdir() if path.is_dir()):
                    if not (delta_dir / "_delta_log").exists():
                        continue
                    with contextlib.suppress(Exception):
                        import duckdb

                        conn = duckdb.connect(":memory:")
                        scan = self._lakehouse_scan_sql(delta_dir, "delta")
                        cursor = conn.execute(f"SELECT * FROM {scan} LIMIT 5000")
                        columns = [desc[0] for desc in (cursor.description or [])]
                        rows = cursor.fetchall()
                        conn.close()
                        records = [dict(zip(columns, row, strict=True)) for row in rows]
                        ingested += self._index_lakehouse_records(
                            rag, Document, records, delta_dir.name, layer
                        )
            logger.info("vector store populated", documents=ingested)
        except Exception as exc:
            logger.warning("vector store ingest failed", error=str(exc))

    def _index_lakehouse_records(
        self,
        rag: Any,
        document_cls: Any,
        records: list[dict[str, Any]],
        table_name: str,
        layer: str,
    ) -> int:
        docs = [
            document_cls(
                text=" | ".join(
                    f"{key}: {value}" for key, value in record.items() if value is not None
                )[:512],
                metadata={"table": table_name, "layer": layer},
            )
            for record in records
        ]
        if not docs:
            return 0
        rag.store.upsert(docs)
        lexical_name = (
            "reviews"
            if "review" in table_name
            else "keywords"
            if "keyword" in table_name
            else "movies"
        )
        lexical = self._lexical_backends.get(lexical_name, self._lexical_backend)
        if lexical is not None:
            lexical.index(docs)
        return len(docs)

    def _init_ai(self) -> None:
        try:
            from dataenginex.domains.ai.llm import get_llm_provider

            self.llm = get_llm_provider(self.config.ai.llm.provider, model=self.config.ai.llm.model)
            prov = self.config.ai.llm.provider
            model = self.config.ai.llm.model
            logger.info("llm provider initialised", provider=prov, model=model)
        except Exception as exc:
            self.llm = None
            prov = self.config.ai.llm.provider
            model = self.config.ai.llm.model
            logger.warning("llm provider unavailable", provider=prov, model=model, error=str(exc))

        # Register tools regardless of LLM availability — predict/search_similar are LLM-free.
        try:
            from dataenginex.domains.ai.tools.builtin import register_builtin_tools

            self._ingest_lakehouse_to_vector_store()
            register_builtin_tools(
                lakehouse_dir=self._dex_dir / "lakehouse",
                models_dir=self._dex_dir / "models",
                vector_store=self._vector_store,
                embed_fn=self._embed_fn,
                lexical_backend=self._lexical_backend,
            )
            logger.info("builtin tools registered")
        except Exception as exc:
            logger.warning("tool registration failed", error=str(exc))

        if self.llm is None:
            return

        try:
            from typing import cast

            import dataenginex.domains.ai.agents.builtin  # noqa: F401
            from dataenginex.domains.ai.agents import agent_registry
            from dataenginex.domains.ai.tools import tool_registry

            for name, agent_cfg in self.config.ai.agents.items():
                agent_llm = self.llm
                if agent_cfg.model:
                    with contextlib.suppress(Exception):
                        from dataenginex.domains.ai.llm import get_llm_provider as _get

                        agent_llm = _get(self.config.ai.llm.provider, model=agent_cfg.model)
                cls: Any = cast(Any, agent_registry.get(agent_cfg.runtime))
                self.agents[name] = cls(
                    llm=agent_llm,
                    system_prompt=agent_cfg.system_prompt,
                    tools=tool_registry,
                    max_iterations=agent_cfg.max_iterations,
                    timeout_seconds=agent_cfg.timeout_seconds,
                    name=name,
                )
                logger.info("agent initialized", agent=name, runtime=agent_cfg.runtime)
        except Exception as exc:
            logger.warning("agent initialization failed", error=str(exc))

    def _init_ai_layer(self) -> None:
        try:
            from dataenginex.domains.ai.memory.base import ShortTermMemory
            from dataenginex.domains.ai.observability.metrics import AgentMetrics
            from dataenginex.domains.ai.runtime.sandbox import Sandbox

            max_entries = int(os.getenv("DEX_AI_MEMORY_MAX_ENTRIES", "100"))
            self.ai_memory = ShortTermMemory(max_entries=max_entries)
            self.ai_metrics = AgentMetrics()
            self.sandbox = Sandbox()
            self._init_model_router()
            logger.info("AI layer initialized", memory_max_entries=max_entries)
        except Exception as exc:
            logger.warning("AI layer init failed", error=str(exc))

    def _init_privacy_guard(self) -> None:
        """Build a ``PrivacyGuard`` from ``dex.yaml secops.guard``.

        Also constructs a :class:`~dataenginex.secops.AuditLogger` when
        ``secops.audit.enabled`` is ``True``, stored on ``self.secops_audit``.
        An empty ``db_path`` uses in-memory DuckDB; a relative path is
        resolved under ``<project>/.dex/``; absolute paths are used as-is.

        Stored on ``self.privacy_guard``. Used by ``_init_model_router`` to
        wrap every LLM provider, and exposed for downstream consumers
        (dex-studio's ``/secops/*`` UI, custom integrations).
        """
        from dataenginex.secops import (
            AuditLogger,
            PrivacyGuard,
            PrivacyGuardConfig,
        )

        guard_dict = self.config.secops.guard.model_dump()
        guard_cfg = PrivacyGuardConfig.from_dict(guard_dict)

        audit_logger: AuditLogger | None = None
        if self.config.secops.audit.enabled:
            audit_logger = AuditLogger()
            self.secops_audit = audit_logger
            logger.info(
                "secops_audit.initialized",
            )
        else:
            logger.info("secops_audit disabled — audit logging not configured")

        self.privacy_guard = PrivacyGuard(
            config=guard_cfg,
            audit_logger=audit_logger,
        )
        en = guard_cfg.enabled
        strat = len(guard_cfg.strategies)
        logger.info("privacy guard initialised", enabled=en, strategies=strat)

    def _init_spark(self) -> None:
        """Initialise Spark subsystem — cluster, SQL, catalog, streaming, ML, lineage."""
        try:
            from dataenginex.spark.catalog.adapter import SparkCatalogAdapter
            from dataenginex.spark.connect.client import SparkConnectClient
            from dataenginex.spark.datasource.registry import DataSourceRegistry
            from dataenginex.spark.lineage.openlineage_projection import OpenLineageProjection
            from dataenginex.spark.metrics.resource_accounting import ResourceAccounting
            from dataenginex.spark.ml.mllib_provider import MLlibProvider
            from dataenginex.spark.sql.spark_sql_executor import SparkSQLExecutor
            from dataenginex.spark.streaming.query_manager import StreamingQueryManager

            project_id = ProjectId(self.config.project.name)

            # Spark Connect client — local[*] for development
            spark_url = getattr(self.config, "spark_master", None) or "local[*]"
            self.spark_client = SparkConnectClient(
                server_url=spark_url,
                project_id=project_id,
                session_config={
                    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
                    "spark.sql.catalog.spark_catalog": (
                        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
                    ),
                },
            )

            # Spark SQL executor
            self.spark_sql = SparkSQLExecutor(
                project_id=project_id,
                spark_client=self.spark_client,
            )

            # Spark Catalog adapter
            catalog_type = getattr(self.config, "spark_catalog_type", "file")
            self.spark_catalog = SparkCatalogAdapter(
                catalog_type=catalog_type,
                connection_config={"lakehouse_path": str(self._dex_dir / "lakehouse")},
            )

            # Streaming query manager
            self.spark_streaming = StreamingQueryManager(project_id=project_id)

            # MLlib provider with MLflow
            mlflow_uri = getattr(self.config, "mlflow_tracking_uri", None)
            self.spark_ml = MLlibProvider(
                project_id=project_id,
                mlflow_tracking_uri=mlflow_uri,
            )

            # Resource accounting
            self.spark_metrics = ResourceAccounting(project_id=project_id)

            # OpenLineage
            openlineage_url = getattr(self.config, "openlineage_url", None)
            if openlineage_url:
                self.spark_lineage = OpenLineageProjection(
                    project_id=project_id,
                    openlineage_url=openlineage_url,
                )

            # Data source registry
            self.spark_datasource_registry = DataSourceRegistry(project_id=project_id)

            logger.info(
                "spark subsystem initialised",
                project=project_id,
                master=spark_url,
                catalog=catalog_type,
            )
        except Exception as exc:
            logger.warning("spark subsystem init failed", error=str(exc))

    def _init_model_router(self) -> None:
        import os

        from dataenginex.domains.ai.providers import BaseProvider
        from dataenginex.providers.model.guarded import GuardedProvider
        from dataenginex.providers.model.router import ModelRouter

        def _wrap(name: str, provider: BaseProvider) -> BaseProvider:
            """Wrap *provider* with the guard. Local targets bypass at call time."""
            return GuardedProvider(provider, self.privacy_guard, target=name)

        providers: dict[str, BaseProvider] = {}
        if os.environ.get("ANTHROPIC_API_KEY"):
            from dataenginex.providers.model.anthropic import AnthropicProvider

            providers["anthropic"] = _wrap("anthropic", AnthropicProvider())
            logger.info("llm provider registered", provider="anthropic")
        if os.environ.get("OPENAI_API_KEY"):
            from dataenginex.providers.model.openai import OpenAIProvider

            providers["openai"] = _wrap("openai", OpenAIProvider())
            logger.info("llm provider registered", provider="openai")

        from dataenginex.providers.model.ollama import OllamaProvider

        providers.setdefault("ollama", _wrap("ollama", OllamaProvider()))
        logger.info("llm provider registered", provider="ollama")

        if providers:
            self.model_router = ModelRouter(providers)
            logger.info("model router initialized", providers=list(providers.keys()))
        else:
            logger.warning("no llm providers configured — model router not initialized")

    def _load_plugins(self) -> None:
        try:
            for plugin in discover():
                with contextlib.suppress(ValueError):
                    self.plugins.register(plugin)
            logger.info("plugins loaded", count=self.plugins.count)
        except Exception:
            logger.warning("plugin discovery failed")
