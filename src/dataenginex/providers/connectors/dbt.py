"""dbt connector — runs a dbt model and reads back its materialized output.

Supports two dbt adapters based on the engine config:
- **duckdb** (default): Uses dbt-duckdb, reads from a local DuckDB file.
- **spark**: Uses dbt-spark, reads via Spark from the configured warehouse.

Install dbt with::

    uv sync --group dbt
    # or: pip install dbt-core dbt-duckdb dbt-spark

Usage in dex.yaml::

    data:
      sources:
        transformed_users:
          type: dbt
          connection:
            project_dir: ./dbt_project
            model: users_cleaned
            engine: duckdb  # or spark
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry
from dataenginex.providers.connectors._utils import rows_to_dicts

logger = structlog.get_logger()

_DBT_AVAILABLE = shutil.which("dbt") is not None

_IMPORT_ERROR = (
    "dbt CLI not found. Install dbt externally: "
    "https://docs.getdbt.com/docs/core/pip-install  "
    "For DuckDB: pip install dbt-core dbt-duckdb  "
    "For Spark: pip install dbt-core dbt-spark pyspark"
)


def _generate_duckdb_profile(
    target_database: str,
    target: str = "dev",
) -> dict[str, Any]:
    """Generate a dbt profiles.yml dict for dbt-duckdb."""
    return {
        "version": 1,
        "profiles": {
            "dex_project": {
                "target": target,
                "outputs": {
                    target: {
                        "type": "duckdb",
                        "path": target_database,
                        "threads": 1,
                    }
                },
            }
        },
    }


def _generate_spark_profile(
    warehouse: str,
    target: str = "dev",
    master: str = "local[*]",
    schema: str = "default",
) -> dict[str, Any]:
    """Generate a dbt profiles.yml dict for dbt-spark."""
    return {
        "version": 1,
        "profiles": {
            "dex_project": {
                "target": target,
                "outputs": {
                    target: {
                        "type": "spark",
                        "method": "shell",
                        "host": "localhost",
                        "port": 10000,
                        "schema": schema,
                        "connect_retries": 3,
                        "connect_timeout": 10,
                        "execution_project": schema,
                    }
                },
            }
        },
    }


@connector_registry.decorator("dbt")
class DbtConnector(BaseConnector):
    """dbt connector — runs a model and reads its materialized output.

    Supports both dbt-duckdb and dbt-spark adapters based on the ``engine`` param.

    Args:
        project_dir: Path to the dbt project root (must contain dbt_project.yml).
        model: dbt model name to run and read.
        engine: dbt adapter engine — "duckdb" (default) or "spark".
        target_database: Path to the DuckDB file (duckdb engine only).
                         Defaults to ``{project_dir}/dev.duckdb``.
        warehouse: Warehouse path (spark engine only). Defaults to ".dex/lakehouse".
        master: Spark master URL (spark engine only). Defaults to "local[*]".
        profiles_dir: Path to the dbt profiles directory.
                      Defaults to ``project_dir``.
        target: dbt target name (default ``"dev"``).
        schema: dbt schema/database name (default ``"default"``).
    """

    def __init__(
        self,
        project_dir: str,
        model: str,
        engine: str = "duckdb",
        target_database: str | None = None,
        warehouse: str = ".dex/lakehouse",
        master: str = "local[*]",
        profiles_dir: str | None = None,
        target: str = "dev",
        schema: str = "default",
        timeout_s: float = 600,
        **kwargs: Any,
    ) -> None:
        if not _DBT_AVAILABLE:
            raise ImportError(_IMPORT_ERROR)
        self._project_dir = Path(project_dir)
        self._model = model
        self._engine = engine
        self._target_db = (
            Path(target_database) if target_database else self._project_dir / "dev.duckdb"
        )
        self._warehouse = warehouse
        self._master = master
        self._profiles_dir = Path(profiles_dir) if profiles_dir else self._project_dir
        self._target = target
        self._schema = schema
        self._timeout_s = timeout_s
        self._profile_dir: Path | None = None

    def connect(self) -> None:
        project_file = self._project_dir / "dbt_project.yml"
        if not project_file.exists():
            msg = f"dbt project not found: {project_file}"
            raise FileNotFoundError(msg)
        # Write the generated profile to a temp dir so dbt can find it
        self._profile_dir = Path(tempfile.mkdtemp(prefix="dex_dbt_"))
        profile_yml = self._profile_dir / "profiles.yml"
        import yaml

        if self._engine == "spark":
            profile = _generate_spark_profile(
                warehouse=self._warehouse,
                target=self._target,
                master=self._master,
                schema=self._schema,
            )
        else:
            profile = _generate_duckdb_profile(
                target_database=str(self._target_db),
                target=self._target,
            )
        profile_yml.write_text(yaml.dump(profile), encoding="utf-8")
        logger.debug(
            "dbt connector ready",
            engine=self._engine,
            project=str(self._project_dir),
            model=self._model,
        )

    def disconnect(self) -> None:
        # ponytail: profile dir is in /tmp, OS cleans it up
        self._profile_dir = None

    def read(
        self,
        *,
        table: str | None = None,
        default: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        model = table or self._model
        self._run_dbt(model)
        if self._engine == "spark":
            return self._read_model_spark(model, default=default)
        return self._read_model_duckdb(model, default=default)

    def write(self, data: Any, *, table: str = "output", **kwargs: Any) -> None:
        msg = "DbtConnector is read-only — writes are managed by dbt models"
        raise NotImplementedError(msg)

    def health_check(self) -> bool:
        return (self._project_dir / "dbt_project.yml").exists()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _run_dbt(self, model: str) -> None:
        profiles_dir = str(self._profile_dir) if self._profile_dir else str(self._profiles_dir)
        cmd = [
            "dbt",
            "run",
            "--select",
            model,
            "--project-dir",
            str(self._project_dir),
            "--profiles-dir",
            profiles_dir,
            "--target",
            self._target,
        ]
        try:
            proc = subprocess.run(  # noqa: S603,S607
                cmd, capture_output=True, text=True, timeout=self._timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"dbt run timed out for model '{model}' after {self._timeout_s}s"
            raise RuntimeError(msg) from exc
        if proc.returncode != 0:
            msg = f"dbt run failed for model '{model}':\n{proc.stdout}\n{proc.stderr}"
            raise RuntimeError(msg)
        logger.info("dbt run complete", model=model, engine=self._engine)

    def _read_model_duckdb(self, model: str, *, default: Any) -> list[dict[str, Any]]:
        import duckdb

        if not self._target_db.exists():
            if default is not None:
                return list(default)
            msg = f"dbt target database not found: {self._target_db}. Run dbt first."
            raise FileNotFoundError(msg)
        conn = duckdb.connect(str(self._target_db), read_only=True)
        try:
            result = conn.execute(f"SELECT * FROM {model}")  # noqa: S608
            rows = rows_to_dicts(result)
        except duckdb.CatalogException:
            if default is not None:
                conn.close()
                return list(default)
            conn.close()
            raise
        conn.close()
        logger.info("dbt model read", model=model, engine="duckdb", rows=len(rows))
        return rows

    def _read_model_spark(self, model: str, *, default: Any) -> list[dict[str, Any]]:
        try:
            from pyspark.sql import SparkSession

            spark = (
                SparkSession.builder
                .master(self._master)
                .appName(f"dex-dbt-{model}")
                .config("spark.sql.warehouse.dir", self._warehouse)
                .getOrCreate()
            )
            df = spark.table(f"{self._schema}.{model}")
            rows = [row.asDict() for row in df.collect()]
            spark.stop()
            logger.info("dbt model read", model=model, engine="spark", rows=len(rows))
            return rows
        except Exception:
            if default is not None:
                return list(default)
            raise
