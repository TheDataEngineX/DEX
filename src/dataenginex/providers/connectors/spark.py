"""PySpark connector — reads Spark-supported sources into DataEngineX pipelines.

PySpark is optional (install with: uv sync --group data).  SparkConnector
converts a Spark DataFrame → Arrow → list[dict] for handoff to the
DuckDB-based pipeline runner.

Usage in dex.yaml::

    data:
      sources:
        my_lake:
          type: spark
          connection:
            master: "local[*]"
            format: parquet
            path: "s3a://bucket/path/"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry
from dataenginex.providers.connectors._utils import NOT_CONNECTED

logger = structlog.get_logger()

try:
    import pyspark  # noqa: F401

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False

if TYPE_CHECKING:
    import pyspark.sql

_IMPORT_ERROR = "PySpark is required for SparkConnector. Install it with: uv sync --group data"


@connector_registry.decorator("spark")
class SparkConnector(BaseConnector):
    """PySpark connector — reads any Spark-supported source into DataEngineX.

    Converts a Spark DataFrame to Arrow then to list[dict] for pipeline use.
    PySpark is optional (install with: uv sync --group data).

    Args:
        master: Spark master URL. Use ``"local[*]"`` for local testing or
                ``"spark://host:7077"`` for a standalone cluster.
        app_name: Spark application name shown in the Spark UI.
        format: Data format for the Spark reader
                (``parquet``, ``csv``, ``json``, ``delta``, ``iceberg``).
        path: Source path — file, directory, or glob.
              Can also be passed per-call to ``read(table=...)``.
        **options: Extra Spark ``DataFrameReader`` options forwarded verbatim.
    """

    def __init__(
        self,
        master: str = "local[*]",
        app_name: str = "dex",
        format: str = "parquet",
        path: str | None = None,
        **options: Any,
    ) -> None:
        if not _PYSPARK_AVAILABLE:
            raise ImportError(_IMPORT_ERROR)
        self._master = master
        self._app_name = app_name
        self._format = format
        self._path = path
        self._options = options
        self._spark: pyspark.sql.SparkSession | None = None

    def connect(self) -> None:
        from pyspark.sql import SparkSession

        self._spark = (
            SparkSession.builder.master(self._master).appName(self._app_name).getOrCreate()
        )
        logger.debug("spark session started", master=self._master, app=self._app_name)

    def disconnect(self) -> None:
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            logger.debug("spark session stopped")

    def read(
        self,
        *,
        table: str | None = None,
        default: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self._spark is None:
            raise RuntimeError(NOT_CONNECTED)
        source = table or self._path
        if not source:
            msg = "No path specified — set 'path' in connector config or pass 'table' to read()"
            raise ValueError(msg)
        try:
            reader = self._spark.read.format(self._format)
            if self._options:
                reader = reader.options(**self._options)
            df = reader.load(source)
        except Exception:
            if default is not None:
                return list(default)
            raise
        arrow_table = df.toArrow()
        col_names = arrow_table.schema.names
        col_data = arrow_table.to_pydict()
        result = [{col: col_data[col][i] for col in col_names} for i in range(arrow_table.num_rows)]
        logger.info("spark read", path=source, format=self._format, rows=len(result))
        return result

    def write(self, data: Any, *, table: str = "output", **kwargs: Any) -> None:
        if self._spark is None:
            raise RuntimeError(NOT_CONNECTED)
        if not isinstance(data, list):
            msg = f"Unsupported data type for Spark write: {type(data)}"
            raise TypeError(msg)
        if not data:
            return
        df = self._spark.createDataFrame(data)
        df.write.format(self._format).mode("overwrite").save(table)
        logger.info("spark write", path=table, format=self._format, rows=len(data))

    def health_check(self) -> bool:
        if self._spark is None:
            return False
        try:
            self._spark.sql("SELECT 1")
            return True
        except Exception:
            return False
