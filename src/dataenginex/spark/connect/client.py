"""Spark Connect client wrapper (§20.4).

Wraps Spark Connect client for managed execution. DEX never exposes raw
SparkSession or DataFrame objects through the public API.

Supports both embedded PySpark (local mode) and Spark Connect (remote).
"""

from __future__ import annotations

from typing import Any

import structlog

from dataenginex.foundation.ids import ProjectId, RunId

logger = structlog.get_logger()

__all__ = ["SparkConnectClient"]

try:
    from pyspark.sql import SparkSession

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False


def _delta_maven_package() -> str:
    """Maven coordinate for the installed Delta Lake Python package.

    ``delta.configure_spark_with_delta_pip()`` derives the artifact suffix
    from ``pyspark.__version__`` (e.g. ``delta-spark_4.2_2.13`` for pyspark
    4.2.0) — but Delta only publishes Spark-line-suffixed artifacts up to
    the Spark minor version it was actually built against, which lags the
    newest installed PySpark. That produced a real 404 during dependency
    resolution here (pyspark 4.2.0 + delta-spark 4.1.0: no
    ``delta-spark_4.2_2.13`` on Maven Central). The unsuffixed artifact
    (``delta-spark_2.13:<version>``) exists for every Delta release and is
    what this pins to instead — verified against Maven Central and a real
    Delta write+read under pyspark 4.2.0.
    """
    import importlib_metadata

    delta_version = importlib_metadata.version("delta_spark")
    return f"io.delta:delta-spark_2.13:{delta_version}"


class SparkConnectClient:
    """Managed Spark Connect client (§20.4).

    Each project execution receives a project-scoped session configuration.
    Sessions are not security boundaries — DEX authorization still applies.

    Supports two modes:
    - Embedded PySpark: local[*] mode for development
    - Spark Connect: remote connection to Spark cluster
    """

    def __init__(
        self,
        server_url: str = "local[*]",
        project_id: ProjectId | None = None,
        session_config: dict[str, Any] | None = None,
    ) -> None:
        self.server_url = server_url
        self.project_id = project_id
        self.session_config = session_config or {}
        self._session: Any = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to Spark Connect server.

        For embedded mode, server_url should be 'local[*]' or similar.
        For remote mode, server_url should be 'spark://host:port' or
        'sc://host:port' for Spark Connect protocol.
        """
        if not _PYSPARK_AVAILABLE:
            msg = "PySpark is required. Install with: uv sync --group spark"
            raise ImportError(msg)

        try:
            builder = SparkSession.builder.master(self.server_url)
            builder = builder.appName(f"dex-{self.project_id or 'default'}")

            # Delta Lake requires these two configs regardless of how the jar
            # itself gets resolved (see _delta_maven_package for why we don't
            # use delta.configure_spark_with_delta_pip's own suffix logic).
            builder = builder.config(
                "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
            )
            builder = builder.config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            builder = builder.config("spark.jars.packages", _delta_maven_package())

            if self.server_url.startswith("local"):
                # macOS commonly resolves the machine hostname to its LAN IP
                # rather than loopback (observed live: "Your hostname ...
                # resolves to a loopback address ... using 192.168.x.x
                # instead"). A local-mode executor then fetches driver
                # dependencies (e.g. the Delta jars just pulled in via
                # spark.jars.packages) over that LAN IP, which intermittently
                # fails to self-connect and crashes SparkContext startup.
                # Pin to loopback — the same fix tests/conftest.py's `spark`
                # fixture already applies for exactly this reason.
                builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
                builder = builder.config("spark.driver.host", "127.0.0.1")

            # Apply session configuration (may override the Delta defaults above)
            for key, value in self.session_config.items():
                builder = builder.config(key, value)

            # Spark Connect specific config
            if self.server_url.startswith("sc://"):
                # Remote Spark Connect mode
                builder = builder.config("spark.connect.grpc.arrow.maxBatchSize", "128m")
                builder = builder.config("spark.connect.reconnectionBackoff", "1s")

            self._session = builder.getOrCreate()
            self._connected = True

            logger.info(
                "spark session started",
                server_url=self.server_url,
                project_id=str(self.project_id),
            )
        except Exception as exc:
            logger.error("spark session failed", error=str(exc))
            raise

    def disconnect(self) -> None:
        """Disconnect from Spark Connect server."""
        if self._session is not None:
            try:
                self._session.stop()
                logger.info("spark session stopped")
            except Exception as exc:
                logger.warning("spark session stop failed", error=str(exc))
            finally:
                self._session = None
                self._connected = False

    def execute_sql(
        self,
        query: str,
        run_id: RunId | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute SQL through Spark Connect.

        Returns execution result metadata (not raw DataFrame).
        """
        if not self._connected or self._session is None:
            msg = "Not connected — call connect() first"
            raise RuntimeError(msg)

        try:
            # Apply parameters if provided
            if parameters:
                for key, value in parameters.items():
                    self._session.sql(f"SET {key} = '{value}'")

            # Execute query
            result_df = self._session.sql(query)
            row_count = result_df.count()

            # Collect results for small result sets
            results = []
            if row_count <= 1000:
                results = [row.asDict() for row in result_df.collect()]

            logger.info(
                "sql executed",
                query_length=len(query),
                row_count=row_count,
                run_id=str(run_id),
            )

            return {
                "status": "executed",
                "query": query,
                "run_id": run_id,
                "row_count": row_count,
                "results": results,
                "columns": result_df.columns,
            }
        except Exception as exc:
            logger.error("sql execution failed", query=query[:100], error=str(exc))
            return {
                "status": "error",
                "query": query,
                "run_id": run_id,
                "error": str(exc),
            }

    def submit_pipeline(self, pipeline_path: str, run_id: RunId) -> dict[str, Any]:
        """Submit a Spark Declarative Pipeline for execution.

        In local mode, this runs the pipeline as a spark-submit job.
        In cluster mode, this submits to the Spark cluster.
        """
        if not self._connected or self._session is None:
            msg = "Not connected — call connect() first"
            raise RuntimeError(msg)

        try:
            # For local mode, execute the pipeline script directly
            if self.server_url.startswith("local"):
                # Import and execute the pipeline
                import importlib.util
                import sys

                spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
                if spec is None or spec.loader is None:
                    msg = f"Could not load pipeline: {pipeline_path}"
                    raise RuntimeError(msg)

                module = importlib.util.module_from_spec(spec)
                sys.modules["pipeline"] = module
                spec.loader.exec_module(module)

                logger.info("pipeline executed locally", path=pipeline_path, run_id=str(run_id))
                return {
                    "status": "executed",
                    "pipeline": pipeline_path,
                    "run_id": run_id,
                    "mode": "local",
                }
            else:
                # Remote submission via spark-submit
                import subprocess

                cmd = [
                    "spark-submit",
                    "--master",
                    self.server_url,
                    "--name",
                    f"dex-pipeline-{run_id}",
                    pipeline_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                if result.returncode == 0:
                    logger.info("pipeline submitted", path=pipeline_path, run_id=str(run_id))
                    return {
                        "status": "submitted",
                        "pipeline": pipeline_path,
                        "run_id": run_id,
                        "mode": "remote",
                    }
                else:
                    msg = f"Pipeline submission failed: {result.stderr}"
                    raise RuntimeError(msg)

        except Exception as exc:
            logger.error("pipeline submission failed", path=pipeline_path, error=str(exc))
            return {
                "status": "error",
                "pipeline": pipeline_path,
                "run_id": run_id,
                "error": str(exc),
            }

    def get_session_config(self) -> dict[str, Any]:
        """Get project-scoped session configuration."""
        return {
            "server_url": self.server_url,
            "project_id": self.project_id,
            **self.session_config,
        }

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Spark."""
        return self._connected and self._session is not None

    def get_spark_session(self) -> Any:
        """Get the underlying SparkSession (internal use only)."""
        return self._session
