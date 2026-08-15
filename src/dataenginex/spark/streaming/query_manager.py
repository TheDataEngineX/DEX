"""Streaming query manager (§20.7).

Tracks active streaming queries for a project and coordinates with DEX.
Integrates with Spark Structured Streaming.
"""

from __future__ import annotations

from typing import Any

import structlog

from dataenginex.foundation.ids import ProjectId

logger = structlog.get_logger()

__all__ = ["StreamingQueryManager"]

try:
    from pyspark.sql import SparkSession

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False


class StreamingQueryManager:
    """Tracks active streaming queries per project (§20.7).

    Provides:
    - Query registration and tracking
    - Status monitoring
    - Checkpoint management
    - Resource cleanup
    """

    def __init__(
        self,
        project_id: ProjectId,
        spark_session: Any | None = None,
        checkpoint_base_path: str = "s3a://dex-checkpoints",
    ) -> None:
        self.project_id = project_id
        self._spark = spark_session
        self._queries: dict[str, dict] = {}
        self._streaming_queries: dict[str, Any] = {}
        self._checkpoint_base_path = checkpoint_base_path

    def connect(self) -> None:
        """Initialize Spark session if not provided."""
        if self._spark is None and _PYSPARK_AVAILABLE:
            try:
                self._spark = SparkSession.builder.getOrCreate()
                logger.info("spark session initialized for streaming")
            except Exception as exc:
                logger.warning("could not initialize spark session", error=str(exc))

    def register(
        self,
        query_id: str,
        config: dict,
        query: Any | None = None,
    ) -> None:
        """Register a streaming query for tracking.

        Args:
            query_id: Unique query identifier
            config: Query configuration (source, sink, options)
            query: Optional active StreamingQuery object
        """
        self._queries[query_id] = {
            "config": config,
            "status": "registered",
            "project_id": self.project_id,
            "checkpoint_path": f"{self._checkpoint_base_path}/{self.project_id}/{query_id}",
        }

        if query is not None:
            self._streaming_queries[query_id] = query
            self._queries[query_id]["status"] = "active"

        logger.info(
            "streaming query registered",
            query_id=query_id,
            project_id=self.project_id,
        )

    def deregister(self, query_id: str) -> None:
        """Deregister a streaming query.

        Args:
            query_id: Query identifier to deregister
        """
        # Stop query if active
        if query_id in self._streaming_queries:
            try:
                self._streaming_queries[query_id].stop()
            except Exception as exc:
                logger.warning("failed to stop query", query_id=query_id, error=str(exc))
            del self._streaming_queries[query_id]

        # Remove from registry
        self._queries.pop(query_id, None)
        logger.info("streaming query deregistered", query_id=query_id)

    def active_queries(self) -> list[dict]:
        """Get list of active streaming queries.

        Returns:
            List of query dicts with id, status, and config
        """
        return [{"id": k, **v} for k, v in self._queries.items()]

    def get_status(self, query_id: str) -> dict | None:
        """Get status of a specific query.

        Args:
            query_id: Query identifier

        Returns:
            Query status dict or None if not found
        """
        return self._queries.get(query_id)

    def update_status(self, query_id: str, status: str, metadata: dict | None = None) -> None:
        """Update query status.

        Args:
            query_id: Query identifier
            status: New status (active, completed, failed, etc.)
            metadata: Optional additional metadata
        """
        if query_id in self._queries:
            self._queries[query_id]["status"] = status
            if metadata:
                self._queries[query_id].update(metadata)
            logger.info("query status updated", query_id=query_id, status=status)

    def start_query(
        self,
        query_id: str,
        source_config: dict,
        sink_config: dict,
        processing_time: str = "10 seconds",
    ) -> dict:
        """Start a streaming query.

        Args:
            query_id: Unique query identifier
            source_config: Source configuration (type, options)
            sink_config: Sink configuration (type, options, checkpoint)
            processing_time: Micro-batch processing time

        Returns:
            Query start result
        """
        if self._spark is None:
            msg = "No Spark session available"
            raise RuntimeError(msg)

        try:
            # Build streaming DataFrame
            df = self._build_streaming_df(source_config)

            # Configure sink
            query = self._configure_sink(df, sink_config, processing_time, query_id)

            # Register query
            self.register(
                query_id,
                {
                    "source": source_config,
                    "sink": sink_config,
                    "processing_time": processing_time,
                },
                query,
            )

            logger.info(
                "streaming query started",
                query_id=query_id,
                source_type=source_config.get("type"),
                sink_type=sink_config.get("type"),
            )

            return {
                "status": "started",
                "query_id": query_id,
                "run_id": query.id,
            }

        except Exception as exc:
            logger.error("failed to start streaming query", query_id=query_id, error=str(exc))
            return {
                "status": "error",
                "query_id": query_id,
                "error": str(exc),
            }

    def stop_query(self, query_id: str) -> dict:
        """Stop a streaming query.

        Args:
            query_id: Query identifier

        Returns:
            Stop result
        """
        if query_id not in self._streaming_queries:
            return {"status": "error", "error": "Query not found"}

        try:
            self._streaming_queries[query_id].stop()
            self.update_status(query_id, "stopped")
            del self._streaming_queries[query_id]

            return {"status": "stopped", "query_id": query_id}
        except Exception as exc:
            logger.error("failed to stop query", query_id=query_id, error=str(exc))
            return {"status": "error", "error": str(exc)}

    def get_query_metrics(self, query_id: str) -> dict:
        """Get metrics for a streaming query.

        Args:
            query_id: Query identifier

        Returns:
            Query metrics (rows processed, batch info, etc.)
        """
        if query_id not in self._streaming_queries:
            return {"error": "Query not found"}

        try:
            query = self._streaming_queries[query_id]
            progress = query.recentProgress

            return {
                "query_id": query_id,
                "status": "active" if query.isActive else "completed",
                "num_input_rows": sum(p.get("numInputRows", 0) for p in progress),
                "num_batches": len(progress),
                "input_rows_per_second": query.lastProgress.get("inputRowsPerSecond", 0)
                if query.lastProgress
                else 0,
                "processed_rows_per_second": query.lastProgress.get("processedRowsPerSecond", 0)
                if query.lastProgress
                else 0,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _build_streaming_df(self, source_config: dict) -> Any:
        """Build streaming DataFrame from source config.

        Args:
            source_config: Source configuration

        Returns:
            Streaming DataFrame
        """
        source_type = source_config.get("type", "kafka")
        options = source_config.get("options", {})

        if source_type == "kafka":
            return (
                self._spark.readStream.format("kafka")
                .option(
                    "kafka.bootstrap.servers", options.get("bootstrap.servers", "localhost:9092")
                )
                .option("subscribe", options.get("topic", "default"))
                .option("startingOffsets", options.get("startingOffsets", "latest"))
                .load()
            )
        elif source_type == "delta":
            return self._spark.readStream.format("delta").load(options.get("path", ""))
        elif source_type == "rate":
            return (
                self._spark.readStream.format("rate")
                .option("rowsPerSecond", options.get("rowsPerSecond", 1))
                .load()
            )
        else:
            msg = f"Unsupported source type: {source_type}"
            raise ValueError(msg)

    def _configure_sink(
        self,
        df: Any,
        sink_config: dict,
        processing_time: str,
        query_id: str,
    ) -> Any:
        """Configure sink for streaming query.

        Args:
            df: Streaming DataFrame
            sink_config: Sink configuration
            processing_time: Processing time interval
            query_id: Query identifier

        Returns:
            StreamingQuery object
        """
        sink_type = sink_config.get("type", "delta")
        options = sink_config.get("options", {})
        checkpoint_path = f"{self._checkpoint_base_path}/{self.project_id}/{query_id}"

        if sink_type == "delta":
            return (
                df.writeStream.format("delta")
                .outputMode(options.get("outputMode", "append"))
                .option("checkpointLocation", checkpoint_path)
                .trigger(processingTime=processing_time)
                .start(options.get("path", ""))
            )
        elif sink_type == "kafka":
            return (
                df.writeStream.format("kafka")
                .option(
                    "kafka.bootstrap.servers", options.get("bootstrap.servers", "localhost:9092")
                )
                .option("topic", options.get("topic", "output"))
                .option("checkpointLocation", checkpoint_path)
                .trigger(processingTime=processing_time)
                .start()
            )
        elif sink_type == "console":
            return (
                df.writeStream.format("console")
                .outputMode(options.get("outputMode", "append"))
                .option("checkpointLocation", checkpoint_path)
                .trigger(processingTime=processing_time)
                .start()
            )
        else:
            msg = f"Unsupported sink type: {sink_type}"
            raise ValueError(msg)

    def cleanup_checkpoints(self, query_id: str | None = None) -> dict:
        """Clean up checkpoint files.

        Args:
            query_id: Optional specific query to clean up (all if None)

        Returns:
            Cleanup result
        """
        import shutil
        from pathlib import Path

        cleaned = []
        if query_id:
            checkpoint_path = Path(f"{self._checkpoint_base_path}/{self.project_id}/{query_id}")
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                cleaned.append(query_id)
        else:
            # Clean all checkpoints for this project
            project_checkpoint_path = Path(f"{self._checkpoint_base_path}/{self.project_id}")
            if project_checkpoint_path.exists():
                for path in project_checkpoint_path.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                        cleaned.append(path.name)

        logger.info("checkpoints cleaned", queries=cleaned)
        return {"cleaned": cleaned}
