"""Resource accounting for Spark jobs (§20.10).

Tracks resource usage, costs, and performance metrics for Spark operations.
Integrates with Spark's built-in metrics system.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from dataenginex.foundation.ids import ProjectId

logger = structlog.get_logger()

__all__ = ["ResourceAccounting"]

try:
    from pyspark import SparkContext

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False


@dataclass
class SparkMetrics:
    """Spark job metrics."""

    job_id: int = 0
    stage_id: int = 0
    task_count: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    shuffle_read: int = 0
    shuffle_write: int = 0
    executor_run_time: int = 0
    gc_time: int = 0
    peak_execution_memory: int = 0


@dataclass
class ResourceUsage:
    """Resource usage summary."""

    cpu_time_ms: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    network_bytes: int = 0
    cost_estimate: float = 0.0


class ResourceAccounting:
    """Tracks resource usage for Spark operations (§20.10).

    Provides:
    - Job-level metrics tracking
    - Resource usage aggregation
    - Cost estimation
    - Performance monitoring
    """

    def __init__(
        self,
        project_id: ProjectId,
        cost_per_gb_hour: float = 0.01,
        cost_per_cpu_hour: float = 0.005,
    ) -> None:
        self.project_id = project_id
        self.cost_per_gb_hour = cost_per_gb_hour
        self.cost_per_cpu_hour = cost_per_cpu_hour
        self._metrics: dict[str, SparkMetrics] = {}
        self._resource_usage: dict[str, ResourceUsage] = {}
        self._total_cost: float = 0.0

    def record_job_metrics(self, run_id: str, metrics: dict) -> None:
        """Record metrics for a Spark job.

        Args:
            run_id: Run identifier
            metrics: Job metrics dict
        """
        spark_metrics = SparkMetrics(
            job_id=metrics.get("job_id", 0),
            stage_id=metrics.get("stage_id", 0),
            task_count=metrics.get("taskCount", 0),
            input_bytes=metrics.get("inputBytes", 0),
            output_bytes=metrics.get("outputBytes", 0),
            shuffle_read=metrics.get("shuffleRead", 0),
            shuffle_write=metrics.get("shuffleWrite", 0),
            executor_run_time=metrics.get("executorRunTime", 0),
            gc_time=metrics.get("gcTime", 0),
            peak_execution_memory=metrics.get("peakExecutionMemory", 0),
        )
        self._metrics[run_id] = spark_metrics

        # Calculate resource usage
        resource_usage = ResourceUsage(
            cpu_time_ms=spark_metrics.executor_run_time,
            memory_bytes=spark_metrics.peak_execution_memory,
        )
        self._resource_usage[run_id] = resource_usage

        # Estimate cost
        cost = self._estimate_cost(resource_usage)
        resource_usage.cost_estimate = cost
        self._total_cost += cost

        logger.info(
            "job metrics recorded",
            run_id=run_id,
            job_id=spark_metrics.job_id,
            input_bytes=spark_metrics.input_bytes,
            executor_time_ms=spark_metrics.executor_run_time,
            cost_estimate=cost,
        )

    def get_job_metrics(self, run_id: str) -> dict | None:
        """Get metrics for a specific job.

        Args:
            run_id: Run identifier

        Returns:
            Job metrics dict or None
        """
        metrics = self._metrics.get(run_id)
        if metrics is None:
            return None

        return {
            "job_id": metrics.job_id,
            "stage_id": metrics.stage_id,
            "task_count": metrics.task_count,
            "input_bytes": metrics.input_bytes,
            "output_bytes": metrics.output_bytes,
            "shuffle_read": metrics.shuffle_read,
            "shuffle_write": metrics.shuffle_write,
            "executor_run_time_ms": metrics.executor_run_time,
            "gc_time_ms": metrics.gc_time,
            "peak_execution_memory": metrics.peak_execution_memory,
        }

    def get_resource_usage(self, run_id: str) -> dict | None:
        """Get resource usage for a specific job.

        Args:
            run_id: Run identifier

        Returns:
            Resource usage dict or None
        """
        usage = self._resource_usage.get(run_id)
        if usage is None:
            return None

        return {
            "cpu_time_ms": usage.cpu_time_ms,
            "memory_bytes": usage.memory_bytes,
            "disk_bytes": usage.disk_bytes,
            "network_bytes": usage.network_bytes,
            "cost_estimate": usage.cost_estimate,
        }

    def get_total_cost(self) -> float:
        """Get total cost estimate for all tracked jobs.

        Returns:
            Total cost in dollars
        """
        return self._total_cost

    def get_usage_summary(self) -> dict:
        """Get summary of all resource usage.

        Returns:
            Usage summary dict
        """
        total_cpu_ms = sum(u.cpu_time_ms for u in self._resource_usage.values())
        total_memory = sum(u.memory_bytes for u in self._resource_usage.values())
        total_jobs = len(self._metrics)

        return {
            "total_jobs": total_jobs,
            "total_cpu_time_ms": total_cpu_ms,
            "total_memory_bytes": total_memory,
            "total_cost_estimate": self._total_cost,
            "avg_cost_per_job": self._total_cost / total_jobs if total_jobs > 0 else 0,
        }

    def _estimate_cost(self, usage: ResourceUsage) -> float:
        """Estimate cost for resource usage.

        Args:
            usage: Resource usage

        Returns:
            Cost estimate in dollars
        """
        # CPU cost
        cpu_hours = usage.cpu_time_ms / (1000 * 3600)
        cpu_cost = cpu_hours * self.cost_per_cpu_hour * 4  # Assume 4 cores

        # Memory cost (assume 1 hour minimum)
        memory_gb = usage.memory_bytes / (1024**3)
        memory_cost = memory_gb * self.cost_per_gb_hour

        return cpu_cost + memory_cost

    def get_performance_metrics(self, run_id: str) -> dict:
        """Get performance metrics for a job.

        Args:
            run_id: Run identifier

        Returns:
            Performance metrics
        """
        metrics = self._metrics.get(run_id)
        if metrics is None:
            return {}

        # Calculate derived metrics
        total_bytes = metrics.input_bytes + metrics.output_bytes
        throughput_mbps = (
            total_bytes / (1024 * 1024) / (metrics.executor_run_time / 1000)
            if metrics.executor_run_time > 0
            else 0
        )

        gc_ratio = (
            metrics.gc_time / metrics.executor_run_time if metrics.executor_run_time > 0 else 0
        )

        return {
            "throughput_mbps": throughput_mbps,
            "gc_ratio": gc_ratio,
            "shuffle_ratio": (
                (metrics.shuffle_read + metrics.shuffle_write) / total_bytes
                if total_bytes > 0
                else 0
            ),
            "task_efficiency": (
                metrics.executor_run_time / metrics.task_count if metrics.task_count > 0 else 0
            ),
        }

    def collect_spark_metrics(self) -> dict:
        """Collect metrics from Spark's metrics system.

        Returns:
            Spark system metrics
        """
        if not _PYSPARK_AVAILABLE:
            return {"error": "PySpark not available"}

        try:
            from pyspark import SparkContext

            sc = SparkContext.getActive()
            if not sc:
                return {"error": "No active SparkContext"}

            sc = sc[0]

            # Get executor metrics
            executor_metrics = sc._jsc.sc().statusTracker().getExecutorInfos()

            return {
                "executor_count": len(executor_metrics),
                "status": "collected",
            }
        except Exception as exc:
            logger.warning("failed to collect spark metrics", error=str(exc))
            return {"error": str(exc)}

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._metrics.clear()
        self._resource_usage.clear()
        self._total_cost = 0.0
        logger.info("resource accounting reset", project_id=self.project_id)
