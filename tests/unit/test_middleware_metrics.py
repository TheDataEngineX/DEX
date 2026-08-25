"""Tests for dataenginex.runtime.telemetry — Prometheus metric objects."""

from __future__ import annotations

from dataenginex.runtime.telemetry import (
    ai_agent_iterations,
    ai_tokens_total,
    ai_tool_calls_total,
    ml_drift_score,
    ml_model_predictions_total,
    pipeline_run_duration_seconds,
    pipeline_runs_total,
    quality_gate_evaluations_total,
)


class TestDomainMetricsImport:
    """Verify all declared metrics are importable and have the expected Prometheus type."""

    def test_pipeline_runs_total_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(pipeline_runs_total, Counter)

    def test_pipeline_run_duration_is_histogram(self) -> None:
        from prometheus_client import Histogram

        assert isinstance(pipeline_run_duration_seconds, Histogram)

    def test_quality_gate_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(quality_gate_evaluations_total, Counter)

    def test_ai_tokens_total_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(ai_tokens_total, Counter)

    def test_ai_tool_calls_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(ai_tool_calls_total, Counter)

    def test_ai_agent_iterations_is_histogram(self) -> None:
        from prometheus_client import Histogram

        assert isinstance(ai_agent_iterations, Histogram)

    def test_ml_drift_score_is_gauge(self) -> None:
        from prometheus_client import Gauge

        assert isinstance(ml_drift_score, Gauge)

    def test_ml_predictions_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(ml_model_predictions_total, Counter)


class TestDomainMetricsUsage:
    """Basic usage — label sets and observe calls must not raise."""

    def test_pipeline_runs_increment(self) -> None:
        pipeline_runs_total.labels(pipeline="test_pipe", status="success").inc()

    def test_pipeline_duration_observe(self) -> None:
        pipeline_run_duration_seconds.labels(pipeline="test_pipe").observe(1.23)

    def test_quality_gate_increment(self) -> None:
        quality_gate_evaluations_total.labels(
            pipeline="ingest", gate="completeness", result="pass"
        ).inc()

    def test_ai_tokens_increment(self) -> None:
        ai_tokens_total.labels(provider="ollama", model="llama3", direction="input").inc(100)

    def test_ai_tool_calls_increment(self) -> None:
        ai_tool_calls_total.labels(tool="query", status="ok").inc()

    def test_ai_agent_iterations_observe(self) -> None:
        ai_agent_iterations.labels(agent="assistant").observe(3)

    def test_ml_drift_score_set(self) -> None:
        ml_drift_score.labels(pipeline="churn", feature="spend", method="psi").set(0.12)

    def test_ml_predictions_increment(self) -> None:
        ml_model_predictions_total.labels(
            model="churn_model", version="1.0", status="success"
        ).inc()
