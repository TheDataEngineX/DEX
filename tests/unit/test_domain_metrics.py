"""Tests for dataenginex.runtime.telemetry."""

from __future__ import annotations

from prometheus_client import REGISTRY

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


class TestMetricRegistration:
    def test_all_metrics_registered(self) -> None:
        names = {
            m.describe()[0].name
            for m in (
                pipeline_runs_total,
                pipeline_run_duration_seconds,
                ml_drift_score,
                ml_model_predictions_total,
                ai_tokens_total,
                ai_agent_iterations,
                ai_tool_calls_total,
                quality_gate_evaluations_total,
            )
        }
        assert "dex_domain_pipeline_runs" in names or "dex_domain_pipeline_runs_total" in names

    def test_registry_exposes_dex_metrics(self) -> None:
        collected = list(REGISTRY.collect())
        metric_names = {m.name for m in collected}
        dex_metrics = {n for n in metric_names if n.startswith("dex_")}
        assert "dex_domain_pipeline_runs" in dex_metrics
        assert "dex_ml_drift_score" in dex_metrics
        assert "dex_ai_tokens" in dex_metrics


class TestPipelineMetrics:
    def test_counter_increments(self) -> None:
        # Renamed from dex_pipeline_runs_total — that name collided with
        # dex_studio/metrics.py's own (differently-labeled) counter of the
        # same name; whichever module's Counter got constructed first won
        # the registry slot, breaking the other's .labels() calls entirely.
        before = _counter_value(
            "dex_domain_pipeline_runs_total", {"pipeline": "t1", "status": "success"}
        )
        pipeline_runs_total.labels(pipeline="t1", status="success").inc()
        after = _counter_value(
            "dex_domain_pipeline_runs_total", {"pipeline": "t1", "status": "success"}
        )
        assert after == before + 1

    def test_histogram_observes(self) -> None:
        pipeline_run_duration_seconds.labels(pipeline="t1").observe(1.5)
        pipeline_run_duration_seconds.labels(pipeline="t1").observe(3.0)


class TestAIMetrics:
    def test_tokens_counter_by_direction(self) -> None:
        labels = {"provider": "anthropic", "model": "claude-opus-4-7"}
        ai_tokens_total.labels(**labels, direction="input").inc(100)
        ai_tokens_total.labels(**labels, direction="output").inc(50)

    def test_tool_calls_labels(self) -> None:
        ai_tool_calls_total.labels(tool="search", status="ok").inc()
        ai_tool_calls_total.labels(tool="search", status="error").inc()

    def test_agent_iterations_histogram(self) -> None:
        ai_agent_iterations.labels(agent="coder").observe(5)


class TestMLMetrics:
    def test_drift_gauge_set(self) -> None:
        ml_drift_score.labels(pipeline="p1", feature="age", method="psi").set(0.15)

    def test_predictions_counter(self) -> None:
        ml_model_predictions_total.labels(model="m1", version="1", status="ok").inc()


class TestGovernanceMetrics:
    def test_quality_gate_counter(self) -> None:
        quality_gate_evaluations_total.labels(
            pipeline="p1", gate="completeness", result="pass"
        ).inc()


def _counter_value(metric: str, labels: dict[str, str]) -> float:
    """Return the current value for a labeled counter from REGISTRY."""
    for family in REGISTRY.collect():
        if family.name == metric.removesuffix("_total"):
            for sample in family.samples:
                if sample.labels == labels and sample.name == metric:
                    return float(sample.value)
    return 0.0
