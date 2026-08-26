"""Tests for ML Prometheus metrics."""

from __future__ import annotations

from dataenginex.domains.ml.metrics import (
    model_drift_alerts_total,
    model_drift_psi,
    model_prediction_latency_seconds,
    model_prediction_total,
)


class TestMLMetrics:
    def test_prediction_total_inc(self) -> None:
        model_prediction_total.labels(model="m1", version="1", status="ok").inc()
        assert model_prediction_total.labels(model="m1", version="1", status="ok")._value.get() == 1

    def test_prediction_latency_observe(self) -> None:
        model_prediction_latency_seconds.labels(model="m1", version="1").observe(0.05)
        assert model_prediction_latency_seconds.labels(model="m1", version="1")._sum.get() == 0.05

    def test_drift_psi_set(self) -> None:
        model_drift_psi.labels(model="m1", feature="f1").set(0.25)
        assert model_drift_psi.labels(model="m1", feature="f1")._value.get() == 0.25

    def test_drift_alerts_inc(self) -> None:
        model_drift_alerts_total.labels(model="m1", severity="high").inc()
        assert model_drift_alerts_total.labels(model="m1", severity="high")._value.get() == 1
