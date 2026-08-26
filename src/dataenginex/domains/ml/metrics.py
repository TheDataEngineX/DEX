"""Prometheus metrics for ML model serving and drift monitoring.

Provides pre-configured counters, histograms, and gauges for model
prediction latency, throughput, and data drift tracking.

Metrics:
    model_prediction_total: Total predictions by model/version/status.
    model_prediction_latency_seconds: Prediction latency histogram.
    model_drift_psi: PSI drift score gauge per model/feature.
    model_drift_alerts_total: Drift alert counter by model/severity.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "model_drift_alerts_total",
    "model_drift_psi",
    "model_prediction_latency_seconds",
    "model_prediction_total",
]

# ---------------------------------------------------------------------------
# Model serving metrics
# ---------------------------------------------------------------------------

model_prediction_total = Counter(
    "model_prediction_total",
    "Total model predictions",
    ["model", "version", "status"],
)

model_prediction_latency_seconds = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model", "version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Drift monitoring metrics
# ---------------------------------------------------------------------------

model_drift_psi = Gauge(
    "model_drift_psi",
    "Population Stability Index (PSI) drift score",
    ["model", "feature"],
)

model_drift_alerts_total = Counter(
    "model_drift_alerts_total",
    "Total drift alerts fired",
    ["model", "severity"],
)
