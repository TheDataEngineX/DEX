"""Domain telemetry — Prometheus instruments for pipelines, ML, AI, quality.

Lives in the runtime rather than beside the HTTP middleware because domains
emit these. A metric is an observation about work being done, so putting the
declarations behind an interface-layer package would have forced every domain
that counts something to import the web tier (§5.5).

Declarations only: no domain reads these back, and the registry they attach to
is chosen by whoever scrapes it.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "ai_agent_iterations",
    "ai_tokens_total",
    "ai_tool_calls_total",
    "ml_drift_score",
    "ml_model_predictions_total",
    "pipeline_run_duration_seconds",
    "pipeline_runs_total",
    "quality_gate_evaluations_total",
]

pipeline_runs_total = Counter(
    "dex_domain_pipeline_runs_total",
    "Total pipeline runs",
    ["pipeline", "status"],
)

pipeline_run_duration_seconds = Histogram(
    "dex_pipeline_run_duration_seconds",
    "Pipeline run duration in seconds",
    ["pipeline"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0),
)

ml_drift_score = Gauge(
    "dex_ml_drift_score",
    "Drift score (PSI or equivalent) for a feature",
    ["pipeline", "feature", "method"],
)

ml_model_predictions_total = Counter(
    "dex_ml_model_predictions_total",
    "Total model predictions served",
    ["model", "version", "status"],
)

ai_tokens_total = Counter(
    "dex_ai_tokens_total",
    "Total AI tokens consumed",
    ["provider", "model", "direction"],
)

ai_agent_iterations = Histogram(
    "dex_ai_agent_iterations",
    "Agent loop iterations to completion",
    ["agent"],
    buckets=(1, 2, 3, 5, 8, 13, 21, 34),
)

ai_tool_calls_total = Counter(
    "dex_ai_tool_calls_total",
    "Total AI tool calls",
    ["tool", "status"],
)

quality_gate_evaluations_total = Counter(
    "dex_quality_gate_evaluations_total",
    "Quality gate evaluations",
    ["pipeline", "gate", "result"],
)
