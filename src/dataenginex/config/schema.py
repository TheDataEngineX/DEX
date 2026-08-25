"""Pydantic models for dex.yaml — the unified config schema.

Every section is a Pydantic BaseModel with defaults so that
only ``project.name`` is required. Progressive disclosure:
add sections as you need them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataenginex.config.defaults import (
    DEFAULT_AGENT_RUNTIME,
    DEFAULT_DRIFT_METHOD,
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_FEATURE_STORE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_SERVING_ENGINE,
    DEFAULT_TRACKER,
    DEFAULT_VECTORSTORE_BACKEND,
)

# --- Project ---


class FrozenModel(BaseModel):
    """Base for all dex.yaml config models — immutable per architecture convention.

    Mutating a loaded config in place (e.g. ``cfg.schedule = x``) is disallowed;
    construct a new instance instead, e.g. ``cfg.model_copy(update={"schedule": x})``.
    """

    model_config = ConfigDict(frozen=True)


class ProjectConfig(FrozenModel):
    """Top-level project metadata."""

    name: str
    version: str = "0.1.0"
    description: str = ""


# --- Data Layer ---


class SourceConfig(FrozenModel):
    """A named data source."""

    type: str  # csv, duckdb, postgres, s3, rest, kafka, etc.
    path: str | None = None
    query: str | None = None
    url: str | None = None
    connection: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class TransformStepConfig(FrozenModel):
    """A single transform step in a pipeline."""

    type: str  # filter, derive, cast, deduplicate, sql, rename, drop_columns, fill_null, aggregate
    condition: str | None = None  # filter
    expression: str | None = None  # derive, window
    name: str | None = None  # derive, window: output column name
    columns: dict[str, str] | list[str] | None = None  # cast: {col: type}, drop_columns: [col, ...]
    key: str | list[str] | None = None  # deduplicate
    sql: str | None = None  # sql
    mapping: dict[str, str] | None = None  # rename: {old_name: new_name}
    defaults: dict[str, Any] | None = None  # fill_null: {col: default_value}
    group_by: list[str] | None = None  # aggregate
    agg_exprs: dict[str, str] | None = None  # aggregate: {output_col: "expression"}
    partition_by: list[str] | None = None  # window
    order_by: str | None = None  # window
    options: dict[str, Any] = Field(default_factory=dict)


class QualityCheckConfig(FrozenModel):
    """Quality checks applied after transforms."""

    completeness: float | None = None  # min non-null ratio (0.0-1.0)
    uniqueness: list[str] | None = None  # columns that must be unique
    row_count_min: int | None = None  # minimum number of rows required
    freshness_hours: float | None = None
    custom_sql: str | None = None


class SparkConfig(FrozenModel):
    """Spark-specific engine configuration."""

    master: str = "local[*]"
    warehouse: str = ".dex/lakehouse"
    file_format: str = "parquet"  # parquet, delta, iceberg
    executor_memory: str | None = None
    executor_cores: int | None = None
    driver_memory: str | None = None
    packages: list[str] = Field(default_factory=list)
    conf: dict[str, str] = Field(default_factory=dict)


class PipelineConfig(FrozenModel):
    """A named data pipeline."""

    source: str  # reference to a named source
    engine: Literal["duckdb", "spark", "auto"] = "duckdb"
    transforms: list[TransformStepConfig] = Field(default_factory=list)
    quality: QualityCheckConfig | None = None
    destination: str | None = None
    target: dict[str, str] | None = None  # {layer: "silver", format: "parquet"|"delta"}
    depends_on: list[str] = Field(default_factory=list)
    publish_to: list[str] = Field(default_factory=list)
    orm_sink: dict[str, str] | None = None
    schedule: str | None = None  # cron expression
    on_failure: Literal["fail", "skip", "stale"] = "fail"  # ponytail: fail-fast default
    spark: SparkConfig | None = None  # per-pipeline spark overrides


class DataConfig(FrozenModel):
    """Data layer configuration."""

    engine: Literal["duckdb", "spark", "auto"] = "duckdb"
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    pipelines: dict[str, PipelineConfig] = Field(default_factory=dict)
    spark: SparkConfig | None = None  # project-level spark defaults


# --- ML Layer ---


class TrackerConfig(FrozenModel):
    """Experiment tracker backend config."""

    backend: str = DEFAULT_TRACKER
    uri: str | None = None  # for mlflow


class FeatureStoreConfig(FrozenModel):
    """Feature store backend config."""

    backend: str = DEFAULT_FEATURE_STORE
    options: dict[str, Any] = Field(default_factory=dict)


class DriftConfig(FrozenModel):
    """Drift detection configuration."""

    monitor: list[str] = Field(default_factory=list)
    method: str = DEFAULT_DRIFT_METHOD
    threshold: float = DEFAULT_DRIFT_THRESHOLD


class ServingConfig(FrozenModel):
    """Model serving config."""

    engine: str = DEFAULT_SERVING_ENGINE
    endpoints: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentConfig(FrozenModel):
    """An ML experiment definition."""

    model_type: str = "sklearn"
    target: str = ""
    features: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class MlConfig(FrozenModel):
    """ML layer configuration."""

    tracker: str = DEFAULT_TRACKER
    tracking: TrackerConfig = Field(default_factory=TrackerConfig)
    features: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    experiments: dict[str, ExperimentConfig] = Field(default_factory=dict)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


# --- AI Layer ---


class LLMConfig(FrozenModel):
    """LLM provider configuration."""

    provider: str = DEFAULT_LLM_PROVIDER
    model: str = DEFAULT_LLM_MODEL
    fallback: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(FrozenModel):
    """Retrieval strategy configuration."""

    strategy: str = DEFAULT_RETRIEVAL_STRATEGY
    top_k: int = 10
    reranker: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(FrozenModel):
    """Vector store backend configuration."""

    backend: str = DEFAULT_VECTORSTORE_BACKEND
    embedding_model: str = "all-MiniLM-L6-v2"
    options: dict[str, Any] = Field(default_factory=dict)


class CollectionConfig(FrozenModel):
    """A named vector collection."""

    source: str | None = None  # pipeline that populates it
    column: str | None = None  # text column on the source table to embed
    embedding_model: str | None = None  # override default
    chunk_size: int = 512
    chunk_overlap: int = 50


class AgentConfig(FrozenModel):
    """An AI agent definition."""

    runtime: str = DEFAULT_AGENT_RUNTIME
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    model: str | None = None  # override default LLM
    max_iterations: int = 10
    # Aggregate wall-clock budget across the whole run() loop. Per-call LLM
    # timeouts (LLMConfig.timeout_seconds) bound a single request, not the
    # sum of up to max_iterations of them — without this, a caller-side
    # timeout (e.g. dex-studio's HTTP request deadline) can never actually
    # fire before the loop finishes on its own.
    timeout_seconds: float | None = None
    memory: Literal["short_term", "episodic", "long_term"] = "short_term"


class AiConfig(FrozenModel):
    """AI layer configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    collections: dict[str, CollectionConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)


# --- SecOps ---


class PolicyRule(FrozenModel):
    """A data governance policy rule."""

    name: str
    description: str = ""
    rule: str  # SQL assertion — must return a single truthy row to pass
    severity: Literal["error", "warn", "info"] = "warn"
    tables: list[str] = Field(default_factory=list)


class AlertRule(FrozenModel):
    """An operational alert definition."""

    name: str
    condition: str
    severity: Literal["critical", "error", "warn", "info"] = "warn"
    channels: list[str] = Field(default_factory=list)


class CostsConfig(FrozenModel):
    """Cost tracking configuration."""

    track: bool = True
    breakdown_by: list[str] = Field(default_factory=lambda: ["pipeline", "model", "agent"])


class CompactionConfig(FrozenModel):
    """Lakehouse compaction configuration."""

    enabled: bool = True
    schedule: str = "0 2 * * 0"
    layers: list[str] = Field(default_factory=lambda: ["bronze", "silver", "gold"])
    strategy: str = "merge_small_files"
    target_file_size_mb: int = 128


class AlertingChannelConfig(FrozenModel):
    """A single alerting output channel."""

    type: str
    level: str = "WARNING"


class AlertingConfig(FrozenModel):
    """Alerting channels configuration."""

    enabled: bool = True
    channels: dict[str, AlertingChannelConfig] = Field(default_factory=dict)


class PiiConfig(FrozenModel):
    """PII detection configuration."""

    scan: bool = False
    patterns: list[str] = Field(default_factory=lambda: ["email", "ssn", "phone", "credit_card"])
    action: Literal["warn", "mask", "block"] = "warn"
    masking: dict[str, Any] | None = None  # per-field masking strategy overrides


class AuditConfig(FrozenModel):
    """Audit logging configuration.

    Attributes:
        enabled: Master switch. When ``True`` a :class:`~dataenginex.secops.AuditLogger`
            is created and attached to the :class:`~dataenginex.secops.PrivacyGuard`
            so every PII detection/masking event is persisted.
        db_path: SQLite database path for the audit log.  Leave empty (default)
            for in-memory-only logging (data lost on restart). Set to a relative
            path (e.g. ``"secops_audit.db"``) and it is resolved under the
            project's ``.dex/`` directory; use an absolute path for a custom
            location.
    """

    enabled: bool = False
    db_path: str = ""  # empty → :memory: (no persistence)


class GuardConfig(FrozenModel):
    """Outbound LLM call guard (``PrivacyGuard``) configuration.

    Wraps every external LLM provider call with PII detection and either
    masking or blocking. Local providers (Ollama, etc.) bypass guarding
    by default since their data never leaves the machine.

    Attributes:
        enabled: Master switch. When ``False`` the guard logs once and
            passes all prompts through unchanged.
        allow_local: When ``True``, prompts to local providers bypass
            scanning. Disable to scan local calls too (e.g. for audit-only).
        block_on_detect: When ``True``, prompts containing PII raise
            ``PrivacyBlocked`` instead of being masked.
        log_all_outbound: Emit a structlog entry for every outbound call.
        strategies: Map of PII type → masking strategy, e.g.
            ``{"email": "hash", "ssn": "redact"}``. PII types absent from
            this map fall back to the masker's default (``REDACT``).
        local_targets: Provider names treated as local (bypass scan).
    """

    enabled: bool = True
    allow_local: bool = True
    block_on_detect: bool = False
    log_all_outbound: bool = True
    strategies: dict[str, str] = Field(default_factory=dict)
    local_targets: list[str] = Field(default_factory=lambda: ["ollama", "local"])


class SecopsConfig(FrozenModel):
    """Security operations configuration."""

    pii: PiiConfig = Field(default_factory=PiiConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    policies: list[PolicyRule] | None = None
    alerts: list[AlertRule] | None = None


# --- Observability ---


class ObservabilityConfig(FrozenModel):
    """Observability configuration."""

    metrics: bool = True
    tracing: bool = False
    log_level: str = DEFAULT_LOG_LEVEL
    costs: CostsConfig | None = None
    compaction: CompactionConfig | None = None
    alerting: AlertingConfig | None = None


# --- Root Config ---


class DexConfig(FrozenModel):
    """Root configuration model — one ``dex.yaml`` defines everything.

    v0.7 adds apiVersion, kind, and spec-level engine/spark/governance config.
    Only ``project`` is required for backward compatibility. All other sections
    have sensible defaults.
    """

    # v0.7 manifest fields
    apiVersion: str = "dex/v0.7"
    kind: str = "Project"
    # v0.7 spec-level config
    # auto: resolve from pipeline engine; lite: duckdb; spark-local: local[*];
    # spark-connect: remote; home-server: single-laptop; distributed: cluster
    runtime_profile: str = "auto"
    imports: list[str] = Field(default_factory=list)
    governance_external_data: str = "deny"  # deny, approve, allow
    governance_classification: str = "internal"  # public, internal, confidential, restricted
    resource_max_cpu: int | None = None
    resource_max_memory: str | None = None

    # Existing config sections
    project: ProjectConfig
    data: DataConfig = Field(default_factory=DataConfig)
    ml: MlConfig = Field(default_factory=MlConfig)
    ai: AiConfig = Field(default_factory=AiConfig)
    secops: SecopsConfig = Field(default_factory=SecopsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
