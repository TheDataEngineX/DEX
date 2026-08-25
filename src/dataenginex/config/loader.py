"""Load, resolve, validate, and layer dex.yaml configurations.

Usage::

    from dataenginex.config.loader import load_config

    cfg = load_config(Path("dex.yaml"))
    cfg = load_config(Path("dex.yaml"), overlay=Path("dex.prod.yaml"))
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from dataenginex.config.schema import DexConfig
from dataenginex.config.settings import DexSettings
from dataenginex.foundation.errors import ConfigError

logger = structlog.get_logger()

# Matches ${VAR} and ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def resolve_env_vars(text: str, env: dict[str, str] | None = None) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in *text*.

    Args:
        text: Raw YAML/config text containing ``${VAR}`` placeholders.
        env:  Variable map to resolve against.  Defaults to ``os.environ``
              when *None*.  Callers that have loaded a :class:`DexSettings`
              instance should pass ``settings.as_env_dict()`` here so
              ``.env`` file values are included without mutating the process
              environment.

    Raises:
        ConfigError: If a variable has no value and no default.
    """
    _env: dict[str, str] = env if env is not None else dict(os.environ)

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        value = _env.get(var_name)
        if value is not None:
            return value
        if default is not None:
            return default
        msg = (
            f"Environment variable '{var_name}' is not set and has no default. "
            f"Use ${{{var_name}:-default}} to provide a fallback."
        )
        raise ConfigError(msg)

    return _ENV_PATTERN.sub(_replace, text)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base*. Overlay values win."""
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Read a YAML file, resolve env vars, and return the parsed dict."""
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise ConfigError(msg)

    raw_text = path.read_text(encoding="utf-8")

    try:
        resolved_text = resolve_env_vars(raw_text, env=env)
    except ConfigError:
        raise
    except Exception as exc:
        msg = f"Failed to resolve env vars in {path}: {exc}"
        raise ConfigError(msg) from exc

    try:
        data = yaml.safe_load(resolved_text)
    except yaml.YAMLError as exc:
        msg = f"Failed to parse YAML in {path}: {exc}"
        raise ConfigError(msg) from exc

    if not isinstance(data, dict):
        msg = f"Config file {path} must be a YAML mapping, got {type(data).__name__}"
        raise ConfigError(msg)

    return data


def load_config(
    path: Path,
    *,
    overlay: Path | None = None,
) -> DexConfig:
    """Load a ``dex.yaml`` and return a validated ``DexConfig``.

    Parameters:
        path: Path to the base config file.
        overlay: Optional overlay file (e.g. ``dex.prod.yaml``).

    Raises:
        ConfigError: If the file is missing or cannot be parsed.
    """
    env = DexSettings.for_config(path).as_env_dict()
    data = _load_yaml(path, env=env)

    if overlay is not None:
        overlay_data = _load_yaml(overlay, env=env)
        data = _deep_merge(data, overlay_data)

    try:
        config = DexConfig.model_validate(data)
    except ValidationError as exc:
        msg = f"Config validation failed: {exc}"
        raise ConfigError(msg) from exc

    logger.info("config loaded", path=str(path), project=config.project.name)
    return config


def _validate_pipelines(config: DexConfig) -> list[str]:
    """Check pipeline source and depends_on references."""
    errors: list[str] = []
    source_names = set(config.data.sources.keys())
    pipeline_names = set(config.data.pipelines.keys())
    pipeline_destinations = {
        cfg.destination for cfg in config.data.pipelines.values() if cfg.destination
    }

    for pipe_name, pipe_cfg in config.data.pipelines.items():
        if (
            pipe_cfg.source
            and pipe_cfg.source not in source_names
            and pipe_cfg.source not in pipeline_destinations
        ):
            errors.append(f"Pipeline '{pipe_name}' references undefined source '{pipe_cfg.source}'")
        for dep in pipe_cfg.depends_on:
            if dep not in pipeline_names:
                errors.append(f"Pipeline '{pipe_name}' depends_on undefined pipeline '{dep}'")
    return errors


def _validate_registries(config: DexConfig) -> list[str]:
    """Check backends and tools against registries (warnings)."""
    # Import builtins to ensure they're registered before checking.
    import dataenginex.domains.ai.agents.builtin  # noqa: F401
    import dataenginex.domains.ml.features.builtin  # noqa: F401
    import dataenginex.domains.ml.serving_engine.builtin  # noqa: F401
    import dataenginex.domains.ml.tracking.builtin  # noqa: F401
    from dataenginex.domains.ai.agents import agent_registry
    from dataenginex.domains.ai.tools import tool_registry
    from dataenginex.domains.ai.tools.builtin import BUILTIN_TOOL_NAMES
    from dataenginex.domains.ml.features import feature_store_registry
    from dataenginex.domains.ml.serving_engine import serving_registry
    from dataenginex.domains.ml.tracking import tracker_registry
    from dataenginex.runtime.registry import BackendRegistry

    warnings: list[str] = []

    checks: list[tuple[str, BackendRegistry[Any], str]] = [
        (config.ml.tracking.backend, tracker_registry, "tracker backend"),
        (config.ml.features.backend, feature_store_registry, "feature store backend"),
        (config.ml.serving.engine, serving_registry, "serving engine"),
    ]
    for value, registry, label in checks:
        if value not in registry:
            warnings.append(
                f"Warning: {label} '{value}' not found in registry (available: {registry.list()})"
            )

    for agent_name, agent_cfg in config.ai.agents.items():
        if agent_cfg.runtime not in agent_registry:
            warnings.append(
                f"Warning: agent '{agent_name}' runtime "
                f"'{agent_cfg.runtime}' not found in registry "
                f"(available: {agent_registry.list()})"
            )
        for tool_name in agent_cfg.tools:
            if tool_name not in tool_registry._tools and tool_name not in BUILTIN_TOOL_NAMES:
                warnings.append(
                    f"Warning: agent '{agent_name}' references unknown tool '{tool_name}'"
                )

    return warnings


def _validate_secops(config: DexConfig) -> list[str]:
    """Validate secops.guard strategy entries against known enums.

    Returns hard errors (not warnings) because an unknown PII type or masking
    strategy will raise at engine init time — better to surface them early via
    ``dex validate``.
    """
    from dataenginex.secops.masking import MaskingStrategy
    from dataenginex.secops.pii import PIIType

    valid_pii = {t.value for t in PIIType}
    valid_strategies = {s.value for s in MaskingStrategy}
    errors: list[str] = []

    for raw_type, raw_strategy in config.secops.guard.strategies.items():
        if raw_type not in valid_pii:
            errors.append(
                f"secops.guard.strategies: unknown PII type '{raw_type}' "
                f"(valid: {sorted(valid_pii)})"
            )
        if raw_strategy not in valid_strategies:
            errors.append(
                f"secops.guard.strategies['{raw_type}']: unknown masking strategy "
                f"'{raw_strategy}' (valid: {sorted(valid_strategies)})"
            )
    return errors


def validate_config(config: DexConfig) -> list[str]:
    """Run cross-reference validation on a loaded config.

    Returns a list of error/warning messages (empty = valid).
    Pipeline source/depends_on mismatches are errors.
    Unknown registry backends are warnings (extras register at import time).
    """
    errors = _validate_pipelines(config)
    errors.extend(_validate_registries(config))
    errors.extend(_validate_secops(config))

    if errors:
        logger.warning("config validation issues", count=len(errors))

    return errors
