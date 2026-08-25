"""Data Source V2 registry (§20.12).

Manages custom data source implementations for Spark.
Supports registering and discovering data sources.
"""

from __future__ import annotations

from typing import Any, cast

import structlog

from dataenginex.foundation.ids import ProjectId

logger = structlog.get_logger()

__all__ = ["DataSourceRegistry"]

try:
    from pyspark.sql import SparkSession  # noqa: F401

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False


class DataSourceRegistry:
    """Registry for custom Spark data sources (§20.12).

    Provides:
    - Data source registration
    - Discovery and loading
    - Capability detection
    - Configuration management
    """

    def __init__(self, project_id: ProjectId) -> None:
        self.project_id = project_id
        self._sources: dict[str, dict[str, Any]] = {}
        self._loaded_sources: dict[str, Any] = {}

    def register(
        self,
        name: str,
        source_class: str,
        description: str = "",
        capabilities: list[str] | None = None,
        config_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom data source.

        Args:
            name: Source name (e.g., 'my_custom_source')
            source_class: Fully qualified class name
            description: Source description
            capabilities: Supported capabilities (read, write, stream, etc.)
            config_schema: Configuration schema
        """
        self._sources[name] = {
            "name": name,
            "source_class": source_class,
            "description": description,
            "capabilities": capabilities or ["read"],
            "config_schema": config_schema or {},
        }
        logger.info(
            "data source registered",
            name=name,
            source_class=source_class,
        )

    def unregister(self, name: str) -> None:
        """Unregister a data source.

        Args:
            name: Source name to unregister
        """
        self._sources.pop(name, None)
        self._loaded_sources.pop(name, None)
        logger.info("data source unregistered", name=name)

    def get_source(self, name: str) -> dict[str, Any] | None:
        """Get source configuration.

        Args:
            name: Source name

        Returns:
            Source config dict or None
        """
        return self._sources.get(name)

    def list_sources(self) -> list[dict[str, Any]]:
        """List all registered sources.

        Returns:
            List of source configs
        """
        return list(self._sources.values())

    def load_source(self, name: str, spark_session: Any | None = None) -> Any:
        """Load and instantiate a data source.

        Args:
            name: Source name
            spark_session: Optional Spark session

        Returns:
            Instantiated data source
        """
        if name in self._loaded_sources:
            return self._loaded_sources[name]

        source_config = self._sources.get(name)
        if not source_config:
            msg = f"Data source not registered: {name}"
            raise ValueError(msg)

        try:
            # Import and instantiate source class
            module_path, class_name = source_config["source_class"].rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            source_class = getattr(module, class_name)

            # Instantiate with optional spark session
            source = (
                source_class(spark=spark_session)
                if spark_session
                else source_class()
            )

            self._loaded_sources[name] = source
            logger.info("data source loaded", name=name)
            return source

        except Exception as exc:
            logger.error("failed to load data source", name=name, error=str(exc))
            raise

    def read(
        self,
        source_name: str,
        options: dict[str, Any] | None = None,
        spark_session: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Read data from a registered source.

        Args:
            source_name: Source name
            options: Read options
            spark_session: Optional Spark session

        Returns:
            List of records
        """
        source = self.load_source(source_name, spark_session)

        # Check if source supports read
        source_config = self._sources.get(source_name, {})
        if "read" not in source_config.get("capabilities", []):
            msg = f"Source does not support read: {source_name}"
            raise ValueError(msg)

        # Call read method
        if hasattr(source, "read"):
            return cast(list[dict[str, Any]], source.read(**(options or {})))
        elif hasattr(source, "load"):
            return cast(list[dict[str, Any]], source.load(**(options or {})))
        else:
            msg = f"Source has no read/load method: {source_name}"
            raise ValueError(msg)

    def write(
        self,
        source_name: str,
        data: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        spark_session: Any | None = None,
    ) -> dict[str, Any]:
        """Write data to a registered source.

        Args:
            source_name: Source name
            data: Data to write
            options: Write options
            spark_session: Optional Spark session

        Returns:
            Write result
        """
        source = self.load_source(source_name, spark_session)

        # Check if source supports write
        source_config = self._sources.get(source_name, {})
        if "write" not in source_config.get("capabilities", []):
            msg = f"Source does not support write: {source_name}"
            raise ValueError(msg)

        # Call write method
        if hasattr(source, "write"):
            return cast(dict[str, Any], source.write(data, **(options or {})))
        elif hasattr(source, "save"):
            return cast(dict[str, Any], source.save(data, **(options or {})))
        else:
            msg = f"Source has no write/save method: {source_name}"
            raise ValueError(msg)

    def get_capabilities(self, source_name: str) -> list[str]:
        """Get capabilities for a source.

        Args:
            source_name: Source name

        Returns:
            List of capabilities
        """
        source_config = self._sources.get(source_name, {})
        return cast(list[str], source_config.get("capabilities", []))

    def validate_config(self, source_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration for a source.

        Args:
            source_name: Source name
            config: Configuration to validate

        Returns:
            Validation result
        """
        source_config = self._sources.get(source_name, {})
        schema = source_config.get("config_schema", {})

        # Simple validation
        errors = []
        for key, rules in schema.items():
            if rules.get("required") and key not in config:
                errors.append(f"Missing required field: {key}")
            if key in config:
                value = config[key]
                if "type" in rules:
                    expected_type = rules["type"]
                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Field {key} must be a string")
                    elif expected_type == "int" and not isinstance(value, int):
                        errors.append(f"Field {key} must be an integer")
                    elif expected_type == "bool" and not isinstance(value, bool):
                        errors.append(f"Field {key} must be a boolean")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }
