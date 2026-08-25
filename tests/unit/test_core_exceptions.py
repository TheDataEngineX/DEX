"""Tests for the unified exception hierarchy."""

from __future__ import annotations

from dataenginex.foundation.errors import (
    BackendNotInstalledError,
    ConfigError,
    ConfigValidationError,
    DataEngineXError,
    PipelineError,
    RegistryError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from DataEngineXError."""

    def test_base_is_exception(self) -> None:
        assert issubclass(DataEngineXError, Exception)

    def test_config_error_inherits_base(self) -> None:
        assert issubclass(ConfigError, DataEngineXError)

    def test_config_validation_inherits_config(self) -> None:
        assert issubclass(ConfigValidationError, ConfigError)

    def test_pipeline_error_inherits_base(self) -> None:
        assert issubclass(PipelineError, DataEngineXError)

    def test_registry_error_inherits_base(self) -> None:
        assert issubclass(RegistryError, DataEngineXError)

    def test_backend_not_installed_inherits_base(self) -> None:
        assert issubclass(BackendNotInstalledError, DataEngineXError)

    def test_backend_not_installed_message(self) -> None:
        err = BackendNotInstalledError(backend="qdrant", extra="vectors")
        assert "qdrant" in str(err)
        assert "pip install dataenginex[vectors]" in str(err)

    def test_config_validation_error_fields(self) -> None:
        err = ConfigValidationError(
            field="ai.vectorstore.backend",
            message="unknown backend 'foo'",
        )
        assert "ai.vectorstore.backend" in str(err)
        assert "unknown backend 'foo'" in str(err)
