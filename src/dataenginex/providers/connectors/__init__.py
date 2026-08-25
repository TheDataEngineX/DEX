"""Connector registry and public API."""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.runtime.registry import BackendRegistry

connector_registry: BackendRegistry[BaseConnector] = BackendRegistry("connector")

from dataenginex.providers.connectors.csv import CsvConnector  # noqa: E402, F401
from dataenginex.providers.connectors.dbt import DbtConnector  # noqa: E402, F401
from dataenginex.providers.connectors.delta import DeltaConnector  # noqa: E402, F401
from dataenginex.providers.connectors.duckdb import DuckDBConnector  # noqa: E402, F401
from dataenginex.providers.connectors.http import HttpConnector  # noqa: E402, F401
from dataenginex.providers.connectors.kafka import KafkaConnector  # noqa: E402, F401
from dataenginex.providers.connectors.parquet import ParquetConnector  # noqa: E402, F401
from dataenginex.providers.connectors.rest import RestApiConnector  # noqa: E402, F401
from dataenginex.providers.connectors.spark import SparkConnector  # noqa: E402, F401
from dataenginex.providers.connectors.sse import SseConnector  # noqa: E402, F401

__all__ = [
    "BaseConnector",
    "connector_registry",
    "CsvConnector",
    "DbtConnector",
    "DeltaConnector",
    "DuckDBConnector",
    "HttpConnector",
    "KafkaConnector",
    "ParquetConnector",
    "RestApiConnector",
    "SparkConnector",
    "SseConnector",
]
