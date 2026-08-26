"""Feature store registry.

Built-in feature store uses DuckDB. Feast available via ``[feast]`` extra.
"""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseFeatureStore
from dataenginex.runtime.registry import BackendRegistry

feature_store_registry: BackendRegistry[BaseFeatureStore] = BackendRegistry("feature_store")
