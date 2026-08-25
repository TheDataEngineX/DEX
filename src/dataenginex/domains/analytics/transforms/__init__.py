"""Transform registry and public API."""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseTransform
from dataenginex.runtime.registry import BackendRegistry

transform_registry: BackendRegistry[BaseTransform] = BackendRegistry("transform")

# Imported for the side effect of registering, exactly as the connector package
# does. Without this the registry is empty at import time and every lookup
# fails with "not registered" for a transform that is right here.
from dataenginex.domains.analytics.transforms import sql as _sql  # noqa: E402, F401

__all__ = ["BaseTransform", "transform_registry"]
