"""Model serving engine registry.

Built-in serving engine wraps the existing ModelServer.
BentoML available via ``[bentoml]`` extra.
"""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseServingEngine
from dataenginex.runtime.registry import BackendRegistry

serving_registry: BackendRegistry[BaseServingEngine] = BackendRegistry("serving")
