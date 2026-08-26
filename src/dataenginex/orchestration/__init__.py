"""Orchestration — scheduling and execution coordination."""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseOrchestrator
from dataenginex.runtime.registry import BackendRegistry

orchestrator_registry: BackendRegistry[BaseOrchestrator] = BackendRegistry("orchestrator")

__all__ = [
    "BaseOrchestrator",
    "orchestrator_registry",
]
