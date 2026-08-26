"""Experiment tracking registry.

Built-in tracker uses JSON storage. MLflow available via ``[mlflow]`` extra.
"""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseTracker
from dataenginex.runtime.registry import BackendRegistry

tracker_registry: BackendRegistry[BaseTracker] = BackendRegistry("tracker")
