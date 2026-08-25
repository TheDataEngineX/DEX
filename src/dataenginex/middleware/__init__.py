"""Middleware — structured logging and HTTP metrics.

Public API::

    from dataenginex.middleware import configure_logging, get_logger, get_metrics

Domain telemetry moved to ``dataenginex.runtime.telemetry`` in 0.6. Re-exporting
it here meant a domain that wanted to count a pipeline run imported the web
tier to get the counter, which §5.5 forbids.
"""

from __future__ import annotations

from dataenginex.middleware.logging_config import APP_VERSION, configure_logging, get_logger
from dataenginex.middleware.metrics import get_metrics

__all__ = [
    "APP_VERSION",
    "configure_logging",
    "get_logger",
    "get_metrics",
]
