"""Execution backends (§7.9, §13.6)."""

from dataenginex.domains.execution.backends import (
    BackendError,
    InProcessBackend,
    OperationHandler,
    SubprocessBackend,
    estimate_from_history,
)
from dataenginex.domains.execution.handlers import (
    ConnectorFactory,
    HandlerError,
    default_handlers,
    register_default_handlers,
)

__all__ = [
    "BackendError",
    "ConnectorFactory",
    "HandlerError",
    "InProcessBackend",
    "OperationHandler",
    "SubprocessBackend",
    "default_handlers",
    "estimate_from_history",
    "register_default_handlers",
]
