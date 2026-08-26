"""Client-facing interfaces: gateway, CLI, SDK (§12, §13)."""

from dataenginex.interfaces.embedded import EmbeddedGateway
from dataenginex.interfaces.gateway import (
    Command,
    CommandResult,
    CursorPage,
    DexGateway,
    ErrorCode,
    GatewayError,
    ProjectSummary,
    Query,
    RunSummary,
)

__all__ = [
    "Command",
    "CommandResult",
    "CursorPage",
    "DexGateway",
    "EmbeddedGateway",
    "ErrorCode",
    "GatewayError",
    "ProjectSummary",
    "Query",
    "RunSummary",
]
