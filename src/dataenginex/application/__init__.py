"""Application services — the layer between interfaces and the runtime (§5.3).

Per §5.5 the arrows point inward::

    interfaces -> application -> foundation <- runtime, providers

This package may depend on foundation contracts and on runtime command
contracts. It must not import concrete providers — a DuckDB connection, an HTTP
client, a model runtime — because that would make the services untestable
without the world attached and would put wiring somewhere other than
``bootstrap/``.

§5.4 states the reason this layer exists: *"EmbeddedGateway invokes local
application services."* The gateway is a transport. It maps errors to stable
codes, applies pagination, and delegates. When query logic lives in the gateway
instead, every new transport re-implements it, and the shared logic drifts
between them.
"""

from dataenginex.application.approval_service import (
    ApprovalAlreadyDecided,
    ApprovalDigestMismatch,
    ApprovalExpired,
    ApprovalService,
    ApprovalView,
)
from dataenginex.application.catalog_service import (
    CatalogEntry,
    CatalogService,
)
from dataenginex.application.export_service import (
    ExportService,
)
from dataenginex.application.governance import (
    AuditEventView,
    DecisionView,
    GovernanceQueryService,
    PolicyView,
)
from dataenginex.application.projects import (
    ProjectService,
    ProjectView,
    PublishRejected,
    RevisionSummary,
)
from dataenginex.application.resources import (
    ResourceService,
    WorkloadService,
    WorkloadSummary,
)
from dataenginex.application.runs import (
    PolicyDenied,
    RunAccepted,
    RunService,
    RunView,
)
from dataenginex.application.schedules import (
    ScheduleFired,
    ScheduleService,
    ScheduleView,
)
from dataenginex.application.services import (
    ApplicationError,
    NotFoundError,
    Service,
)
from dataenginex.application.workspace_service import (
    WorkspaceService,
    WorkspaceView,
)

__all__ = [
    "ApplicationError",
    "ApprovalAlreadyDecided",
    "ApprovalDigestMismatch",
    "ApprovalExpired",
    "ApprovalService",
    "ApprovalView",
    "AuditEventView",
    "CatalogEntry",
    "CatalogService",
    "DecisionView",
    "ExportService",
    "GovernanceQueryService",
    "NotFoundError",
    "PolicyView",
    "PolicyDenied",
    "ProjectService",
    "ProjectView",
    "PublishRejected",
    "ResourceService",
    "RevisionSummary",
    "RunAccepted",
    "RunService",
    "RunView",
    "ScheduleFired",
    "ScheduleService",
    "ScheduleView",
    "Service",
    "WorkspaceService",
    "WorkspaceView",
    "WorkloadService",
    "WorkloadSummary",
]
