"""Shared plumbing for application services (§5.3).

Application services are the middle layer §5.4 requires: *"EmbeddedGateway
invokes local application services."* Before this existed the gateway wrote SQL
against the control store directly, which works but puts query logic in the
transport. Every method added that way is another reason for the next one to go
there too, and the end state of that road is a 1300-line object with a protocol
bolted to the front — the exact thing the greenfield rewrite is replacing.

Two rules hold everywhere in this package:

**Services depend on foundation contracts, not concrete providers (§5.5).**
A service takes a ``ControlStore`` because reading the control plane *is* its
job, but it never imports a DuckDB connection, an HTTP client, or a model
runtime. Those arrive through ``providers/`` and are wired in ``bootstrap/``.

**Commands and queries stay apart (§13.3).** A command changes state, is
audited, and carries an idempotency key. A query changes nothing and carries a
cursor. Mixing them produces "read" paths that quietly mutate, which makes both
caching and auditing unsound.
"""

from __future__ import annotations

from typing import Any

from dataenginex.foundation import ProjectId, RevisionId
from dataenginex.runtime.state import ControlStore

__all__ = ["ApplicationError", "NotFoundError", "Service"]


class ApplicationError(RuntimeError):
    """A service could not complete a request.

    Distinct from ``GatewayError``: this layer does not know about transports or
    HTTP status codes. The gateway maps these onto stable error codes (§13.5),
    which is what keeps the application layer usable from the CLI and the SDK
    without dragging web semantics along.
    """


class NotFoundError(ApplicationError):
    """The requested subject does not exist."""


class Service:
    """Base for application services.

    Holds the control store and the read helpers every service needs.
    Deliberately thin — a base class that grows behaviour becomes the god object
    by another route, so the shared surface stays at "how do I read a row".
    """

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    # --- shared reads -------------------------------------------------------

    def active_revision(self, project_id: ProjectId) -> RevisionId:
        """The project's published revision (ADR-0003).

        Raises rather than falling back to working files. "Which definition did
        this run use?" must always have an answer, and a fallback is how that
        answer becomes "whatever was on disk at the time".
        """
        row = self.store.query_one(
            "SELECT active_revision_id FROM projects WHERE project_id = ?", (project_id,)
        )
        if row is None:
            raise NotFoundError(f"no project {project_id}")
        if not row["active_revision_id"]:
            raise NotFoundError(f"project {project_id} has no published revision")
        return RevisionId(row["active_revision_id"])

    def require_row(self, sql: str, params: tuple[Any, ...], *, subject: str) -> Any:
        """Fetch exactly one row or raise.

        Centralised so every "not found" reads the same way to a caller. A
        service that returns ``None`` for a missing subject pushes the check to
        every call site, and one of them will forget.
        """
        row = self.store.query_one(sql, params)
        if row is None:
            raise NotFoundError(subject)
        return row
