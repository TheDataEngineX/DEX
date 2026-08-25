"""Approval service (§4.13).

Manages human approval workflow for consequential actions.
Approvals are first-class stateful resources with operation-digest binding.
"""

from __future__ import annotations

from typing import Any

from dataenginex.application.services import ApplicationError, Service
from dataenginex.foundation.ids import ApprovalId, PrincipalId, new_id
from dataenginex.foundation.projects import utcnow

__all__ = ["ApprovalService", "ApprovalView"]


class ApprovalView:
    """Read-only projection of an approval."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.approval_id = row["approval_id"]
        self.action = row["action"]
        self.project_id = row["project_id"]
        self.requested_by = row["requested_by"]
        self.state = row["state"]
        self.operation_digest = row.get("operation_digest", "")
        self.requested_at = row.get("requested_at", "")
        self.decided_at = row.get("decided_at")
        self.decided_by = row.get("decided_by")
        self.expiry = row.get("expiry")


class ApprovalAlreadyDecided(ApplicationError):
    """This approval has already been decided."""


class ApprovalExpired(ApplicationError):
    """This approval has expired."""


class ApprovalDigestMismatch(ApplicationError):
    """The operation has changed since approval was requested."""


class ApprovalService(Service):
    """Human approval workflow (§4.13)."""

    def request_approval(
        self,
        *,
        project_id: str,
        action: str,
        requested_by: PrincipalId,
        operation_digest: str = "",
        summary: str = "",
        expiry_hours: int = 24,
    ) -> ApprovalId:
        """Request human approval for an action."""
        approval_id = ApprovalId(new_id("apr"))
        now = utcnow()
        self.store.query_one(
            "INSERT INTO approvals "
            "(approval_id, project_id, action, requested_by, state, "
            "operation_digest, summary, requested_at, expiry) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
            (
                approval_id, project_id, action, requested_by,
                operation_digest, summary, now.isoformat(),
                (now + __import__("datetime").timedelta(hours=expiry_hours)).isoformat(),
            ),
        )
        return approval_id

    def approve(
        self,
        approval_id: ApprovalId,
        decided_by: PrincipalId,
        *,
        comment: str = "",
    ) -> None:
        """Approve a pending request."""
        row = self.require_row(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (approval_id,),
            subject=f"approval {approval_id}",
        )
        if row["state"] != "pending":
            raise ApprovalAlreadyDecided(f"approval {approval_id} is {row['state']}")
        if row.get("expiry") and utcnow().isoformat() > row["expiry"]:
            raise ApprovalExpired(f"approval {approval_id} has expired")

        self.store.query_one(
            "UPDATE approvals SET state = 'approved', decided_by = ?, decided_at = ?, comment = ? "
            "WHERE approval_id = ?",
            (decided_by, utcnow().isoformat(), comment, approval_id),
        )

    def reject(
        self,
        approval_id: ApprovalId,
        decided_by: PrincipalId,
        *,
        reason: str = "",
    ) -> None:
        """Reject a pending request."""
        row = self.require_row(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (approval_id,),
            subject=f"approval {approval_id}",
        )
        if row["state"] != "pending":
            raise ApprovalAlreadyDecided(f"approval {approval_id} is {row['state']}")

        self.store.query_one(
            "UPDATE approvals SET state = 'rejected', decided_by = ?, decided_at = ?, comment = ? "
            "WHERE approval_id = ?",
            (decided_by, utcnow().isoformat(), reason, approval_id),
        )

    def get_approval(self, approval_id: ApprovalId) -> ApprovalView:
        """Fetch an approval by ID."""
        row = self.require_row(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (approval_id,),
            subject=f"approval {approval_id}",
        )
        return ApprovalView(row)

    def list_pending(self, project_id: str | None = None) -> list[ApprovalView]:
        """List pending approvals, optionally filtered by project."""
        if project_id:
            rows = self.store.query(
                "SELECT * FROM approvals WHERE state = 'pending' AND project_id = ? "
                "ORDER BY requested_at",
                (project_id,),
            )
        else:
            rows = self.store.query(
                "SELECT * FROM approvals WHERE state = 'pending' ORDER BY requested_at",
                (),
            )
        return [ApprovalView(dict(r)) for r in rows]

    def check_digest(
        self, approval_id: ApprovalId, current_operation_digest: str
    ) -> bool:
        """Verify the operation hasn't changed since approval was requested (§4.13).

        A changed operation invalidates the approval.
        """
        row = self.require_row(
            "SELECT operation_digest FROM approvals WHERE approval_id = ?",
            (approval_id,),
            subject=f"approval {approval_id}",
        )
        stored_digest = row.get("operation_digest", "")
        if not stored_digest:
            return True  # No digest binding
        return bool(stored_digest == current_operation_digest)
