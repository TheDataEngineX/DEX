"""Secret management types (§9.6).

Secret values never appear in YAML, revision bundles, API responses, logs,
lineage, telemetry, exports, or error messages. Only references do. Providers
receive secrets just in time and only for the assigned operation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from dataenginex.foundation.ids import PrincipalId, ProjectId, new_id
from dataenginex.foundation.projects import FrozenModel, utcnow

__all__ = [
    "SecretReferenceV2",
    "SecretRotationPolicy",
]


class SecretRotationPolicy(StrEnum):
    """How secrets are rotated (§9.6)."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"


class SecretReferenceV2(FrozenModel):
    """Enhanced secret reference with full v0.7 metadata (§9.6).

    This replaces the simpler SecretReference in identity.py for new code.
    The key rule: there is deliberately no field that could hold a secret value.
    """

    secret_id: str = Field(default_factory=lambda: new_id("sec"))
    name: str
    project_id: ProjectId
    workspace_id: str | None = None
    provider: str = "keyring"
    owner: PrincipalId | None = None
    rotation_policy: SecretRotationPolicy = SecretRotationPolicy.MANUAL
    rotation_due: datetime | None = None
    # Principals permitted to resolve this reference; empty means project-scoped
    # default rules apply.
    permitted_consumers: tuple[PrincipalId, ...] = ()
    # Secret categories for policy matching (e.g. "database", "api_key", "oauth")
    categories: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utcnow)
    last_rotated_at: datetime | None = None
    # Whether the secret is currently enabled (disabled secrets cannot be resolved)
    enabled: bool = True
