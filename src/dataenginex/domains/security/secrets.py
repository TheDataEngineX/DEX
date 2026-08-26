"""Secret providers: just-in-time resolution (§9.6).

Invariant 5 — secret values never appear in project revisions, metadata, logs,
lineage, telemetry, or exports. Only references do. This module is where a
reference becomes a value, and it is deliberately the narrowest surface in the
codebase:

* Resolution requires the attempt's capability token. Not ambient state, not a
  process-wide singleton — the caller must prove *this* attempt was authorized
  for *this* reference. A provider that reads permissions from a global cannot
  express "the worker got only what its attempt needed".
* The result is a lease with an expiry, not a plain string. A resolved secret
  that outlives its attempt is a standing credential.
* Errors name the reference, never the value. The obvious mistake here is an
  exception message that helpfully includes what it found.

``KeyringSecretProvider`` is the default because the OS keyring is already
encrypted at rest and already unlocked by the user's login — reimplementing that
badly is how local tools end up with a plaintext credentials file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import timedelta
from typing import Protocol

from dataenginex.foundation import (
    CapabilityToken,
    SecretLease,
    SecretReference,
    utcnow,
)

__all__ = [
    "DEFAULT_LEASE_TTL",
    "EnvSecretProvider",
    "KeyringSecretProvider",
    "MemorySecretProvider",
    "SecretAccessError",
    "SecretBackend",
    "SecretNotFoundError",
]

# Short by default: a lease should cover one operation, not one session.
DEFAULT_LEASE_TTL = timedelta(minutes=15)


class SecretAccessError(RuntimeError):
    """Resolution was refused.

    The message names the reference and the reason. It never contains a secret
    value, and it never chains the backend's raw error, which on some keyrings
    echoes the stored payload.
    """


class SecretNotFoundError(SecretAccessError):
    """No secret is stored under this reference."""


class SecretBackend(Protocol):
    """Where secret bytes actually live.

    Split from the provider so the authorization logic below is written once
    and shared by every storage mechanism.
    """

    def read(self, service: str, name: str) -> str | None: ...


def _authorize(reference: SecretReference, capability: CapabilityToken) -> None:
    """Check a capability token may resolve this reference (§9.4, §9.6).

    Every branch fails closed. Ordered cheapest-first, but the order carries no
    security meaning — any single failure refuses.
    """
    if capability.is_expired():
        raise SecretAccessError(
            f"cannot resolve secret {reference.name!r}: capability token has expired"
        )

    if capability.project_id != reference.project_id:
        raise SecretAccessError(
            f"cannot resolve secret {reference.name!r}: token is scoped to a different project"
        )

    # The token must name this reference. A token that carries no secret refs
    # resolves nothing — "unspecified" is not "all".
    if reference.name not in capability.secret_refs:
        raise SecretAccessError(
            f"cannot resolve secret {reference.name!r}: not in the token's secret scope"
        )

    if (
        reference.permitted_consumers
        and capability.principal_id not in reference.permitted_consumers
    ):
        raise SecretAccessError(
            f"cannot resolve secret {reference.name!r}: principal is not a permitted consumer"
        )


class _BaseSecretProvider:
    """Shared authorization and lease construction.

    Subclasses supply storage; the rules above are not theirs to reinterpret.
    """

    def __init__(self, *, ttl: timedelta = DEFAULT_LEASE_TTL) -> None:
        self._ttl = ttl

    def resolve(self, reference: SecretReference, capability: CapabilityToken) -> SecretLease:
        _authorize(reference, capability)
        value = self._read(reference)
        if value is None:
            raise SecretNotFoundError(f"no secret stored for reference {reference.name!r}")
        return SecretLease(
            reference_name=reference.name,
            value=value,
            expires_at=utcnow() + self._ttl,
        )

    def _read(self, reference: SecretReference) -> str | None:  # pragma: no cover - abstract
        raise NotImplementedError


class KeyringSecretProvider(_BaseSecretProvider):
    """Resolves from the OS keyring (§9.6).

    ``keyring`` is imported lazily so importing this module on a headless
    machine without a backend does not fail — the error belongs at the point
    someone actually asks for a secret, where it can name the reference.
    """

    def __init__(
        self,
        *,
        service_prefix: str = "dataenginex",
        ttl: timedelta = DEFAULT_LEASE_TTL,
    ) -> None:
        super().__init__(ttl=ttl)
        self._service_prefix = service_prefix

    def _read(self, reference: SecretReference) -> str | None:
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SecretAccessError(
                f"cannot resolve secret {reference.name!r}: no keyring backend installed"
            ) from exc

        service = f"{self._service_prefix}:{reference.project_id}"
        try:
            value: str | None = keyring.get_password(service, reference.name)
        except Exception as exc:  # pragma: no cover - backend dependent
            raise SecretAccessError(
                f"keyring lookup failed for secret {reference.name!r}"
            ) from exc
        return value


class EnvSecretProvider(_BaseSecretProvider):
    """Resolves from environment variables.

    For CI and container deployments where secrets arrive as env vars. The
    variable name is derived, not configurable per reference, so a project
    cannot point a reference at an arbitrary variable and read the host's
    environment.
    """

    def __init__(
        self,
        *,
        prefix: str = "DEX_SECRET_",
        ttl: timedelta = DEFAULT_LEASE_TTL,
    ) -> None:
        super().__init__(ttl=ttl)
        self._prefix = prefix

    def _read(self, reference: SecretReference) -> str | None:
        variable = self._prefix + reference.name.upper().replace("-", "_").replace(".", "_")
        return os.environ.get(variable)


class MemorySecretProvider(_BaseSecretProvider):
    """In-memory secrets for tests and ephemeral runs.

    Enforces the same authorization rules as the others, so a test that passes
    here exercises the actual gate rather than a permissive stub.
    """

    def __init__(
        self,
        secrets: Mapping[str, str] | None = None,
        *,
        ttl: timedelta = DEFAULT_LEASE_TTL,
    ) -> None:
        super().__init__(ttl=ttl)
        self._secrets = dict(secrets or {})

    def store(self, name: str, value: str) -> None:
        self._secrets[name] = value

    def _read(self, reference: SecretReference) -> str | None:
        return self._secrets.get(reference.name)
