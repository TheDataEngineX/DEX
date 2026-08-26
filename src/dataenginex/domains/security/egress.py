"""Network egress enforcement (§9.7).

Default deny. A project reaches an external host only if that host is declared
in its manifest *and* the capability token for the running attempt carries it.
Both checks are required and neither is redundant: the manifest declaration is
what a human reviewed at publish time, and the token scope is what bounds this
particular attempt. A compromised operation holding a valid token still cannot
reach a host the project never declared.

Denials are structured rather than a bare ``False`` — the caller has to write an
audit record naming the destination and the reason, and reconstructing that from
a boolean is guesswork.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence

from dataenginex.foundation import (
    CapabilityToken,
    NetworkDestination,
    Obligation,
    ObligationType,
)
from dataenginex.foundation.policy import extract_host
from dataenginex.foundation.projects import FrozenModel

__all__ = [
    "EgressDecision",
    "EgressGuard",
    "extract_host",
    "redaction_obligation",
]

# Link-local and cloud metadata endpoints. Reaching these from project code is
# nearly always SSRF rather than intent, so they are refused ahead of any
# project declaration.
_DEFAULT_BLOCKED: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)


class EgressDecision(FrozenModel):
    """The outcome of one egress check.

    Carries the reason and the matched rule so the audit record and the user's
    error message can both be built from it without re-deriving anything.
    """

    allowed: bool
    host: str
    reason: str
    matched_rule: str | None = None
    obligations: tuple[Obligation, ...] = ()




def _host_matches(host: str, pattern: str) -> bool:
    """Glob match of a host against a rule pattern.

    ``*.example.com`` deliberately does not match bare ``example.com``: a
    wildcard for subdomains should not silently grant the apex, which is
    usually a different service.
    """
    return fnmatch.fnmatch(host, pattern)


def _token_permits(capability: CapabilityToken, host: str) -> bool:
    """Whether the attempt's token carries this host.

    Fails closed on an empty scope — a token with no destinations grants no
    egress at all, which is the correct reading of "narrowly scoped" (§9.4).
    """
    if not capability.destinations:
        return False
    return any(_host_matches(host, d.lower()) for d in capability.destinations)


class EgressGuard:
    """Checks outbound destinations against declared policy (§9.7).

    ``declared`` comes from the compiled project manifest; the compiler has
    already collected the declared destinations at publish time, so this is the
    runtime half of a check that starts at compile time.
    """

    def __init__(
        self,
        declared: Sequence[NetworkDestination] = (),
        *,
        blocked: Iterable[str] | None = None,
        obligations: Sequence[Obligation] = (),
    ) -> None:
        self._declared = tuple(declared)
        # Explicit blocks beat declarations. Passing an empty iterable disables
        # the defaults, which is why the sentinel is None rather than ().
        self._blocked = (
            _DEFAULT_BLOCKED if blocked is None else frozenset(b.lower() for b in blocked)
        )
        self._obligations = tuple(obligations)

    def check(
        self,
        target: str,
        *,
        capability: CapabilityToken | None = None,
        operation: str | None = None,
    ) -> EgressDecision:
        """Whether ``target`` may be reached.

        ``capability`` is optional only so that compile-time and diagnostic
        callers can check a destination without an attempt in flight. At
        runtime it is always supplied, and omitting it skips the per-attempt
        bound — never do that on an execution path.
        """
        host = extract_host(target)
        if not host:
            return EgressDecision(
                allowed=False,
                host="",
                reason=f"could not parse a host from {target!r}",
            )

        for pattern in self._blocked:
            if _host_matches(host, pattern):
                return EgressDecision(
                    allowed=False,
                    host=host,
                    reason=f"{host} is blocked by installation policy",
                    matched_rule=pattern,
                )

        declaration = self._find_declaration(host)
        if declaration is None:
            return EgressDecision(
                allowed=False,
                host=host,
                reason=(
                    f"{host} is not a declared destination for this project "
                    "(§9.7 default deny)"
                ),
            )

        if (
            operation is not None
            and declaration.operations
            and operation not in declaration.operations
        ):
            return EgressDecision(
                allowed=False,
                host=host,
                reason=f"{host} is declared but not for operation {operation!r}",
                matched_rule=declaration.host,
            )

        if capability is not None and not _token_permits(capability, host):
            return EgressDecision(
                allowed=False,
                host=host,
                reason=f"capability token does not carry {host} in its destination scope",
                matched_rule=declaration.host,
            )

        return EgressDecision(
            allowed=True,
            host=host,
            reason=f"{host} declared for purpose {declaration.purpose or 'unspecified'}",
            matched_rule=declaration.host,
            obligations=self._obligations,
        )

    def _find_declaration(self, host: str) -> NetworkDestination | None:
        for destination in self._declared:
            if _host_matches(host, destination.host.lower()):
                return destination
        return None


def redaction_obligation(fields: Sequence[str]) -> Obligation:
    """Build the obligation attached to a permitted-but-redacted egress."""
    return Obligation(obligation_type=ObligationType.REDACT_FIELDS, parameters=tuple(fields))
