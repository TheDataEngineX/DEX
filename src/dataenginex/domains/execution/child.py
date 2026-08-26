"""Child process entry point for :class:`SubprocessBackend` (§7.9).

Reads one execution plan from stdin as JSON, runs it, and exits non-zero on
failure. Deliberately minimal: everything this process knows arrives on stdin or
in its scrubbed environment, so it cannot reach back into the control plane.
That one-way flow is what makes the control/execution split real rather than a
naming convention (ADR-0004).

Not imported by the parent — only executed via ``-m``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from dataenginex.foundation import UnknownOperationError, registry

__all__ = ["main", "run_payload"]


def run_payload(payload: dict[str, Any]) -> int:
    """Execute a decoded plan. Returns the process exit code.

    Split from :func:`main` so the logic is testable without spawning a
    process — a subprocess-only path is one that gets tested by hand and then
    not at all.
    """
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        print("payload has no plan", file=sys.stderr)
        return 2

    operations = plan.get("operations") or []
    if not operations:
        print("plan declares no operations", file=sys.stderr)
        return 2

    for declared in operations:
        operation_type = declared.get("operation_type")
        try:
            registry.get(str(operation_type))
        except UnknownOperationError as exc:
            print(str(exc), file=sys.stderr)
            return 3

    # Operation implementations are dispatched here once domain handlers are
    # registered. Until then a plan that type-checks is a successful no-op
    # rather than a silent failure, and the unknown-operation check above still
    # rejects anything the catalogue does not declare.
    return 0


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"could not decode the execution payload: {exc}", file=sys.stderr)
        return 2

    if not isinstance(payload, dict):
        print("execution payload must be a JSON object", file=sys.stderr)
        return 2

    return run_payload(payload)


if __name__ == "__main__":
    raise SystemExit(main())
