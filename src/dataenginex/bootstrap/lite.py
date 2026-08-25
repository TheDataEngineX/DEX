"""Lite profile — everything in one process, no external services (§11.3).

This is the assembly point, and the only place in the package where concrete
implementations are chosen. Every other layer names a Protocol from
``foundation/contracts.py``; here the Protocols become objects.

That is why the wiring is worth a module of its own rather than a helper on the
gateway. When the store moves to Postgres, or execution moves off-process, the
edit lands here and nowhere else — and the architecture test asserts that no
other layer imports ``providers/``, so the claim stays true.

Usage::

    from dataenginex.bootstrap import lite

    with lite() as dex:
        dex.start_run(command, workload="daily_load")
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dataenginex.bootstrap.settings import Settings
from dataenginex.domains.execution.backends import InProcessBackend
from dataenginex.domains.execution.handlers import (
    HandlerError,
    ResultSink,
    register_default_handlers,
)
from dataenginex.domains.security import DEFAULT_POLICY_SET, GovernanceService
from dataenginex.foundation import ExecutionContext
from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.foundation.policy import Policy
from dataenginex.interfaces.embedded import EmbeddedGateway
from dataenginex.providers.connectors import connector_registry
from dataenginex.runtime.planning import store_interactive_result
from dataenginex.runtime.queue import DurableQueue
from dataenginex.runtime.state import ControlStore

__all__ = [
    "build_lite_backend",
    "build_lite_gateway",
    "lite",
    "open_connector",
    "open_control_store",
]


def open_control_store(settings: Settings | None = None) -> ControlStore:
    """Open and migrate the control store.

    Migration runs on open rather than behind a separate command because a
    store on an older schema is not usable, and making the caller remember to
    migrate turns a guaranteed step into an occasional bug.
    """
    resolved = settings or Settings()
    store = ControlStore(resolved.control_db_path, timeout=resolved.store_timeout_seconds)
    store.migrate()
    return store


def build_lite_gateway(
    store: ControlStore,
    *,
    policies: Sequence[Policy] = DEFAULT_POLICY_SET,
) -> EmbeddedGateway:
    """Wire an in-process gateway over an already-open store.

    Takes the store rather than opening one so a caller that already has state
    — the CLI, a test, a Studio process — assembles the gateway over *that*
    store. Two stores over one deployment would mean two views of the same
    runs.

    The policy set is a parameter because it is a deployment decision. The
    default denies anything not explicitly permitted (§9.7), which is the right
    default and the wrong thing to hardcode.
    """
    return EmbeddedGateway(
        store,
        governance=GovernanceService(store, policies=policies),
        queue=DurableQueue(store),
    )


def open_connector(config: Mapping[str, str]) -> BaseConnector:
    """Resolve a resource's declared config to a connector (§4.6).

    Lives here because it names ``providers/``, which no other layer may. The
    handlers that use it take it as an argument, so moving a project onto a
    different connector set is a change to this function and to nothing else.
    """
    settings = {key: value for key, value in config.items() if key != "type"}
    kind = config.get("type") or "duckdb"
    try:
        cls = connector_registry.get(kind)
    except KeyError as exc:
        available = ", ".join(sorted(connector_registry.list()))
        raise HandlerError(
            f"no connector for resource type {kind!r}; this installation has: {available}"
        ) from exc
    return cls(**settings)


def build_lite_backend(store: ControlStore | None = None) -> InProcessBackend:
    """The execution backend a Lite worker runs plans through.

    In-process because Lite is one process by definition (§11.3). A deployment
    that wants a workload's crash to be survivable wires ``SubprocessBackend``
    here instead — which is the whole reason this choice lives in bootstrap.

    Without a *store* the interactive handlers are left unregistered. A worker
    that cannot store a preview should refuse the work rather than run a query
    and drop the answer: the run then fails with "no handler registered", which
    is at least a true statement about that worker.
    """
    sink = _result_sink(store) if store is not None else None
    return register_default_handlers(InProcessBackend(), open_connector, sink)


def _result_sink(store: ControlStore) -> ResultSink:
    """Bind a control store into the callable interactive handlers are given."""

    def sink(
        context: ExecutionContext,
        payload: dict[str, Any],
        row_count: int,
        truncated: bool,
    ) -> None:
        store_interactive_result(store, context, payload, row_count, truncated)

    return sink


@contextmanager
def lite(
    state_dir: Path | str | None = None,
    *,
    policies: Sequence[Policy] = DEFAULT_POLICY_SET,
) -> Iterator[EmbeddedGateway]:
    """Open a complete Lite deployment and close it afterwards.

    The context manager is the point: the control store holds a SQLite
    connection per thread, and a caller that forgets to close leaks them for
    the life of the process.
    """
    store = open_control_store(Settings.from_env(state_dir=state_dir))
    try:
        yield build_lite_gateway(store, policies=policies)
    finally:
        store.close()


