"""``dex studio`` — serve the web UI against a wired gateway (§5.6).

Studio is handed a gateway; it does not build one (§5.5). This command is the
host that does the handing: it opens the control store, opens the project, and
injects the gateway before uvicorn serves the first request.

That ordering is the point. Studio had no host — ``set_gateway`` had no caller
outside tests — so anyone running the app directly got a process in which every
ported route answered 503 and no page could show a run.

Studio lives in its own distribution (``dex-studio``). Importing it lazily keeps
the core installable without it, and turns "not installed" into a sentence
rather than an ImportError traceback.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import structlog

from dataenginex.bootstrap import build_lite_gateway, open_control_store
from dataenginex.bootstrap.settings import Settings
from dataenginex.foundation import PrincipalId
from dataenginex.interfaces import Command

log = structlog.get_logger().bind(src="studio")

__all__ = ["studio"]

# Who the UI acts as before a session exists. Lite binds to loopback and
# associates with the OS user (§9.2), so there is exactly one human; naming them
# keeps the publish this command performs attributable.
_LOCAL_PRINCIPAL = "prin_studio_local"

_STATE_DIR_HELP = "Where the control store lives (default: $DEX_STATE_DIR or .dex)."


@click.command()
@click.option(
    "--project",
    type=click.Path(exists=True, path_type=Path),
    help="Project directory or dex.yaml to open. Defaults to $DEX_CONFIG_PATH.",
)
@click.option("--host", default="127.0.0.1", help="Bind address. Loopback by default (§9.2).")
@click.option("--port", default=7860, type=int, help="Port to serve on.")
@click.option("--state-dir", help=_STATE_DIR_HELP)
def studio(project: Path | None, host: str, port: int, state_dir: str | None) -> None:
    """Serve DEX Studio."""
    try:
        import uvicorn

        # Untyped to mypy because Studio ships no ``py.typed`` — it is a
        # separate distribution, and the core does not depend on it.
        from dex_studio._gateway import (
            select_project,
            set_gateway,
        )
        from dex_studio.app import create_app
    except ImportError as exc:  # pragma: no cover - depends on what is installed
        raise click.ClickException(
            "dex-studio is not installed in this environment. "
            "Install it with `pip install dex-studio`."
        ) from exc

    store = open_control_store(Settings.from_env(state_dir=Path(state_dir) if state_dir else None))
    gateway = build_lite_gateway(store)
    set_gateway(gateway)

    configured = os.environ.get("DEX_CONFIG_PATH")
    manifest = project or (Path(configured) if configured else None)
    if manifest is not None:
        # Opening publishes, so a project's resources and workloads exist before
        # the first page reads them. Without it the UI renders an empty project,
        # indistinguishable from one that genuinely contains nothing.
        result = gateway.open_project(
            Command(principal_id=PrincipalId(_LOCAL_PRINCIPAL)), source=str(manifest)
        )
        select_project(str(result.subject_id))
        click.echo(result.message)
        # The routes not yet ported still read the manifest through the engine,
        # which finds it here.
        os.environ["DEX_CONFIG_PATH"] = str(manifest)
    else:
        click.echo("No project selected — open one from the onboarding page.")

    click.echo(f"DEX Studio on http://{host}:{port}")
    try:
        uvicorn.run(create_app(), host=host, port=port)
    finally:
        # This process opened the store, so this process closes it.
        store.close()
