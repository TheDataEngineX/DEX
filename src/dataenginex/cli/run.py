"""``dex run`` — queue a workload for a worker to execute (§5.6, §7).

This command no longer runs anything itself. It opens the project, asks the
gateway to start a run, and reports the run id; a worker picks the work up from
the durable queue.

The previous version constructed a ``PipelineRunner`` and executed the pipeline
inside the CLI process. That meant the work died with the terminal, left no run
record anything else could see, and could not be cancelled, retried, or leased —
so two people running the same pipeline ran it twice, concurrently, over the
same outputs.
"""

from __future__ import annotations

from pathlib import Path

import click
import structlog

from dataenginex.bootstrap import build_lite_gateway, open_control_store
from dataenginex.bootstrap.settings import Settings
from dataenginex.foundation import PrincipalId, ProjectId
from dataenginex.interfaces import Command, GatewayError, Query

logger = structlog.get_logger()

__all__ = ["run"]

# Who the CLI acts as. Lite associates with the OS user (§9.2); naming the
# principal keeps a queued run attributable to something.
_LOCAL_PRINCIPAL = "prin_cli_local"

_STATE_DIR_HELP = "Where the control store lives (default: $DEX_STATE_DIR or .dex)."


@click.command()
@click.argument("pipeline", required=False)
@click.option("--all", "run_all", is_flag=True, help="Queue every workload the project declares")
@click.option("--config", "config_path", default="dex.yaml", help="Project directory or dex.yaml")
@click.option("--state-dir", help=_STATE_DIR_HELP)
def run(pipeline: str | None, run_all: bool, config_path: str, state_dir: str | None) -> None:
    """Queue workloads declared in dex.yaml.

    Nothing executes until a worker claims the run — start one with
    ``dex worker start``.
    """
    if not run_all and not pipeline:
        raise click.UsageError("Specify a workload name or use --all")

    settings = Settings.from_env(state_dir=Path(state_dir) if state_dir else None)
    store = open_control_store(settings)
    try:
        gateway = build_lite_gateway(store)
        principal = PrincipalId(_LOCAL_PRINCIPAL)

        opened = gateway.open_project(Command(principal_id=principal), source=config_path)
        project_id = ProjectId(str(opened.subject_id))
        click.echo(opened.message)

        if run_all:
            query = Query(principal_id=principal, project_id=project_id, limit=500)
            names = [w.name for w in gateway.list_workloads(query).items]
        else:
            names = [str(pipeline)]
        if not names:
            raise click.ClickException("This project declares no workloads.")

        failed = False
        for name in names:
            try:
                accepted = gateway.start_run(
                    Command(principal_id=principal, project_id=project_id), workload=name
                )
            except GatewayError as exc:
                # One refusal must not silence the rest: a policy denial on one
                # workload says nothing about the others.
                click.echo(f"{name}: refused — {exc.message}", err=True)
                failed = True
                continue
            click.echo(f"{name}: queued as {accepted.subject_id}")

        click.echo("\nNothing has run yet. Start a worker with `dex worker start`.")
        if failed:
            raise SystemExit(1)
    finally:
        store.close()
