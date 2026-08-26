"""Entry point for the ``dex`` CLI.

Usage::

    dex --help
    dex validate .
    dex version
"""

from __future__ import annotations

from pathlib import Path

import click
import structlog

from dataenginex.foundation import ValidationSeverity
from dataenginex.runtime.compiler import ProjectCompiler

logger = structlog.get_logger()


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    col_w = max(len(r[0]) for r in rows) + 2
    click.echo(f"\n{title}")
    click.echo("-" * (col_w + 20))
    for key, val in rows:
        click.echo(f"  {key:<{col_w}}{val}")
    click.echo()


@click.group()
@click.version_option(package_name="dataenginex")
def dex() -> None:
    """DataEngineX — unified Data + ML + AI framework."""


@dex.command()
@click.argument(
    "project_path",
    type=click.Path(exists=True, path_type=Path),
    default=".",
)
def validate(project_path: Path) -> None:
    """Compile a project and report what the compiler found (§6.8, §12.5).

    Runs the same eleven stages a publish runs, so "valid" here means the same
    thing it will mean at publish time. A validate that used a lighter check
    would be worse than none: it would certify projects the publisher then
    rejects.
    """
    # A directory is the project. Accepting the manifest path too is for the
    # muscle memory of anyone typing `dex validate dex.yaml`.
    root = project_path.parent if project_path.is_file() else project_path

    result = ProjectCompiler(root).compile()

    errors = [i for i in result.report.issues if i.severity is ValidationSeverity.ERROR]
    warnings = [i for i in result.report.issues if i.severity is ValidationSeverity.WARNING]

    for issue in warnings:
        where = f" [{issue.location}]" if issue.location else ""
        click.echo(f"  ! {issue.code}: {issue.message}{where}")

    if errors:
        for issue in errors:
            where = f" [{issue.location}]" if issue.location else ""
            click.echo(f"  x {issue.code}: {issue.message}{where}", err=True)
        click.echo(f"\n{len(errors)} error(s). Project is not publishable.", err=True)
        raise SystemExit(1)

    _print_table(
        f"Project: {result.manifest.metadata.name}",
        [
            ("Profile", result.manifest.spec.profile),
            ("Workloads", str(len(result.workloads))),
            ("Resources", str(len(result.resources))),
            ("Execution order", " -> ".join(result.execution_order) or "(none)"),
            ("Capabilities", ", ".join(result.required_capabilities) or "(none)"),
            ("Destinations", ", ".join(result.declared_destinations) or "(none, default deny)"),
            ("Content hash", result.content_hash),
            # Absent means nothing pins the dependencies, which makes the
            # revision reproducible only by luck. Worth saying out loud.
            ("Dependency lock", result.dependency_lock_hash or "(none)"),
            ("Warnings", str(len(warnings))),
        ],
    )
    click.echo("Project is valid.")


@dex.command()
def version() -> None:
    """Show DataEngineX version and environment info."""
    import importlib.metadata
    import platform
    import sys

    ver = importlib.metadata.version("dataenginex")
    click.echo(f"DataEngineX {ver}")
    click.echo(f"Python {sys.version}")
    click.echo(f"Platform {platform.platform()}")


from dataenginex.cli.run import run  # noqa: E402
from dataenginex.cli.runtime import runtime  # noqa: E402
from dataenginex.cli.secops import secops  # noqa: E402
from dataenginex.cli.studio import studio  # noqa: E402
from dataenginex.cli.train import train  # noqa: E402
from dataenginex.cli.worker import worker  # noqa: E402

dex.add_command(run)
dex.add_command(runtime)
dex.add_command(worker)
dex.add_command(secops)
dex.add_command(studio)
dex.add_command(train)

if __name__ == "__main__":
    dex()
