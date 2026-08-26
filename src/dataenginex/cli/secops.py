"""`dex secops` — inspect and test the PrivacyGuard from the command line.

Subcommands::

    dex secops status            # show the policies this installation enforces
    dex secops scan "some text"  # run PII detection and show matches
"""

from __future__ import annotations

from pathlib import Path

import click

_STATE_DIR_HELP = "Where the control store lives (default: $DEX_STATE_DIR or .dex)."

# Who the CLI acts as when reading the control plane (§9.2).
_LOCAL_PRINCIPAL = "prin_cli_local"


@click.group()
def secops() -> None:
    """SecOps — PrivacyGuard inspection and PII scanning."""


@secops.command()
@click.option("--state-dir", help=_STATE_DIR_HELP)
def status(state_dir: str | None) -> None:
    """Show the policies this installation actually enforces (§9.3).

    Read from the policy engine rather than from a ``secops:`` block in
    dex.yaml. The config block was reported here but evaluated nowhere, so the
    command described a security posture that no request was ever subject to.
    """
    from dataenginex.bootstrap import build_lite_gateway, open_control_store
    from dataenginex.bootstrap.settings import Settings
    from dataenginex.foundation import PrincipalId
    from dataenginex.interfaces import Query

    settings = Settings.from_env(state_dir=Path(state_dir) if state_dir else None)
    store = open_control_store(settings)
    try:
        gateway = build_lite_gateway(store)
        policies = gateway.list_policies(Query(principal_id=PrincipalId(_LOCAL_PRINCIPAL))).items

        _section(f"Policies ({len(policies)})")
        for policy in sorted(policies, key=lambda p: -p.priority):
            actions = ", ".join(policy.actions) if policy.actions else "*"
            _row(
                f"{policy.effect.value:<8} {policy.name}",
                f"{actions}  (priority {policy.priority}, max risk {policy.max_risk_level})",
            )
        if not policies:
            # Not a benign empty list: with no rule matching, the engine's
            # default deny refuses everything.
            _row("(none)", "default deny — nothing is permitted")
        click.echo()
    finally:
        store.close()


@secops.command()
@click.argument("text")
@click.option(
    "--target",
    default="openai",
    show_default=True,
    help="Provider target name (affects local-bypass logic).",
)
def scan(text: str, target: str) -> None:
    """Scan TEXT for PII using the default guard configuration.

    Prints each match (type, confidence, matched value) and shows the masked
    output the guard would send to the provider.

    No ``--config``: detection is a property of the guard, not of a project, and
    the option invited the reading that a project could weaken it.
    """
    from dataenginex.secops import PrivacyGuard, PrivacyGuardConfig

    guard = PrivacyGuard(config=PrivacyGuardConfig())

    result = guard.process(text, target=target)

    if result.bypassed_local:
        click.echo(click.style(f"⊘  Bypassed — '{target}' is a local provider", fg="yellow"))
        return

    if result.detections:
        _section(f"Detections ({len(result.detections)})")
        for m in result.detections:
            conf = f"{m.confidence:.0%}" if m.confidence is not None else ""
            click.echo(
                f"  {click.style(m.pii_type.value, fg='red', bold=True):<20}"
                f"  {conf:<8}"
                f"  {m.value!r}"
            )
        click.echo()
        if result.blocked:
            click.echo(click.style("✗  BLOCKED — prompt would not be sent", fg="red", bold=True))
        else:
            _section("Masked output")
            click.echo(f"  {result.safe_prompt}")
    else:
        click.echo(click.style("✓  No PII detected", fg="green"))
        click.echo(f"  {text}")

    click.echo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    click.echo(click.style(f"  {title}", bold=True))
    click.echo(click.style("  " + "─" * (len(title) + 2), fg="bright_black"))


def _row(key: str, value: str) -> None:
    click.echo(f"    {key:<28}{value}")


def _yn(flag: bool) -> str:
    return click.style("yes", fg="green") if flag else click.style("no", fg="red")
