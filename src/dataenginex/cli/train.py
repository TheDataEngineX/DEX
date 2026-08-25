"""``dex train`` — train ML models from config.

Usage::

    dex train churn_model --config dex.yaml
    dex train --all --config dex.yaml
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import click
import structlog

logger = structlog.get_logger()


@click.command("train")
@click.argument("experiment", required=False)
@click.option("--all", "run_all", is_flag=True, help="Train all experiments.")
@click.option(
    "--config",
    "config_path",
    default="dex.yaml",
    show_default=True,
    help="Path to dex.yaml config.",
)
@click.option(
    "--model-dir",
    default=".dex/models",
    show_default=True,
    help="Directory for model artifacts.",
)
def train(
    experiment: str | None,
    run_all: bool,
    config_path: str,
    model_dir: str,
) -> None:
    """Train ML models defined in dex.yaml."""
    from dataenginex.config.loader import load_config

    config = load_config(Path(config_path))

    if not config.ml.experiments:
        click.echo("No experiments defined in config.")
        return

    if not experiment and not run_all:
        click.echo("Specify an experiment name or use --all.")
        click.echo(f"Available: {list(config.ml.experiments.keys())}")
        return

    experiments_to_run = _resolve_experiments(experiment, run_all, config)

    rows = _run_experiments(experiments_to_run, config, model_dir)
    click.echo("\nTraining Results")
    click.echo(f"  {'Experiment':<30}{'Status':<20}{'Score':<12}Version")
    click.echo("-" * 72)
    for row in rows:
        click.echo(f"  {row['name']:<30}{row['status']:<20}{row['score']:<12}{row['version']}")


def _resolve_experiments(
    experiment: str | None,
    run_all: bool,
    config: Any,
) -> dict[str, Any]:
    """Resolve which experiments to run from config."""
    if run_all:
        return dict(config.ml.experiments)
    if experiment:
        if experiment not in config.ml.experiments:
            available = list(config.ml.experiments.keys())
            msg = f"Experiment '{experiment}' not found. Available: {available}"
            raise click.ClickException(msg)
        return {experiment: config.ml.experiments[experiment]}
    return {}


def _run_experiments(
    experiments: dict[str, Any],
    config: Any,
    model_dir: str,
) -> list[dict[str, str]]:
    """Run all experiments and return rows for tabular display."""
    from dataenginex.domains.ml.registry import ModelRegistry
    from dataenginex.domains.ml.tracking import tracker_registry

    # Import to trigger registry
    from dataenginex.domains.ml.tracking.builtin import BuiltinTracker as _  # noqa: F401

    tracker_cls = tracker_registry.get(config.ml.tracking.backend)
    tracker_kwargs: dict[str, Any] = {"storage_dir": f"{model_dir}/tracking"}
    if config.ml.tracking.uri:
        tracker_kwargs["uri"] = config.ml.tracking.uri
    tracker = tracker_cls(**tracker_kwargs)

    model_registry = ModelRegistry(persist_path=f"{model_dir}/registry.json")
    rows: list[dict[str, str]] = []

    for exp_name, exp_config in experiments.items():
        log = logger.bind(experiment=exp_name)
        log.info("training starting", model_type=exp_config.model_type)

        try:
            result = _train_experiment(
                exp_name,
                exp_config,
                tracker,
                model_registry,
                model_dir,
                log,
            )
            rows.append(
                {
                    "name": exp_name,
                    "status": "OK",
                    "score": str(result.get("score", "N/A")),
                    "version": result.get("version", "?"),
                }
            )
        except Exception as e:
            log.error("training failed", error=str(e))
            rows.append({"name": exp_name, "status": f"FAIL: {e}", "score": "-", "version": "-"})

    return rows


def _train_experiment(
    name: str,
    exp_config: Any,
    tracker: Any,
    model_registry: Any,
    model_dir: str,
    log: Any,
) -> dict[str, Any]:
    """Train a single experiment. Returns result dict."""
    from dataenginex.domains.ml.registry import ModelArtifact

    # Create tracker experiment + run
    exp_id = tracker.create_experiment(name)
    run_id = tracker.start_run(exp_id, run_name=f"{name}-train")

    # Log params
    tracker.log_params(run_id, exp_config.params)

    # For config-driven training, we track the experiment.
    # Real training would load data from a pipeline and use sklearn.
    version = "1.0.0"
    log.info("experiment tracked", experiment_id=exp_id, run_id=run_id)
    tracker.log_metrics(run_id, {"status": 1.0})
    tracker.end_run(run_id, status="FINISHED")

    # Register model artifact
    artifact_path = f"{model_dir}/{name}/model.pkl"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)

    artifact = ModelArtifact(
        name=name,
        version=version,
        artifact_path=artifact_path,
        parameters=exp_config.params,
    )
    with contextlib.suppress(ValueError):
        model_registry.register(artifact)

    return {"score": "tracked", "version": version}
