"""Built-in JSON-backed experiment tracker.

Stores experiments and runs as JSON files in a directory.
Zero-dependency, works out of the box for local development.
For production: use MLflow via ``[mlflow]`` extra.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from dataenginex import _json
from dataenginex.domains.ml.tracking import tracker_registry
from dataenginex.foundation.plugin_contracts import BaseTracker

logger = structlog.get_logger()


@tracker_registry.decorator("builtin", is_default=True)
class BuiltinTracker(BaseTracker):
    """JSON-backed experiment tracker.

    Args:
        storage_dir: Directory for experiment/run JSON files.
    """

    def __init__(self, storage_dir: str = ".dex/tracking", **kwargs: Any) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load existing experiments and runs from disk."""
        exp_file = self._dir / "experiments.json"
        runs_file = self._dir / "runs.json"
        if exp_file.exists():
            self._experiments = _json.loads(exp_file.read_text())
        if runs_file.exists():
            self._runs = _json.loads(runs_file.read_text())

    def _save(self) -> None:
        """Persist experiments and runs to disk."""
        (self._dir / "experiments.json").write_text(
            _json.dumps(self._experiments, indent=2, default=str)
        )
        (self._dir / "runs.json").write_text(_json.dumps(self._runs, indent=2, default=str))

    def create_experiment(self, name: str) -> str:
        """Create an experiment, return its ID."""
        with self._lock:
            for exp_id, exp in self._experiments.items():
                if exp["name"] == name:
                    return exp_id

            exp_id = str(uuid.uuid4())[:8]
            self._experiments[exp_id] = {
                "name": name,
                "created_at": datetime.now(tz=UTC).isoformat(),
            }
            self._save()
            logger.info("experiment created", name=name, experiment_id=exp_id)
            return exp_id

    def start_run(
        self,
        experiment_id: str,
        run_name: str | None = None,
    ) -> str:
        """Start a run within an experiment, return run ID."""
        with self._lock:
            if experiment_id not in self._experiments:
                msg = f"Experiment '{experiment_id}' not found"
                raise KeyError(msg)

            run_id = str(uuid.uuid4())[:8]
            self._runs[run_id] = {
                "experiment_id": experiment_id,
                "name": run_name or f"run-{run_id}",
                "status": "RUNNING",
                "started_at": datetime.now(tz=UTC).isoformat(),
                "ended_at": None,
                "params": {},
                "metrics": {},
            }
            self._save()
            logger.info("run started", run_id=run_id, experiment_id=experiment_id)
            return run_id

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """End a run with given status."""
        with self._lock:
            if run_id not in self._runs:
                msg = f"Run '{run_id}' not found"
                raise KeyError(msg)
            self._runs[run_id]["status"] = status
            self._runs[run_id]["ended_at"] = datetime.now(tz=UTC).isoformat()
            self._save()
            logger.info("run ended", run_id=run_id, status=status)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log parameters for a run."""
        with self._lock:
            if run_id not in self._runs:
                msg = f"Run '{run_id}' not found"
                raise KeyError(msg)
            self._runs[run_id]["params"].update(params)
            self._save()

    def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics for a run at optional step."""
        with self._lock:
            if run_id not in self._runs:
                msg = f"Run '{run_id}' not found"
                raise KeyError(msg)
            run_metrics = self._runs[run_id]["metrics"]
            for key, value in metrics.items():
                if key not in run_metrics:
                    run_metrics[key] = []
                run_metrics[key].append({"value": value, "step": step})
            self._save()

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        """List all runs for an experiment."""
        return [
            {"run_id": rid, **run}
            for rid, run in self._runs.items()
            if run["experiment_id"] == experiment_id
        ]

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments with their IDs and names."""
        return [{"id": exp_id, "name": exp["name"]} for exp_id, exp in self._experiments.items()]
