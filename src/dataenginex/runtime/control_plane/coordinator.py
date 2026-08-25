"""Control plane coordinator (§7.2).

Orchestrates the full lifecycle of workload runs: admission, policy,
planning, execution, commit, and recovery.
"""

from __future__ import annotations

from typing import Any

from dataenginex.foundation.ids import AttemptId, RunId
from dataenginex.foundation.workloads import AttemptState, RunState, can_transition
from dataenginex.runtime.state import ControlStore

__all__ = ["ControlPlaneCoordinator"]


class ControlPlaneCoordinator:
    """Coordinates workload run lifecycle through the control plane (§7.2).

    This is the central orchestrator that ties together policy evaluation,
    scheduling, execution, and commit. It does NOT duplicate Spark's internal
    task scheduler — it manages DEX-level run admission and worker ownership.
    """

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    def admit_run(self, run_id: RunId) -> RunState:
        """Admit a run into the system: validate revision, check policy, enqueue.

        Returns the resulting state after admission.
        """
        run = self._get_run(run_id)
        if not can_transition(run["state"], RunState.AWAITING_POLICY):
            raise RuntimeError(
                f"Run {run_id} in state {run['state']} cannot be admitted"
            )
        self._update_run_state(run_id, RunState.AWAITING_POLICY)
        return RunState.AWAITING_POLICY

    def complete_policy(self, run_id: RunId, approved: bool) -> RunState:
        """Policy evaluation completed. Advance to planning or fail."""
        if approved:
            self._update_run_state(run_id, RunState.PLANNING)
            return RunState.PLANNING
        else:
            self._update_run_state(run_id, RunState.FAILED)
            return RunState.FAILED

    def enqueue_run(self, run_id: RunId) -> RunState:
        """Planning complete, enqueue for worker pickup."""
        self._update_run_state(run_id, RunState.QUEUED)
        return RunState.QUEUED

    def claim_run(self, run_id: RunId, worker_id: str, attempt_id: AttemptId) -> RunState:
        """A worker claims a queued run."""
        self._update_run_state(run_id, RunState.LEASED)
        self._create_attempt(attempt_id, run_id, worker_id)
        return RunState.LEASED

    def start_execution(self, run_id: RunId) -> RunState:
        """Worker begins actual execution."""
        self._update_run_state(run_id, RunState.RUNNING)
        return RunState.RUNNING

    def commit_outputs(self, run_id: RunId) -> RunState:
        """Outputs validated and committed. Run completes."""
        self._update_run_state(run_id, RunState.COMPLETED)
        return RunState.COMPLETED

    def fail_run(self, run_id: RunId, error: str) -> RunState:
        """Mark a run as failed."""
        self._update_run_state(run_id, RunState.FAILED)
        self.store.query_one(
            "UPDATE runs SET error = ? WHERE run_id = ?",
            (error, run_id),
        )
        return RunState.FAILED

    def cancel_run(self, run_id: RunId) -> RunState:
        """Cancel a run (cooperative, then forceful after timeout)."""
        self._update_run_state(run_id, RunState.CANCELLED)
        return RunState.CANCELLED

    def _get_run(self, run_id: RunId) -> dict[str, Any]:
        row = self.store.query_one(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        )
        if row is None:
            raise RuntimeError(f"Run {run_id} not found")
        return dict(row)

    def _update_run_state(self, run_id: RunId, new_state: RunState) -> None:
        self.store.query_one(
            "UPDATE runs SET state = ? WHERE run_id = ?",
            (new_state.value, run_id),
        )

    def _create_attempt(
        self, attempt_id: AttemptId, run_id: RunId, worker_id: str
    ) -> None:
        run = self._get_run(run_id)
        self.store.query_one(
            "INSERT INTO attempts "
            "(attempt_id, run_id, project_id, revision_id, worker_id, state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                attempt_id, run_id, run["project_id"],
                run["revision_id"], worker_id, AttemptState.PENDING.value,
            ),
        )
