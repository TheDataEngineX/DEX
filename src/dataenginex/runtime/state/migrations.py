"""Numbered, deterministic control-store migrations (§16.6).

Migrations are plain SQL in a numbered list, applied in order inside one
transaction each, and recorded in ``schema_migrations``. No Alembic: it exists
to diff models against a live database across branches, and this store has one
owner, one embedded engine, and a linear history. A list of statements is the
whole requirement.

The rule that keeps this honest: **migrations are append-only**. An applied
migration is never edited — a mistake is corrected by adding the next number.
Editing one silently diverges every database that already ran it.

Schema areas follow §8.2. Notable shapes:

- ``runs`` and ``attempts`` are separate tables, not one (§4.10). Retry history
  survives because a new attempt inserts a row instead of updating one.
- ``outbox_events`` is written in the *same* transaction as the state change it
  describes (§8.3), closing the dual-write hole where the DB commits but the
  event publication is lost.
- ``audit_events`` is append-only, enforced by triggers (invariant 9).
- ``queue_items`` carries the lease columns inline. A separate lease table would
  need a second write to claim work, and claiming has to be one atomic
  statement (§7.6).
"""

from __future__ import annotations

from typing import Final

__all__ = ["MIGRATIONS", "Migration", "latest_version"]


class Migration:
    """One numbered schema change.

    Not a Pydantic model: migrations are module-level constants read by the
    store, never validated input, and keeping them plain avoids importing the
    domain layer into schema definitions.
    """

    __slots__ = ("name", "statements", "version")

    def __init__(self, version: int, name: str, statements: tuple[str, ...]) -> None:
        self.version = version
        self.name = name
        self.statements = statements


_INITIAL: Final = (
    # --- installations, workspaces, principals (§8.2) ----------------------
    """
    CREATE TABLE installations (
        installation_id TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        settings_json   TEXT NOT NULL DEFAULT '{}',
        trust_roots_json TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE workspaces (
        workspace_id    TEXT PRIMARY KEY,
        installation_id TEXT NOT NULL REFERENCES installations(installation_id),
        name            TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        budgets_json    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX idx_workspaces_installation ON workspaces(installation_id)",
    """
    CREATE TABLE principals (
        principal_id    TEXT PRIMARY KEY,
        principal_type  TEXT NOT NULL,
        name            TEXT NOT NULL,
        display_name    TEXT NOT NULL DEFAULT '',
        trust_level     TEXT NOT NULL DEFAULT 'untrusted',
        delegated_from  TEXT REFERENCES principals(principal_id),
        roles_json      TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL,
        disabled        INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE memberships (
        workspace_id    TEXT NOT NULL REFERENCES workspaces(workspace_id),
        principal_id    TEXT NOT NULL REFERENCES principals(principal_id),
        role            TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        PRIMARY KEY (workspace_id, principal_id)
    )
    """,
    # --- projects and revisions (§4.4-4.5) --------------------------------
    """
    CREATE TABLE projects (
        project_id          TEXT PRIMARY KEY,
        workspace_id        TEXT NOT NULL REFERENCES workspaces(workspace_id),
        name                TEXT NOT NULL,
        description         TEXT NOT NULL DEFAULT '',
        created_at          TEXT NOT NULL,
        active_revision_id  TEXT,
        archived            INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE UNIQUE INDEX idx_projects_workspace_name ON projects(workspace_id, name)",
    """
    CREATE TABLE project_revisions (
        revision_id             TEXT PRIMARY KEY,
        project_id              TEXT NOT NULL REFERENCES projects(project_id),
        parent_revision_id      TEXT REFERENCES project_revisions(revision_id),
        content_hash            TEXT NOT NULL,
        created_by              TEXT NOT NULL,
        created_at              TEXT NOT NULL,
        manifest_schema_version TEXT NOT NULL,
        dependency_lock_hash    TEXT,
        capability_requirements_json TEXT NOT NULL DEFAULT '[]',
        validation_report_json  TEXT NOT NULL DEFAULT '{}',
        status                  TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_revisions_project ON project_revisions(project_id, status)",
    # Content hash is unique per project: republishing identical content should
    # reuse the revision rather than fork history.
    """
    CREATE UNIQUE INDEX idx_revisions_content
        ON project_revisions(project_id, content_hash)
    """,
    """
    CREATE TABLE revision_files (
        revision_id TEXT NOT NULL REFERENCES project_revisions(revision_id),
        path        TEXT NOT NULL,
        digest      TEXT NOT NULL,
        size_bytes  INTEGER NOT NULL,
        media_type  TEXT NOT NULL,
        PRIMARY KEY (revision_id, path)
    )
    """,
    # --- resources (§4.6) --------------------------------------------------
    """
    CREATE TABLE resources (
        resource_id     TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        revision_id     TEXT NOT NULL,
        resource_type   TEXT NOT NULL,
        name            TEXT NOT NULL,
        description     TEXT NOT NULL DEFAULT '',
        labels_json     TEXT NOT NULL DEFAULT '{}',
        owner           TEXT,
        classification  TEXT NOT NULL,
        sensitivity_json TEXT NOT NULL DEFAULT '{}',
        lifecycle_state TEXT NOT NULL,
        version         TEXT,
        snapshot_ref    TEXT,
        created_at      TEXT NOT NULL,
        facets_json     TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX idx_resources_project_type ON resources(project_id, resource_type)",
    # Unique per revision, not per project: a resource is "the users table as
    # declared by revision X" (§4.6), so every revision redeclares the whole
    # set. Project-wide uniqueness would make publishing a second revision fail
    # on its own unchanged resources.
    "CREATE UNIQUE INDEX idx_resources_revision_name ON resources(project_id, revision_id, name)",
    """
    CREATE TABLE resource_versions (
        resource_id  TEXT NOT NULL REFERENCES resources(resource_id),
        version      TEXT NOT NULL,
        digest       TEXT,
        created_at   TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (resource_id, version)
    )
    """,
    # Invariant 6: cross-project access requires an explicit row here.
    """
    CREATE TABLE resource_grants (
        grant_id            TEXT PRIMARY KEY,
        resource_id         TEXT NOT NULL REFERENCES resources(resource_id),
        grantee_project_id  TEXT NOT NULL REFERENCES projects(project_id),
        actions_json        TEXT NOT NULL DEFAULT '[]',
        granted_by          TEXT NOT NULL,
        granted_at          TEXT NOT NULL,
        expires_at          TEXT
    )
    """,
    """
    CREATE UNIQUE INDEX idx_grants_resource_grantee
        ON resource_grants(resource_id, grantee_project_id)
    """,
    # --- workload definitions, schedules, triggers -------------------------
    """
    CREATE TABLE workload_definitions (
        workload_id     TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        revision_id     TEXT NOT NULL REFERENCES project_revisions(revision_id),
        name            TEXT NOT NULL,
        kind            TEXT NOT NULL,
        definition_json TEXT NOT NULL,
        continuous      INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX idx_workloads_revision_name
        ON workload_definitions(revision_id, name)
    """,
    """
    CREATE TABLE schedules (
        schedule_id     TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        workload_id     TEXT NOT NULL REFERENCES workload_definitions(workload_id),
        cron            TEXT NOT NULL,
        timezone        TEXT NOT NULL DEFAULT 'UTC',
        enabled         INTEGER NOT NULL DEFAULT 1,
        next_fire_at    TEXT,
        last_fired_at   TEXT
    )
    """,
    "CREATE INDEX idx_schedules_next_fire ON schedules(enabled, next_fire_at)",
    """
    CREATE TABLE triggers (
        trigger_id      TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        workload_id     TEXT NOT NULL REFERENCES workload_definitions(workload_id),
        trigger_type    TEXT NOT NULL,
        config_json     TEXT NOT NULL DEFAULT '{}',
        enabled         INTEGER NOT NULL DEFAULT 1
    )
    """,
    # --- runs, attempts, queue (§4.10, §7.6) -------------------------------
    """
    CREATE TABLE runs (
        run_id          TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        revision_id     TEXT NOT NULL REFERENCES project_revisions(revision_id),
        workload_name   TEXT NOT NULL,
        kind            TEXT NOT NULL,
        state           TEXT NOT NULL,
        trigger_type    TEXT NOT NULL,
        requested_by    TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        started_at      TEXT,
        completed_at    TEXT,
        attempt_count   INTEGER NOT NULL DEFAULT 0,
        idempotency_key TEXT,
        error           TEXT
    )
    """,
    "CREATE INDEX idx_runs_project_state ON runs(project_id, state)",
    "CREATE INDEX idx_runs_created ON runs(created_at)",
    # §13.4: a resubmitted command with the same key returns the original run
    # instead of starting a second one.
    """
    CREATE UNIQUE INDEX idx_runs_idempotency
        ON runs(project_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL
    """,
    """
    CREATE TABLE task_runs (
        task_run_id     TEXT PRIMARY KEY,
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        task_name       TEXT NOT NULL,
        state           TEXT NOT NULL,
        depends_on_json TEXT NOT NULL DEFAULT '[]',
        started_at      TEXT,
        completed_at    TEXT,
        error           TEXT
    )
    """,
    "CREATE INDEX idx_task_runs_run ON task_runs(run_id)",
    """
    CREATE TABLE attempts (
        attempt_id          TEXT PRIMARY KEY,
        run_id              TEXT NOT NULL REFERENCES runs(run_id),
        project_id          TEXT NOT NULL,
        revision_id         TEXT NOT NULL,
        attempt_number      INTEGER NOT NULL,
        state               TEXT NOT NULL,
        principal_id        TEXT NOT NULL,
        capability_token_id TEXT,
        worker_id           TEXT,
        environment_id      TEXT,
        planned_resources_json TEXT NOT NULL DEFAULT '{}',
        observed_resources_json TEXT NOT NULL DEFAULT '{}',
        input_artifacts_json  TEXT NOT NULL DEFAULT '[]',
        output_artifacts_json TEXT NOT NULL DEFAULT '[]',
        policy_decisions_json TEXT NOT NULL DEFAULT '[]',
        started_at          TEXT,
        last_heartbeat_at   TEXT,
        completed_at        TEXT,
        error               TEXT,
        error_class         TEXT,
        checkpoint_ref      TEXT,
        trace_id            TEXT,
        commit_token        TEXT
    )
    """,
    "CREATE UNIQUE INDEX idx_attempts_run_number ON attempts(run_id, attempt_number)",
    "CREATE INDEX idx_attempts_state ON attempts(state)",
    # Lease columns live here so a worker claims work in one UPDATE (§7.6).
    """
    CREATE TABLE queue_items (
        queue_item_id   TEXT PRIMARY KEY,
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        attempt_id      TEXT REFERENCES attempts(attempt_id),
        project_id      TEXT NOT NULL,
        revision_id     TEXT NOT NULL,
        workload_kind   TEXT NOT NULL,
        priority        INTEGER NOT NULL DEFAULT 100,
        resource_request_json TEXT NOT NULL DEFAULT '{}',
        not_before      TEXT,
        retry_count     INTEGER NOT NULL DEFAULT 0,
        idempotency_key TEXT,
        state           TEXT NOT NULL,
        leased_by       TEXT,
        lease_expires_at TEXT,
        created_at      TEXT NOT NULL
    )
    """,
    # The claim query orders by (priority, not_before) over ready rows.
    """
    CREATE INDEX idx_queue_ready
        ON queue_items(state, priority, not_before)
    """,
    "CREATE INDEX idx_queue_lease_expiry ON queue_items(lease_expires_at)",
    # --- workers and leases -------------------------------------------------
    """
    CREATE TABLE workers (
        worker_id       TEXT PRIMARY KEY,
        pool            TEXT NOT NULL,
        hostname        TEXT NOT NULL DEFAULT '',
        pid             INTEGER,
        state           TEXT NOT NULL,
        started_at      TEXT NOT NULL,
        last_heartbeat_at TEXT
    )
    """,
    "CREATE INDEX idx_workers_pool_state ON workers(pool, state)",
    """
    CREATE TABLE worker_capabilities (
        worker_id   TEXT NOT NULL REFERENCES workers(worker_id),
        capability  TEXT NOT NULL,
        PRIMARY KEY (worker_id, capability)
    )
    """,
    """
    CREATE TABLE leases (
        lease_id        TEXT PRIMARY KEY,
        queue_item_id   TEXT NOT NULL REFERENCES queue_items(queue_item_id),
        attempt_id      TEXT NOT NULL REFERENCES attempts(attempt_id),
        worker_id       TEXT NOT NULL REFERENCES workers(worker_id),
        acquired_at     TEXT NOT NULL,
        expires_at      TEXT NOT NULL,
        released_at     TEXT,
        lost            INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX idx_leases_expiry ON leases(released_at, expires_at)",
    """
    CREATE TABLE heartbeats (
        attempt_id  TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
        worker_id   TEXT NOT NULL,
        beat_at     TEXT NOT NULL,
        progress_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    # --- policy, decisions, approvals (§4.12-4.13) -------------------------
    """
    CREATE TABLE policies (
        policy_id       TEXT PRIMARY KEY,
        project_id      TEXT REFERENCES projects(project_id),
        name            TEXT NOT NULL,
        version         TEXT NOT NULL,
        effect          TEXT NOT NULL,
        definition_json TEXT NOT NULL,
        priority        INTEGER NOT NULL DEFAULT 100,
        created_at      TEXT NOT NULL,
        enabled         INTEGER NOT NULL DEFAULT 1
    )
    """,
    "CREATE INDEX idx_policies_project ON policies(project_id, enabled)",
    """
    CREATE TABLE policy_decisions (
        decision_id             TEXT PRIMARY KEY,
        policy_set_version      TEXT NOT NULL,
        input_context_digest    TEXT NOT NULL,
        effect                  TEXT NOT NULL,
        obligations_json        TEXT NOT NULL DEFAULT '[]',
        matched_policies_json   TEXT NOT NULL DEFAULT '[]',
        reason                  TEXT NOT NULL DEFAULT '',
        evaluated_by            TEXT NOT NULL,
        evaluated_at            TEXT NOT NULL,
        project_id              TEXT,
        principal_id            TEXT,
        action                  TEXT
    )
    """,
    "CREATE INDEX idx_decisions_project ON policy_decisions(project_id, evaluated_at)",
    """
    CREATE TABLE approvals (
        approval_id     TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        requested_by    TEXT NOT NULL,
        action_summary  TEXT NOT NULL,
        operation_digest TEXT NOT NULL,
        affected_resources_json TEXT NOT NULL DEFAULT '[]',
        destination_json TEXT,
        risk_level      INTEGER NOT NULL,
        state           TEXT NOT NULL,
        approver_id     TEXT,
        requested_at    TEXT NOT NULL,
        decided_at      TEXT,
        expires_at      TEXT,
        reason          TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX idx_approvals_state ON approvals(project_id, state)",
    # --- artifacts, checkpoints, environments (§4.8, §8.7, §10.7) ----------
    """
    CREATE TABLE artifact_records (
        artifact_id     TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        revision_id     TEXT NOT NULL,
        logical_name    TEXT NOT NULL,
        digest          TEXT NOT NULL,
        size_bytes      INTEGER NOT NULL,
        media_type      TEXT NOT NULL,
        provider        TEXT NOT NULL,
        provider_uri    TEXT NOT NULL,
        produced_by_attempt TEXT REFERENCES attempts(attempt_id),
        classification  TEXT NOT NULL,
        retention_state TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        labels_json     TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX idx_artifacts_project_name
        ON artifact_records(project_id, logical_name)
    """,
    "CREATE INDEX idx_artifacts_digest ON artifact_records(digest)",
    # Invariant 4: one physical location holds exactly one digest. A second
    # write to the same URI with different bytes violates the constraint rather
    # than silently clobbering.
    """
    CREATE UNIQUE INDEX idx_artifacts_location
        ON artifact_records(provider, provider_uri)
    """,
    """
    CREATE TABLE checkpoint_records (
        checkpoint_id   TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        stream_name     TEXT NOT NULL,
        cursor_json     TEXT NOT NULL,
        created_at      TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX idx_checkpoints_stream
        ON checkpoint_records(project_id, stream_name)
    """,
    """
    CREATE TABLE environment_records (
        environment_id  TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        content_hash    TEXT NOT NULL,
        python_version  TEXT NOT NULL,
        lock_digest     TEXT NOT NULL,
        path            TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        state           TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX idx_environments_hash
        ON environment_records(content_hash)
    """,
    # --- events (§4.14, §8.3) ----------------------------------------------
    """
    CREATE TABLE metadata_events (
        event_id        TEXT PRIMARY KEY,
        occurred_at     TEXT NOT NULL,
        producer        TEXT NOT NULL,
        installation_id TEXT,
        workspace_id    TEXT,
        project_id      TEXT,
        revision_id     TEXT,
        principal_id    TEXT,
        correlation_id  TEXT,
        schema_version  TEXT NOT NULL,
        event_type      TEXT NOT NULL,
        subject_id      TEXT,
        subject_type    TEXT,
        payload_json    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX idx_metadata_events_project
        ON metadata_events(project_id, occurred_at)
    """,
    "CREATE INDEX idx_metadata_events_subject ON metadata_events(subject_id)",
    "CREATE INDEX idx_metadata_events_correlation ON metadata_events(correlation_id)",
    # Invariant 9: append-only.
    """
    CREATE TABLE audit_events (
        event_id        TEXT PRIMARY KEY,
        occurred_at     TEXT NOT NULL,
        producer        TEXT NOT NULL,
        installation_id TEXT,
        workspace_id    TEXT,
        project_id      TEXT,
        revision_id     TEXT,
        principal_id    TEXT,
        correlation_id  TEXT,
        schema_version  TEXT NOT NULL,
        event_type      TEXT NOT NULL,
        action          TEXT NOT NULL,
        outcome         TEXT NOT NULL,
        target_id       TEXT,
        target_type     TEXT,
        destination     TEXT,
        policy_decision_id TEXT,
        detail_json     TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX idx_audit_events_project ON audit_events(project_id, occurred_at)",
    """
    CREATE INDEX idx_audit_events_principal
        ON audit_events(principal_id, occurred_at)
    """,
    # Defence in depth for invariant 9: even a direct sqlite3 session cannot
    # rewrite history through ordinary statements.
    """
    CREATE TRIGGER audit_events_no_update
    BEFORE UPDATE ON audit_events
    BEGIN
        SELECT RAISE(ABORT, 'audit_events is append-only');
    END
    """,
    """
    CREATE TRIGGER audit_events_no_delete
    BEFORE DELETE ON audit_events
    BEGIN
        SELECT RAISE(ABORT, 'audit_events is append-only');
    END
    """,
    # The outbox: written in the same transaction as the state change (§8.3).
    """
    CREATE TABLE outbox_events (
        outbox_id       TEXT PRIMARY KEY,
        event_kind      TEXT NOT NULL,
        event_type      TEXT NOT NULL,
        payload_json    TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        dispatched_at   TEXT,
        attempts        INTEGER NOT NULL DEFAULT 0,
        last_error      TEXT
    )
    """,
    # The dispatcher scans undispatched rows in creation order.
    """
    CREATE INDEX idx_outbox_pending
        ON outbox_events(dispatched_at, created_at)
    """,
    # --- lineage projection (§8.5) -----------------------------------------
    """
    CREATE TABLE lineage_edges (
        edge_id     TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL,
        source_type TEXT NOT NULL,
        target_id   TEXT NOT NULL,
        target_type TEXT NOT NULL,
        relation    TEXT NOT NULL,
        project_id  TEXT NOT NULL,
        revision_id TEXT,
        run_id      TEXT,
        created_at  TEXT NOT NULL,
        attributes_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX idx_lineage_source ON lineage_edges(source_id, relation)",
    "CREATE INDEX idx_lineage_target ON lineage_edges(target_id, relation)",
    "CREATE INDEX idx_lineage_project ON lineage_edges(project_id)",
)


_STREAMS: Final = (
    # A record that could not be processed after its retries ran out (§14.8).
    #
    # Kept rather than dropped, and kept out of the main path rather than
    # retried forever: one poison record that fails deterministically blocks a
    # stream indefinitely if the cursor cannot advance past it, and silently
    # discarding it loses data nobody knows is missing. Parking it here does
    # neither.
    """
    CREATE TABLE stream_dead_letters (
        dead_letter_id  TEXT PRIMARY KEY,
        project_id      TEXT NOT NULL REFERENCES projects(project_id),
        stream_name     TEXT NOT NULL,
        run_id          TEXT,
        -- Where in the source this record sat, so it can be replayed once
        -- whatever broke it is fixed.
        cursor_json     TEXT NOT NULL,
        payload_json    TEXT NOT NULL,
        error           TEXT NOT NULL,
        attempts        INTEGER NOT NULL,
        -- Both times (§8.8). Event time is when the thing happened; created_at
        -- is when we gave up on it, and the two are not the same fact.
        event_time      TEXT,
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_dead_letters_stream ON stream_dead_letters(project_id, stream_name)",
)


_INTERACTIVE: Final[tuple[str, ...]] = (
    # An interactive run carries its own operations (§7.3). A SQL preview or a
    # schema inspection is composed when the user asks for it, so there is no
    # declared workload to look up — but the run still pins a revision, so it
    # still cannot see anything the project did not declare.
    #
    # Held on the run rather than written into workload_definitions: that table
    # is the revision's declared workload set, which Studio lists and schedules
    # against. Injecting a row per SQL preview would put throwaway work in the
    # user's pipeline list.
    "ALTER TABLE runs ADD COLUMN ad_hoc_plan_json TEXT",
    # Results of interactive runs. §7.3: "Results may be ephemeral until
    # explicitly saved" — so they land here, with an expiry, rather than in the
    # artifact store where content-addressed permanence is the whole point.
    #
    # Persisted rather than returned in-process because §13.8 requires that
    # "persisted state remains queryable through ordinary APIs": the browser
    # that asked may be gone, and another tab may be the one that asks for it.
    """
    CREATE TABLE interactive_results (
        run_id       TEXT PRIMARY KEY REFERENCES runs(run_id),
        project_id   TEXT NOT NULL REFERENCES projects(project_id),
        -- Rows, columns, and whatever else the operation produced, as JSON.
        -- Capped by the handler; this table is not a place to park a dataset.
        payload_json TEXT NOT NULL,
        row_count    INTEGER NOT NULL DEFAULT 0,
        truncated    INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL,
        -- After this, the result is garbage. Reads filter on it rather than
        -- trusting a sweeper to have run: an expired preview must not be
        -- served just because nothing has deleted it yet.
        expires_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_interactive_results_expiry ON interactive_results(expires_at)",
)

_WORKFLOW_STEPS = (
    """
    CREATE TABLE IF NOT EXISTS workflow_steps (
        run_id        TEXT NOT NULL,
        step_name     TEXT NOT NULL,
        step_type     TEXT NOT NULL,
        state         TEXT NOT NULL,
        error         TEXT,
        started_at    TEXT,
        completed_at  TEXT,
        PRIMARY KEY (run_id, step_name)
    )
    """,
)


MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(1, "initial_control_schema", _INITIAL),
    Migration(2, "stream_dead_letters", _STREAMS),
    Migration(3, "interactive_workloads", _INTERACTIVE),
    Migration(4, "workflow_steps", _WORKFLOW_STEPS),
)


def latest_version() -> int:
    """Highest migration version defined."""
    return max(m.version for m in MIGRATIONS)
