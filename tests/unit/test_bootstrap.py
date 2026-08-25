"""Assembly (§5.5, §11.3).

Two things are worth asserting about wiring. That it produces something that
actually works — a gateway whose store is migrated and whose queue and
governance are attached — and that the defaults are the safe ones. A bootstrap
that silently assembled a permissive deployment would be worse than none, since
every caller would inherit the mistake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.bootstrap import Settings, build_lite_gateway, lite, open_control_store
from dataenginex.domains.security import GovernanceService
from dataenginex.foundation import (
    AuthorizationRequest,
    NetworkDestination,
    Policy,
    PolicyEffect,
    PrincipalId,
    PrincipalType,
    ProjectId,
    RevisionId,
    RiskLevel,
    utcnow,
)
from dataenginex.interfaces import Command
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_boot")
REVISION = RevisionId("rev_boot")
ALICE = PrincipalId("prin_alice")


def _seed(store: ControlStore) -> None:
    """The installation -> workspace -> project chain a run needs to exist."""
    now = utcnow().isoformat()
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO installations (installation_id, name, created_at) VALUES (?,?,?)",
            ("inst_1", "test", now),
        )
        tx.execute(
            "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
            "VALUES (?,?,?,?)",
            ("ws_1", "inst_1", "default", now),
        )
        tx.execute(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?,?,?,?)",
            (PROJECT, "ws_1", "bootstrap-project", now),
        )
        tx.execute(
            "INSERT INTO project_revisions (revision_id, project_id, content_hash, "
            "created_by, created_at, manifest_schema_version, status) VALUES (?,?,?,?,?,?,?)",
            (REVISION, PROJECT, "sha256:boot", ALICE, now, "dex/v1alpha1", "published"),
        )
        tx.execute(
            "UPDATE projects SET active_revision_id = ? WHERE project_id = ?",
            (REVISION, PROJECT),
        )
        tx.execute(
            "INSERT INTO principals (principal_id, principal_type, name, created_at) "
            "VALUES (?,?,?,?)",
            (ALICE, PrincipalType.HUMAN.value, "alice", now),
        )


class TestSettings:
    def test_the_control_db_lands_under_the_state_dir(self, tmp_path: Path) -> None:
        settings = Settings(state_dir=tmp_path / "state")
        assert settings.control_db_path == tmp_path / "state" / "control.db"

    def test_an_explicit_state_dir_beats_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller knows where the project is; an ambient variable does not.

        Without this precedence a test using a temp dir would write into
        whatever ``DEX_STATE_DIR`` happened to be set to in the shell.
        """
        monkeypatch.setenv("DEX_STATE_DIR", "/somewhere/else")
        assert Settings.from_env(state_dir=tmp_path).state_dir == tmp_path

    def test_the_environment_is_used_when_no_state_dir_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEX_STATE_DIR", str(tmp_path / "from_env"))
        assert Settings.from_env().state_dir == tmp_path / "from_env"


class TestOpenControlStore:
    def test_the_store_is_migrated_on_open(self, tmp_path: Path) -> None:
        """An unmigrated store is unusable, so opening one must not be possible.

        Asserted by querying a table that only exists after migration: if
        ``migrate()`` were left to the caller, this would raise.
        """
        store = open_control_store(Settings(state_dir=tmp_path))
        try:
            assert store.query("SELECT COUNT(*) AS n FROM projects")[0]["n"] == 0
        finally:
            store.close()

    def test_the_database_file_is_created_under_the_state_dir(self, tmp_path: Path) -> None:
        settings = Settings(state_dir=tmp_path / "nested" / "state")
        store = open_control_store(settings)
        try:
            assert settings.control_db_path.exists()
        finally:
            store.close()


class TestLiteGateway:
    def test_it_denies_beyond_a_project_by_default(self, tmp_path: Path) -> None:
        """Default deny is the load-bearing default (§9.7).

        ``DEFAULT_POLICY_SET`` permits a project's own reads, local artifacts,
        and workloads — and nothing above risk level 2. The boundary is what
        this asserts: transmitting externally is level 3 and still needs
        explicit configuration, so widening the set to let a project run itself
        did not quietly open the egress path too.

        A bootstrap that defaulted the other way would hand every caller an open
        deployment.
        """
        store = open_control_store(Settings(state_dir=tmp_path))
        try:
            _seed(store)
            governance = GovernanceService(store)

            decision = governance.authorize(
                AuthorizationRequest(
                    principal_id=ALICE,
                    project_id=PROJECT,
                    revision_id=REVISION,
                    action="notify",
                    risk_level=RiskLevel.TRANSMIT_EXTERNAL,
                    destination=NetworkDestination(host="example.invalid"),
                ),
                auto_request_approval=False,
            )

            assert decision.effect is not PolicyEffect.PERMIT
        finally:
            store.close()

    def test_the_default_policy_set_permits_a_project_its_own_workloads(
        self, tmp_path: Path
    ) -> None:
        """Otherwise a fresh install can open a project and never run it.

        Nothing writes policies, so ``run:<workload>`` matched no rule and every
        run was refused by default deny — on an installation that had just been
        told to run exactly that workload.
        """
        store = open_control_store(Settings(state_dir=tmp_path))
        try:
            _seed(store)
            gateway = build_lite_gateway(store)

            accepted = gateway.start_run(
                Command(principal_id=ALICE, project_id=PROJECT), workload="daily_load"
            )

            assert accepted.subject_id
            assert store.query("SELECT COUNT(*) AS n FROM queue_items")[0]["n"] == 1
        finally:
            store.close()

    def test_a_permitted_run_is_queued(self, tmp_path: Path) -> None:
        """The assembled parts have to work together, not merely exist.

        This exercises governance, the run service, and the queue through one
        call — the wiring is only correct if the run reaches the queue.
        """
        store = open_control_store(Settings(state_dir=tmp_path))
        try:
            _seed(store)
            gateway = build_lite_gateway(
                store,
                policies=[
                    Policy(name="permit-runs", effect=PolicyEffect.PERMIT, actions=("run:*",))
                ],
            )
            result = gateway.start_run(
                Command(principal_id=ALICE, project_id=PROJECT),
                workload="daily_load",
            )
            assert result.subject_id
            assert store.query("SELECT COUNT(*) AS n FROM queue_items")[0]["n"] == 1
        finally:
            store.close()


class TestLiteContextManager:
    def test_it_closes_the_store_on_exit(self, tmp_path: Path) -> None:
        """A leaked connection per thread outlives the caller that made it."""
        with lite(tmp_path) as gateway:
            store = gateway.store
            assert store.query("SELECT COUNT(*) AS n FROM projects")[0]["n"] == 0

        assert getattr(store._local, "conn", None) is None

    def test_it_closes_the_store_even_when_the_body_raises(self, tmp_path: Path) -> None:
        """The cleanup path is the one that matters — it runs on the bad day."""
        with (  # noqa: PT012 - the raise inside the block is the subject
            pytest.raises(RuntimeError, match="boom"),
            lite(tmp_path) as gateway,
        ):
            store = gateway.store
            raise RuntimeError("boom")

        assert getattr(store._local, "conn", None) is None
