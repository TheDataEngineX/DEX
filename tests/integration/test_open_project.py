"""Opening a project directory registers and publishes it (§6.3).

Nothing created installations, workspaces, or projects outside test fixtures, so
``publish`` — which requires the row to exist — could not succeed on a real
install: the control store held no project, no revision, and no resources, and
every gateway read answered empty. These tests hold that path open.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.bootstrap import build_lite_gateway, open_control_store
from dataenginex.bootstrap.settings import Settings
from dataenginex.foundation import PrincipalId
from dataenginex.interfaces import Command, Query
from dataenginex.runtime.state import ControlStore

PRINCIPAL = PrincipalId("prin_test")

MANIFEST = """apiVersion: dex/v1alpha1
kind: Project
metadata:
  name: store-analytics
spec:
  resources:
    - name: orders_csv
      type: csv
      config:
        path: "{csv_path}"
  workloads:
    - name: load
      kind: batch
      operations:
        - name: load-orders
          type: ingest
          inputs: [orders_csv]
          outputs: [orders]
"""


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    opened = open_control_store(Settings(state_dir=tmp_path))
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    csv = root / "orders.csv"
    csv.write_text("order_id,amount\n1,10\n")
    (root / "dex.yaml").write_text(MANIFEST.format(csv_path=csv))
    return root


def _query(project_id: str) -> Query:
    return Query(principal_id=PRINCIPAL, project_id=project_id)


def open_project(store: ControlStore, manifest: Path) -> tuple[str, str | None]:
    """Open *manifest* through the gateway, as a host process does.

    Returns the project id and its active revision id, or ``None`` when the
    manifest did not compile — the shape the assertions below are written
    against, taken from the command result and the project row rather than from
    a second return value the gateway has no reason to carry.
    """
    gateway = build_lite_gateway(store)
    result = gateway.open_project(Command(principal_id=PRINCIPAL), source=str(manifest))
    project_id = str(result.subject_id)
    row = store.query_one(
        "SELECT active_revision_id FROM projects WHERE project_id = ?", (project_id,)
    )
    return project_id, (row["active_revision_id"] if row else None)


def test_opening_a_directory_publishes_a_revision(
    store: ControlStore, project_dir: Path
) -> None:
    project_id, revision = open_project(store, project_dir)

    assert revision is not None
    assert build_lite_gateway(store).get_project(_query(project_id))


def test_the_published_resources_are_queryable(
    store: ControlStore, project_dir: Path
) -> None:
    """The point of publishing: the manifest becomes something the UI can read."""
    project_id, _ = open_project(store, project_dir)

    page = build_lite_gateway(store).list_resources(_query(project_id))

    assert [r.name for r in page.items] == ["orders_csv"]


def test_the_workload_is_runnable(store: ControlStore, project_dir: Path) -> None:
    project_id, _ = open_project(store, project_dir)

    page = build_lite_gateway(store).list_workloads(_query(project_id))

    assert [w.name for w in page.items] == ["load"]


def test_reopening_the_same_directory_finds_the_same_project(
    store: ControlStore, project_dir: Path
) -> None:
    """A restart must not fork a second project that shares the manifest.

    Two projects over one directory would split the run history in half, each
    page showing whichever half its process happened to open.
    """
    first, _ = open_project(store, project_dir)
    second, _ = open_project(store, project_dir)

    assert first == second


def test_republishing_unchanged_content_does_not_fork_history(
    store: ControlStore, project_dir: Path
) -> None:
    _, first = open_project(store, project_dir)
    _, second = open_project(store, project_dir)

    assert first is not None
    assert second is not None
    assert first == second


def test_an_edited_manifest_publishes_a_new_revision(
    store: ControlStore, project_dir: Path
) -> None:
    _, first = open_project(store, project_dir)
    manifest = project_dir / "dex.yaml"
    manifest.write_text(manifest.read_text().replace("outputs: [orders]", "outputs: [orders_v2]"))

    _, second = open_project(store, project_dir)

    assert first is not None
    assert second is not None
    assert first != second


def test_a_broken_manifest_still_registers_the_project(
    store: ControlStore, project_dir: Path
) -> None:
    """Otherwise the user gets an error with no page on which to fix it."""
    (project_dir / "dex.yaml").write_text("apiVersion: dex/v1alpha1\nkind: Project\n")

    project_id, revision = open_project(store, project_dir)

    assert revision is None
    assert build_lite_gateway(store).get_project(_query(project_id))


def test_the_manifest_path_may_be_given_instead_of_the_directory(
    store: ControlStore, project_dir: Path
) -> None:
    """Studio stores a ``dex.yaml`` path, not a directory."""
    project_id, revision = open_project(store, project_dir / "dex.yaml")

    assert revision is not None
    assert project_id
