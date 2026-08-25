"""Project compiler and revision lifecycle.

The tests that matter most here are the ones proving the compiler *rejects*
things: cycles, undeclared egress, inline secrets, missing capabilities. A
compiler that only works on valid input is the design the superseded engine
already had.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.foundation import (
    PrincipalId,
    ProjectId,
    PublicationStatus,
    RiskLevel,
    SideEffectClass,
)
from dataenginex.runtime.compiler import (
    ProjectManifest,
    PublicationError,
    RevisionService,
    compile_project,
    parse_size,
)
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_test")
PRINCIPAL = PrincipalId("prin_test")
TS = "2026-08-03T00:00:00+00:00"

MINIMAL = """
apiVersion: dex/v1alpha1
kind: Project
metadata:
  name: demo
spec:
  limits:
    cpu: 4
    memory: 6GiB
"""


def write_project(root: Path, manifest: str, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dex.yaml").write_text(textwrap.dedent(manifest).lstrip())
    for name, content in (files or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip())
    return root


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        with s.transaction() as tx:
            tx.execute(
                "INSERT INTO installations (installation_id, name, created_at) "
                "VALUES ('inst_1', 'test', ?)",
                (TS,),
            )
            tx.execute(
                "INSERT INTO workspaces (workspace_id, installation_id, name, "
                "created_at) VALUES ('ws_1', 'inst_1', 'default', ?)",
                (TS,),
            )
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                "VALUES (?, 'ws_1', 'demo', ?)",
                (PROJECT, TS),
            )
        yield s


# --- size parsing -----------------------------------------------------------


def test_parse_size_distinguishes_binary_and_decimal() -> None:
    assert parse_size("1GiB") == 1024**3
    assert parse_size("1GB") == 1000**3
    assert parse_size("512MiB") == 512 * 1024**2


def test_parse_size_rejects_nonsense() -> None:
    for value in ("", "GiB", "12 parsecs", "-4GiB"):
        with pytest.raises(ValueError):
            parse_size(value)


# --- stage 1: schema --------------------------------------------------------


def test_minimal_project_compiles(tmp_path: Path) -> None:
    result = compile_project(write_project(tmp_path / "p", MINIMAL))
    assert result.ok, result.report.errors
    assert result.content_hash.startswith("sha256:")


def test_missing_manifest_is_an_error(tmp_path: Path) -> None:
    result = compile_project(tmp_path / "nothing-here")
    assert not result.ok
    assert result.report.errors[0].code == "E_NO_MANIFEST"


def test_wrong_kind_is_rejected(tmp_path: Path) -> None:
    result = compile_project(
        write_project(tmp_path / "p", "apiVersion: dex/v1alpha1\nkind: Pipeline\n")
    )
    assert not result.ok
    assert result.report.errors[0].code == "E_WRONG_KIND"


def test_unknown_api_version_is_rejected(tmp_path: Path) -> None:
    result = compile_project(
        write_project(
            tmp_path / "p",
            "apiVersion: dex/v99\nkind: Project\nmetadata:\n  name: demo\n",
        )
    )
    assert not result.ok
    assert any("apiVersion" in e.message for e in result.report.errors)


def test_invalid_yaml_is_reported_not_raised(tmp_path: Path) -> None:
    result = compile_project(write_project(tmp_path / "p", "kind: [unclosed\n"))
    assert not result.ok
    assert result.report.errors[0].code == "E_YAML"


# --- §6.7: forbidden configuration content ---------------------------------


def test_inline_secret_in_resource_config_is_rejected(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      resources:
        - name: warehouse
          type: connection
          config:
            token: ghp_abcdefghij0123456789ABCDEFGHIJKLMNOP
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any("inline secret" in e.message for e in result.report.errors)


def test_secret_reference_is_allowed(tmp_path: Path) -> None:
    # The whole point of ${secret:...} is naming a secret without carrying one.
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      resources:
        - name: warehouse
          type: connection
          config:
            token: ${secret:warehouse_token}
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert result.ok, result.report.errors


# --- stage 2: imports -------------------------------------------------------


def test_imports_are_merged(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      imports:
        - resources/*.yaml
    """
    root = write_project(
        tmp_path / "p",
        manifest,
        {
            "resources/data.yaml": """
            resources:
              - name: customers
                type: dataset
            """
        },
    )
    result = compile_project(root)
    assert result.ok, result.report.errors
    assert [r.name for r in result.resources] == ["customers"]


def test_import_escaping_the_project_is_rejected(tmp_path: Path) -> None:
    # A revision must be self-contained; reaching outside breaks portability
    # and content addressing.
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      imports:
        - ../secrets/*.yaml
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert result.report.errors[0].code == "E_IMPORT_ESCAPE"


def test_missing_import_is_a_warning_not_an_error(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      imports:
        - pipelines/*.yaml
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert result.ok
    assert any(i.code == "W_IMPORT_EMPTY" for i in result.report.issues)


# --- stage 4: capabilities --------------------------------------------------


def test_unknown_required_capability_fails(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      capabilities:
        required:
          - quantum.annealing
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert result.report.errors[0].code == "E_CAPABILITY"


def test_unknown_optional_capability_only_warns(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      capabilities:
        optional:
          - quantum.annealing
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert result.ok
    assert any(i.code == "W_CAPABILITY" for i in result.report.issues)


# --- stage 5: static permission analysis (§9.7) -----------------------------


def test_transmitting_operation_without_a_destination_is_rejected(tmp_path: Path) -> None:
    # `notify` always sends. Naming no destination is not "local", it is a
    # transmission nobody wrote down, and it fails at compile time rather than
    # at the moment the data would have left.
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: alert
          operations:
            - type: notify
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any(e.code == "E_EGRESS_UNRESOLVED" for e in result.report.errors)


def test_reading_a_local_file_needs_no_network_policy(tmp_path: Path) -> None:
    """`ingest` is EXTERNAL_READ but a local path transmits nothing (§9.7).

    Requiring an allow list here would make every offline project declare
    network access it never uses, which is how allow lists become decorative.
    """
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      resources:
        - name: orders_csv
          type: dataset
          config:
            path: data/orders.csv
      workloads:
        - name: load
          operations:
            - type: ingest
              outputs: [orders_csv]
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert result.ok, result.report.errors


def test_reading_from_a_remote_host_still_needs_a_policy(tmp_path: Path) -> None:
    """The same operation type against a URL is egress and must be declared."""
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      resources:
        - name: orders_api
          type: dataset
          config:
            url: https://api.example.com/orders
      workloads:
        - name: load
          operations:
            - type: ingest
              outputs: [orders_api]
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any(e.code == "E_EGRESS_UNDECLARED" for e in result.report.errors)


def test_declared_destination_is_permitted(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      network:
        default: deny
        allow:
          - host: hooks.slack.com
            operations: [notify]
            purpose: alerting
      workloads:
        - name: alert
          operations:
            - type: notify
              parameters:
                destination: hooks.slack.com
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert result.ok, result.report.errors
    assert result.declared_destinations == ("hooks.slack.com",)


def test_undeclared_destination_is_denied(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      network:
        default: deny
        allow:
          - host: hooks.slack.com
      workloads:
        - name: leak
          operations:
            - type: export
              parameters:
                destination: attacker.example.com
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any(e.code == "E_EGRESS_DENIED" for e in result.report.errors)


# --- stage 8: workload graph ------------------------------------------------


def test_dependency_cycle_is_detected(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: a
          depends_on: [b]
        - name: b
          depends_on: [a]
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any(e.code == "E_GRAPH_CYCLE" for e in result.report.errors)


def test_missing_dependency_is_detected(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: a
          depends_on: [ghost]
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any(e.code == "E_MISSING_DEPENDENCY" for e in result.report.errors)


def test_execution_order_is_topological(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: report
          depends_on: [clean]
        - name: clean
          depends_on: [ingest]
        - name: ingest
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert result.ok, result.report.errors
    assert result.execution_order == ("ingest", "clean", "report")


def test_execution_order_is_deterministic(tmp_path: Path) -> None:
    # Independent workloads must order identically on every machine, or the
    # content hash stops meaning anything.
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: zebra
        - name: alpha
        - name: mango
    """
    root = write_project(tmp_path / "p", manifest)
    first = compile_project(root)
    second = compile_project(root)
    assert first.execution_order == second.execution_order
    assert first.execution_order == ("alpha", "mango", "zebra")


# --- stage 10: canonical hashing --------------------------------------------


def test_identical_projects_hash_identically(tmp_path: Path) -> None:
    a = compile_project(write_project(tmp_path / "a", MINIMAL))
    b = compile_project(write_project(tmp_path / "b", MINIMAL))
    assert a.content_hash == b.content_hash


def test_key_order_does_not_change_the_hash(tmp_path: Path) -> None:
    reordered = """
    kind: Project
    apiVersion: dex/v1alpha1
    metadata:
      name: demo
    spec:
      limits:
        memory: 6GiB
        cpu: 4
    """
    a = compile_project(write_project(tmp_path / "a", MINIMAL))
    b = compile_project(write_project(tmp_path / "b", reordered))
    assert a.content_hash == b.content_hash


def test_content_change_changes_the_hash(tmp_path: Path) -> None:
    changed = MINIMAL.replace("memory: 6GiB", "memory: 8GiB")
    a = compile_project(write_project(tmp_path / "a", MINIMAL))
    b = compile_project(write_project(tmp_path / "b", changed))
    assert a.content_hash != b.content_hash


# --- stage 11: IR -----------------------------------------------------------


def test_operations_get_declared_side_effects(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: pipeline
          operations:
            - type: transform
            - type: delete
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    operations = {o.operation_type: o for o in result.workloads[0].operations}

    assert operations["transform"].side_effect_class is SideEffectClass.LOCAL_WRITE
    assert operations["transform"].retry_safe

    # Destructive work is never silently replayed and always needs approval.
    assert operations["delete"].side_effect_class is SideEffectClass.DESTRUCTIVE
    assert not operations["delete"].retry_safe
    assert operations["delete"].risk_level is RiskLevel.CONSEQUENTIAL
    assert operations["delete"].requires_approval


def test_unknown_operation_type_is_rejected(tmp_path: Path) -> None:
    manifest = """
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: demo
    spec:
      workloads:
        - name: pipeline
          operations:
            - type: rm_rf
    """
    result = compile_project(write_project(tmp_path / "p", manifest))
    assert not result.ok
    assert any(e.code == "E_UNKNOWN_OPERATION" for e in result.report.errors)


# --- revision lifecycle (§6.3) ---------------------------------------------


def test_draft_then_publish_sets_active_pointer(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    root = write_project(tmp_path / "p", MINIMAL)

    draft, compiled = service.create_draft(PROJECT, root, PRINCIPAL)
    assert draft.status is PublicationStatus.DRAFT
    assert compiled.ok

    published = service.publish(draft.revision_id, PRINCIPAL)
    assert published.status is PublicationStatus.PUBLISHED

    active = service.active(PROJECT)
    assert active is not None
    assert active.revision_id == draft.revision_id


def test_invalid_project_cannot_be_published(store: ControlStore, tmp_path: Path) -> None:
    # Fail closed. The superseded engine ran invalid config anyway.
    service = RevisionService(store)
    root = write_project(
        tmp_path / "p",
        """
        apiVersion: dex/v1alpha1
        kind: Project
        metadata:
          name: demo
        spec:
          workloads:
            - name: a
              depends_on: [b]
            - name: b
              depends_on: [a]
        """,
    )
    draft, compiled = service.create_draft(PROJECT, root, PRINCIPAL)
    assert not compiled.ok

    with pytest.raises(PublicationError, match="validation error"):
        service.publish(draft.revision_id, PRINCIPAL)

    assert service.active(PROJECT) is None


def test_publishing_supersedes_the_previous_revision(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    first, _ = service.create_draft(PROJECT, write_project(tmp_path / "v1", MINIMAL), PRINCIPAL)
    service.publish(first.revision_id, PRINCIPAL)

    second, _ = service.create_draft(
        PROJECT,
        write_project(tmp_path / "v2", MINIMAL.replace("6GiB", "8GiB")),
        PRINCIPAL,
    )
    service.publish(second.revision_id, PRINCIPAL)

    assert service.get(first.revision_id).status is PublicationStatus.SUPERSEDED
    assert service.get(second.revision_id).status is PublicationStatus.PUBLISHED
    active = service.active(PROJECT)
    assert active is not None
    assert active.revision_id == second.revision_id


def test_second_revision_records_its_parent(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    first, _ = service.create_draft(PROJECT, write_project(tmp_path / "v1", MINIMAL), PRINCIPAL)
    service.publish(first.revision_id, PRINCIPAL)

    second, _ = service.create_draft(
        PROJECT,
        write_project(tmp_path / "v2", MINIMAL.replace("6GiB", "8GiB")),
        PRINCIPAL,
    )
    assert second.parent_revision_id == first.revision_id


def test_rollback_repoints_without_mutating_history(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    first, _ = service.create_draft(PROJECT, write_project(tmp_path / "v1", MINIMAL), PRINCIPAL)
    service.publish(first.revision_id, PRINCIPAL)
    second, _ = service.create_draft(
        PROJECT,
        write_project(tmp_path / "v2", MINIMAL.replace("6GiB", "8GiB")),
        PRINCIPAL,
    )
    service.publish(second.revision_id, PRINCIPAL)

    service.rollback(PROJECT, first.revision_id, PRINCIPAL)

    active = service.active(PROJECT)
    assert active is not None
    assert active.revision_id == first.revision_id
    # The rolled-away revision still exists — history is not rewritten.
    assert service.get(second.revision_id).status is PublicationStatus.SUPERSEDED
    assert len(service.history(PROJECT)) == 2


def test_rollback_is_audited(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    first, _ = service.create_draft(PROJECT, write_project(tmp_path / "v1", MINIMAL), PRINCIPAL)
    service.publish(first.revision_id, PRINCIPAL)
    second, _ = service.create_draft(
        PROJECT,
        write_project(tmp_path / "v2", MINIMAL.replace("6GiB", "8GiB")),
        PRINCIPAL,
    )
    service.publish(second.revision_id, PRINCIPAL)
    service.rollback(PROJECT, first.revision_id, PRINCIPAL)

    events = store.query("SELECT action FROM audit_events")
    assert [e["action"] for e in events] == ["project.rollback"]


def test_cannot_publish_twice(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    draft, _ = service.create_draft(PROJECT, write_project(tmp_path / "p", MINIMAL), PRINCIPAL)
    service.publish(draft.revision_id, PRINCIPAL)

    with pytest.raises(PublicationError, match="only drafts"):
        service.publish(draft.revision_id, PRINCIPAL)


def test_cannot_roll_back_to_a_draft(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    published, _ = service.create_draft(PROJECT, write_project(tmp_path / "v1", MINIMAL), PRINCIPAL)
    service.publish(published.revision_id, PRINCIPAL)
    draft, _ = service.create_draft(
        PROJECT,
        write_project(tmp_path / "v2", MINIMAL.replace("6GiB", "8GiB")),
        PRINCIPAL,
    )

    with pytest.raises(PublicationError, match="draft"):
        service.rollback(PROJECT, draft.revision_id, PRINCIPAL)


def test_publish_emits_a_metadata_event(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    draft, _ = service.create_draft(PROJECT, write_project(tmp_path / "p", MINIMAL), PRINCIPAL)
    service.publish(draft.revision_id, PRINCIPAL)

    events = store.query("SELECT event_type FROM metadata_events")
    assert [e["event_type"] for e in events] == ["ProjectRevisionPublished"]
    assert len(store.pending_outbox()) == 1


def test_revision_captures_source_file_digests(store: ControlStore, tmp_path: Path) -> None:
    service = RevisionService(store)
    draft, _ = service.create_draft(PROJECT, write_project(tmp_path / "p", MINIMAL), PRINCIPAL)

    assert [f.path for f in draft.files] == ["dex.yaml"]
    assert draft.files[0].digest.startswith("sha256:")


def test_manifest_roundtrips_through_aliases() -> None:
    manifest = ProjectManifest.model_validate(
        {"apiVersion": "dex/v1alpha1", "kind": "Project", "metadata": {"name": "demo"}}
    )
    assert manifest.api_version == "dex/v1alpha1"
    assert manifest.model_dump(by_alias=True)["apiVersion"] == "dex/v1alpha1"
