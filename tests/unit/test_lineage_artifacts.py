"""Lineage, artifacts, retention, and standards projections (§8.5-8.10).

The properties under test are the ones that make provenance trustworthy: edges
that actually get written (the old implementation never set a parent, so every
graph was empty), artifacts that cannot be silently overwritten, and deletion
that knows what it would break.
"""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from dataenginex.domains.governance import (
    ArtifactError,
    FilesystemArtifactStore,
    LineageService,
    RetentionPolicy,
    RetentionService,
)
from dataenginex.foundation import (
    ArtifactDescriptor,
    ArtifactId,
    Classification,
    LineageEdge,
    LineageNodeType,
    LineageRelation,
    ProjectId,
    RetentionState,
    RevisionId,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_test")
REVISION = RevisionId("rev_test")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        with s.transaction() as tx:
            tx.execute(
                "INSERT INTO installations (installation_id, name, created_at) VALUES (?,?,?)",
                ("inst_1", "test", utcnow().isoformat()),
            )
            tx.execute(
                "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
                "VALUES (?,?,?,?)",
                ("ws_1", "inst_1", "default", utcnow().isoformat()),
            )
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                "VALUES (?,?,?,?)",
                (PROJECT, "ws_1", "test-project", utcnow().isoformat()),
            )
        yield s


@pytest.fixture
def lineage(store: ControlStore) -> LineageService:
    return LineageService(store)


@pytest.fixture
def artifacts(store: ControlStore, tmp_path: Path) -> FilesystemArtifactStore:
    return FilesystemArtifactStore(store, tmp_path / "artifacts")


def descriptor(name: str = "output.parquet", **kwargs: object) -> ArtifactDescriptor:
    defaults: dict[str, object] = {
        "project_id": PROJECT,
        "revision_id": REVISION,
        "logical_name": name,
    }
    defaults.update(kwargs)
    return ArtifactDescriptor(**defaults)  # type: ignore[arg-type]


# --- lineage recording ------------------------------------------------------


def test_a_run_actually_writes_edges(lineage: LineageService) -> None:
    """The regression that motivates this module: graphs used to be empty."""
    edges = lineage.record_run(
        run_id="run_1",
        project_id=PROJECT,
        revision_id=REVISION,
        consumed=["res_raw"],
        produced=["art_clean"],
    )

    assert edges
    assert lineage.edges_for("run_1")


def test_run_links_output_to_input_directly(lineage: LineageService) -> None:
    """A derived artifact must be traceable without replaying the run."""
    lineage.record_run(
        run_id="run_1",
        project_id=PROJECT,
        revision_id=REVISION,
        consumed=["res_raw"],
        produced=["art_clean"],
    )

    derived = lineage.edges_for("art_clean", relations=[LineageRelation.DERIVED_FROM])

    assert [e.target_id for e in derived] == ["res_raw"]


def test_upstream_traverses_multiple_hops(lineage: LineageService) -> None:
    lineage.record_run(
        run_id="run_1", project_id=PROJECT, revision_id=REVISION,
        consumed=["res_raw"], produced=["art_stage1"],
    )
    lineage.record_run(
        run_id="run_2", project_id=PROJECT, revision_id=REVISION,
        consumed=["art_stage1"], produced=["art_stage2"],
    )

    assert "res_raw" in lineage.upstream("art_stage2")


def test_deletion_impact_finds_derived_outputs(lineage: LineageService) -> None:
    """Deleting a source without this leaves data that outlives the request."""
    lineage.record_run(
        run_id="run_1", project_id=PROJECT, revision_id=REVISION,
        consumed=["res_pii"], produced=["art_model"],
    )
    lineage.record_run(
        run_id="run_2", project_id=PROJECT, revision_id=REVISION,
        consumed=["art_model"], produced=["art_report"],
    )

    impact = lineage.deletion_impact("res_pii")

    assert "art_model" in impact
    assert "art_report" in impact


def test_traversal_terminates_on_a_cycle(lineage: LineageService) -> None:
    """Provenance should be acyclic, but a bug must not hang a sweep."""
    lineage.record(
        [
            LineageEdge(
                source_id="a", source_type=LineageNodeType.ARTIFACT,
                target_id="b", target_type=LineageNodeType.ARTIFACT,
                relation=LineageRelation.DERIVED_FROM, project_id=PROJECT,
            ),
            LineageEdge(
                source_id="b", source_type=LineageNodeType.ARTIFACT,
                target_id="a", target_type=LineageNodeType.ARTIFACT,
                relation=LineageRelation.DERIVED_FROM, project_id=PROJECT,
            ),
        ]
    )

    # Terminates, and reports b exactly once without revisiting the start node.
    assert lineage.upstream("a") == ("b",)


def test_transmission_is_recorded_in_the_graph(lineage: LineageService) -> None:
    """'What data reached this host' is a traversal, not a log search."""
    lineage.record_transmission(
        source_id="art_export",
        destination="api.vendor.com",
        project_id=PROJECT,
        policy_decision_id="dec_1",
    )

    edges = lineage.edges_for("art_export", relations=[LineageRelation.TRANSMITTED_TO])

    assert edges[0].target_id == "api.vendor.com"
    assert edges[0].attributes["policy_decision_id"] == "dec_1"


def test_recording_nothing_is_a_no_op(lineage: LineageService) -> None:
    assert lineage.record([]) == 0


def test_invalid_direction_is_rejected(lineage: LineageService) -> None:
    with pytest.raises(ValueError, match="direction must be"):
        lineage.edges_for("x", direction="sideways")


# --- projections (§8.6) -----------------------------------------------------


def test_openlineage_projection_carries_inputs_and_outputs(
    lineage: LineageService,
) -> None:
    lineage.record_run(
        run_id="run_1", project_id=PROJECT, revision_id=REVISION,
        consumed=["res_raw"], produced=["art_clean"],
    )

    event = lineage.to_openlineage("run_1")

    assert event["run"]["runId"] == "run_1"
    assert {i["name"] for i in event["inputs"]} == {"res_raw"}
    assert {o["name"] for o in event["outputs"]} == {"art_clean"}


def test_openlineage_keeps_dex_facts_in_a_custom_facet(
    lineage: LineageService,
) -> None:
    """A projection that drops the DEX detail is useless for compliance."""
    lineage.record_run(
        run_id="run_1", project_id=PROJECT, revision_id=REVISION, produced=["art_x"]
    )
    lineage.record_transmission(
        source_id="art_x", destination="api.vendor.com", project_id=PROJECT, run_id="run_1"
    )

    facet = lineage.to_openlineage("run_1")["run"]["facets"]["dex"]

    assert facet["projectId"] == PROJECT
    assert facet["revisionId"] == REVISION
    assert facet["transmissions"] == ["api.vendor.com"]


def test_openlineage_for_an_unknown_run_is_empty(lineage: LineageService) -> None:
    assert lineage.to_openlineage("run_missing") == {}


def test_prov_export_separates_entities_activities_and_agents(
    lineage: LineageService,
) -> None:
    """W3C PROV's three-way split is the whole point of the format."""
    lineage.record_run(
        run_id="run_1", project_id=PROJECT, revision_id=REVISION,
        consumed=["res_raw"], produced=["art_clean"],
    )

    doc = lineage.to_prov(PROJECT)

    assert "dex:run_1" in doc["activity"]
    assert "dex:res_raw" in doc["entity"]
    assert "dex:art_clean" in doc["entity"]
    assert doc["prefix"]["prov"] == "http://www.w3.org/ns/prov#"


def test_prov_maps_relations_to_standard_terms(lineage: LineageService) -> None:
    lineage.record_run(
        run_id="run_1", project_id=PROJECT, revision_id=REVISION,
        consumed=["res_raw"], produced=["art_clean"],
    )

    relations = lineage.to_prov(PROJECT)["wasInfluencedBy"]
    kinds = {r["prov:type"] for r in relations.values()}

    assert "used" in kinds
    assert "wasGeneratedBy" in kinds
    assert "wasDerivedFrom" in kinds


# --- artifacts (§8.7, §14.3) ------------------------------------------------


def test_stored_artifact_round_trips(artifacts: FilesystemArtifactStore) -> None:
    reference = artifacts.put(descriptor(), io.BytesIO(b"hello world"))

    with artifacts.open(reference) as handle:
        assert handle.read() == b"hello world"


def test_digest_is_computed_not_trusted(artifacts: FilesystemArtifactStore) -> None:
    """A caller must not be able to claim an address for bytes it did not write."""
    reference = artifacts.put(descriptor(), io.BytesIO(b"hello world"))

    assert artifacts.verify(reference)
    assert str(reference.digest).startswith("sha256:")


def test_identical_content_deduplicates(artifacts: FilesystemArtifactStore) -> None:
    """Same bytes, one identity — regardless of logical name."""
    first = artifacts.put(descriptor("a.bin"), io.BytesIO(b"same"))
    second = artifacts.put(descriptor("b.bin"), io.BytesIO(b"same"))

    assert first.artifact_id == second.artifact_id
    assert first.digest == second.digest


def test_different_content_creates_a_new_version_not_an_overwrite(
    artifacts: FilesystemArtifactStore,
) -> None:
    """Invariant 4: artifacts are never silently overwritten."""
    first = artifacts.put(descriptor("out.bin"), io.BytesIO(b"v1"))
    second = artifacts.put(descriptor("out.bin"), io.BytesIO(b"v2"))

    assert first.artifact_id != second.artifact_id

    versions = artifacts.history(PROJECT, "out.bin")
    assert len(versions) == 2
    # The earlier version is still readable — that is what "not overwritten" means.
    with artifacts.open(first) as handle:
        assert handle.read() == b"v1"


def test_latest_returns_the_newest_version(artifacts: FilesystemArtifactStore) -> None:
    artifacts.put(descriptor("out.bin"), io.BytesIO(b"v1"))
    artifacts.put(descriptor("out.bin"), io.BytesIO(b"v2"))

    latest = artifacts.latest(PROJECT, "out.bin")

    assert latest is not None
    with artifacts.open(latest.reference) as handle:
        assert handle.read() == b"v2"


def test_large_content_is_streamed_not_buffered(
    artifacts: FilesystemArtifactStore,
) -> None:
    """Artifacts routinely exceed RAM; chunking must actually work."""
    payload = b"x" * (3 * 1024 * 1024)

    reference = artifacts.put(descriptor("big.bin"), io.BytesIO(payload))

    assert reference.size_bytes == len(payload)
    assert artifacts.verify(reference)


def test_no_staging_files_are_left_behind(
    artifacts: FilesystemArtifactStore, tmp_path: Path
) -> None:
    """A partial write must not linger where a reader could find it."""
    artifacts.put(descriptor(), io.BytesIO(b"content"))

    staging = tmp_path / "artifacts" / ".staging"
    assert not staging.exists() or not list(staging.iterdir())


def test_failed_write_leaves_nothing_behind(
    artifacts: FilesystemArtifactStore, tmp_path: Path
) -> None:
    """An exception mid-stream must not publish a truncated artifact."""

    class Exploding(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise OSError("disk gone")

    with pytest.raises(OSError, match="disk gone"):
        artifacts.put(descriptor(), Exploding())  # type: ignore[arg-type]

    staging = tmp_path / "artifacts" / ".staging"
    assert not staging.exists() or not list(staging.iterdir())


def test_missing_bytes_raise_rather_than_return_empty(
    artifacts: FilesystemArtifactStore,
) -> None:
    """Silent empty content would look like a legitimately empty artifact."""
    reference = artifacts.put(descriptor(), io.BytesIO(b"content"))
    Path(reference.provider_uri).unlink()

    with pytest.raises(ArtifactError, match="missing"):
        artifacts.open(reference)


def test_verify_detects_corruption(artifacts: FilesystemArtifactStore) -> None:
    reference = artifacts.put(descriptor(), io.BytesIO(b"original"))
    Path(reference.provider_uri).write_bytes(b"tampered")

    assert not artifacts.verify(reference)


def test_logical_name_is_separate_from_location(
    artifacts: FilesystemArtifactStore,
) -> None:
    """Location is incidental; identity is the digest."""
    reference = artifacts.put(descriptor("reports/q1.parquet"), io.BytesIO(b"data"))
    stored = artifacts.get(reference.artifact_id)

    assert stored is not None
    assert stored.logical_name == "reports/q1.parquet"
    assert stored.digest.value in reference.provider_uri


# --- retention (§8.10, §9.8) ------------------------------------------------


@pytest.fixture
def retention(store: ControlStore, artifacts: FilesystemArtifactStore) -> RetentionService:
    return RetentionService(
        store,
        artifacts,
        policy=RetentionPolicy(
            default_days=30,
            per_classification={Classification.RESTRICTED: 1},
        ),
    )


def test_fresh_artifacts_are_not_expired(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    artifacts.put(descriptor(), io.BytesIO(b"data"))

    assert retention.expired() == ()


def test_artifacts_expire_per_classification(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    """A shorter window for restricted data is the point of the policy."""
    artifacts.put(
        descriptor("secret.bin", classification=Classification.RESTRICTED),
        io.BytesIO(b"sensitive"),
    )

    expired = retention.expired(now=utcnow() + timedelta(days=2))

    assert len(expired) == 1
    assert expired[0].logical_name == "secret.bin"


def test_legal_hold_survives_a_retention_sweep(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    """An artifact on hold is not eligible no matter how old."""
    reference = artifacts.put(
        descriptor("held.bin", classification=Classification.RESTRICTED),
        io.BytesIO(b"evidence"),
    )
    retention.place_legal_hold(reference.artifact_id)

    assert retention.expired(now=utcnow() + timedelta(days=999)) == ()


def test_deletion_is_two_phase(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    """Marking first means a crash cannot orphan a record from its bytes."""
    reference = artifacts.put(descriptor(), io.BytesIO(b"data"))

    retention.mark_for_deletion([reference.artifact_id])

    marked = artifacts.get(reference.artifact_id)
    assert marked is not None
    assert marked.retention_state is RetentionState.PENDING_DELETION
    # Bytes still present until purge runs.
    assert Path(reference.provider_uri).exists()

    retention.purge()

    purged = artifacts.get(reference.artifact_id)
    assert purged is not None
    assert purged.retention_state is RetentionState.DELETED
    assert not Path(reference.provider_uri).exists()


def test_one_record_owns_one_path(
    store: ControlStore, artifacts: FilesystemArtifactStore
) -> None:
    """Invariant 4's storage half: no two records share a location.

    This is what makes purge unambiguous. Deduplication returns the *existing*
    artifact rather than adding a second row, so a delete can never orphan
    bytes another record still points at.
    """
    first = artifacts.put(descriptor("a.bin"), io.BytesIO(b"shared"))
    second = artifacts.put(descriptor("b.bin"), io.BytesIO(b"shared"))

    assert first.artifact_id == second.artifact_id

    rows = store.query(
        "SELECT provider_uri FROM artifact_records WHERE provider_uri = ?",
        (first.provider_uri,),
    )
    assert len(rows) == 1

    # And the schema refuses a second record at the same location outright.
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as tx:
        tx.execute(
            "INSERT INTO artifact_records (artifact_id, project_id, revision_id, "
            "logical_name, digest, size_bytes, media_type, provider, provider_uri, "
            "classification, retention_state, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "art_second", PROJECT, REVISION, "b.bin", str(first.digest),
                first.size_bytes, "application/octet-stream", "filesystem",
                first.provider_uri, Classification.INTERNAL.value,
                RetentionState.ACTIVE.value, utcnow().isoformat(),
            ),
        )


def test_dry_run_purge_changes_nothing(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    reference = artifacts.put(descriptor(), io.BytesIO(b"data"))
    retention.mark_for_deletion([reference.artifact_id])

    planned = retention.purge(dry_run=True)

    assert reference.artifact_id in planned
    assert Path(reference.provider_uri).exists()


def test_marking_an_artifact_on_hold_is_refused(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    """The guard is in the WHERE clause so a concurrent hold still wins."""
    reference = artifacts.put(descriptor(), io.BytesIO(b"data"))
    retention.place_legal_hold(reference.artifact_id)

    retention.mark_for_deletion([reference.artifact_id])

    still_held = artifacts.get(reference.artifact_id)
    assert still_held is not None
    assert still_held.retention_state is RetentionState.LEGAL_HOLD


def test_usage_excludes_deleted_artifacts(
    artifacts: FilesystemArtifactStore, retention: RetentionService
) -> None:
    reference = artifacts.put(descriptor(), io.BytesIO(b"0123456789"))
    assert retention.usage_bytes(PROJECT) == 10

    retention.mark_for_deletion([reference.artifact_id])
    retention.purge()

    assert retention.usage_bytes(PROJECT) == 0


def test_marking_nothing_is_a_no_op(retention: RetentionService) -> None:
    assert retention.mark_for_deletion([]) == 0


def test_unknown_artifact_lookup_returns_none(
    artifacts: FilesystemArtifactStore,
) -> None:
    assert artifacts.get(ArtifactId("art_missing")) is None
