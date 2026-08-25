"""Content-addressed artifact storage and retention (§8.7, §8.10, §14.3).

Two ideas carry this module.

**Identity is the digest, location is incidental.** Bytes are stored under
``<digest[:2]>/<digest>`` so identical content written twice occupies one file
and gets one identity, regardless of the logical name either writer used. That
is what makes "is this the same output?" answerable without a byte comparison,
and it is why the digest is computed on the way in rather than trusted from the
caller.

**Publication is atomic.** Bytes land in a temporary file and are renamed into
place only after the digest is known (§14.3). A reader therefore never observes
a partially written artifact — on POSIX the rename is atomic, so the file either
exists complete or does not exist.

Invariant 4: artifacts are never silently overwritten. Writing different bytes
under an existing logical name creates a new artifact version; writing identical
bytes is a no-op that returns the existing record.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO

from dataenginex.foundation import (
    Artifact,
    ArtifactDescriptor,
    ArtifactId,
    ArtifactReference,
    AttemptId,
    Classification,
    Digest,
    DigestAlgorithm,
    ProjectId,
    RetentionState,
    RevisionId,
    digest_stream,
    new_id,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

__all__ = ["ArtifactError", "FilesystemArtifactStore", "RetentionPolicy", "RetentionService"]

CHUNK_SIZE = 1 << 20


class ArtifactError(RuntimeError):
    """An artifact could not be stored or retrieved."""


class FilesystemArtifactStore:
    """Content-addressed artifacts on local disk (§13.6).

    Implements the ``ArtifactStore`` protocol. The control store holds the
    metadata; this class owns only the bytes, which is what lets an artifact be
    re-homed to object storage later without touching its identity.
    """

    provider = "filesystem"

    def __init__(self, store: ControlStore, root: Path) -> None:
        self.store = store
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # --- writing ------------------------------------------------------------

    def put(self, descriptor: ArtifactDescriptor, content: BinaryIO) -> ArtifactReference:
        """Store bytes and register the artifact.

        The digest is computed while streaming to a temporary file, so content
        larger than memory is handled and the caller cannot claim a digest that
        does not match what was written.
        """
        staged, digest, size = self._stage(content)

        try:
            existing = self._find_by_digest(descriptor.project_id, digest)
            if existing is not None:
                # Identical content already stored: a no-op, not a new version.
                staged.unlink(missing_ok=True)
                return existing.reference

            final = self._path_for(digest)
            final.parent.mkdir(parents=True, exist_ok=True)
            # Atomic publication — a reader sees all of it or none of it.
            os.replace(staged, final)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise

        artifact = Artifact(
            artifact_id=ArtifactId(new_id("art")),
            project_id=descriptor.project_id,
            revision_id=descriptor.revision_id,
            logical_name=descriptor.logical_name,
            digest=digest,
            size_bytes=size,
            media_type=descriptor.media_type,
            provider=self.provider,
            provider_uri=str(final),
            produced_by_attempt=descriptor.produced_by_attempt,
            classification=descriptor.classification,
        )
        self._register(artifact)
        return artifact.reference

    def _stage(self, content: BinaryIO) -> tuple[Path, Digest, int]:
        """Stream to a temp file, returning its path, digest, and size."""
        staging = self.root / ".staging"
        staging.mkdir(parents=True, exist_ok=True)

        # mkstemp rather than NamedTemporaryFile: the file must outlive this
        # block so it can be renamed into place once the digest is known.
        descriptor, name = tempfile.mkstemp(dir=staging)
        path = Path(name)
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while chunk := content.read(CHUNK_SIZE):
                    handle.write(chunk)
                    size += len(chunk)
            with path.open("rb") as written:
                digest = digest_stream(written)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path, digest, size

    def _path_for(self, digest: Digest) -> Path:
        # Two-character fan-out: a flat directory with a million artifacts is
        # slow to list on every filesystem worth supporting.
        return self.root / digest.value[:2] / digest.value

    def _register(self, artifact: Artifact) -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO artifact_records (artifact_id, project_id, revision_id, "
                "logical_name, digest, size_bytes, media_type, provider, provider_uri, "
                "produced_by_attempt, classification, retention_state, created_at, "
                "labels_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact.artifact_id,
                    artifact.project_id,
                    artifact.revision_id,
                    artifact.logical_name,
                    str(artifact.digest),
                    artifact.size_bytes,
                    artifact.media_type,
                    artifact.provider,
                    artifact.provider_uri,
                    artifact.produced_by_attempt,
                    artifact.classification.value,
                    artifact.retention_state.value,
                    artifact.created_at.isoformat(),
                    json.dumps(artifact.labels),
                ),
            )

    # --- reading ------------------------------------------------------------

    def open(self, reference: ArtifactReference) -> BinaryIO:
        """Open stored bytes for reading.

        Verifies the file exists but not its digest — re-hashing on every read
        would make large artifacts unusable. Use :meth:`verify` when integrity
        matters more than latency.
        """
        path = Path(reference.provider_uri)
        if not path.exists():
            raise ArtifactError(f"artifact {reference.artifact_id} is missing at {path}")
        return path.open("rb")

    def verify(self, reference: ArtifactReference) -> bool:
        """Re-hash stored bytes and compare against the recorded digest."""
        path = Path(reference.provider_uri)
        if not path.exists():
            return False
        with path.open("rb") as handle:
            return digest_stream(handle) == reference.digest

    def get(self, artifact_id: ArtifactId) -> Artifact | None:
        row = self.store.query_one(
            "SELECT * FROM artifact_records WHERE artifact_id = ?", (artifact_id,)
        )
        return _row_to_artifact(row) if row is not None else None

    def history(self, project_id: ProjectId, logical_name: str) -> tuple[Artifact, ...]:
        """Every version stored under a logical name, oldest first."""
        rows = self.store.query(
            "SELECT * FROM artifact_records WHERE project_id = ? AND logical_name = ? "
            "ORDER BY created_at",
            (project_id, logical_name),
        )
        return tuple(_row_to_artifact(row) for row in rows)

    def latest(self, project_id: ProjectId, logical_name: str) -> Artifact | None:
        versions = self.history(project_id, logical_name)
        return versions[-1] if versions else None

    def _find_by_digest(self, project_id: ProjectId, digest: Digest) -> Artifact | None:
        row = self.store.query_one(
            "SELECT * FROM artifact_records WHERE project_id = ? AND digest = ?",
            (project_id, str(digest)),
        )
        return _row_to_artifact(row) if row is not None else None


class RetentionPolicy:
    """How long artifacts of a classification are kept (§9.8)."""

    __slots__ = ("default_days", "per_classification")

    def __init__(
        self,
        *,
        default_days: int = 90,
        per_classification: dict[Classification, int] | None = None,
    ) -> None:
        self.default_days = default_days
        self.per_classification = per_classification or {}

    def days_for(self, classification: Classification) -> int:
        return self.per_classification.get(classification, self.default_days)

    def expires_at(self, artifact: Artifact) -> datetime:
        return artifact.created_at + timedelta(days=self.days_for(artifact.classification))


class RetentionService:
    """Retention sweeps and deletion with impact analysis (§8.10).

    Deletion is two-phase: an artifact is marked ``PENDING_DELETION`` first and
    only then has its bytes removed. A single-phase delete that crashes between
    the file removal and the metadata update leaves a record pointing at nothing,
    which reads as corruption rather than as a completed deletion.
    """

    def __init__(
        self,
        store: ControlStore,
        artifacts: FilesystemArtifactStore,
        *,
        policy: RetentionPolicy | None = None,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.policy = policy or RetentionPolicy()

    def expired(self, *, now: datetime | None = None) -> tuple[Artifact, ...]:
        """Active artifacts past their retention window.

        Legal holds are excluded by the state filter: an artifact on hold is not
        eligible no matter how old it is.
        """
        moment = now or utcnow()
        rows = self.store.query(
            "SELECT * FROM artifact_records WHERE retention_state = ?",
            (RetentionState.ACTIVE.value,),
        )
        candidates = tuple(_row_to_artifact(row) for row in rows)
        return tuple(a for a in candidates if self.policy.expires_at(a) <= moment)

    def mark_for_deletion(self, artifact_ids: Sequence[ArtifactId]) -> int:
        """Move artifacts to ``PENDING_DELETION``.

        Refuses to touch anything under legal hold — the guard is in the WHERE
        clause so a concurrent hold placed mid-sweep still wins.
        """
        if not artifact_ids:
            return 0
        with self.store.transaction() as tx:
            cursor = tx.executemany(
                "UPDATE artifact_records SET retention_state = ? "
                "WHERE artifact_id = ? AND retention_state = ?",
                [
                    (RetentionState.PENDING_DELETION.value, aid, RetentionState.ACTIVE.value)
                    for aid in artifact_ids
                ],
            )
            return int(cursor.rowcount)

    def place_legal_hold(self, artifact_id: ArtifactId) -> None:
        """Exempt an artifact from retention sweeps."""
        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE artifact_records SET retention_state = ? WHERE artifact_id = ?",
                (RetentionState.LEGAL_HOLD.value, artifact_id),
            )

    def purge(self, *, dry_run: bool = False) -> tuple[ArtifactId, ...]:
        """Delete bytes for artifacts already marked pending.

        One record owns one path: ``idx_artifacts_location`` makes
        ``(provider, provider_uri)`` unique, and identical content
        deduplicates onto the *existing* artifact rather than creating a second
        row. Removing the file is therefore unambiguous — there is no other
        record that could still be pointing at it.
        """
        rows = self.store.query(
            "SELECT * FROM artifact_records WHERE retention_state = ?",
            (RetentionState.PENDING_DELETION.value,),
        )
        purged: list[ArtifactId] = []

        for row in rows:
            artifact = _row_to_artifact(row)
            if dry_run:
                purged.append(artifact.artifact_id)
                continue

            Path(artifact.provider_uri).unlink(missing_ok=True)
            with self.store.transaction() as tx:
                tx.execute(
                    "UPDATE artifact_records SET retention_state = ? WHERE artifact_id = ?",
                    (RetentionState.DELETED.value, artifact.artifact_id),
                )
            purged.append(artifact.artifact_id)

        return tuple(purged)

    def usage_bytes(self, project_id: ProjectId | None = None) -> int:
        """Total bytes held by non-deleted artifacts."""
        if project_id is None:
            row = self.store.query_one(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM artifact_records "
                "WHERE retention_state != ?",
                (RetentionState.DELETED.value,),
            )
        else:
            row = self.store.query_one(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM artifact_records "
                "WHERE retention_state != ? AND project_id = ?",
                (RetentionState.DELETED.value, project_id),
            )
        return int(row["total"]) if row else 0


def _row_to_artifact(row: sqlite3.Row) -> Artifact:
    algorithm, _, value = str(row["digest"]).partition(":")
    return Artifact(
        artifact_id=ArtifactId(row["artifact_id"]),
        project_id=ProjectId(row["project_id"]),
        revision_id=RevisionId(row["revision_id"]),
        logical_name=row["logical_name"],
        digest=Digest(algorithm=DigestAlgorithm(algorithm), value=value),
        size_bytes=int(row["size_bytes"]),
        media_type=row["media_type"],
        provider=row["provider"],
        provider_uri=row["provider_uri"],
        produced_by_attempt=(
            AttemptId(row["produced_by_attempt"]) if row["produced_by_attempt"] else None
        ),
        classification=Classification(row["classification"]),
        retention_state=RetentionState(row["retention_state"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        labels=json.loads(row["labels_json"]),
    )
