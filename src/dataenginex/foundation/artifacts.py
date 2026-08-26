"""Artifacts: materialized, content-addressed outputs (§4.8).

The central separation here is logical identity from physical location. A
``Digest`` says what the bytes are; a provider URI says where a copy currently
sits. Conflating them makes artifacts unmovable between providers and makes
"is this the same output?" unanswerable without fetching the data.

Invariant 4 (§4.16): artifacts are never silently overwritten. A new digest
creates a new artifact version.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import BinaryIO

from pydantic import Field

from dataenginex.foundation.ids import (
    ArtifactId,
    AttemptId,
    ProjectId,
    RevisionId,
    new_id,
)
from dataenginex.foundation.projects import FrozenModel, utcnow
from dataenginex.foundation.resources import Classification

__all__ = [
    "Artifact",
    "ArtifactDescriptor",
    "ArtifactReference",
    "Digest",
    "DigestAlgorithm",
    "RetentionState",
    "digest_bytes",
    "digest_stream",
]


class DigestAlgorithm(StrEnum):
    SHA256 = "sha256"


class Digest(FrozenModel):
    """Content address of a byte sequence.

    Rendered as ``sha256:abc123...`` so it stays self-describing when it lands
    in a log line or a lineage export and the algorithm can change later
    without ambiguity.
    """

    algorithm: DigestAlgorithm = DigestAlgorithm.SHA256
    value: str

    def __str__(self) -> str:
        return f"{self.algorithm.value}:{self.value}"


def digest_bytes(data: bytes) -> Digest:
    return Digest(value=hashlib.sha256(data).hexdigest())


def digest_stream(stream: BinaryIO, chunk_size: int = 1 << 20) -> Digest:
    """Digest a stream without loading it into memory.

    Artifacts are routinely larger than RAM (Parquet extracts, model weights),
    so hashing is chunked. Leaves the stream at EOF; callers seek if they need
    to re-read.
    """
    hasher = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        hasher.update(chunk)
    return Digest(value=hasher.hexdigest())


class RetentionState(StrEnum):
    """Where an artifact sits in its retention lifecycle (§8.10)."""

    ACTIVE = "active"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"
    LEGAL_HOLD = "legal_hold"


class ArtifactDescriptor(FrozenModel):
    """What a producer declares *before* bytes are written (§13.6).

    Passed to ``ArtifactStore.put`` alongside the content stream. Carries no
    digest and no location: the store computes the former and chooses the
    latter, which is what keeps producers from inventing their own addresses.
    """

    project_id: ProjectId
    revision_id: RevisionId
    logical_name: str
    media_type: str = "application/octet-stream"
    classification: Classification = Classification.INTERNAL
    produced_by_attempt: AttemptId | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class ArtifactReference(FrozenModel):
    """A resolvable pointer to stored bytes (§13.6).

    ``provider_uri`` is opaque to the core — only the owning provider parses
    it. Keeping the digest alongside lets a reader verify integrity without
    trusting the location.
    """

    artifact_id: ArtifactId
    digest: Digest
    provider: str
    provider_uri: str
    size_bytes: int = Field(ge=0)


class Artifact(FrozenModel):
    """The durable record of a produced output (§4.8).

    Records digest, size, media type, provider URI, producing execution,
    project revision, classification, retention state, and lineage links.
    """

    artifact_id: ArtifactId = Field(default_factory=lambda: ArtifactId(new_id("art")))
    project_id: ProjectId
    revision_id: RevisionId
    logical_name: str
    digest: Digest
    size_bytes: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    provider: str
    provider_uri: str
    produced_by_attempt: AttemptId | None = None
    classification: Classification = Classification.INTERNAL
    retention_state: RetentionState = RetentionState.ACTIVE
    created_at: datetime = Field(default_factory=utcnow)
    labels: dict[str, str] = Field(default_factory=dict)

    @property
    def reference(self) -> ArtifactReference:
        return ArtifactReference(
            artifact_id=self.artifact_id,
            digest=self.digest,
            provider=self.provider,
            provider_uri=self.provider_uri,
            size_bytes=self.size_bytes,
        )

    def same_content_as(self, other: Artifact) -> bool:
        """Whether two artifacts hold identical bytes.

        Used by the commit protocol (§14.3) to turn a duplicate write into a
        no-op instead of a new version.
        """
        return self.digest == other.digest
