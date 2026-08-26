"""Plugin manifests (§10.3).

A plugin declares what it is and what it needs *before* any of its code runs.
That ordering is the entire point: the existing registry imports and
instantiates an entry point to find out what it does, which means a malicious or
merely broken plugin executes before anyone decided to trust it.

Here the manifest is a data file, validated as data. Discovery reads it, the
user approves the capabilities it asks for, and only then is the module
imported (§10.4).

Two rules that look like bureaucracy and are not:

* **Installation is not authorization.** A plugin present on disk is inert until
  a project grants it capabilities. Pip-installing a package must not silently
  extend what a project may do.
* **Declared capabilities are a ceiling, not a wish.** The runtime issues
  capability tokens bounded by this declaration, so a plugin that asks for
  ``data.batch`` cannot later reach the network because it never declared
  ``ai.remote``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, field_validator

from dataenginex.foundation import (
    Determinism,
    FrozenModel,
    IdempotencyStrategy,
    IOContract,
    Operation,
    ResourceRequest,
    RiskLevel,
    SideEffectClass,
    TrustLevel,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    "PLUGIN_API_VERSION",
    "ConnectorDeclaration",
    "ManifestError",
    "OperationDeclaration",
    "PluginManifest",
    "PluginMetadata",
    "PluginSpec",
    "load_manifest",
    "validate_manifest",
]

PLUGIN_API_VERSION = "dex/v1alpha1"

# Lowercase, hyphen-separated. Restrictive because a plugin name reaches log
# lines, file paths, and entry-point keys.
_NAME = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

# Capabilities the core knows how to grant. A plugin asking for anything else is
# rejected rather than granted something approximate.
KNOWN_CAPABILITIES = frozenset(
    {
        "data.batch",
        "data.stream",
        "analytics.sql",
        "scheduling.local",
        "ml.training",
        "ml.serving",
        "ai.local",
        "ai.remote",
        "governance.policy",
    }
)


class ManifestError(ValueError):
    """A plugin manifest could not be read or is structurally invalid."""


class PluginMetadata(FrozenModel):
    """Identity of a plugin (§10.3)."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError(
                f"plugin name {value!r} must be lowercase alphanumeric with hyphens"
            )
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if not _SEMVER.match(value):
            raise ValueError(f"plugin version {value!r} must be semantic (e.g. 1.2.3)")
        return value


class OperationDeclaration(FrozenModel):
    """An operation a plugin provides (§10.6).

    Mirrors the fields of a core operation because the compiler and runtime must
    treat plugin operations identically — a plugin operation that skipped the
    side-effect or risk declaration would be invisible to policy.
    """

    type: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    side_effect: SideEffectClass = SideEffectClass.READ
    determinism: Determinism = Determinism.DETERMINISTIC
    idempotency: IdempotencyStrategy = IdempotencyStrategy.NATURAL
    risk_level: RiskLevel = RiskLevel.READ_PROJECT_DATA
    implementation: str = ""
    timeout_seconds: int = Field(default=3600, gt=0)
    memory_mb: int = Field(default=512, gt=0)

    def to_operation(self) -> Operation:
        """Project onto the core's ``Operation`` type."""
        return Operation(
            operation_type=self.type,
            inputs=tuple(IOContract(name=n, resource_type="any") for n in self.inputs),
            outputs=tuple(IOContract(name=n, resource_type="any") for n in self.outputs),
            side_effect_class=self.side_effect,
            determinism=self.determinism,
            idempotency=self.idempotency,
            risk_level=self.risk_level,
            resource_request=ResourceRequest(
                memory_mb=self.memory_mb, timeout_seconds=self.timeout_seconds
            ),
            implementation_ref=self.implementation or None,
        )


class ConnectorDeclaration(FrozenModel):
    """A connector a plugin provides (§10.5)."""

    name: str
    direction: str = "source"
    formats: tuple[str, ...] = ()
    implementation: str = ""
    supports_incremental: bool = False
    supports_schema_discovery: bool = False

    @field_validator("direction")
    @classmethod
    def _check_direction(cls, value: str) -> str:
        allowed = {"source", "sink", "bidirectional"}
        if value not in allowed:
            raise ValueError(f"connector direction must be one of {sorted(allowed)}")
        return value


class PluginSpec(FrozenModel):
    """What the plugin needs and provides (§10.3)."""

    capabilities: tuple[str, ...] = ()
    operations: tuple[OperationDeclaration, ...] = ()
    connectors: tuple[ConnectorDeclaration, ...] = ()
    # Hosts this plugin may reach, subject to the project's own egress policy.
    network: tuple[str, ...] = ()
    # Secret reference names, never values (invariant 5).
    secrets: tuple[str, ...] = ()
    entry_point: str = ""
    python_requires: str = ""
    dependencies: tuple[str, ...] = ()


class PluginManifest(FrozenModel):
    """The full declaration (§10.3)."""

    apiVersion: str = PLUGIN_API_VERSION  # noqa: N815 - matches the on-disk key
    kind: str = "Plugin"
    metadata: PluginMetadata
    spec: PluginSpec = Field(default_factory=PluginSpec)
    # Assigned by the installation, never self-declared: a plugin claiming to be
    # first-party would be trusting its own assertion (§10.8).
    trust_level: TrustLevel = TrustLevel.UNTRUSTED

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        if value != "Plugin":
            raise ValueError(f"expected kind 'Plugin', got {value!r}")
        return value


def load_manifest(path: Path) -> PluginManifest:
    """Read and parse a manifest file.

    ``yaml.safe_load`` rather than ``load``: a plugin manifest is untrusted
    input, and full YAML can construct arbitrary Python objects.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ManifestError(f"could not read plugin manifest at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"plugin manifest at {path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"plugin manifest at {path} must be a mapping")

    try:
        return PluginManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestError(f"plugin manifest at {path} is invalid: {exc}") from exc


def _error(code: str, message: str, location: str) -> ValidationIssue:
    return ValidationIssue(
        severity=ValidationSeverity.ERROR, code=code, message=message, location=location
    )


def _warn(code: str, message: str, location: str) -> ValidationIssue:
    return ValidationIssue(
        severity=ValidationSeverity.WARNING, code=code, message=message, location=location
    )


def _check_envelope(manifest: PluginManifest) -> list[ValidationIssue]:
    """apiVersion, capabilities, entry point, and trust."""
    issues: list[ValidationIssue] = []

    if manifest.apiVersion != PLUGIN_API_VERSION:
        issues.append(
            _error(
                "E_PLUGIN_API_VERSION",
                f"unsupported apiVersion {manifest.apiVersion!r}; "
                f"expected {PLUGIN_API_VERSION!r}",
                "apiVersion",
            )
        )

    unknown = sorted(set(manifest.spec.capabilities) - KNOWN_CAPABILITIES)
    if unknown:
        issues.append(
            _error(
                "E_PLUGIN_UNKNOWN_CAPABILITY",
                f"declares unknown capabilities {unknown}; the installation cannot grant them",
                "spec.capabilities",
            )
        )

    if not manifest.spec.entry_point:
        issues.append(
            _error(
                "E_PLUGIN_NO_ENTRY_POINT",
                "manifest declares no entry point, so nothing can be loaded",
                "spec.entry_point",
            )
        )

    if manifest.spec.network and manifest.trust_level is TrustLevel.UNTRUSTED:
        issues.append(
            _warn(
                "W_PLUGIN_UNTRUSTED_EGRESS",
                f"untrusted plugin requests network access to {list(manifest.spec.network)}; "
                "review before granting",
                "spec.network",
            )
        )

    return issues


def _check_operation(
    declaration: OperationDeclaration,
    manifest: PluginManifest,
    known_operations: frozenset[str],
    seen: set[str],
) -> list[ValidationIssue]:
    """One operation's declaration (§10.6)."""
    location = f"spec.operations.{declaration.type}"
    issues: list[ValidationIssue] = []

    if declaration.type in known_operations:
        issues.append(
            _error(
                "E_PLUGIN_OPERATION_CONFLICT",
                f"operation {declaration.type!r} is already provided by the core; "
                "a plugin may not redefine it",
                location,
            )
        )

    if declaration.type in seen:
        issues.append(
            _error(
                "E_PLUGIN_DUPLICATE_OPERATION",
                f"operation {declaration.type!r} is declared twice",
                location,
            )
        )

    if not declaration.implementation:
        issues.append(
            _error(
                "E_PLUGIN_NO_IMPLEMENTATION",
                f"operation {declaration.type!r} declares no implementation reference",
                location,
            )
        )

    # An operation acting externally with no declared destination cannot be
    # checked against egress policy at compile time (invariant 7).
    acts_externally = declaration.side_effect in (
        SideEffectClass.EXTERNAL_READ,
        SideEffectClass.EXTERNAL_WRITE,
    )
    if acts_externally and not manifest.spec.network:
        issues.append(
            _error(
                "E_PLUGIN_EGRESS_UNDECLARED",
                f"operation {declaration.type!r} acts externally but the plugin "
                "declares no network destinations (§9.7 default deny)",
                location,
            )
        )

    if declaration.risk_level >= RiskLevel.MODIFY_EXTERNAL:
        issues.append(
            _warn(
                "W_PLUGIN_HIGH_RISK",
                f"operation {declaration.type!r} is risk level "
                f"{int(declaration.risk_level)}; runs will require human approval",
                location,
            )
        )

    return issues


def validate_manifest(
    manifest: PluginManifest, *, known_operations: frozenset[str] = frozenset()
) -> ValidationReport:
    """Check a manifest against installation rules (§10.4).

    Returns a report rather than raising, so a user sees every problem at once
    instead of fixing them one run at a time. Errors block enabling the plugin;
    warnings do not.
    """
    issues = _check_envelope(manifest)

    seen: set[str] = set()
    for declaration in manifest.spec.operations:
        issues.extend(_check_operation(declaration, manifest, known_operations, seen))
        seen.add(declaration.type)

    for connector in manifest.spec.connectors:
        if not connector.implementation:
            issues.append(
                _error(
                    "E_PLUGIN_NO_IMPLEMENTATION",
                    f"connector {connector.name!r} declares no implementation reference",
                    f"spec.connectors.{connector.name}",
                )
            )

    return ValidationReport(issues=tuple(issues))


def manifest_from_dict(data: dict[str, Any]) -> PluginManifest:
    """Build a manifest from an already-decoded mapping."""
    try:
        return PluginManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(f"invalid plugin manifest: {exc}") from exc
