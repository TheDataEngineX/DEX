"""Plugin manifests, discovery, authorization, and environments (§10)."""

from dataenginex.domains.plugins.discovery import (
    ENTRY_POINT_GROUP,
    EnvironmentSpec,
    InstalledPlugin,
    PluginDiscovery,
    PluginGrant,
    PluginNotAuthorizedError,
    current_environment,
    environment_id,
)
from dataenginex.domains.plugins.manifest import (
    KNOWN_CAPABILITIES,
    PLUGIN_API_VERSION,
    ConnectorDeclaration,
    ManifestError,
    OperationDeclaration,
    PluginManifest,
    PluginMetadata,
    PluginSpec,
    load_manifest,
    validate_manifest,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "KNOWN_CAPABILITIES",
    "PLUGIN_API_VERSION",
    "ConnectorDeclaration",
    "EnvironmentSpec",
    "InstalledPlugin",
    "ManifestError",
    "OperationDeclaration",
    "PluginDiscovery",
    "PluginGrant",
    "PluginManifest",
    "PluginMetadata",
    "PluginNotAuthorizedError",
    "PluginSpec",
    "current_environment",
    "environment_id",
    "load_manifest",
    "validate_manifest",
]
