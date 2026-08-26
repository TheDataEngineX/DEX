"""Plugin discovery, authorization, and environments (§10.4, §10.7, §10.8).

The ordering here is the security property. Discovery reads manifests as data;
validation runs on that data; a project grants capabilities; only then is the
plugin's module imported. The previous implementation imported an entry point to
find out what it was, which means untrusted code ran before anyone decided to
trust it.

Three separate states, deliberately not collapsed:

* **Installed** — present on disk, manifest readable. Inert.
* **Enabled** — manifest validated, installation-level approval given.
* **Granted** — a specific project has authorized specific capabilities.

A plugin can be installed and enabled and still do nothing for a project that
has not granted it anything. That is what makes ``pip install`` safe: it cannot
silently extend what any existing project may do.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from dataenginex.domains.plugins.manifest import (
    ManifestError,
    PluginManifest,
    load_manifest,
    validate_manifest,
)
from dataenginex.foundation import ProjectId, TrustLevel, ValidationReport

__all__ = [
    "ENTRY_POINT_GROUP",
    "EnvironmentSpec",
    "InstalledPlugin",
    "PluginDiscovery",
    "PluginGrant",
    "PluginNotAuthorizedError",
    "environment_id",
]

ENTRY_POINT_GROUP = "dataenginex.plugins"
MANIFEST_FILENAME = "dex-plugin.yaml"


class PluginNotAuthorizedError(PermissionError):
    """A plugin was asked to act beyond what a project granted it."""


@dataclass(frozen=True)
class InstalledPlugin:
    """A plugin found on disk, with its validation verdict.

    Holds the manifest and report but never the imported module — loading is a
    separate, explicit step so that discovery is safe to run over untrusted
    packages.
    """

    manifest: PluginManifest
    report: ValidationReport
    source: str
    entry_point: str = ""

    @property
    def name(self) -> str:
        return self.manifest.metadata.name

    @property
    def usable(self) -> bool:
        """Whether this plugin may be enabled at all."""
        return self.report.ok


@dataclass
class PluginGrant:
    """What one project has authorized one plugin to do (§10.4).

    Capabilities are intersected with the manifest's declaration at grant time,
    so a grant can never exceed what the plugin asked for even if a caller
    passes something wider.
    """

    project_id: ProjectId
    plugin_name: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    destinations: frozenset[str] = field(default_factory=frozenset)

    def permits(self, capability: str) -> bool:
        return capability in self.capabilities


class PluginDiscovery:
    """Finds, validates, and authorizes plugins (§10.4).

    ``search_paths`` are directories containing ``dex-plugin.yaml`` files;
    entry points are also scanned, but only for their *metadata* — an entry
    point is not called until :meth:`load` is invoked for an authorized plugin.
    """

    def __init__(
        self,
        *,
        search_paths: Sequence[Path] = (),
        known_operations: frozenset[str] = frozenset(),
        trusted: dict[str, TrustLevel] | None = None,
    ) -> None:
        self._search_paths = tuple(search_paths)
        self._known_operations = known_operations
        # Trust is assigned by the installation, never self-declared (§10.8).
        self._trusted = trusted or {}
        self._installed: dict[str, InstalledPlugin] = {}
        self._enabled: set[str] = set()
        self._grants: dict[tuple[ProjectId, str], PluginGrant] = {}

    # --- discovery ----------------------------------------------------------

    def discover(self) -> tuple[InstalledPlugin, ...]:
        """Find every plugin manifest without importing anything.

        A manifest that cannot be parsed is skipped rather than fatal — one
        broken plugin must not make the installation unusable.
        """
        found: list[InstalledPlugin] = []

        for directory in self._search_paths:
            for path in sorted(directory.glob(f"*/{MANIFEST_FILENAME}")):
                installed = self._read(path)
                if installed is not None:
                    found.append(installed)

        found.extend(self._discover_entry_points())

        for plugin in found:
            self._installed[plugin.name] = plugin
        return tuple(found)

    def _read(self, path: Path) -> InstalledPlugin | None:
        try:
            manifest = load_manifest(path)
        except ManifestError:
            return None

        # Trust comes from the installation's table, overriding whatever the
        # file claims about itself.
        assigned = self._trusted.get(manifest.metadata.name, TrustLevel.UNTRUSTED)
        manifest = manifest.model_copy(update={"trust_level": assigned})

        return InstalledPlugin(
            manifest=manifest,
            report=validate_manifest(manifest, known_operations=self._known_operations),
            source=str(path),
            entry_point=manifest.spec.entry_point,
        )

    def _discover_entry_points(self) -> list[InstalledPlugin]:
        """Scan entry-point metadata without calling ``load()``.

        ``ep.load()`` imports the module, which is exactly what must not happen
        before validation. Only the declared name and value are read here.
        """
        found: list[InstalledPlugin] = []
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            if entry.name in self._installed:
                continue
            # Without a manifest there is nothing to validate, so the plugin is
            # recorded as unusable rather than trusted by default.
            manifest = PluginManifest.model_validate(
                {
                    "metadata": {"name": entry.name, "version": "0.0.0"},
                    "spec": {"entry_point": entry.value},
                }
            )
            report = validate_manifest(manifest, known_operations=self._known_operations)
            found.append(
                InstalledPlugin(
                    manifest=manifest,
                    report=report,
                    source=f"entry-point:{entry.value}",
                    entry_point=entry.value,
                )
            )
        return found

    # --- enabling and granting ----------------------------------------------

    def enable(self, name: str) -> InstalledPlugin:
        """Approve a plugin at installation level (§10.4).

        Refuses anything whose manifest did not validate. This is the gate the
        old implementation lacked entirely.
        """
        plugin = self._installed.get(name)
        if plugin is None:
            raise KeyError(f"no plugin named {name!r} was discovered")
        if not plugin.usable:
            codes = ", ".join(i.code for i in plugin.report.errors)
            raise PluginNotAuthorizedError(
                f"plugin {name!r} cannot be enabled; its manifest is invalid ({codes})"
            )
        self._enabled.add(name)
        return plugin

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled

    def grant(
        self,
        project_id: ProjectId,
        plugin_name: str,
        capabilities: Sequence[str],
        *,
        destinations: Sequence[str] = (),
    ) -> PluginGrant:
        """Authorize a plugin for one project (§10.4).

        The grant is intersected with the manifest, so asking for more than the
        plugin declared silently narrows rather than escalating.
        """
        if not self.is_enabled(plugin_name):
            raise PluginNotAuthorizedError(
                f"plugin {plugin_name!r} must be enabled before it can be granted capabilities"
            )

        plugin = self._installed[plugin_name]
        declared = frozenset(plugin.manifest.spec.capabilities)
        declared_hosts = frozenset(plugin.manifest.spec.network)

        grant = PluginGrant(
            project_id=project_id,
            plugin_name=plugin_name,
            capabilities=frozenset(capabilities) & declared,
            destinations=frozenset(destinations) & declared_hosts,
        )
        self._grants[(project_id, plugin_name)] = grant
        return grant

    def grant_for(self, project_id: ProjectId, plugin_name: str) -> PluginGrant | None:
        return self._grants.get((project_id, plugin_name))

    def revoke(self, project_id: ProjectId, plugin_name: str) -> None:
        self._grants.pop((project_id, plugin_name), None)

    # --- loading ------------------------------------------------------------

    def load(self, project_id: ProjectId, plugin_name: str) -> Any:
        """Import an authorized plugin's entry point (§10.4).

        The last step, and the only one that executes plugin code. Every
        precondition is re-checked here rather than trusted from the caller:
        this function is the boundary, so it cannot assume the boundary was
        already respected.
        """
        if not self.is_enabled(plugin_name):
            raise PluginNotAuthorizedError(f"plugin {plugin_name!r} is not enabled")

        grant = self.grant_for(project_id, plugin_name)
        if grant is None:
            raise PluginNotAuthorizedError(
                f"project {project_id} has not granted plugin {plugin_name!r} any capabilities"
            )

        plugin = self._installed[plugin_name]
        if not plugin.entry_point:
            raise PluginNotAuthorizedError(f"plugin {plugin_name!r} declares no entry point")

        module_name, _, attribute = plugin.entry_point.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attribute) if attribute else module

    def __iter__(self) -> Iterator[InstalledPlugin]:
        return iter(self._installed.values())

    def __len__(self) -> int:
        return len(self._installed)


@dataclass(frozen=True)
class EnvironmentSpec:
    """A content-addressed project environment (§10.7, ADR-0010).

    Identity is the hash of the interpreter version plus the exact dependency
    set. Two projects requesting the same dependencies share one environment;
    changing a single pin produces a different id and therefore a different
    environment, which is what makes "the environment a run used" a recordable
    fact rather than whatever happened to be installed that day.
    """

    python_version: str
    dependencies: tuple[str, ...]

    @property
    def environment_id(self) -> str:
        return environment_id(self.python_version, self.dependencies)

    def to_json(self) -> str:
        return json.dumps(
            {
                "python_version": self.python_version,
                # Sorted so declaration order cannot change the identity.
                "dependencies": sorted(self.dependencies),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def environment_id(python_version: str, dependencies: Sequence[str]) -> str:
    """Content hash of an environment specification (§10.7)."""
    canonical = json.dumps(
        {"python_version": python_version, "dependencies": sorted(dependencies)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "env-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def current_environment(dependencies: Sequence[str] = ()) -> EnvironmentSpec:
    """The environment this process is running in."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return EnvironmentSpec(python_version=version, dependencies=tuple(dependencies))
