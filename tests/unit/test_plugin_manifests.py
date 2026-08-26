"""Plugin manifests, discovery, and authorization (§10).

The property under test throughout is the *ordering*: a manifest is validated as
data before any plugin code is imported, and installation never implies
authorization. The old registry called ``ep.load()`` during discovery, which
executes untrusted code before anyone has decided to trust it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dataenginex.domains.plugins import (
    PLUGIN_API_VERSION,
    EnvironmentSpec,
    ManifestError,
    PluginDiscovery,
    PluginManifest,
    PluginNotAuthorizedError,
    environment_id,
    load_manifest,
    validate_manifest,
)
from dataenginex.foundation import ProjectId, RiskLevel, SideEffectClass, TrustLevel
from dataenginex.foundation import registry as core_operations

PROJECT = ProjectId("proj_test")
CORE_TYPES = frozenset(op.operation_type for op in core_operations)


def manifest_dict(name: str = "acme-connector", **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "apiVersion": PLUGIN_API_VERSION,
        "kind": "Plugin",
        "metadata": {"name": name, "version": "1.0.0"},
        "spec": {
            "capabilities": ["data.batch"],
            "entry_point": "acme_connector:Plugin",
        },
    }
    base.update(overrides)
    return base


def write_manifest(root: Path, name: str = "acme-connector", **overrides: object) -> Path:
    """Write a manifest whose declared name matches its directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dex-plugin.yaml"
    path.write_text(yaml.safe_dump(manifest_dict(name, **overrides)))
    return path


# --- manifest parsing -------------------------------------------------------


def test_valid_manifest_loads(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path))

    assert manifest.metadata.name == "acme-connector"
    assert manifest.spec.capabilities == ("data.batch",)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dex-plugin.yaml"
    path.write_text("{{{not yaml")

    with pytest.raises(ManifestError, match="not valid YAML"):
        load_manifest(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="could not read"):
        load_manifest(tmp_path / "absent.yaml")


def test_yaml_cannot_construct_arbitrary_objects(tmp_path: Path) -> None:
    """safe_load, not load — a manifest is untrusted input."""
    path = tmp_path / "dex-plugin.yaml"
    path.write_text("!!python/object/apply:os.system ['echo pwned']\n")

    with pytest.raises(ManifestError):
        load_manifest(path)


@pytest.mark.parametrize("name", ["Acme", "acme_connector", "a", "9lives", "x" * 70])
def test_invalid_plugin_names_are_rejected(name: str) -> None:
    """The name reaches log lines, paths, and entry-point keys."""
    with pytest.raises(ValueError, match="must be lowercase"):
        PluginManifest.model_validate(manifest_dict(metadata={"name": name, "version": "1.0.0"}))


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "latest"])
def test_non_semver_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="must be semantic"):
        PluginManifest.model_validate(
            manifest_dict(metadata={"name": "acme", "version": version})
        )


def test_wrong_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected kind 'Plugin'"):
        PluginManifest.model_validate(manifest_dict(kind="Project"))


def test_a_plugin_cannot_declare_its_own_trust_level(tmp_path: Path) -> None:
    """§10.8: trust is assigned by the installation, never self-asserted."""
    write_manifest(tmp_path, trust_level="first_party")
    discovery = PluginDiscovery(search_paths=[tmp_path])

    found = discovery.discover()

    assert found[0].manifest.trust_level is TrustLevel.UNTRUSTED


def test_installation_assigns_trust(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    discovery = PluginDiscovery(
        search_paths=[tmp_path], trusted={"acme-connector": TrustLevel.VERIFIED}
    )

    found = discovery.discover()

    assert found[0].manifest.trust_level is TrustLevel.VERIFIED


# --- manifest validation ----------------------------------------------------


def test_unknown_capability_is_an_error() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(spec={"capabilities": ["god.mode"], "entry_point": "x:Y"})
    )

    report = validate_manifest(manifest)

    assert not report.ok
    assert any(i.code == "E_PLUGIN_UNKNOWN_CAPABILITY" for i in report.errors)


def test_wrong_api_version_is_an_error() -> None:
    manifest = PluginManifest.model_validate(manifest_dict(apiVersion="dex/v0"))

    report = validate_manifest(manifest)

    assert any(i.code == "E_PLUGIN_API_VERSION" for i in report.errors)


def test_missing_entry_point_is_an_error() -> None:
    manifest = PluginManifest.model_validate(manifest_dict(spec={"capabilities": []}))

    report = validate_manifest(manifest)

    assert any(i.code == "E_PLUGIN_NO_ENTRY_POINT" for i in report.errors)


def test_a_plugin_cannot_redefine_a_core_operation() -> None:
    """Silently replacing `delete` with a low-risk version is a policy bypass."""
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={
                "entry_point": "x:Y",
                "operations": [
                    {"type": "delete", "implementation": "x:delete", "risk_level": 0}
                ],
            }
        )
    )

    report = validate_manifest(manifest, known_operations=CORE_TYPES)

    assert any(i.code == "E_PLUGIN_OPERATION_CONFLICT" for i in report.errors)


def test_duplicate_operations_are_rejected() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={
                "entry_point": "x:Y",
                "operations": [
                    {"type": "scrape", "implementation": "x:a"},
                    {"type": "scrape", "implementation": "x:b"},
                ],
            }
        )
    )

    report = validate_manifest(manifest, known_operations=CORE_TYPES)

    assert any(i.code == "E_PLUGIN_DUPLICATE_OPERATION" for i in report.errors)


def test_operation_without_an_implementation_is_rejected() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(spec={"entry_point": "x:Y", "operations": [{"type": "scrape"}]})
    )

    report = validate_manifest(manifest)

    assert any(i.code == "E_PLUGIN_NO_IMPLEMENTATION" for i in report.errors)


def test_external_operation_without_declared_network_is_rejected() -> None:
    """Invariant 7: an undeclared destination cannot be policy-checked."""
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={
                "entry_point": "x:Y",
                "operations": [
                    {
                        "type": "scrape",
                        "implementation": "x:scrape",
                        "side_effect": SideEffectClass.EXTERNAL_READ.value,
                    }
                ],
            }
        )
    )

    report = validate_manifest(manifest)

    assert any(i.code == "E_PLUGIN_EGRESS_UNDECLARED" for i in report.errors)


def test_external_operation_with_declared_network_is_accepted() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={
                "entry_point": "x:Y",
                "capabilities": ["data.batch"],
                "network": ["api.acme.com"],
                "operations": [
                    {
                        "type": "scrape",
                        "implementation": "x:scrape",
                        "side_effect": SideEffectClass.EXTERNAL_READ.value,
                    }
                ],
            }
        )
    )

    report = validate_manifest(manifest, known_operations=CORE_TYPES)

    assert report.ok


def test_high_risk_operation_warns_without_blocking() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={
                "entry_point": "x:Y",
                "capabilities": ["data.batch"],
                "operations": [
                    {
                        "type": "purge",
                        "implementation": "x:purge",
                        "risk_level": int(RiskLevel.CONSEQUENTIAL),
                    }
                ],
            }
        )
    )

    report = validate_manifest(manifest, known_operations=CORE_TYPES)

    assert report.ok
    assert any(i.code == "W_PLUGIN_HIGH_RISK" for i in report.warnings)


def test_untrusted_plugin_requesting_egress_warns() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={"entry_point": "x:Y", "capabilities": [], "network": ["api.acme.com"]}
        )
    )

    report = validate_manifest(manifest)

    assert any(i.code == "W_PLUGIN_UNTRUSTED_EGRESS" for i in report.warnings)


def test_declared_operation_projects_onto_the_core_type() -> None:
    manifest = PluginManifest.model_validate(
        manifest_dict(
            spec={
                "entry_point": "x:Y",
                "operations": [
                    {
                        "type": "scrape",
                        "implementation": "x:scrape",
                        "inputs": ["url"],
                        "outputs": ["page"],
                        "memory_mb": 256,
                    }
                ],
            }
        )
    )

    operation = manifest.spec.operations[0].to_operation()

    assert operation.operation_type == "scrape"
    assert [c.name for c in operation.inputs] == ["url"]
    assert operation.resource_request.memory_mb == 256


# --- discovery does not execute code ----------------------------------------


def test_discovery_finds_manifests(tmp_path: Path) -> None:
    write_manifest(tmp_path, "acme-connector")
    write_manifest(tmp_path, "beta-connector")

    found = PluginDiscovery(search_paths=[tmp_path]).discover()

    assert {p.name for p in found} >= {"acme-connector", "beta-connector"}


def test_discovery_does_not_import_plugin_code(tmp_path: Path) -> None:
    """The whole point of §10.4: an entry point pointing at a module that does
    not exist must still discover cleanly, because nothing is imported."""
    write_manifest(tmp_path, spec={
        "capabilities": ["data.batch"],
        "entry_point": "nonexistent_module_xyz:Plugin",
    })

    found = PluginDiscovery(search_paths=[tmp_path]).discover()

    assert found[0].name == "acme-connector"


def test_a_broken_manifest_does_not_break_discovery(tmp_path: Path) -> None:
    """One bad plugin must not make the installation unusable."""
    write_manifest(tmp_path, "good-plugin")
    broken = tmp_path / "bad-plugin"
    broken.mkdir()
    (broken / "dex-plugin.yaml").write_text("{{{")

    found = PluginDiscovery(search_paths=[tmp_path]).discover()

    assert "good-plugin" in {p.name for p in found}


# --- installation is not authorization --------------------------------------


def test_a_discovered_plugin_is_not_enabled(tmp_path: Path) -> None:
    """pip install must not extend what any project may do."""
    write_manifest(tmp_path)
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()

    assert not discovery.is_enabled("acme-connector")


def test_an_invalid_plugin_cannot_be_enabled(tmp_path: Path) -> None:
    write_manifest(tmp_path, spec={"capabilities": ["god.mode"], "entry_point": "x:Y"})
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()

    with pytest.raises(PluginNotAuthorizedError, match="manifest is invalid"):
        discovery.enable("acme-connector")


def test_enabling_an_unknown_plugin_fails(tmp_path: Path) -> None:
    discovery = PluginDiscovery(search_paths=[tmp_path])

    with pytest.raises(KeyError):
        discovery.enable("ghost")


def test_an_enabled_plugin_still_needs_a_project_grant(tmp_path: Path) -> None:
    """Enabled installation-wide is not authorized for a given project."""
    write_manifest(tmp_path)
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()
    discovery.enable("acme-connector")

    with pytest.raises(PluginNotAuthorizedError, match="has not granted"):
        discovery.load(PROJECT, "acme-connector")


def test_a_grant_cannot_exceed_the_manifest(tmp_path: Path) -> None:
    """Asking for more than declared narrows rather than escalating."""
    write_manifest(tmp_path)
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()
    discovery.enable("acme-connector")

    grant = discovery.grant(PROJECT, "acme-connector", ["data.batch", "ai.remote"])

    assert grant.capabilities == frozenset({"data.batch"})
    assert not grant.permits("ai.remote")


def test_destinations_are_also_intersected(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        spec={
            "capabilities": ["data.batch"],
            "entry_point": "x:Y",
            "network": ["api.acme.com"],
        },
    )
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()
    discovery.enable("acme-connector")

    grant = discovery.grant(
        PROJECT, "acme-connector", ["data.batch"], destinations=["api.acme.com", "evil.com"]
    )

    assert grant.destinations == frozenset({"api.acme.com"})


def test_cannot_grant_before_enabling(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()

    with pytest.raises(PluginNotAuthorizedError, match="must be enabled"):
        discovery.grant(PROJECT, "acme-connector", ["data.batch"])


def test_revoking_removes_authorization(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()
    discovery.enable("acme-connector")
    discovery.grant(PROJECT, "acme-connector", ["data.batch"])

    discovery.revoke(PROJECT, "acme-connector")

    assert discovery.grant_for(PROJECT, "acme-connector") is None


def test_a_grant_to_one_project_does_not_authorize_another(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    discovery = PluginDiscovery(search_paths=[tmp_path])
    discovery.discover()
    discovery.enable("acme-connector")
    discovery.grant(PROJECT, "acme-connector", ["data.batch"])

    other = ProjectId("proj_other")

    assert discovery.grant_for(other, "acme-connector") is None
    with pytest.raises(PluginNotAuthorizedError):
        discovery.load(other, "acme-connector")


# --- content-addressed environments (§10.7) ---------------------------------


def test_identical_dependencies_share_an_environment_id() -> None:
    first = EnvironmentSpec(python_version="3.13.1", dependencies=("pandas==2.0", "numpy==1.26"))
    second = EnvironmentSpec(python_version="3.13.1", dependencies=("numpy==1.26", "pandas==2.0"))

    assert first.environment_id == second.environment_id


def test_changing_a_pin_changes_the_environment() -> None:
    """A run's environment must be a recordable fact, not whatever was installed."""
    first = EnvironmentSpec(python_version="3.13.1", dependencies=("pandas==2.0",))
    second = EnvironmentSpec(python_version="3.13.1", dependencies=("pandas==2.1",))

    assert first.environment_id != second.environment_id


def test_changing_the_interpreter_changes_the_environment() -> None:
    first = EnvironmentSpec(python_version="3.13.1", dependencies=())
    second = EnvironmentSpec(python_version="3.12.0", dependencies=())

    assert first.environment_id != second.environment_id


def test_environment_id_is_stable_across_calls() -> None:
    assert environment_id("3.13.1", ["a", "b"]) == environment_id("3.13.1", ["b", "a"])
