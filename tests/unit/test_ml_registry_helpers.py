"""Tests for ML registry helpers: ModelArtifact, ModelRegistry."""

from __future__ import annotations

from dataenginex.domains.ml.registry import VERSION_AUTO, ModelArtifact, ModelRegistry, ModelStage


class TestModelArtifact:
    def test_to_dict(self) -> None:
        a = ModelArtifact(name="m", version="1.0.0", stage=ModelStage.DEVELOPMENT)
        d = a.to_dict()
        assert d["name"] == "m"
        assert d["version"] == "1.0.0"
        assert d["stage"] == "development"
        assert "created_at" in d
        assert d["promoted_at"] is None

    def test_defaults(self) -> None:
        a = ModelArtifact(name="m", version="1.0.0")
        assert a.stage == ModelStage.DEVELOPMENT
        assert a.metrics == {}
        assert a.parameters == {}
        assert a.tags == []


class TestModelRegistry:
    def test_in_memory(self) -> None:
        reg = ModelRegistry()
        a = ModelArtifact(name="m", version="1.0.0")
        reg.register(a)
        assert "m" in reg.list_models()
        assert "1.0.0" in reg.list_versions("m")

    def test_get_roundtrip(self) -> None:
        reg = ModelRegistry()
        a = ModelArtifact(name="m", version="1.0.0", metrics={"acc": 0.9})
        reg.register(a)
        got = reg.get("m", "1.0.0")
        assert got is not None
        assert got.name == "m"
        assert got.metrics == {"acc": 0.9}

    def test_get_missing(self) -> None:
        reg = ModelRegistry()
        assert reg.get("m", "1.0.0") is None

    def test_get_latest(self) -> None:
        reg = ModelRegistry()
        reg.register(ModelArtifact(name="m", version="1.0.0"))
        reg.register(ModelArtifact(name="m", version="1.1.0"))
        latest = reg.get_latest("m")
        assert latest is not None
        assert latest.version == "1.1.0"

    def test_get_production(self) -> None:
        reg = ModelRegistry()
        reg.register(ModelArtifact(name="m", version="1.0.0"))
        reg.promote("m", "1.0.0", ModelStage.PRODUCTION)
        prod = reg.get_production("m")
        assert prod is not None
        assert prod.stage == ModelStage.PRODUCTION

    def test_duplicate_raises(self) -> None:
        reg = ModelRegistry()
        reg.register(ModelArtifact(name="m", version="1.0.0"))
        import pytest

        with pytest.raises(ValueError, match="already registered"):
            reg.register(ModelArtifact(name="m", version="1.0.0"))

    def test_upsert(self) -> None:
        reg = ModelRegistry()
        reg.register(ModelArtifact(name="m", version="1.0.0"))
        reg.register(ModelArtifact(name="m", version="1.0.0"), upsert=True)
        versions = reg.list_versions("m")
        assert len(versions) == 2

    def test_auto_version(self) -> None:
        reg = ModelRegistry()
        reg.register(ModelArtifact(name="m", version="1.0.0"))
        reg.register(ModelArtifact(name="m", version=VERSION_AUTO))
        versions = reg.list_versions("m")
        assert "1.0.1" in versions

    def test_promote_missing(self) -> None:
        reg = ModelRegistry()
        import pytest

        with pytest.raises(ValueError, match="not found"):
            reg.promote("m", "1.0.0", ModelStage.PRODUCTION)

    def test_close(self) -> None:
        reg = ModelRegistry()
        reg.close()
