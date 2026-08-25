"""Tests for PrivacyGuard wiring: dex.yaml schema + engine integration."""

from __future__ import annotations

from typing import Any

import pytest

from dataenginex.config.schema import (
    DexConfig,
    GuardConfig,
    ProjectConfig,
    SecopsConfig,
)
from dataenginex.secops import (
    MaskingStrategy,
    PIIType,
    PrivacyGuardConfig,
)

# ---------------------------------------------------------------------------
# GuardConfig — Pydantic schema
# ---------------------------------------------------------------------------


class TestGuardConfigSchema:
    def test_defaults(self) -> None:
        cfg = GuardConfig()
        assert cfg.enabled is True
        assert cfg.allow_local is True
        assert cfg.block_on_detect is False
        assert cfg.log_all_outbound is True
        assert cfg.strategies == {}
        assert "ollama" in cfg.local_targets

    def test_round_trip_via_dict(self) -> None:
        raw = {
            "enabled": True,
            "allow_local": False,
            "block_on_detect": True,
            "log_all_outbound": True,
            "strategies": {"email": "hash", "ssn": "redact"},
            "local_targets": ["my_self_hosted"],
        }
        cfg = GuardConfig(**raw)
        assert cfg.allow_local is False
        assert cfg.block_on_detect is True
        assert cfg.strategies == {"email": "hash", "ssn": "redact"}
        assert cfg.local_targets == ["my_self_hosted"]

    def test_secops_config_includes_guard_block(self) -> None:
        secops = SecopsConfig()
        assert hasattr(secops, "guard")
        assert isinstance(secops.guard, GuardConfig)

    def test_dex_config_secops_guard_default(self) -> None:
        cfg = DexConfig(project=ProjectConfig(name="test"))
        assert cfg.secops.guard.enabled is True

    def test_dex_config_secops_guard_overrides_load(self) -> None:
        cfg = DexConfig(
            project=ProjectConfig(name="test"),
            secops=SecopsConfig(
                guard=GuardConfig(
                    enabled=False,
                    strategies={"phone": "partial"},
                ),
            ),
        )
        assert cfg.secops.guard.enabled is False
        assert cfg.secops.guard.strategies == {"phone": "partial"}


# ---------------------------------------------------------------------------
# PrivacyGuardConfig.from_dict — loose dict → typed dataclass
# ---------------------------------------------------------------------------


class TestPrivacyGuardConfigFromDict:
    def test_empty_dict_gives_defaults(self) -> None:
        cfg = PrivacyGuardConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.allow_local is True
        assert cfg.block_on_detect is False
        assert cfg.strategies == {}
        assert "ollama" in cfg.local_targets

    def test_string_strategies_converted_to_enums(self) -> None:
        cfg = PrivacyGuardConfig.from_dict(
            {
                "strategies": {"email": "hash", "ssn": "redact", "phone": "partial"},
            }
        )
        assert cfg.strategies[PIIType.EMAIL] == MaskingStrategy.HASH
        assert cfg.strategies[PIIType.SSN] == MaskingStrategy.REDACT
        assert cfg.strategies[PIIType.PHONE] == MaskingStrategy.PARTIAL

    def test_unknown_pii_type_raises(self) -> None:
        with pytest.raises(ValueError):
            PrivacyGuardConfig.from_dict({"strategies": {"not_a_pii_type": "hash"}})

    def test_unknown_masking_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            PrivacyGuardConfig.from_dict({"strategies": {"email": "obfuscate_lol"}})

    def test_local_targets_list_becomes_frozenset(self) -> None:
        cfg = PrivacyGuardConfig.from_dict({"local_targets": ["a", "b", "c"]})
        assert cfg.local_targets == frozenset({"a", "b", "c"})

    def test_local_targets_none_uses_default(self) -> None:
        cfg = PrivacyGuardConfig.from_dict({"local_targets": None})
        assert "ollama" in cfg.local_targets

    def test_all_flags_round_trip(self) -> None:
        raw = {
            "enabled": False,
            "allow_local": False,
            "block_on_detect": True,
            "log_all_outbound": False,
            "strategies": {"email": "hash"},
            "local_targets": ["x"],
        }
        cfg = PrivacyGuardConfig.from_dict(raw)
        assert cfg.enabled is False
        assert cfg.allow_local is False
        assert cfg.block_on_detect is True
        assert cfg.log_all_outbound is False
        assert cfg.strategies == {PIIType.EMAIL: MaskingStrategy.HASH}
        assert cfg.local_targets == frozenset({"x"})


# ---------------------------------------------------------------------------
# DexEngine — privacy_guard attribute + provider wrapping
# ---------------------------------------------------------------------------


@pytest.fixture
def _minimal_config_path(tmp_path: Any) -> Any:
    """Write a minimal dex.yaml — just enough for DexEngine to start."""
    yaml = tmp_path / "dex.yaml"
    yaml.write_text(
        """
project:
  name: PrivacyGuardWiringTest
  version: 0.1.0
data:
  engine: duckdb
secops:
  guard:
    enabled: true
    allow_local: true
    strategies:
      email: hash
"""
    )
    return yaml


class TestEngineWiring:
    def test_engine_exposes_privacy_guard(self, _minimal_config_path: Any) -> None:
        from dataenginex.engine import DexEngine
        from dataenginex.secops import PrivacyGuard

        engine = DexEngine(_minimal_config_path)
        assert hasattr(engine, "privacy_guard")
        assert isinstance(engine.privacy_guard, PrivacyGuard)

    def test_engine_guard_reflects_dex_yaml(self, _minimal_config_path: Any) -> None:
        from dataenginex.engine import DexEngine

        engine = DexEngine(_minimal_config_path)
        assert engine.privacy_guard.config.enabled is True
        assert engine.privacy_guard.config.strategies[PIIType.EMAIL] == MaskingStrategy.HASH

    def test_engine_providers_are_wrapped(self, _minimal_config_path: Any) -> None:
        from dataenginex.engine import DexEngine
        from dataenginex.providers.model.guarded import GuardedProvider

        engine = DexEngine(_minimal_config_path)
        if not hasattr(engine, "model_router"):
            pytest.skip("model_router not initialized (no providers available)")
        for name, provider in engine.model_router._providers.items():  # noqa: SLF001
            assert isinstance(provider, GuardedProvider), (
                f"provider {name!r} should be guard-wrapped"
            )
            assert provider.target == name

    def test_guarded_ollama_bypasses_local(self, _minimal_config_path: Any) -> None:
        from dataenginex.engine import DexEngine

        engine = DexEngine(_minimal_config_path)
        # Verify ollama is wrapped and would bypass scan on local
        if not hasattr(engine, "model_router"):
            pytest.skip("model_router not initialized")
        ollama = engine.model_router._providers.get("ollama")  # noqa: SLF001
        if ollama is None:
            pytest.skip("ollama provider not present")
        # Process a known-PII prompt with target=ollama via the engine's guard
        result = engine.privacy_guard.process("Email alice@example.com", target="ollama")
        assert result.bypassed_local is True

    def test_engine_with_block_on_detect(self, tmp_path: Any) -> None:
        """Engine configured with block_on_detect routes through correctly."""
        from dataenginex.engine import DexEngine
        from dataenginex.secops import PrivacyBlocked

        yaml = tmp_path / "dex.yaml"
        yaml.write_text(
            """
project:
  name: BlockTest
  version: 0.1.0
secops:
  guard:
    enabled: true
    allow_local: false
    block_on_detect: true
"""
        )
        engine = DexEngine(yaml)
        assert engine.privacy_guard.config.block_on_detect is True

        # Direct guard invocation (no real provider needed) — verifies the
        # blocked path is configured end-to-end from yaml → engine → guard.
        result = engine.privacy_guard.process(
            "Email me at alice@example.com",
            target="anthropic",
        )
        assert result.blocked is True

        # And via GuardedProvider with a fake inner:
        from dataenginex.domains.ai.providers import BaseProvider
        from dataenginex.providers.model.guarded import GuardedProvider

        class _Spy(BaseProvider):
            calls = 0

            def generate(self, prompt: str, **kwargs: Any) -> str:
                _Spy.calls += 1
                return prompt

        wrapped = GuardedProvider(_Spy(), engine.privacy_guard, target="anthropic")
        with pytest.raises(PrivacyBlocked):
            wrapped.generate("SSN 123-45-6789")
        assert _Spy.calls == 0  # blocked before reaching inner
