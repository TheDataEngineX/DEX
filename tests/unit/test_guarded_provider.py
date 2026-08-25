"""Tests for ai/routing/guarded.py — GuardedProvider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dataenginex.domains.ai.providers import BaseProvider
from dataenginex.providers.model.guarded import GuardedProvider
from dataenginex.secops.guard import PrivacyBlocked, PrivacyGuard, PrivacyGuardConfig


def _mock_provider(response: str = "ok") -> BaseProvider:
    p = MagicMock(spec=BaseProvider)
    p.generate.return_value = response
    return p


def _guard(block: bool = False) -> PrivacyGuard:
    cfg = PrivacyGuardConfig(enabled=True, block_on_detect=block)
    return PrivacyGuard(config=cfg)


class TestGuardedProviderInit:
    def test_inner_accessible(self) -> None:
        inner = _mock_provider()
        gp = GuardedProvider(inner, _guard(), target="openai")
        assert gp.inner is inner

    def test_target_stored(self) -> None:
        gp = GuardedProvider(_mock_provider(), _guard(), target="anthropic")
        assert gp.target == "anthropic"

    def test_target_derived_from_class(self) -> None:
        class MyCustomProvider(BaseProvider):
            def generate(self, prompt: str, **kw: object) -> str:
                return ""

        gp = GuardedProvider(MyCustomProvider(), _guard())
        assert gp.target == "mycustom"

    def test_derive_target_strips_provider_suffix(self) -> None:
        class OpenAIProvider(BaseProvider):
            def generate(self, prompt: str, **kw: object) -> str:
                return ""

        gp = GuardedProvider(OpenAIProvider(), _guard())
        assert gp.target == "openai"


class TestGuardedProviderGenerate:
    def test_passes_prompt_through(self) -> None:
        inner = _mock_provider("hello")
        gp = GuardedProvider(inner, _guard(), target="local")
        result = gp.generate("safe prompt")
        assert result == "hello"
        inner.generate.assert_called_once()

    def test_clean_prompt_reaches_inner(self) -> None:
        inner = _mock_provider()
        gp = GuardedProvider(inner, _guard(), target="ollama")
        gp.generate("no sensitive data here")
        assert inner.generate.called

    def test_blocked_prompt_raises_privacy_blocked(self) -> None:
        inner = _mock_provider()
        guard = PrivacyGuard(config=PrivacyGuardConfig(enabled=True, block_on_detect=True))
        gp = GuardedProvider(inner, guard, target="openai")
        with pytest.raises(PrivacyBlocked):
            gp.generate("my SSN is 123-45-6789 and email is alice@example.com")
        inner.generate.assert_not_called()

    def test_kwargs_forwarded_to_inner(self) -> None:
        inner = _mock_provider("response")
        gp = GuardedProvider(inner, _guard(), target="local")
        gp.generate("prompt", temperature=0.5, max_tokens=100)
        inner.generate.assert_called_once()
        _, kw = inner.generate.call_args
        assert "temperature" in kw
        assert "max_tokens" in kw

    def test_local_target_bypasses_scan(self) -> None:
        inner = _mock_provider("local ok")
        cfg = PrivacyGuardConfig(enabled=True, allow_local=True, block_on_detect=True)
        guard = PrivacyGuard(config=cfg)
        gp = GuardedProvider(inner, guard, target="ollama")
        # Even with PII — local target is bypassed
        result = gp.generate("email: alice@example.com")
        assert result == "local ok"


class TestDeriveTarget:
    def test_fallback_is_nonempty_string(self) -> None:
        class Provider(BaseProvider):
            def generate(self, p: str, **k: object) -> str:
                return ""

        result = GuardedProvider._derive_target(Provider())
        assert isinstance(result, str) and len(result) > 0

    def test_strips_provider_suffix(self) -> None:
        class AnthropicProvider(BaseProvider):
            def generate(self, p: str, **k: object) -> str:
                return ""

        result = GuardedProvider._derive_target(AnthropicProvider())
        assert result == "anthropic"
