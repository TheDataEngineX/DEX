"""Tests for LLM helpers: MockProvider, get_llm_provider, dataclasses."""

from __future__ import annotations

import pytest

from dataenginex.domains.ai.llm import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    MockProvider,
    get_llm_provider,
)


class TestChatMessage:
    def test_fields(self) -> None:
        m = ChatMessage(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.model == "llama3.1:8b"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.top_p == 0.9
        assert cfg.timeout_seconds == 120

    def test_custom(self) -> None:
        cfg = LLMConfig(model="gpt-4", temperature=0.1)
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.1


class TestLLMResponse:
    def test_defaults(self) -> None:
        r = LLMResponse(text="hello")
        assert r.text == "hello"
        assert r.finish_reason == "stop"
        assert r.total_tokens == 0
        assert r.metadata == {}


class TestMockProvider:
    def test_generate(self) -> None:
        p = MockProvider()
        resp = p.generate("test prompt")
        assert "mock LLM response" in resp.text
        assert resp.model == "mock-model"
        assert resp.total_tokens > 0
        assert len(p.call_history) == 1

    def test_chat(self) -> None:
        p = MockProvider()
        msgs = [ChatMessage(role="user", content="hello")]
        resp = p.chat(msgs)
        assert "messages=1" in resp.text
        assert len(p.call_history) == 1

    def test_is_available(self) -> None:
        p = MockProvider()
        assert p.is_available() is True

    def test_custom_response(self) -> None:
        p = MockProvider(default_response="custom")
        resp = p.generate("test")
        assert resp.text.startswith("custom")

    def test_generate_with_context(self) -> None:
        p = MockProvider()
        resp = p.generate_with_context("question?", "some context")
        assert "messages=" in resp.text


class TestGetLLMProvider:
    def test_mock(self) -> None:
        p = get_llm_provider("mock")
        assert isinstance(p, MockProvider)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("nonexistent")
