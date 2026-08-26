"""Extended tests for dataenginex.domains.ai.llm — providers, config, and factory."""

from __future__ import annotations

import pytest

from dataenginex.domains.ai.llm import (
    ChatMessage,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    MockProvider,
    OllamaProvider,
    VLLMProvider,
    get_llm_provider,
)

# ── Data models ───────────────────────────────────────────────────────────────


class TestChatMessage:
    def test_fields(self) -> None:
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_system_role(self) -> None:
        m = ChatMessage(role="system", content="You are a helpful assistant.")
        assert m.role == "system"

    def test_assistant_role(self) -> None:
        m = ChatMessage(role="assistant", content="I can help.")
        assert m.role == "assistant"


class TestLLMConfig:
    def test_defaults(self) -> None:
        c = LLMConfig()
        assert c.model is not None
        assert c.max_tokens > 0

    def test_custom_values(self) -> None:
        c = LLMConfig(model="llama3:8b", max_tokens=256)
        assert c.model == "llama3:8b"
        assert c.max_tokens == 256


class TestLLMResponse:
    def test_fields(self) -> None:
        r = LLMResponse(text="hello")
        assert r.text == "hello"

    def test_optional_metadata(self) -> None:
        r = LLMResponse(text="hi", model="llama3", total_tokens=42)
        assert r.model == "llama3"
        assert r.total_tokens == 42

    def test_default_empty_text(self) -> None:
        r = LLMResponse(text="")
        assert r.text == ""


# ── LLMProvider ABC ───────────────────────────────────────────────────────────


class TestLLMProviderABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_generate_with_context(self) -> None:
        p = MockProvider(default_response="ctx response")
        result = p.generate_with_context("question", "some context")
        assert isinstance(result, LLMResponse)
        assert result.text.startswith("ctx response")

    def test_generate_with_context_custom_system_prompt(self) -> None:
        p = MockProvider()
        result = p.generate_with_context("q", "ctx", system_prompt="You are helpful.")
        assert isinstance(result, LLMResponse)


# ── MockProvider ──────────────────────────────────────────────────────────────


class TestMockProviderExtended:
    def test_default_response_prefix(self) -> None:
        p = MockProvider(default_response="fixed")
        r = p.generate("any prompt")
        assert r.text.startswith("fixed")  # appends metadata suffix

    def test_response_is_llm_response(self) -> None:
        p = MockProvider()
        r = p.generate("hello")
        assert isinstance(r, LLMResponse)

    def test_call_history_tracked(self) -> None:
        p = MockProvider()
        p.generate("first")
        p.generate("second")
        assert len(p.call_history) == 2

    def test_is_available(self) -> None:
        assert MockProvider().is_available() is True

    def test_chat_returns_response(self) -> None:
        p = MockProvider(default_response="chat reply")
        msgs = [ChatMessage(role="user", content="hi")]
        r = p.chat(msgs)
        assert r.text.startswith("chat reply")

    def test_chat_history_tracked(self) -> None:
        p = MockProvider()
        p.chat([ChatMessage(role="user", content="hello")])
        p.chat([ChatMessage(role="user", content="bye")])
        assert len(p.call_history) == 2

    def test_tokens_usage_populated(self) -> None:
        p = MockProvider()
        r = p.generate("test prompt")
        assert r.total_tokens is not None and r.total_tokens > 0

    def test_model_in_response(self) -> None:
        p = MockProvider()
        r = p.generate("test")
        assert r.model is not None


# ── OllamaProvider (no server needed — just init tests) ──────────────────────


class TestOllamaProviderInit:
    def test_init_defaults(self) -> None:
        p = OllamaProvider()
        assert p is not None

    def test_init_custom_base_url(self) -> None:
        p = OllamaProvider(base_url="http://my-server:11434")
        assert "my-server" in p.base_url

    def test_is_available_false_without_server(self) -> None:
        p = OllamaProvider(base_url="http://127.0.0.1:19999")
        assert p.is_available() is False


# ── VLLMProvider ──────────────────────────────────────────────────────────────


class TestVLLMProvider:
    def test_init(self) -> None:
        p = VLLMProvider(model="mistral-7b")
        assert p is not None

    def test_is_openai_compatible_subclass(self) -> None:
        from dataenginex.domains.ai.llm import OpenAICompatibleProvider

        assert isinstance(VLLMProvider(), OpenAICompatibleProvider)


# ── get_llm_provider factory ──────────────────────────────────────────────────


class TestGetLLMProviderExtended:
    def test_mock_provider(self) -> None:
        p = get_llm_provider("mock")
        assert isinstance(p, MockProvider)

    def test_ollama_provider(self) -> None:
        p = get_llm_provider("ollama")
        assert isinstance(p, OllamaProvider)

    def test_unknown_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            get_llm_provider("nonexistent_llm_xyz")

    def test_case_insensitive(self) -> None:
        p = get_llm_provider("MOCK")
        assert isinstance(p, MockProvider)

    def test_vllm_provider(self) -> None:
        p = get_llm_provider("vllm", model="test-model")
        assert isinstance(p, VLLMProvider)
