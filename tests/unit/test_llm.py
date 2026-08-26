"""Tests for dataenginex.ml.llm — LLM provider abstraction & Ollama adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dataenginex.domains.ai.llm import (
    ChatMessage,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    MockProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    get_llm_provider,
    llm_request_latency_seconds,
    llm_tokens_total,
)

# ============================================================================
# LLMConfig
# ============================================================================


class TestLLMConfig:
    """Test LLM configuration dataclass."""

    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.model == "llama3.1:8b"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048

    def test_custom_values(self) -> None:
        cfg = LLMConfig(model="mistral:7b", temperature=0.3, max_tokens=1024)
        assert cfg.model == "mistral:7b"
        assert cfg.temperature == 0.3


# ============================================================================
# MockProvider
# ============================================================================


class TestMockProvider:
    """Test the deterministic mock LLM provider."""

    def test_generate(self) -> None:
        mock = MockProvider()
        resp = mock.generate("Hello, world!")
        assert isinstance(resp, LLMResponse)
        assert "mock LLM response" in resp.text
        assert resp.model == "mock-model"
        assert resp.total_tokens > 0

    def test_chat(self) -> None:
        mock = MockProvider()
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is Python?"),
        ]
        resp = mock.chat(messages)
        assert "messages=2" in resp.text
        assert resp.total_tokens > 0

    def test_is_available(self) -> None:
        mock = MockProvider()
        assert mock.is_available() is True

    def test_call_history(self) -> None:
        mock = MockProvider()
        mock.generate("prompt1")
        mock.generate("prompt2")
        mock.chat([ChatMessage(role="user", content="hi")])
        assert len(mock.call_history) == 3
        assert mock.call_history[0]["type"] == "generate"
        assert mock.call_history[2]["type"] == "chat"

    def test_custom_default_response(self) -> None:
        mock = MockProvider(default_response="custom answer")
        resp = mock.generate("question")
        assert "custom answer" in resp.text

    def test_generate_with_context(self) -> None:
        mock = MockProvider()
        resp = mock.generate_with_context(
            question="What is DataEngineX?",
            context="DataEngineX is a data engineering platform.",
        )
        assert isinstance(resp, LLMResponse)
        assert len(mock.call_history) == 1
        assert mock.call_history[0]["type"] == "chat"


# ============================================================================
# OllamaProvider
# ============================================================================


class TestOllamaProvider:
    """Test Ollama provider (without a running server)."""

    def test_init(self) -> None:
        provider = OllamaProvider(model="llama3.1:8b")
        assert provider.config.model == "llama3.1:8b"
        assert "localhost:11434" in provider.base_url

    def test_is_available_returns_false_without_server(self) -> None:
        provider = OllamaProvider(base_url="http://localhost:99999")
        assert provider.is_available() is False

    def test_generate_raises_connection_error_without_server(self) -> None:
        import httpx

        provider = OllamaProvider(base_url="http://localhost:99999")
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            provider.generate("test prompt")

    def test_chat_raises_connection_error_without_server(self) -> None:
        import httpx

        provider = OllamaProvider(base_url="http://localhost:99999")
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            provider.chat([ChatMessage(role="user", content="hi")])

    def test_list_models_returns_empty_without_server(self) -> None:
        provider = OllamaProvider(base_url="http://localhost:99999")
        models = provider.list_models()
        assert models == []


# ============================================================================
# LLMProvider ABC
# ============================================================================


class TestLLMProviderABC:
    """Test that LLMProvider cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]


# ============================================================================
# OpenAICompatibleProvider
# ============================================================================


class TestOpenAICompatibleProvider:
    """Test the OpenAI-compatible provider."""

    def test_init(self) -> None:
        provider = OpenAICompatibleProvider(api_key="test-key", model="gpt-4o-mini")
        assert provider.config.model == "gpt-4o-mini"
        assert provider.base_url == "https://api.openai.com"

    def test_custom_base_url(self) -> None:
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            base_url="https://api.groq.com/openai",
            model="llama3-8b",
        )
        assert provider.base_url == "https://api.groq.com/openai"
        assert provider.config.model == "llama3-8b"

    def test_headers_contain_auth(self) -> None:
        provider = OpenAICompatibleProvider(api_key="sk-test123")
        headers = provider._headers()
        assert headers["Authorization"] == "Bearer sk-test123"
        assert headers["Content-Type"] == "application/json"

    def test_chat_raises_connection_error_without_server(self) -> None:
        import httpx

        provider = OpenAICompatibleProvider(
            api_key="test-key",
            base_url="http://localhost:99999",
        )
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            provider.chat([ChatMessage(role="user", content="hi")])

    def test_generate_raises_connection_error_without_server(self) -> None:
        import httpx

        provider = OpenAICompatibleProvider(
            api_key="test-key",
            base_url="http://localhost:99999",
        )
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            provider.generate("test prompt")

    def test_is_available_returns_false_without_server(self) -> None:
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            base_url="http://localhost:99999",
        )
        assert provider.is_available() is False

    def test_chat_with_mocked_response(self) -> None:
        provider = OpenAICompatibleProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello from GPT!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            resp = provider.chat([ChatMessage(role="user", content="Hello")])

        assert resp.text == "Hello from GPT!"
        assert resp.model == "gpt-4o-mini"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.total_tokens == 15


# ============================================================================
# Factory function
# ============================================================================


class TestGetLLMProvider:
    """Test the get_llm_provider factory function."""

    def test_create_mock_provider(self) -> None:
        llm = get_llm_provider("mock")
        assert isinstance(llm, MockProvider)
        assert llm.is_available() is True

    def test_create_ollama_provider(self) -> None:
        llm = get_llm_provider("ollama", model="llama3.1:8b")
        assert isinstance(llm, OllamaProvider)

    def test_create_openai_provider(self) -> None:
        llm = get_llm_provider("openai", api_key="test-key", model="gpt-4o")
        assert isinstance(llm, OpenAICompatibleProvider)
        assert llm.config.model == "gpt-4o"

    def test_case_insensitive(self) -> None:
        llm = get_llm_provider("Mock")
        assert isinstance(llm, MockProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("unknown_provider")


# ============================================================================
# Prometheus Metrics
# ============================================================================


class TestLLMMetrics:
    """Test that Prometheus metrics are properly defined."""

    def test_latency_histogram_exists(self) -> None:
        assert llm_request_latency_seconds is not None
        assert llm_request_latency_seconds._name == "llm_request_latency_seconds"

    def test_tokens_counter_exists(self) -> None:
        assert llm_tokens_total is not None
        assert llm_tokens_total._name == "llm_tokens"
