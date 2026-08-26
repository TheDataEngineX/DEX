"""Tests for LiteLLMProvider and VLLMProvider in dataenginex.domains.ai.llm."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataenginex.domains.ai.llm import (
    ChatMessage,
    LiteLLMProvider,
    LLMResponse,
    VLLMProvider,
    get_llm_provider,
)


def _fake_litellm_response(text: str = "hi") -> MagicMock:
    resp = MagicMock()
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    resp.model = "openai/gpt-4o-mini"
    return resp


class TestLiteLLMProvider:
    def test_factory_registration(self) -> None:
        provider = get_llm_provider("litellm", model="openai/gpt-4o-mini")
        assert isinstance(provider, LiteLLMProvider)
        assert provider.config.model == "openai/gpt-4o-mini"

    def test_generate_calls_litellm(self) -> None:
        fake_module = types.ModuleType("litellm")
        fake_module.completion = MagicMock(return_value=_fake_litellm_response("hello"))  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"litellm": fake_module}):
            provider = LiteLLMProvider(model="openai/gpt-4o-mini")
            resp = provider.generate("hi there")
        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello"
        assert resp.prompt_tokens == 3
        assert resp.completion_tokens == 2
        call_kwargs: dict[str, Any] = fake_module.completion.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["messages"][-1]["content"] == "hi there"

    def test_chat_forwards_message_roles(self) -> None:
        fake_module = types.ModuleType("litellm")
        fake_module.completion = MagicMock(return_value=_fake_litellm_response("ok"))  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"litellm": fake_module}):
            provider = LiteLLMProvider(model="anthropic/claude-3-sonnet")
            provider.chat(
                [
                    ChatMessage(role="system", content="sys"),
                    ChatMessage(role="user", content="u"),
                ],
            )
        messages = fake_module.completion.call_args.kwargs["messages"]  # type: ignore[attr-defined]
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_call_translates_exception(self) -> None:
        fake_module = types.ModuleType("litellm")

        def boom(**_: Any) -> None:
            msg = "upstream fail"
            raise RuntimeError(msg)

        fake_module.completion = boom  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"litellm": fake_module}):
            provider = LiteLLMProvider(model="openai/gpt-4o-mini")
            with pytest.raises(ConnectionError, match="LiteLLM call failed"):
                provider.generate("x")

    def test_is_available_missing_module(self) -> None:
        provider = LiteLLMProvider(model="openai/gpt-4o-mini")
        with patch.dict(sys.modules, {"litellm": None}):
            assert provider.is_available() is False


class TestVLLMProvider:
    def test_factory_registration(self) -> None:
        provider = get_llm_provider("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
        assert isinstance(provider, VLLMProvider)

    def test_defaults(self) -> None:
        provider = VLLMProvider()
        assert provider.base_url == "http://localhost:8000"
        assert provider.config.model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_openai_compatible_inheritance(self) -> None:
        from dataenginex.domains.ai.llm import OpenAICompatibleProvider

        provider = VLLMProvider(base_url="http://vllm:8000")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider._chat_url == "http://vllm:8000/v1/chat/completions"
