"""Anthropic Claude API provider adapter — calls Messages API via httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from dataenginex.domains.ai.providers import BaseProvider

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider.

    Requires ANTHROPIC_API_KEY env var or explicit *api_key* argument.
    Uses the Messages API directly (no anthropic SDK dependency).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using Anthropic Messages API."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set — export ANTHROPIC_API_KEY=sk-ant-...")
        try:
            response = httpx.post(
                _API_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": _API_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["content"][0]["text"])
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Anthropic API not reachable: {exc}") from exc
