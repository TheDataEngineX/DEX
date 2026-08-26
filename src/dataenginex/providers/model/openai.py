"""OpenAI Chat Completions API provider adapter — calls API via httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from dataenginex.domains.ai.providers import BaseProvider

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible chat completions provider.

    Requires OPENAI_API_KEY env var or explicit *api_key* argument.
    Works with any OpenAI-compatible endpoint (set *base_url* for proxies).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using OpenAI Chat Completions API."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set — export OPENAI_API_KEY=sk-...")
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
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
            return str(data["choices"][0]["message"]["content"])
        except httpx.ConnectError as exc:
            raise ConnectionError(f"OpenAI API not reachable: {exc}") from exc
