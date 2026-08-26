"""Ollama local LLM provider adapter — calls generate API via httpx."""

from __future__ import annotations

from typing import Any

import httpx

from dataenginex.domains.ai.providers import BaseProvider

_DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider.

    Calls ``POST /api/generate`` on the running Ollama server.
    No API key required — Ollama runs locally.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = _DEFAULT_HOST,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response using the local Ollama server."""
        try:
            response = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **{k: v for k, v in kwargs.items() if k not in ("model", "prompt", "stream")},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["response"])
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Ollama not reachable at {self.host}: {exc}") from exc
