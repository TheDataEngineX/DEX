"""The model provider contract (§5.3, §9.10).

``BaseProvider`` is what the AI domain means by "something that can answer a
prompt". Concrete providers — Anthropic, OpenAI, Ollama, and the guarded
wrapper that enforces egress policy — implement it from ``providers/model/``.

It lives in the domain rather than beside those implementations because agents,
workflows, and the tool gateway all speak in terms of it. If the contract sat
in the provider package, every one of them would import a vendor module to name
the type it depends on, and the domain would no longer be usable without the
providers installed.

Stdlib only, deliberately: a contract that needed an SDK to express itself
would not be a contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["BaseProvider"]


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM."""
