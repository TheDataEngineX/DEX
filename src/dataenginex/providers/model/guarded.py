"""GuardedProvider — wrap a BaseProvider so every outbound call is PII-guarded.

Decorator-style wrapper. The inner provider's ``generate`` is invoked with
the guard's masked prompt; on a block the wrapper raises
:exc:`~dataenginex.secops.guard.PrivacyBlocked` instead of contacting the
remote service at all.

Example::

    from dataenginex.providers.model.openai import OpenAIProvider
    from dataenginex.providers.model.guarded import GuardedProvider
    from dataenginex.secops import PrivacyGuard

    guard = PrivacyGuard()
    provider = GuardedProvider(OpenAIProvider(api_key="..."), guard, target="openai")
    answer = provider.generate("Email me at alice@example.com")
"""

from __future__ import annotations

from typing import Any

from dataenginex.domains.ai.providers import BaseProvider
from dataenginex.secops.guard import PrivacyBlocked, PrivacyGuard

__all__ = ["GuardedProvider"]


class GuardedProvider(BaseProvider):
    """A :class:`BaseProvider` whose ``generate`` runs through a :class:`PrivacyGuard`.

    The wrapper preserves the inner provider's full ``**kwargs`` API; only
    the prompt is transformed before being passed through.

    Parameters:
        inner: The provider whose calls should be guarded.
        guard: The guard instance to apply.
        target: Logical name for the inner provider (e.g. ``"openai"``,
            ``"anthropic"``). Used by the guard's local-bypass check and
            audit logs. Defaults to the inner class name with ``"provider"``
            stripped, lowercased.
    """

    def __init__(
        self,
        inner: BaseProvider,
        guard: PrivacyGuard,
        target: str = "",
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._target = target or self._derive_target(inner)

    @property
    def inner(self) -> BaseProvider:
        """The wrapped provider (read-only)."""
        return self._inner

    @property
    def target(self) -> str:
        """The provider name used for guard decisions and audit logs."""
        return self._target

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Run the guard, then delegate to the wrapped provider.

        Raises:
            PrivacyBlocked: When the guard's ``block_on_detect`` rejected
                the call. The inner provider is *not* invoked in that case.
        """
        result = self._guard.process(prompt, target=self._target)
        if result.blocked:
            raise PrivacyBlocked(target=self._target, detections=result.detections)
        return self._inner.generate(result.safe_prompt, **kwargs)

    @staticmethod
    def _derive_target(inner: BaseProvider) -> str:
        """Best-effort target name from the inner provider's class name."""
        name = type(inner).__name__.lstrip("_").lower()
        return name.removesuffix("provider") or name
