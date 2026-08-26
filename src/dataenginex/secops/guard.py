"""PrivacyGuard — intercept outbound LLM calls; scan, mask, and audit PII.

The guard sits between the user's prompt and any external LLM provider. It:

  1. Scans every outbound prompt for PII via :class:`PIIDetector.scan_text`.
  2. Masks the matches (or blocks the send entirely) per configuration.
  3. Logs each call so users can see exactly what left their machine.

Local providers (Ollama by default) are bypassed when
:attr:`PrivacyGuardConfig.allow_local` is ``True`` — those calls never leave
the machine in the first place. Cloud providers (OpenAI, Anthropic, etc.)
go through full scan/mask/log.

The guard is composed with a provider via
:class:`~dataenginex.providers.model.guarded.GuardedProvider` rather than
modifying the provider classes directly.

Example::

    from dataenginex.secops import (
        MaskingStrategy, PIIType, PrivacyGuard, PrivacyGuardConfig,
    )
    from dataenginex.providers.model.guarded import GuardedProvider
    from dataenginex.providers.model.openai import OpenAIProvider

    guard = PrivacyGuard(
        config=PrivacyGuardConfig(
            strategies={
                PIIType.EMAIL: MaskingStrategy.HASH,
                PIIType.SSN: MaskingStrategy.REDACT,
            },
        ),
    )
    provider = GuardedProvider(OpenAIProvider(api_key="..."), guard, target="openai")
    # All prompts now flow through the guard
    answer = provider.generate("Contact me at alice@example.com")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from dataenginex.secops.audit import AuditEvent, AuditLogger, AuditOperation
from dataenginex.secops.masking import MaskingEngine, MaskingStrategy
from dataenginex.secops.pii import PIIDetector, PIIType, TextMatch

logger = structlog.get_logger()

__all__ = [
    "GuardResult",
    "PrivacyBlocked",
    "PrivacyGuard",
    "PrivacyGuardConfig",
]

# Provider targets treated as local (never leave the machine).
_DEFAULT_LOCAL_TARGETS: frozenset[str] = frozenset({"ollama", "local"})


class PrivacyBlocked(RuntimeError):
    """Raised when an outbound call is rejected by the guard.

    Carries the original prompt's PII findings so the caller can present a
    useful error (e.g. surface the detected types in studio's quarantine UI
    without re-scanning).
    """

    def __init__(self, target: str, detections: tuple[TextMatch, ...]) -> None:
        types = sorted({m.pii_type.value for m in detections})
        super().__init__(
            f"PrivacyGuard blocked outbound call to {target!r}: "
            f"{len(detections)} PII match(es) ({', '.join(types)})"
        )
        self.target = target
        self.detections = detections


@dataclass
class PrivacyGuardConfig:
    """Behavioural configuration for :class:`PrivacyGuard`.

    Attributes:
        enabled: Master switch. When ``False`` the guard logs a warning on
            first call and passes prompts through unchanged.
        allow_local: When ``True`` (default), prompts to local providers
            (``local_targets``) bypass scanning entirely.
        block_on_detect: When ``True``, prompts containing any PII raise
            :exc:`PrivacyBlocked` instead of being masked.
        log_all_outbound: When ``True`` (default), every outbound call is
            logged via structlog — and, if an :class:`AuditLogger` is
            attached, persisted as an :class:`AuditEvent`.
        strategies: Per-:class:`PIIType` masking strategy overrides. Types
            absent from this map fall back to the masker's default.
        local_targets: Provider names treated as local. Customisable so
            self-hosted endpoints (e.g. private OpenAI-compatible servers)
            can opt out of guarding.
    """

    enabled: bool = True
    allow_local: bool = True
    block_on_detect: bool = False
    log_all_outbound: bool = True
    strategies: dict[PIIType, MaskingStrategy] = field(default_factory=dict)
    local_targets: frozenset[str] = _DEFAULT_LOCAL_TARGETS

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PrivacyGuardConfig:
        """Build a config from a loose dict (e.g. straight from ``dex.yaml``).

        String values in ``strategies`` and ``local_targets`` are converted
        to the appropriate enum / frozenset types. Unknown PII types or
        masking strategies raise ``ValueError`` from the enum constructors.
        """
        raw_strategies = data.get("strategies") or {}
        strategies: dict[PIIType, MaskingStrategy] = {}
        if isinstance(raw_strategies, dict):
            for k, v in raw_strategies.items():
                strategies[PIIType(k)] = MaskingStrategy(v)

        raw_targets = data.get("local_targets")
        if raw_targets is None:
            local_targets = _DEFAULT_LOCAL_TARGETS
        else:
            if not hasattr(raw_targets, "__iter__"):
                raise TypeError(f"local_targets must be iterable, got {type(raw_targets)!r}")
            local_targets = frozenset(str(t) for t in raw_targets)

        return cls(
            enabled=bool(data.get("enabled", True)),
            allow_local=bool(data.get("allow_local", True)),
            block_on_detect=bool(data.get("block_on_detect", False)),
            log_all_outbound=bool(data.get("log_all_outbound", True)),
            strategies=strategies,
            local_targets=local_targets,
        )


@dataclass(frozen=True)
class GuardResult:
    """The outcome of running :meth:`PrivacyGuard.process` on a prompt.

    Attributes:
        safe_prompt: The text that should actually be sent. Equals the
            original when no PII was detected or the guard was bypassed.
        detections: PII matches found in the original prompt. Empty when
            the guard was disabled / bypassed.
        blocked: ``True`` when ``block_on_detect`` rejected the call.
            Callers should raise :class:`PrivacyBlocked`.
        target: The provider name passed to :meth:`process`.
        bypassed_local: ``True`` when the target was a local provider and
            scanning was skipped.
    """

    safe_prompt: str
    detections: tuple[TextMatch, ...]
    target: str
    blocked: bool = False
    bypassed_local: bool = False


class PrivacyGuard:
    """Pre-send PII interception for outbound LLM calls.

    Compose with a :class:`~dataenginex.providers.model.router.BaseProvider` via
    :class:`~dataenginex.providers.model.guarded.GuardedProvider`, or call
    :meth:`process` directly for custom integrations.

    Parameters:
        config: Behavioural config. Defaults to a sensible "mask + log"
            setup with no per-type strategy overrides.
        detector: PII detector instance. A default :class:`PIIDetector`
            is created if omitted.
        masker: Masking engine. A default :class:`MaskingEngine`
            (REDACT strategy) is created if omitted.
        audit_logger: Optional :class:`AuditLogger`. When provided, every
            non-bypassed call writes an audit row (in addition to the
            structlog message).
    """

    def __init__(
        self,
        config: PrivacyGuardConfig | None = None,
        detector: PIIDetector | None = None,
        masker: MaskingEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.config = config or PrivacyGuardConfig()
        self._detector = detector or PIIDetector()
        self._masker = masker or MaskingEngine(default_strategy=MaskingStrategy.REDACT)
        self._audit = audit_logger
        self._warned_disabled = False

    def process(self, prompt: str, *, target: str = "unknown") -> GuardResult:
        """Scan ``prompt`` for PII and return the (possibly masked) safe form.

        Behaviour:
          - Guard disabled → pass-through (logs a one-time warning).
          - ``allow_local`` set and target is local → pass-through.
          - Otherwise: scan, then either mask or block, then log.
        """
        if not self.config.enabled:
            if not self._warned_disabled:
                self._warned_disabled = True
                logger.warning(
                    "privacy_guard.disabled",
                    note="all outbound prompts pass unmodified",
                )
            return GuardResult(safe_prompt=prompt, detections=(), target=target)

        if self.config.allow_local and self._is_local(target):
            return GuardResult(
                safe_prompt=prompt,
                detections=(),
                target=target,
                bypassed_local=True,
            )

        matches = self._detector.scan_text(prompt)
        if not matches:
            self._log(target, original=prompt, safe=prompt, matches=(), blocked=False)
            return GuardResult(safe_prompt=prompt, detections=(), target=target)

        detections = tuple(matches)
        if self.config.block_on_detect:
            self._log(target, original=prompt, safe="", matches=detections, blocked=True)
            return GuardResult(
                safe_prompt=prompt,
                detections=detections,
                target=target,
                blocked=True,
            )

        safe = self._masker.mask_text(prompt, list(matches), strategies=self.config.strategies)
        self._log(target, original=prompt, safe=safe, matches=detections, blocked=False)
        return GuardResult(safe_prompt=safe, detections=detections, target=target)

    def _is_local(self, target: str) -> bool:
        """Return ``True`` if ``target`` is configured as local."""
        return target.lower() in self.config.local_targets

    def _log(
        self,
        target: str,
        *,
        original: str,
        safe: str,
        matches: tuple[TextMatch, ...],
        blocked: bool,
    ) -> None:
        """Emit a structured log entry and (optionally) an audit row."""
        if not self.config.log_all_outbound:
            return
        types = sorted({m.pii_type.value for m in matches})
        logger.info(
            "privacy_guard.outbound",
            target=target,
            blocked=blocked,
            pii_count=len(matches),
            pii_types=types,
            chars_in=len(original),
            chars_out=len(safe),
        )
        if self._audit is not None and matches:
            operation = AuditOperation.PII_ACCESS if blocked else AuditOperation.PII_MASK
            self._audit.log(
                AuditEvent(
                    operation=operation,
                    dataset_name=f"outbound:{target}",
                    pii_fields=types,
                    record_count=1,
                    actor="privacy_guard",
                    metadata={
                        "blocked": blocked,
                        "match_count": len(matches),
                        "chars_in": len(original),
                        "chars_out": len(safe),
                    },
                )
            )
