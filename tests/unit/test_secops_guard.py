"""Tests for PrivacyGuard, GuardedProvider, and string-level PII helpers."""

from __future__ import annotations

from typing import Any

import pytest

from dataenginex.domains.ai.providers import BaseProvider
from dataenginex.providers.model.guarded import GuardedProvider
from dataenginex.secops import (
    AuditLogger,
    MaskingEngine,
    MaskingStrategy,
    PIIDetector,
    PIIType,
    PrivacyBlocked,
    PrivacyGuard,
    PrivacyGuardConfig,
    TextMatch,
)

# ---------------------------------------------------------------------------
# PIIDetector.scan_text
# ---------------------------------------------------------------------------


class TestScanText:
    def test_empty_text_returns_no_matches(self) -> None:
        det = PIIDetector()
        assert det.scan_text("") == []

    def test_text_with_no_pii_returns_no_matches(self) -> None:
        det = PIIDetector()
        assert det.scan_text("Just a regular sentence with no sensitive data.") == []

    def test_detects_email(self) -> None:
        det = PIIDetector()
        matches = det.scan_text("Please reply to alice@example.com when ready.")
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "alice@example.com"

    def test_detects_ssn(self) -> None:
        det = PIIDetector()
        matches = det.scan_text("SSN is 123-45-6789, do not share.")
        assert any(m.pii_type == PIIType.SSN for m in matches)

    def test_detects_credit_card(self) -> None:
        det = PIIDetector()
        matches = det.scan_text("Card: 4111-1111-1111-1111 expires soon.")
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in matches)

    def test_detects_multiple_pii_in_one_string(self) -> None:
        det = PIIDetector()
        matches = det.scan_text("Email: bob@foo.com SSN: 987-65-4321")
        types = {m.pii_type for m in matches}
        assert PIIType.EMAIL in types
        assert PIIType.SSN in types

    def test_matches_returned_sorted_by_start(self) -> None:
        det = PIIDetector()
        text = "first bob@foo.com then later 192.168.1.1"
        matches = det.scan_text(text)
        starts = [m.start for m in matches]
        assert starts == sorted(starts)

    def test_match_offsets_are_correct(self) -> None:
        det = PIIDetector()
        text = "contact: alice@example.com"
        matches = det.scan_text(text)
        assert len(matches) == 1
        assert text[matches[0].start : matches[0].end] == "alice@example.com"

    def test_high_threshold_filters_low_confidence(self) -> None:
        # DOB has confidence 0.60 — should be filtered at threshold 0.85
        det_strict = PIIDetector(confidence_threshold=0.85)
        text = "Born 1990-01-15"
        assert det_strict.scan_text(text) == []
        det_loose = PIIDetector(confidence_threshold=0.5)
        assert any(m.pii_type == PIIType.DATE_OF_BIRTH for m in det_loose.scan_text(text))


# ---------------------------------------------------------------------------
# MaskingEngine.mask_text
# ---------------------------------------------------------------------------


class TestMaskText:
    def test_no_matches_returns_text_unchanged(self) -> None:
        eng = MaskingEngine()
        assert eng.mask_text("hello world", []) == "hello world"

    def test_redacts_email_by_default(self) -> None:
        eng = MaskingEngine(default_strategy=MaskingStrategy.REDACT)
        det = PIIDetector()
        text = "Reply to alice@example.com please"
        masked = eng.mask_text(text, det.scan_text(text))
        assert "alice@example.com" not in masked
        assert "[REDACTED]" in masked

    def test_per_type_strategy_overrides_default(self) -> None:
        eng = MaskingEngine(default_strategy=MaskingStrategy.REDACT)
        det = PIIDetector()
        text = "Reach me at alice@example.com"
        masked = eng.mask_text(
            text,
            det.scan_text(text),
            strategies={PIIType.EMAIL: MaskingStrategy.HASH},
        )
        assert "alice@example.com" not in masked
        assert "[REDACTED]" not in masked  # HASH used instead

    def test_multiple_replacements_preserve_indices(self) -> None:
        # Two emails — masking must not shift the second match's indices
        eng = MaskingEngine(default_strategy=MaskingStrategy.REDACT)
        det = PIIDetector()
        text = "From: a@x.com To: b@y.com"
        masked = eng.mask_text(text, det.scan_text(text))
        assert "a@x.com" not in masked
        assert "b@y.com" not in masked
        assert masked.count("[REDACTED]") == 2

    def test_partial_keeps_last_chars(self) -> None:
        eng = MaskingEngine(default_strategy=MaskingStrategy.PARTIAL, partial_keep_last=4)
        det = PIIDetector()
        text = "ssn 123-45-6789"
        masked = eng.mask_text(text, det.scan_text(text))
        # PARTIAL keeps the last 4 chars of "123-45-6789" → "6789"
        assert masked.endswith("6789")


# ---------------------------------------------------------------------------
# PrivacyGuard.process
# ---------------------------------------------------------------------------


class TestPrivacyGuard:
    def test_disabled_passes_through(self) -> None:
        guard = PrivacyGuard(config=PrivacyGuardConfig(enabled=False))
        result = guard.process("Email alice@example.com", target="openai")
        assert result.safe_prompt == "Email alice@example.com"
        assert result.detections == ()
        assert not result.blocked

    def test_local_target_bypassed_by_default(self) -> None:
        guard = PrivacyGuard()
        result = guard.process("Email alice@example.com", target="ollama")
        assert result.bypassed_local is True
        assert result.safe_prompt == "Email alice@example.com"
        assert result.detections == ()

    def test_local_bypass_can_be_disabled(self) -> None:
        guard = PrivacyGuard(config=PrivacyGuardConfig(allow_local=False))
        result = guard.process("Email alice@example.com", target="ollama")
        assert result.bypassed_local is False
        assert len(result.detections) == 1
        assert "alice@example.com" not in result.safe_prompt

    def test_remote_target_with_no_pii_passes_through(self) -> None:
        guard = PrivacyGuard()
        result = guard.process("What is 2+2?", target="openai")
        assert result.safe_prompt == "What is 2+2?"
        assert result.detections == ()
        assert not result.blocked

    def test_remote_target_with_pii_gets_masked(self) -> None:
        guard = PrivacyGuard()
        result = guard.process("Email me at alice@example.com", target="openai")
        assert "alice@example.com" not in result.safe_prompt
        assert len(result.detections) == 1
        assert result.detections[0].pii_type == PIIType.EMAIL
        assert not result.blocked

    def test_block_on_detect_returns_blocked_result(self) -> None:
        guard = PrivacyGuard(config=PrivacyGuardConfig(block_on_detect=True))
        result = guard.process("SSN 123-45-6789", target="anthropic")
        assert result.blocked is True
        assert result.safe_prompt == "SSN 123-45-6789"  # untouched
        assert len(result.detections) >= 1

    def test_block_on_detect_does_not_fire_on_clean_prompt(self) -> None:
        guard = PrivacyGuard(config=PrivacyGuardConfig(block_on_detect=True))
        result = guard.process("hello world", target="openai")
        assert not result.blocked

    def test_per_type_strategy_used_when_masking(self) -> None:
        cfg = PrivacyGuardConfig(strategies={PIIType.EMAIL: MaskingStrategy.HASH})
        guard = PrivacyGuard(config=cfg)
        result = guard.process("Reach me at alice@example.com", target="openai")
        assert "alice@example.com" not in result.safe_prompt
        assert "[REDACTED]" not in result.safe_prompt  # HASH used instead

    def test_custom_local_targets_bypass(self) -> None:
        cfg = PrivacyGuardConfig(local_targets=frozenset({"my_self_hosted_llm"}))
        guard = PrivacyGuard(config=cfg)
        result = guard.process("a@x.com", target="my_self_hosted_llm")
        assert result.bypassed_local is True

    def test_audit_logger_receives_event_on_mask(self) -> None:
        audit = AuditLogger()
        guard = PrivacyGuard(audit_logger=audit)
        guard.process("Email alice@example.com", target="openai")
        events = audit.events
        assert len(events) == 1
        assert events[0].dataset_name == "outbound:openai"
        assert events[0].actor == "privacy_guard"
        assert "email" in events[0].pii_fields
        audit.close()

    def test_audit_logger_records_block(self) -> None:
        audit = AuditLogger()
        guard = PrivacyGuard(
            config=PrivacyGuardConfig(block_on_detect=True),
            audit_logger=audit,
        )
        guard.process("Email alice@example.com", target="openai")
        events = audit.events
        assert len(events) == 1
        assert events[0].metadata["blocked"] is True
        audit.close()

    def test_audit_logger_skipped_on_clean_prompt(self) -> None:
        audit = AuditLogger()
        guard = PrivacyGuard(audit_logger=audit)
        guard.process("hello world", target="openai")
        # No PII matches → no audit event
        assert audit.events == []
        audit.close()


# ---------------------------------------------------------------------------
# GuardedProvider — wires PrivacyGuard into the provider call path
# ---------------------------------------------------------------------------


class _RecordingProvider(BaseProvider):
    """Test double that captures the prompt it receives."""

    def __init__(self) -> None:
        self.received: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.received.append(prompt)
        return f"echo: {prompt[:32]}"


class TestGuardedProvider:
    def test_target_derived_from_inner_class_name(self) -> None:
        inner = _RecordingProvider()
        wrapped = GuardedProvider(inner, PrivacyGuard())
        # _RecordingProvider → "recording"
        assert wrapped.target == "recording"

    def test_explicit_target_overrides_derivation(self) -> None:
        wrapped = GuardedProvider(_RecordingProvider(), PrivacyGuard(), target="openai")
        assert wrapped.target == "openai"

    def test_clean_prompt_passes_through_unmodified(self) -> None:
        inner = _RecordingProvider()
        wrapped = GuardedProvider(inner, PrivacyGuard(), target="openai")
        wrapped.generate("hello world")
        assert inner.received == ["hello world"]

    def test_pii_in_prompt_is_masked_before_inner_call(self) -> None:
        inner = _RecordingProvider()
        wrapped = GuardedProvider(inner, PrivacyGuard(), target="openai")
        wrapped.generate("Email me at alice@example.com")
        assert len(inner.received) == 1
        assert "alice@example.com" not in inner.received[0]

    def test_local_target_bypasses_guard(self) -> None:
        inner = _RecordingProvider()
        wrapped = GuardedProvider(inner, PrivacyGuard(), target="ollama")
        wrapped.generate("Email alice@example.com")
        # Ollama is local → guard bypassed → inner sees original
        assert inner.received == ["Email alice@example.com"]

    def test_block_on_detect_raises_privacyblocked(self) -> None:
        inner = _RecordingProvider()
        cfg = PrivacyGuardConfig(block_on_detect=True)
        wrapped = GuardedProvider(inner, PrivacyGuard(config=cfg), target="openai")
        with pytest.raises(PrivacyBlocked) as exc:
            wrapped.generate("SSN 123-45-6789")
        # Inner never called
        assert inner.received == []
        # Exception carries detections
        assert len(exc.value.detections) >= 1
        assert exc.value.target == "openai"

    def test_kwargs_pass_through_to_inner(self) -> None:
        class _KwargsProvider(BaseProvider):
            def __init__(self) -> None:
                self.last_kwargs: dict[str, Any] = {}

            def generate(self, prompt: str, **kwargs: Any) -> str:
                self.last_kwargs = kwargs
                return "ok"

        inner = _KwargsProvider()
        wrapped = GuardedProvider(inner, PrivacyGuard(), target="openai")
        wrapped.generate("hello", temperature=0.7, max_tokens=100)
        assert inner.last_kwargs == {"temperature": 0.7, "max_tokens": 100}


# ---------------------------------------------------------------------------
# Overlapping-match deduplication
# ---------------------------------------------------------------------------


class TestOverlappingMatches:
    def test_overlapping_spans_deduplicated(self) -> None:
        # A phone-number-like span and a credit-card-like span overlap;
        # the scanner should keep only the higher-confidence one.
        det = PIIDetector()
        # 4111-1111-1111-1111 matches both CREDIT_CARD (0.85) and PHONE (0.80)
        matches = det.scan_text("Card 4111-1111-1111-1111 is here")
        # CREDIT_CARD should win (higher confidence)
        kept_types = {m.pii_type for m in matches}
        # PHONE pattern matches sub-spans; dedup should prefer CC
        assert PIIType.CREDIT_CARD in kept_types

    def test_textmatch_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        m = TextMatch(pii_type=PIIType.EMAIL, value="a@b.c", start=0, end=5)
        with pytest.raises(FrozenInstanceError):
            m.value = "x"  # type: ignore[misc]
