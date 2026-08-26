"""Unit tests for the dataenginex.secops module."""

from __future__ import annotations

from dataenginex.secops import (
    AuditLogger,
    AuditOperation,
    MaskingEngine,
    MaskingStrategy,
    PIIDetector,
    PIIType,
)

# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------


class TestPIIDetectorNameHints:
    def test_email_field_detected(self) -> None:
        detector = PIIDetector()
        fields = detector.pii_field_names([{"email": "user@example.com"}])
        assert "email" in fields

    def test_phone_field_detected(self) -> None:
        detector = PIIDetector()
        fields = detector.pii_field_names([{"phone_number": "555-1234"}])
        assert "phone_number" in fields

    def test_ssn_field_detected(self) -> None:
        detector = PIIDetector()
        fields = detector.pii_field_names([{"ssn": "123-45-6789"}])
        assert "ssn" in fields

    def test_non_pii_field_not_detected(self) -> None:
        detector = PIIDetector()
        fields = detector.pii_field_names([{"product_id": "ABC-001", "quantity": 5}])
        assert fields == set()


class TestPIIDetectorValuePatterns:
    def test_email_value_detected(self) -> None:
        detector = PIIDetector()
        findings = detector.scan_record({"contact": "reach me at alice@example.com please"})
        assert any(f.pii_type == PIIType.EMAIL for f in findings)

    def test_ssn_value_detected(self) -> None:
        detector = PIIDetector()
        findings = detector.scan_record({"data": "SSN: 123-45-6789"})
        assert any(f.pii_type == PIIType.SSN for f in findings)

    def test_credit_card_value_detected(self) -> None:
        detector = PIIDetector()
        findings = detector.scan_record({"payment": "4111 1111 1111 1111"})
        assert any(f.pii_type == PIIType.CREDIT_CARD for f in findings)

    def test_ip_address_value_detected(self) -> None:
        detector = PIIDetector()
        findings = detector.scan_record({"client": "192.168.1.100"})
        assert any(f.pii_type == PIIType.IP_ADDRESS for f in findings)


class TestPIIDetectorDataset:
    def test_scan_dataset_deduplicates(self) -> None:
        detector = PIIDetector()
        records = [{"email": "a@b.com"}, {"email": "c@d.com"}]
        result = detector.scan_dataset(records)
        assert "email" in result
        assert len(result) == 1  # deduped to one entry per field name

    def test_confidence_threshold_filters(self) -> None:
        # High threshold should suppress name-hint detections (confidence 0.85)
        detector = PIIDetector(confidence_threshold=0.99)
        fields = detector.pii_field_names([{"email": "user@example.com"}])
        # Name hint confidence is 0.85 < 0.99 → not reported
        assert "email" not in fields


# ---------------------------------------------------------------------------
# MaskingEngine
# ---------------------------------------------------------------------------


class TestMaskingEngineRedact:
    def test_redact_replaces_value(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.REDACT)
        result = engine.mask_record({"email": "user@example.com"}, {"email"})
        assert result["email"] == "[REDACTED]"

    def test_non_pii_field_unchanged(self) -> None:
        engine = MaskingEngine()
        result = engine.mask_record({"name": "Alice", "id": 1}, {"name"})
        assert result["id"] == 1


class TestMaskingEngineHash:
    def test_hash_produces_hex_string(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.HASH)
        result = engine.mask_record({"email": "user@example.com"}, {"email"})
        assert isinstance(result["email"], str)
        assert len(result["email"]) == 64  # SHA-256 hex

    def test_hash_is_deterministic(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.HASH)
        r1 = engine.mask_record({"email": "user@example.com"}, {"email"})
        r2 = engine.mask_record({"email": "user@example.com"}, {"email"})
        assert r1["email"] == r2["email"]

    def test_different_values_produce_different_hashes(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.HASH)
        r1 = engine.mask_record({"email": "a@example.com"}, {"email"})
        r2 = engine.mask_record({"email": "b@example.com"}, {"email"})
        assert r1["email"] != r2["email"]


class TestMaskingEnginePartial:
    def test_partial_keeps_last_4(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.PARTIAL)
        result = engine.mask_record({"phone": "555-867-5309"}, {"phone"})
        assert result["phone"].endswith("5309")
        assert "*" in result["phone"]

    def test_partial_short_value_fully_masked(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.PARTIAL, partial_keep_last=4)
        result = engine.mask_record({"pin": "123"}, {"pin"})
        assert result["pin"] == "***"


class TestMaskingEngineTokenize:
    def test_tokenize_produces_tok_prefix(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.TOKENIZE)
        result = engine.mask_record({"email": "user@example.com"}, {"email"})
        assert str(result["email"]).startswith("tok_")

    def test_tokenize_is_deterministic(self) -> None:
        engine = MaskingEngine(default_strategy=MaskingStrategy.TOKENIZE)
        r1 = engine.mask_record({"email": "user@example.com"}, {"email"})
        r2 = engine.mask_record({"email": "user@example.com"}, {"email"})
        assert r1["email"] == r2["email"]


class TestMaskingEngineFieldStrategies:
    def test_per_field_strategy_overrides_default(self) -> None:
        engine = MaskingEngine(
            default_strategy=MaskingStrategy.REDACT,
            field_strategies={"email": MaskingStrategy.HASH},
        )
        result = engine.mask_record({"email": "x@y.com", "phone": "555-0000"}, {"email", "phone"})
        # email → hash (64 hex chars), phone → [REDACTED]
        assert len(str(result["email"])) == 64
        assert result["phone"] == "[REDACTED]"


class TestMaskingEngineNone:
    def test_none_value_redacted(self) -> None:
        engine = MaskingEngine()
        result = engine.mask_record({"email": None}, {"email"})
        assert result["email"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class TestAuditLogger:
    def test_log_scan_appends_event(self) -> None:
        log = AuditLogger()
        log.log_scan("users", ["email"], 100)
        assert len(log.events) == 1
        assert log.events[0].operation == AuditOperation.PII_SCAN

    def test_log_mask_appends_event(self) -> None:
        log = AuditLogger()
        log.log_mask("users", ["email"], 100, strategy="redact")
        assert len(log.events) == 1
        assert log.events[0].operation == AuditOperation.PII_MASK

    def test_events_for_filters_by_dataset(self) -> None:
        log = AuditLogger()
        log.log_scan("users", ["email"], 10)
        log.log_scan("orders", ["phone"], 5)
        assert len(log.events_for("users")) == 1
        assert len(log.events_for("orders")) == 1

    def test_max_history_evicts_oldest(self) -> None:
        log = AuditLogger(max_history=3)
        for i in range(5):
            log.log_scan(f"ds_{i}", [], 1)
        assert len(log.events) == 3
        # Oldest events evicted; most recent retained
        assert log.events[0].dataset_name == "ds_2"

    def test_clear_removes_all_events(self) -> None:
        log = AuditLogger()
        log.log_scan("x", [], 1)
        log.clear()
        assert log.events == []
