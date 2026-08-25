# SecOps

PII detection, masking, audit logging, and outbound-call privacy guard.

```python
from dataenginex.secops import (
    PIIDetector, MaskingEngine, AuditLogger, PrivacyGuard,
)
```

## PIIDetector

Detect PII in text.

```python
detector = PIIDetector()
matches = detector.scan("Email me at alice@example.com")
# -> [PIIField(type=PIIType.EMAIL, value="alice@example.com", confidence=0.95)]
```

### Classes

| Class | Description |
|-------|-------------|
| `PIIDetector` | PII detection engine |
| `PIIField` | Individual PII match record |
| `PIIType` | Enum of PII categories (EMAIL, PHONE, SSN, etc.) |
| `TextMatch` | Text match result |

## MaskingEngine

Mask or redact PII in text.

```python
masker = MaskingEngine(strategy=MaskingStrategy.REDACT)
safe_text = masker.mask("Email me at alice@example.com")
# -> "Email me at [REDACTED]"
```

| Class | Description |
|-------|-------------|
| `MaskingEngine` | PII masking/redaction engine |
| `MaskingStrategy` | Enum: REDACT, HASH, PARTIAL, TOKENIZE |

## AuditLogger

Log audit events to a database or in-memory store.

```python
logger = AuditLogger()
logger.log(action="pipeline.run", resource="clean_users", status="success")
events = logger.get_events(action="pipeline.run")
```

| Class | Description |
|-------|-------------|
| `AuditLogger` | Audit trail logger |
| `AuditEvent` | Audit event record |
| `AuditOperation` | Audit operation type |

## PrivacyGuard

Intercept outbound calls and block requests containing PII.

```python
guard = PrivacyGuard(config=PrivacyGuardConfig(enabled=True))
result = guard.process("Send to openai", target="openai")
if result.blocked:
    raise PrivacyBlocked(result.reason)
```

| Class | Description |
|-------|-------------|
| `PrivacyGuard` | Outbound-call PII guard |
| `PrivacyGuardConfig` | Guard configuration |
| `GuardResult` | Result of guard processing |
| `PrivacyBlocked` | Exception for blocked requests |
