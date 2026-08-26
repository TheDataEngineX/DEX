"""DataSecOps — PII detection, masking, audit logging, and outbound-call guard.

Public API::

    from dataenginex.secops import (
        PIIDetector, PIIField, PIIType, TextMatch,
        MaskingEngine, MaskingStrategy,
        AuditLogger, AuditEvent, AuditOperation,
        PrivacyGuard, PrivacyGuardConfig, GuardResult, PrivacyBlocked,
    )
"""

from __future__ import annotations

from .audit import AuditEvent, AuditLogger, AuditOperation
from .guard import GuardResult, PrivacyBlocked, PrivacyGuard, PrivacyGuardConfig
from .masking import MaskingEngine, MaskingStrategy
from .pii import PIIDetector, PIIField, PIIType, TextMatch

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "AuditOperation",
    "GuardResult",
    "MaskingEngine",
    "MaskingStrategy",
    "PIIDetector",
    "PIIField",
    "PIIType",
    "PrivacyBlocked",
    "PrivacyGuard",
    "PrivacyGuardConfig",
    "TextMatch",
]
