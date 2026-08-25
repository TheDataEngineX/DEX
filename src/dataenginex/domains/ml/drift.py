"""
Drift detection — statistical tests to detect data/model drift.

``DriftDetector`` compares a reference distribution against a current
distribution using Population Stability Index (PSI) and simple
distribution checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "DriftDetector",
    "DriftReport",
]


@dataclass
class DriftReport:
    """Outcome of a drift check for a single feature.

    Attributes:
        feature_name: Name of the feature that was checked.
        psi: Population Stability Index value.
        drift_detected: Whether drift exceeds the configured threshold.
        severity: Drift severity — ``"none"``, ``"moderate"``, or ``"severe"``.
        reference_mean: Mean of the reference distribution.
        current_mean: Mean of the current distribution.
        reference_std: Standard deviation of reference distribution.
        current_std: Standard deviation of current distribution.
        details: Extra context (bins, threshold, etc.).
        checked_at: Timestamp of the drift check.
    """

    feature_name: str
    psi: float  # Population Stability Index
    drift_detected: bool
    severity: str  # "none", "moderate", "severe"
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the drift report to a plain dictionary."""
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 6),
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "reference_std": self.reference_std,
            "current_std": self.current_std,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


class DriftDetector:
    """Detect distribution drift between a reference and current dataset.

    PSI thresholds (industry standard):
        < 0.10  — no significant drift
        0.10-0.25 — moderate drift
        > 0.25  — significant drift

    Parameters
    ----------
    psi_threshold:
        PSI value above which drift is flagged (default 0.20).
    n_bins:
        Number of histogram bins for PSI calculation (default 10).
    """

    def __init__(self, psi_threshold: float = 0.20, n_bins: int = 10) -> None:
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins

    def check_feature(
        self,
        feature_name: str,
        reference: list[float],
        current: list[float],
    ) -> DriftReport:
        """Check drift for a single numeric feature."""
        if not reference or not current:
            return DriftReport(
                feature_name=feature_name,
                psi=0.0,
                drift_detected=False,
                severity="none",
                details={"error": "Empty distribution(s)"},
            )

        import statistics

        ref_mean = statistics.mean(reference)
        cur_mean = statistics.mean(current)
        ref_std = statistics.stdev(reference) if len(reference) > 1 else 0.0
        cur_std = statistics.stdev(current) if len(current) > 1 else 0.0

        psi = self._compute_psi(reference, current)
        severity = self._classify_severity(psi)

        return DriftReport(
            feature_name=feature_name,
            psi=psi,
            drift_detected=psi > self.psi_threshold,
            severity=severity,
            reference_mean=round(ref_mean, 4),
            current_mean=round(cur_mean, 4),
            reference_std=round(ref_std, 4),
            current_std=round(cur_std, 4),
            details={"n_bins": self.n_bins, "threshold": self.psi_threshold},
        )

    def check_dataset(
        self,
        reference: dict[str, list[float]],
        current: dict[str, list[float]],
    ) -> list[DriftReport]:
        """Check drift across all shared features in two datasets.

        Parameters
        ----------
        reference:
            Mapping of ``feature_name → values`` for the reference period.
        current:
            Mapping of ``feature_name → values`` for the current period.
        """
        reports: list[DriftReport] = []
        shared = set(reference.keys()) & set(current.keys())
        for feat in sorted(shared):
            reports.append(self.check_feature(feat, reference[feat], current[feat]))
        return reports

    # -- internal ------------------------------------------------------------

    def _compute_psi(self, reference: list[float], current: list[float]) -> float:
        """Compute PSI using equal-width binning."""
        all_vals = reference + current
        lo = min(all_vals)
        hi = max(all_vals)
        if lo == hi:
            return 0.0

        bin_width = (hi - lo) / self.n_bins
        eps = 1e-6

        ref_counts = [0] * self.n_bins
        cur_counts = [0] * self.n_bins

        for v in reference:
            idx = min(int((v - lo) / bin_width), self.n_bins - 1)
            ref_counts[idx] += 1
        for v in current:
            idx = min(int((v - lo) / bin_width), self.n_bins - 1)
            cur_counts[idx] += 1

        ref_total = len(reference)
        cur_total = len(current)

        psi = 0.0
        for r, c in zip(ref_counts, cur_counts, strict=True):
            ref_pct = (r / ref_total) + eps
            cur_pct = (c / cur_total) + eps
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return max(0.0, psi)

    @staticmethod
    def _classify_severity(psi: float) -> str:
        if psi < 0.10:
            return "none"
        if psi < 0.25:
            return "moderate"
        return "severe"
