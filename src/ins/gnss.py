"""GNSS fix-quality and freshness checks used before IEKF fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GnssQualityDecision:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class GnssQualityGate:
    minimum_satellites: int = 8
    maximum_accuracy_m: float = 10.0
    maximum_fix_age_s: float = 12.0

    def __post_init__(self) -> None:
        if self.minimum_satellites < 1:
            raise ValueError("minimum_satellites must be positive")
        if self.maximum_accuracy_m <= 0 or self.maximum_fix_age_s <= 0:
            raise ValueError("accuracy and fix-age limits must be positive")

    def evaluate(
        self,
        *,
        satellites: float,
        accuracy_m: float,
        age_s: float,
        externally_available: bool = True,
    ) -> GnssQualityDecision:
        if not externally_available:
            return GnssQualityDecision(False, "unavailable")
        if not all(math.isfinite(value) for value in (satellites, accuracy_m, age_s)):
            return GnssQualityDecision(False, "non_finite")
        if age_s > self.maximum_fix_age_s:
            return GnssQualityDecision(False, "stale")
        if satellites < self.minimum_satellites:
            return GnssQualityDecision(False, "satellites")
        if accuracy_m > self.maximum_accuracy_m:
            return GnssQualityDecision(False, "accuracy")
        return GnssQualityDecision(True, "accepted")
