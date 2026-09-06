"""Deterministic navigation-mode arbitration for GNSS outages and recovery.

The state machine deliberately treats GNSS quality as an input decision.  Quality
gating belongs upstream; this module only needs to know whether the current fix
was accepted.  A rejected or missing fix is represented by ``False`` or ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class NavigationMode(str, Enum):
    """Operating modes exposed to the navigation pipeline."""

    GNSS_AIDED = "GNSS_AIDED"
    DEAD_RECKONING = "DEAD_RECKONING"
    TRANSITIONING = "TRANSITIONING"


@dataclass(frozen=True)
class ArbitrationDecision:
    """Mode and GNSS contribution to use for the current sample.

    ``gnss_blend_weight`` is zero in dead reckoning, rises monotonically from
    zero to one during recovery, and is one in fully GNSS-aided operation.
    """

    mode: NavigationMode
    gnss_blend_weight: float
    fix_accepted: bool


class GnssModeArbitrator:
    """Three-state GNSS/dead-reckoning arbitration machine.

    The machine starts in ``GNSS_AIDED`` because the normal pipeline initializes
    before an outage while valid GNSS is available.  The first bad or missing fix
    switches immediately to ``DEAD_RECKONING``.  The first accepted fix after an
    outage begins a linear recovery blend at weight zero.
    """

    def __init__(self, transition_duration_s: float) -> None:
        if not math.isfinite(transition_duration_s) or transition_duration_s <= 0.0:
            raise ValueError("transition_duration_s must be finite and positive")
        self.transition_duration_s = float(transition_duration_s)
        self._mode = NavigationMode.GNSS_AIDED
        self._transition_start_s: float | None = None
        self._last_timestamp_s: float | None = None

    @property
    def mode(self) -> NavigationMode:
        """Current mode without advancing the state machine."""

        return self._mode

    def update(
        self, timestamp_s: float, fix_accepted: bool | None
    ) -> ArbitrationDecision:
        """Advance the state machine for one timestamped GNSS decision.

        Timestamps must be finite and nondecreasing. ``None`` is treated as a
        missing fix and therefore has the same effect as a rejected fix.
        """

        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._last_timestamp_s is not None and timestamp_s < self._last_timestamp_s:
            raise ValueError("timestamps must be nondecreasing")
        self._last_timestamp_s = timestamp_s
        accepted = bool(fix_accepted) if fix_accepted is not None else False

        if not accepted:
            self._mode = NavigationMode.DEAD_RECKONING
            self._transition_start_s = None
            return ArbitrationDecision(self._mode, 0.0, False)

        if self._mode is NavigationMode.DEAD_RECKONING:
            self._mode = NavigationMode.TRANSITIONING
            self._transition_start_s = timestamp_s
            return ArbitrationDecision(self._mode, 0.0, True)

        if self._mode is NavigationMode.TRANSITIONING:
            assert self._transition_start_s is not None
            elapsed_s = max(0.0, timestamp_s - self._transition_start_s)
            weight = min(1.0, elapsed_s / self.transition_duration_s)
            if weight >= 1.0:
                self._mode = NavigationMode.GNSS_AIDED
                self._transition_start_s = None
            return ArbitrationDecision(self._mode, weight, True)

        return ArbitrationDecision(NavigationMode.GNSS_AIDED, 1.0, True)
