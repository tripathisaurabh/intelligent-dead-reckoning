from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlackoutInterval:
    start_index: int
    end_index_inclusive: int
    start_time_s: float
    end_time_s: float
    duration_s: float


def select_blackout_interval(
    time_s: np.ndarray,
    reference_speed_mps: np.ndarray,
    earliest_index: int,
    duration_s: float,
    minimum_speed_mps: float,
    required_moving_fraction: float,
    maximum_gap_s: float,
    context_before_s: float,
    context_after_s: float,
) -> BlackoutInterval:
    """Select the earliest qualifying interval, independent of INS performance."""
    t = np.asarray(time_s, dtype=float)
    dt = np.diff(t)
    for start in range(max(earliest_index, 1), len(t) - 1):
        if t[start] - t[0] < context_before_s:
            continue
        target_end = t[start] + duration_s
        end = int(np.searchsorted(t, target_end, side="left"))
        if end >= len(t) or t[-1] - t[end] < context_after_s:
            break
        if np.any(dt[start:end] > maximum_gap_s):
            continue
        if (
            reference_speed_mps[start] < minimum_speed_mps
            or reference_speed_mps[end] < minimum_speed_mps
        ):
            continue
        moving_fraction = float(np.mean(reference_speed_mps[start : end + 1] >= minimum_speed_mps))
        if moving_fraction >= required_moving_fraction:
            return BlackoutInterval(
                start_index=start,
                end_index_inclusive=end,
                start_time_s=float(t[start]),
                end_time_s=float(t[end]),
                duration_s=float(t[end] - t[start]),
            )
    raise ValueError("No blackout interval satisfies the predefined selection criteria")
