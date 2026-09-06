from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiasEstimate:
    start_index: int
    end_index_exclusive: int
    duration_s: float
    gyro_ypr_bias_rps: np.ndarray
    residual_accel_device_bias_mps2: np.ndarray
    stationary_linear_accel_mean_mps2: np.ndarray
    stationary_linear_accel_std_mps2: np.ndarray
    stationary_linear_accel_rms_mps2: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def stationary_mask(
    reference_speed_mps: np.ndarray,
    gyro_ypr_rps: np.ndarray,
    accel_device_mps2: np.ndarray,
    gravity_mps2: float,
    maximum_reference_speed_mps: float,
    maximum_gyro_norm_rps: float,
    gravity_magnitude_tolerance_mps2: float,
) -> np.ndarray:
    accel_norm = np.linalg.norm(accel_device_mps2, axis=1)
    gyro_norm = np.linalg.norm(gyro_ypr_rps, axis=1)
    return (
        (reference_speed_mps <= maximum_reference_speed_mps)
        & (gyro_norm <= maximum_gyro_norm_rps)
        & (np.abs(accel_norm - gravity_mps2) <= gravity_magnitude_tolerance_mps2)
    )


def estimate_biases_from_stationary_run(
    time_s: np.ndarray,
    accel_device_mps2: np.ndarray,
    gravity_device_mps2: np.ndarray,
    gyro_ypr_rps: np.ndarray,
    mask: np.ndarray,
    minimum_duration_s: float,
) -> BiasEstimate:
    candidates = [
        (start, end)
        for start, end in _runs(mask)
        if end - start > 1 and time_s[end - 1] - time_s[start] >= minimum_duration_s
    ]
    if not candidates:
        raise ValueError("No stationary run satisfies the calibration duration")
    # Longest qualifying run gives the most stable mean. Ties retain the earlier run.
    start, end = max(candidates, key=lambda bounds: time_s[bounds[1] - 1] - time_s[bounds[0]])
    residual = accel_device_mps2[start:end] - gravity_device_mps2[start:end]
    accel_bias = np.mean(residual, axis=0)
    corrected = residual - accel_bias
    return BiasEstimate(
        start_index=int(start),
        end_index_exclusive=int(end),
        duration_s=float(time_s[end - 1] - time_s[start]),
        gyro_ypr_bias_rps=np.mean(gyro_ypr_rps[start:end], axis=0),
        residual_accel_device_bias_mps2=accel_bias,
        stationary_linear_accel_mean_mps2=np.mean(corrected, axis=0),
        stationary_linear_accel_std_mps2=np.std(corrected, axis=0),
        stationary_linear_accel_rms_mps2=float(np.sqrt(np.mean(np.sum(corrected**2, axis=1)))),
    )


# Compatibility name for callers created during the first Phase 1 iteration.
estimate_biases_from_first_stationary_run = estimate_biases_from_stationary_run
