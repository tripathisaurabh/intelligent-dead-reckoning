from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MountingAlignment:
    """Observable phone-to-car tilt estimated before dead reckoning starts.

    ``rotation_car_to_imu`` maps vectors expressed in the vehicle frame into
    the phone IMU frame. Yaw mounting is deliberately left at zero: gravity
    and a low-turn gyroscope window cannot make it observable.
    """

    start_index: int
    end_index_exclusive: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    roll_deg: float
    pitch_deg: float
    mean_gravity_imu_mps2: np.ndarray
    mean_gyro_imu_rps: np.ndarray
    yaw_rate_rms_rps: float
    rotation_car_to_imu: np.ndarray


def _rotation_y(pitch_rad: float) -> np.ndarray:
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_x(roll_rad: float) -> np.ndarray:
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def gravity_to_car_to_imu_rotation(gravity_imu_mps2: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Level a phone from gravity, returning R_car_to_imu, roll and pitch.

    The convention is ``R_car_to_imu = R_y(pitch) @ R_x(roll)`` and vehicle
    gravity is +Z, matching the IO-VNBD gravity channel. Consequently the
    matrix's third column must equal the normalized measured gravity vector.
    """

    gravity = np.asarray(gravity_imu_mps2, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity vector must contain three finite values")
    magnitude = float(np.linalg.norm(gravity))
    if magnitude < 1e-6:
        raise ValueError("gravity vector magnitude is too small")
    x, y, z = gravity / magnitude
    roll = float(np.arcsin(np.clip(-y, -1.0, 1.0)))
    pitch = float(np.arctan2(x, z))
    rotation = _rotation_y(pitch) @ _rotation_x(roll)
    return rotation, roll, pitch


def estimate_mounting_alignment(
    time_s: np.ndarray,
    gravity_imu_mps2: np.ndarray,
    gyro_imu_rps: np.ndarray,
    *,
    search_duration_s: float = 50.0,
    window_duration_s: float = 30.0,
    maximum_yaw_rate_rps: float = 0.03,
    minimum_low_yaw_fraction: float = 0.90,
    gravity_magnitude_mps2: float = 9.80665,
    gravity_tolerance_mps2: float = 0.5,
) -> MountingAlignment:
    """Find the earliest initial low-yaw window and solve observable tilt.

    Window detection uses only IMU data. The gyroscope establishes that the
    car is not turning; averaged gravity establishes phone pitch and roll.
    No VBOX/GNSS values enter this routine.
    """

    time = np.asarray(time_s, dtype=float)
    gravity = np.asarray(gravity_imu_mps2, dtype=float)
    gyro = np.asarray(gyro_imu_rps, dtype=float)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("alignment timestamps must be strictly increasing")
    if gravity.shape != (len(time), 3) or gyro.shape != (len(time), 3):
        raise ValueError("gravity and gyroscope arrays must have shape (N, 3)")
    if not (0.0 < window_duration_s <= search_duration_s):
        raise ValueError("window duration must be positive and within the search horizon")

    search_stop = int(np.searchsorted(time, time[0] + search_duration_s, side="right"))
    if search_stop < 2 or time[search_stop - 1] - time[0] < window_duration_s:
        raise ValueError("recording is too short for the requested alignment window")

    gravity_norm = np.linalg.norm(gravity, axis=1)
    candidates: list[tuple[int, int]] = []
    for start in range(search_stop):
        end = int(np.searchsorted(time, time[start] + window_duration_s, side="right"))
        end = min(end, search_stop)
        if end - start < 2 or time[end - 1] - time[start] < window_duration_s - 1e-6:
            continue
        usable = (
            np.isfinite(gravity[start:end]).all(axis=1)
            & np.isfinite(gyro[start:end]).all(axis=1)
            & (np.abs(gravity_norm[start:end] - gravity_magnitude_mps2) <= gravity_tolerance_mps2)
        )
        low_yaw = np.abs(gyro[start:end, 2]) <= maximum_yaw_rate_rps
        if np.mean(usable & low_yaw) >= minimum_low_yaw_fraction:
            candidates.append((start, end))

    if not candidates:
        raise ValueError("no initial low-yaw-rate alignment window was detected")

    # Earliest qualifying window makes the selection causal and prevents any
    # evaluation-error-driven window tuning.
    start, end = candidates[0]
    mean_gravity = np.mean(gravity[start:end], axis=0)
    mean_gyro = np.mean(gyro[start:end], axis=0)
    rotation, roll, pitch = gravity_to_car_to_imu_rotation(mean_gravity)
    return MountingAlignment(
        start_index=start,
        end_index_exclusive=end,
        start_time_s=float(time[start]),
        end_time_s=float(time[end - 1]),
        duration_s=float(time[end - 1] - time[start]),
        roll_deg=float(np.rad2deg(roll)),
        pitch_deg=float(np.rad2deg(pitch)),
        mean_gravity_imu_mps2=mean_gravity,
        mean_gyro_imu_rps=mean_gyro,
        yaw_rate_rms_rps=float(np.sqrt(np.mean(gyro[start:end, 2] ** 2))),
        rotation_car_to_imu=rotation,
    )
