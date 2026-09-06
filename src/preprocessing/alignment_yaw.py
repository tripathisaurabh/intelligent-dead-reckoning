from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GpsHeadingYawAlignment:
    """Causal car-to-phone yaw inferred during straight-line acceleration."""

    start_index: int
    end_index_exclusive: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    yaw_deg: float
    mean_horizontal_acceleration_imu_mps2: np.ndarray
    speed_start_mps: float
    speed_end_mps: float
    speed_gain_mps: float
    displacement_m: float
    displacement_course_deg: float
    mean_gps_heading_deg: float
    course_heading_difference_deg: float
    yaw_rate_rms_rps: float
    phone_orientation_pitch_median_deg: float
    phone_orientation_near_gimbal_lock: bool
    rotation_car_to_imu: np.ndarray


def _rotation_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _circular_mean_deg(angle_deg: np.ndarray) -> float:
    angle = np.deg2rad(np.asarray(angle_deg, dtype=float))
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * angle)))) % 360.0)


def _angle_difference_deg(first_deg: float, second_deg: float) -> float:
    return float((first_deg - second_deg + 180.0) % 360.0 - 180.0)


def estimate_gps_heading_mounting_yaw(
    time_s: np.ndarray,
    acceleration_imu_mps2: np.ndarray,
    gravity_imu_mps2: np.ndarray,
    gyro_imu_rps: np.ndarray,
    reference_enu_m: np.ndarray,
    gps_speed_mps: np.ndarray,
    gps_heading_deg: np.ndarray,
    phone_orientation_ypr_deg: np.ndarray,
    *,
    blackout_start_index: int,
    initial_rotation_car_to_imu: np.ndarray,
    accelerometer_bias_imu_mps2: np.ndarray | None = None,
    earliest_index: int = 0,
    window_duration_s: float = 2.0,
    minimum_start_speed_mps: float = 0.5,
    minimum_end_speed_mps: float = 4.0,
    minimum_speed_gain_mps: float = 3.0,
    minimum_displacement_m: float = 4.0,
    maximum_course_heading_difference_deg: float = 5.0,
    maximum_yaw_rate_rms_rps: float = 0.08,
) -> GpsHeadingYawAlignment:
    """Estimate mounting yaw without using any samples from the blackout.

    GPS displacement course and heading establish that the vehicle is moving
    straight forward while its speed is increasing. In that interval the mean
    gravity-removed phone acceleration is the vehicle-forward axis expressed
    in phone coordinates, so ``atan2(a_y, a_x)`` is car-to-phone mounting yaw.

    Android Euler yaw is deliberately not used at near-vertical pitch, where
    its yaw/roll decomposition is gimbal-locked. The pitch diagnostic makes
    that exclusion explicit rather than silently trusting an invalid heading.
    """

    time = np.asarray(time_s, dtype=float)
    acceleration = np.asarray(acceleration_imu_mps2, dtype=float)
    gravity = np.asarray(gravity_imu_mps2, dtype=float)
    gyro = np.asarray(gyro_imu_rps, dtype=float)
    enu = np.asarray(reference_enu_m, dtype=float)
    speed = np.asarray(gps_speed_mps, dtype=float)
    heading = np.asarray(gps_heading_deg, dtype=float)
    orientation = np.asarray(phone_orientation_ypr_deg, dtype=float)
    n = len(time)
    if not (1 < blackout_start_index <= n):
        raise ValueError("blackout_start_index must be within the recording")
    if not (0 <= earliest_index < blackout_start_index):
        raise ValueError("earliest_index must precede the blackout")
    if np.any(np.diff(time) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    for name, values, columns in (
        ("acceleration", acceleration, 3),
        ("gravity", gravity, 3),
        ("gyroscope", gyro, 3),
        ("reference ENU", enu, 3),
        ("phone orientation", orientation, 3),
    ):
        if values.shape != (n, columns):
            raise ValueError(f"{name} must have shape (N, 3)")
    if speed.shape != (n,) or heading.shape != (n,):
        raise ValueError("GPS speed and heading must have shape (N,)")
    if not all(np.isfinite(x).all() for x in (time, acceleration, gravity, gyro, enu, speed, heading)):
        raise ValueError("alignment inputs must be finite")

    bias = np.zeros(3) if accelerometer_bias_imu_mps2 is None else np.asarray(
        accelerometer_bias_imu_mps2, dtype=float
    )
    if bias.shape != (3,):
        raise ValueError("accelerometer bias must have shape (3,)")

    candidates: list[tuple[int, int, float, float, float]] = []
    # Scan by endpoint and select the latest qualifying window. This is causal:
    # every candidate ends strictly before the first blackout sample.
    for end in range(earliest_index + 2, blackout_start_index + 1):
        target_start = time[end - 1] - window_duration_s
        start = int(np.searchsorted(time, target_start, side="left"))
        start = max(start, earliest_index)
        if end - start < 2 or time[end - 1] - time[start] < window_duration_s - 1e-6:
            continue
        delta = enu[end - 1, :2] - enu[start, :2]
        displacement = float(np.linalg.norm(delta))
        course = float(np.rad2deg(np.arctan2(delta[0], delta[1])) % 360.0)
        mean_heading = _circular_mean_deg(heading[start:end])
        course_difference = _angle_difference_deg(course, mean_heading)
        yaw_rate_rms = float(np.sqrt(np.mean(gyro[start:end, 2] ** 2)))
        if (
            speed[start] >= minimum_start_speed_mps
            and speed[end - 1] >= minimum_end_speed_mps
            and speed[end - 1] - speed[start] >= minimum_speed_gain_mps
            and displacement >= minimum_displacement_m
            and abs(course_difference) <= maximum_course_heading_difference_deg
            and yaw_rate_rms <= maximum_yaw_rate_rms_rps
        ):
            candidates.append((start, end, displacement, course, mean_heading))

    if not candidates:
        raise ValueError("no causal straight accelerating GPS-heading window was detected")
    start, end, displacement, course, mean_heading = candidates[-1]
    linear_acceleration = acceleration[start:end] - gravity[start:end] - bias
    mean_horizontal = np.mean(linear_acceleration[:, :2], axis=0)
    if np.linalg.norm(mean_horizontal) < 0.2:
        raise ValueError("straight window has insufficient horizontal acceleration for yaw")
    yaw = float(np.arctan2(mean_horizontal[1], mean_horizontal[0]))
    initial_rotation = np.asarray(initial_rotation_car_to_imu, dtype=float)
    if initial_rotation.shape != (3, 3):
        raise ValueError("initial car-to-IMU rotation must have shape (3, 3)")
    pitch_median = float(np.median(orientation[start:end, 1]))
    return GpsHeadingYawAlignment(
        start_index=start,
        end_index_exclusive=end,
        start_time_s=float(time[start]),
        end_time_s=float(time[end - 1]),
        duration_s=float(time[end - 1] - time[start]),
        yaw_deg=float(np.rad2deg(yaw)),
        mean_horizontal_acceleration_imu_mps2=mean_horizontal,
        speed_start_mps=float(speed[start]),
        speed_end_mps=float(speed[end - 1]),
        speed_gain_mps=float(speed[end - 1] - speed[start]),
        displacement_m=displacement,
        displacement_course_deg=course,
        mean_gps_heading_deg=mean_heading,
        course_heading_difference_deg=_angle_difference_deg(course, mean_heading),
        yaw_rate_rms_rps=float(np.sqrt(np.mean(gyro[start:end, 2] ** 2))),
        phone_orientation_pitch_median_deg=pitch_median,
        phone_orientation_near_gimbal_lock=abs(pitch_median) >= 80.0,
        rotation_car_to_imu=_rotation_z(yaw) @ initial_rotation,
    )
