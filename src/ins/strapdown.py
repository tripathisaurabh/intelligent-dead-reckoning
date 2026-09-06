from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from src.preprocessing.frames import bearing_deg_to_enu_velocity, vehicle_to_enu_rotation


@dataclass(frozen=True)
class InsInput:
    time_s: np.ndarray
    accel_device_mps2: np.ndarray
    gravity_device_mps2: np.ndarray
    gyro_ypr_rps: np.ndarray
    accel_bias_device_mps2: np.ndarray
    gyro_bias_ypr_rps: np.ndarray
    initial_speed_mps: float
    initial_bearing_deg: float


@dataclass(frozen=True)
class InsResult:
    time_s: np.ndarray
    position_enu_m: np.ndarray
    velocity_enu_mps: np.ndarray
    linear_accel_enu_mps2: np.ndarray
    rotation_vehicle_to_enu: np.ndarray
    heading_deg: np.ndarray


def propagate_basic_ins(data: InsInput) -> InsResult:
    """Propagate an unfiltered calibrated strapdown INS without reference updates.

    Dataset contract: device axes are treated as vehicle X-forward, Y-left, Z-up.
    Gyroscope source order is yaw, pitch, roll and is reordered to body XYZ as
    roll, pitch, yaw before attitude propagation.
    """
    t = np.asarray(data.time_s, dtype=float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("INS timestamps must be strictly increasing")
    n = len(t)
    for values, name in (
        (data.accel_device_mps2, "accelerometer"),
        (data.gravity_device_mps2, "gravity"),
        (data.gyro_ypr_rps, "gyroscope"),
    ):
        if np.asarray(values).shape != (n, 3):
            raise ValueError(f"{name} must have shape (N, 3)")

    position = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    acceleration_enu = np.zeros((n, 3))
    rotations = np.zeros((n, 3, 3))
    headings = np.zeros(n)

    rotation = vehicle_to_enu_rotation(data.initial_bearing_deg)
    velocity[0] = bearing_deg_to_enu_velocity(data.initial_speed_mps, data.initial_bearing_deg)
    rotations[0] = rotation

    residual_accel = (
        np.asarray(data.accel_device_mps2)
        - np.asarray(data.gravity_device_mps2)
        - np.asarray(data.accel_bias_device_mps2)
    )
    gyro_ypr = np.asarray(data.gyro_ypr_rps) - np.asarray(data.gyro_bias_ypr_rps)
    gyro_body_xyz = gyro_ypr[:, [2, 1, 0]]
    acceleration_enu[0] = rotation @ residual_accel[0]
    headings[0] = data.initial_bearing_deg % 360.0

    for index in range(1, n):
        dt = t[index] - t[index - 1]
        omega_mid = 0.5 * (gyro_body_xyz[index - 1] + gyro_body_xyz[index])
        rotation = rotation @ Rotation.from_rotvec(omega_mid * dt).as_matrix()
        # Suppress numerical loss of orthogonality without adding a navigation correction.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        rotations[index] = rotation
        acceleration_enu[index] = rotation @ residual_accel[index]
        velocity[index] = velocity[index - 1] + 0.5 * (
            acceleration_enu[index - 1] + acceleration_enu[index]
        ) * dt
        position[index] = position[index - 1] + 0.5 * (
            velocity[index - 1] + velocity[index]
        ) * dt
        forward_enu = rotation[:, 0]
        headings[index] = np.rad2deg(np.arctan2(forward_enu[0], forward_enu[1])) % 360.0

    return InsResult(t, position, velocity, acceleration_enu, rotations, headings)
