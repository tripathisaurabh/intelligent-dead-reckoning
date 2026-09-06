import numpy as np
import pytest

from src.preprocessing.alignment_yaw import estimate_gps_heading_mounting_yaw


def _synthetic(yaw_deg: float = 15.0):
    time = np.arange(0.0, 5.0, 0.1)
    n = len(time)
    speed = np.zeros(n)
    speed[10:] = 2.0 * (time[10:] - time[10]) + 0.5
    heading = np.full(n, 30.0)
    bearing = np.deg2rad(30.0)
    velocity = np.column_stack((speed * np.sin(bearing), speed * np.cos(bearing)))
    enu = np.zeros((n, 3))
    enu[1:, :2] = np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * 0.1, axis=0)
    gravity = np.tile([0.0, 0.0, 9.80665], (n, 1))
    acceleration = gravity.copy()
    yaw = np.deg2rad(yaw_deg)
    acceleration[10:, :2] += 2.0 * np.array([np.cos(yaw), np.sin(yaw)])
    gyro = np.zeros((n, 3))
    orientation = np.tile([0.0, -89.0, 0.0], (n, 1))
    return time, acceleration, gravity, gyro, enu, speed, heading, orientation


def test_recovers_known_car_to_phone_yaw_before_blackout():
    values = _synthetic(15.0)
    estimate = estimate_gps_heading_mounting_yaw(
        *values,
        blackout_start_index=40,
        earliest_index=10,
        initial_rotation_car_to_imu=np.eye(3),
        window_duration_s=2.0,
        minimum_end_speed_mps=4.0,
        minimum_speed_gain_mps=3.0,
        minimum_displacement_m=4.0,
    )
    assert estimate.end_index_exclusive <= 40
    assert estimate.yaw_deg == pytest.approx(15.0, abs=1e-10)
    assert estimate.phone_orientation_near_gimbal_lock
    expected_forward = [np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0]
    assert np.allclose(estimate.rotation_car_to_imu[:, 0], expected_forward)


def test_rejects_trace_without_speed_gain():
    values = list(_synthetic())
    values[5][:] = 3.0
    with pytest.raises(ValueError, match="no causal"):
        estimate_gps_heading_mounting_yaw(
            *values,
            blackout_start_index=40,
            earliest_index=10,
            initial_rotation_car_to_imu=np.eye(3),
        )


def test_rejects_window_with_turning_gyro():
    values = list(_synthetic())
    values[3][:, 2] = 0.2
    with pytest.raises(ValueError, match="no causal"):
        estimate_gps_heading_mounting_yaw(
            *values,
            blackout_start_index=40,
            earliest_index=10,
            initial_rotation_car_to_imu=np.eye(3),
        )
