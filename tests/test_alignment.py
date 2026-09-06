import numpy as np

from src.preprocessing.alignment import (
    estimate_mounting_alignment,
    gravity_to_car_to_imu_rotation,
)


def test_gravity_solve_recovers_known_roll_and_pitch():
    roll = np.deg2rad(8.0)
    pitch = np.deg2rad(-5.0)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    rotation = np.array(
        [[cy, sy * sx, sy * cx], [0.0, cx, -sx], [-sy, cy * sx, cy * cx]]
    )
    gravity = rotation @ np.array([0.0, 0.0, 9.80665])

    estimated, estimated_roll, estimated_pitch = gravity_to_car_to_imu_rotation(gravity)

    assert np.allclose(estimated, rotation, atol=1e-12)
    assert np.isclose(estimated_roll, roll)
    assert np.isclose(estimated_pitch, pitch)
    assert np.allclose(estimated[:, 2], gravity / np.linalg.norm(gravity))


def test_window_detection_selects_earliest_qualifying_low_yaw_window():
    time = np.arange(0.0, 12.1, 0.1)
    gravity = np.tile([0.2, -0.1, np.sqrt(9.80665**2 - 0.05)], (len(time), 1))
    gyro = np.zeros((len(time), 3))
    gyro[time < 2.0, 2] = 0.2

    result = estimate_mounting_alignment(
        time,
        gravity,
        gyro,
        search_duration_s=12.0,
        window_duration_s=4.0,
        maximum_yaw_rate_rps=0.03,
        minimum_low_yaw_fraction=1.0,
    )

    assert result.start_time_s == 2.0
    assert np.isclose(result.duration_s, 4.0)
    assert result.yaw_rate_rms_rps == 0.0
    assert np.allclose(result.rotation_car_to_imu[:, 2], gravity[0] / np.linalg.norm(gravity[0]))


def test_window_detection_rejects_continuous_turning():
    time = np.arange(0.0, 10.1, 0.1)
    gravity = np.tile([0.0, 0.0, 9.80665], (len(time), 1))
    gyro = np.tile([0.0, 0.0, 0.2], (len(time), 1))

    try:
        estimate_mounting_alignment(
            time,
            gravity,
            gyro,
            search_duration_s=10.0,
            window_duration_s=5.0,
        )
    except ValueError as error:
        assert "no initial low-yaw-rate" in str(error)
    else:
        raise AssertionError("continuous turning must not produce an alignment")
