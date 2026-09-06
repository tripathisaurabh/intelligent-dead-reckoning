import numpy as np

from src.ins.strapdown import InsInput, propagate_basic_ins


def _input(accel_x=0.0, gyro_yaw=0.0):
    time = np.arange(0.0, 10.1, 0.1)
    gravity = np.tile([0.0, 0.0, 9.80665], (len(time), 1))
    accel = gravity.copy()
    accel[:, 0] += accel_x
    gyro = np.zeros((len(time), 3))
    gyro[:, 0] = gyro_yaw
    return InsInput(time, accel, gravity, gyro, np.zeros(3), np.zeros(3), 0.0, 90.0)


def test_zero_motion_remains_at_origin():
    result = propagate_basic_ins(_input())
    assert np.max(np.abs(result.position_enu_m)) < 1e-10
    assert np.max(np.abs(result.velocity_enu_mps)) < 1e-10


def test_constant_forward_acceleration_matches_kinematics():
    result = propagate_basic_ins(_input(accel_x=1.0))
    assert np.isclose(result.velocity_enu_mps[-1, 0], 10.0, atol=1e-8)
    assert np.isclose(result.position_enu_m[-1, 0], 50.0, atol=1e-8)


def test_zero_gyro_keeps_heading_stable():
    result = propagate_basic_ins(_input())
    assert np.allclose(result.heading_deg, 90.0)


def test_ins_input_has_no_reference_measurement_fields():
    assert not any("reference" in name or "gnss" in name or "vbox" in name for name in InsInput.__dataclass_fields__)
