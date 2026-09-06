import numpy as np

from src.preprocessing.stationary import estimate_biases_from_stationary_run


def test_stationary_bias_removes_gravity_aware_residual():
    time = np.arange(0.0, 10.0, 0.1)
    gravity = np.tile([0.0, 0.0, 9.80665], (len(time), 1))
    accel_bias = np.array([0.1, -0.2, 0.05])
    accel = gravity + accel_bias
    gyro_bias = np.array([0.01, -0.02, 0.03])
    gyro = np.tile(gyro_bias, (len(time), 1))
    result = estimate_biases_from_stationary_run(
        time, accel, gravity, gyro, np.ones(len(time), dtype=bool), minimum_duration_s=5.0
    )
    assert np.allclose(result.residual_accel_device_bias_mps2, accel_bias)
    assert np.allclose(result.gyro_ypr_bias_rps, gyro_bias)
    assert result.stationary_linear_accel_rms_mps2 < 1e-12
