import numpy as np

from src.ins.adaptive_iekf import run_adaptive_iekf
from src.ins.iekf import IEKFInput, IEKFNoise, run_iekf


def test_zero_adapter_output_reproduces_fixed_iekf():
    data = IEKFInput(
        time_s=np.arange(0.0, 0.6, 0.1),
        gyro_body_rps=np.zeros((6, 3)),
        specific_force_body_mps2=np.tile([0.0, 0.0, 9.80665], (6, 1)),
        initial_speed_mps=5.0,
        initial_bearing_deg=20.0,
        initial_gyro_bias_body_rps=np.zeros(3),
        initial_accel_bias_body_mps2=np.zeros(3),
    )
    noise = IEKFNoise(lateral_velocity=10.0, upward_velocity=10.0)
    fixed = run_iekf(data, noise)
    adaptive = run_adaptive_iekf(data, np.zeros((6, 11)), noise)
    assert np.allclose(adaptive.position_enu_m, fixed.position_enu_m)
    assert np.allclose(adaptive.velocity_enu_mps, fixed.velocity_enu_mps)
    assert np.allclose(adaptive.covariance_diagonal, fixed.covariance_diagonal)
