import numpy as np
import torch

from src.ins.adaptive_iekf import run_adaptive_iekf
from src.ins.differentiable_iekf import initialize_state, rollout
from src.ins.iekf import IEKFInput, IEKFNoise, run_iekf
from src.models.adapter_cnn import AdapterOutput


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


def test_nonzero_adapter_matches_differentiable_training_filter():
    """Prevent training/runtime drift in all four learned correction groups."""
    time = np.arange(0.0, 0.6, 0.1)
    gyro = np.array(
        [[0.01, -0.02, 0.03], [0.02, -0.01, 0.04], [0.01, 0.00, 0.05],
         [0.00, 0.01, 0.04], [-0.01, 0.02, 0.03], [-0.02, 0.01, 0.02]],
        dtype=float,
    )
    force = np.array(
        [[0.2, -0.1, 9.7], [0.3, -0.1, 9.8], [0.4, 0.0, 9.9],
         [0.2, 0.1, 9.8], [0.1, 0.0, 9.7], [0.3, -0.1, 9.8]],
        dtype=float,
    )
    raw = np.linspace(-0.4, 0.4, len(time) * 11).reshape(len(time), 11)
    gyro_bias = np.array([0.002, -0.003, 0.004])
    accel_bias = np.array([0.04, -0.03, 0.02])
    bearing = 27.0
    speed = 6.5
    car_rotation = np.array(
        [[0.98, -0.20, 0.0], [0.20, 0.98, 0.0], [0.0, 0.0, 1.0]]
    )
    # Use a proper rotation so the NumPy initializer's SVD normalization does
    # not introduce an artificial difference from the Torch initializer.
    yaw = np.deg2rad(11.5)
    car_rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    data = IEKFInput(
        time_s=time,
        gyro_body_rps=gyro,
        specific_force_body_mps2=force,
        initial_speed_mps=speed,
        initial_bearing_deg=bearing,
        initial_gyro_bias_body_rps=gyro_bias,
        initial_accel_bias_body_mps2=accel_bias,
        initial_rotation_car_to_imu=car_rotation,
    )
    runtime = run_adaptive_iekf(data, raw, IEKFNoise(lateral_velocity=10.0, upward_velocity=10.0))

    dtype = torch.float64
    torch_state = initialize_state(
        torch.tensor([speed], dtype=dtype),
        torch.tensor([bearing], dtype=dtype),
        torch.tensor(gyro_bias[None], dtype=dtype),
        torch.tensor(accel_bias[None], dtype=dtype),
        torch.tensor(car_rotation[None], dtype=dtype),
    )
    raw_steps = torch.tensor(raw[1:][None], dtype=dtype)
    adapters = AdapterOutput(
        torch.exp(np.log(10.0) * raw_steps[..., :2]),
        0.02 * raw_steps[..., 2:5],
        0.50 * raw_steps[..., 5:8],
        1.0 + 0.10 * raw_steps[..., 8:11],
    )
    trained = rollout(
        torch_state,
        torch.tensor(gyro[1:][None], dtype=dtype),
        torch.tensor(force[1:][None], dtype=dtype),
        torch.tensor(np.diff(time)[None], dtype=dtype),
        adapters,
    )

    assert np.allclose(runtime.position_enu_m[-1], trained.position_enu_m[0].detach().numpy(), atol=1e-10)
    assert np.allclose(runtime.velocity_enu_mps[-1], trained.velocity_enu_mps[0].detach().numpy(), atol=1e-10)
    assert np.allclose(runtime.rotation_imu_to_enu[-1], trained.rotation_imu_to_enu[0].detach().numpy(), atol=1e-10)
    assert np.allclose(runtime.gyro_bias_body_rps[-1], trained.gyro_bias_body_rps[0].detach().numpy(), atol=1e-10)
    assert np.allclose(runtime.accel_bias_body_mps2[-1], trained.accel_bias_body_mps2[0].detach().numpy(), atol=1e-10)
    assert np.allclose(runtime.covariance_diagonal[-1], np.diag(trained.covariance[0].detach().numpy()), atol=1e-10)
