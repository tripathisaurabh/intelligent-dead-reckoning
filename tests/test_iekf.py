import numpy as np

from src.preprocessing.frames import vehicle_to_enu_rotation
from src.ins.iekf import (
    IEKFInput,
    IEKFNoise,
    IEKFState,
    propagation_jacobians,
    se23_exp,
    so3_exp,
    update_nonholonomic,
    update_gnss,
    run_iekf,
)


def _state(velocity=(5.0, 2.0, 1.0)):
    return IEKFState(
        rotation_imu_to_enu=np.eye(3),
        velocity_enu_mps=np.array(velocity, dtype=float),
        position_enu_m=np.zeros(3),
        gyro_bias_body_rps=np.zeros(3),
        accel_bias_body_mps2=np.zeros(3),
        rotation_car_to_imu=np.eye(3),
        translation_car_to_imu_m=np.zeros(3),
        covariance=np.eye(21),
    )


def test_se23_exponential_embeds_rotation_velocity_and_position():
    xi = np.array([0.1, -0.2, 0.3, 1.0, 2.0, 3.0, -1.0, 0.5, 2.0])
    result = se23_exp(xi)
    assert result.shape == (5, 5)
    assert np.allclose(result[:3, :3], so3_exp(xi[:3]))
    assert np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(result[:3, :3]), 1.0)
    assert np.allclose(result[3:, :3], 0.0)
    assert np.allclose(result[3:, 3:], np.eye(2))


def test_se23_exponential_zero_rotation_reduces_to_vector_addition():
    xi = np.r_[np.zeros(3), [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = se23_exp(xi)
    assert np.allclose(result[:3, 3], xi[3:6])
    assert np.allclose(result[:3, 4], xi[6:9])


def test_propagation_jacobian_shapes_and_position_velocity_block():
    transition, process = propagation_jacobians(_state(), 0.1)
    assert transition.shape == (21, 21)
    assert process.shape == (21, 18)
    assert np.allclose(transition[6:9, 3:6], 0.1 * np.eye(3))
    assert np.isfinite(transition).all()
    assert np.isfinite(process).all()


def test_nonholonomic_update_reduces_lateral_velocity_and_preserves_covariance():
    state = _state()
    before = np.linalg.norm(state.velocity_enu_mps[1:])
    updated, innovation = update_nonholonomic(
        state, np.zeros(3), IEKFNoise(lateral_velocity=0.01, upward_velocity=0.01)
    )
    after = np.linalg.norm(updated.velocity_enu_mps[1:])
    assert np.allclose(innovation, [2.0, 1.0])
    assert after < before
    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(updated.covariance)) > -1e-10


def test_run_iekf_uses_supplied_car_to_imu_initial_rotation():
    angle = np.deg2rad(10.0)
    car_to_imu = so3_exp(np.array([0.0, angle, 0.0]))
    result = run_iekf(
        IEKFInput(
            time_s=np.array([0.0, 0.1]),
            gyro_body_rps=np.zeros((2, 3)),
            specific_force_body_mps2=np.tile(car_to_imu @ [0.0, 0.0, 9.80665], (2, 1)),
            initial_speed_mps=5.0,
            initial_bearing_deg=0.0,
            initial_gyro_bias_body_rps=np.zeros(3),
            initial_accel_bias_body_mps2=np.zeros(3),
            initial_rotation_car_to_imu=car_to_imu,
        )
    )

    assert np.allclose(result.rotation_car_to_imu[0], car_to_imu)
    assert np.allclose(
        result.rotation_imu_to_enu[0], vehicle_to_enu_rotation(0.0) @ car_to_imu.T
    )
    assert np.linalg.norm(result.velocity_enu_mps[1] - result.velocity_enu_mps[0]) < 1e-3


def test_gnss_update_reduces_horizontal_position_and_velocity_error():
    state = _state(velocity=(8.0, -2.0, 0.0))
    state.position_enu_m = np.array([20.0, -10.0, 0.0])
    measurement_position = np.array([10.0, -5.0, 0.0])
    measurement_velocity = np.array([6.0, 0.0, 0.0])
    before = np.linalg.norm(state.position_enu_m[:2] - measurement_position[:2])
    result = update_gnss(
        state, measurement_position, measurement_velocity,
        position_std_m=3.0, velocity_std_mps=1.0, innovation_gate=1e6,
    )
    assert result.accepted
    assert np.linalg.norm(result.state.position_enu_m[:2] - measurement_position[:2]) < before
    assert np.linalg.norm(result.state.velocity_enu_mps[:2] - measurement_velocity[:2]) < np.sqrt(8.0)
    assert np.allclose(result.state.covariance, result.state.covariance.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(result.state.covariance)) > -1e-10


def test_gnss_update_rejects_large_innovation_without_mutating_state():
    state = _state()
    result = update_gnss(
        state, np.array([1000.0, 1000.0, 0.0]), np.zeros(3),
        position_std_m=3.0, velocity_std_mps=1.0, innovation_gate=18.467,
    )
    assert not result.accepted
    assert result.state is state
    assert np.allclose(state.position_enu_m, 0.0)
