from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.ins.iekf import (
    ERROR_DIM,
    IEKFInput,
    IEKFNoise,
    IEKFResult,
    IEKFState,
    normalize_rotation,
    propagate_state,
    pseudo_measurement_model,
    update_nonholonomic,
)
from src.preprocessing.frames import bearing_deg_to_enu_velocity, vehicle_to_enu_rotation


def run_adaptive_iekf(
    data: IEKFInput,
    adapter_output: np.ndarray,
    noise: IEKFNoise,
) -> IEKFResult:
    """Run the production NumPy IEKF with per-sample learned filter parameters.

    Adapter output is the bounded 11-channel representation emitted by
    ``DilatedCNNAdapter``. It cannot directly set position, velocity, or attitude.
    """

    time = np.asarray(data.time_s, dtype=float)
    gyro = np.asarray(data.gyro_body_rps, dtype=float)
    force = np.asarray(data.specific_force_body_mps2, dtype=float)
    output = np.asarray(adapter_output, dtype=float)
    if output.shape != (len(time), 11):
        raise ValueError("adapter output must have shape (N, 11)")
    if np.any(np.diff(time) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    measurement_scale = np.exp(np.log(10.0) * output[:, 0:2])
    gyro_delta = 0.02 * output[:, 2:5]
    accel_delta = 0.50 * output[:, 5:8]
    accel_scale = 1.0 + 0.10 * output[:, 8:11]
    car_to_imu = normalize_rotation(np.asarray(data.initial_rotation_car_to_imu, dtype=float))
    state = IEKFState(
        rotation_imu_to_enu=vehicle_to_enu_rotation(data.initial_bearing_deg) @ car_to_imu.T,
        velocity_enu_mps=bearing_deg_to_enu_velocity(
            data.initial_speed_mps, data.initial_bearing_deg
        ),
        position_enu_m=np.zeros(3),
        gyro_bias_body_rps=np.asarray(data.initial_gyro_bias_body_rps, dtype=float).copy(),
        accel_bias_body_mps2=np.asarray(data.initial_accel_bias_body_mps2, dtype=float).copy(),
        rotation_car_to_imu=car_to_imu,
        translation_car_to_imu_m=np.asarray(data.initial_translation_car_to_imu_m, dtype=float).copy(),
        covariance=noise.initial_matrix(),
    )
    n = len(time)
    positions, velocities = np.zeros((n, 3)), np.zeros((n, 3))
    rotations, car_rotations = np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    gyro_biases, accel_biases = np.zeros((n, 3)), np.zeros((n, 3))
    lateral_upward, covariance_diagonal = np.zeros((n, 2)), np.zeros((n, ERROR_DIM))
    car_translations = np.zeros((n, 3))
    for index in range(n):
        corrected_gyro = gyro[index] - gyro_delta[index]
        if index:
            # Express the learned correction as an equivalent measurement so
            # the existing, tested IEKF propagation remains the source of truth.
            corrected_force = (
                (force[index] - state.accel_bias_body_mps2 - accel_delta[index])
                * accel_scale[index]
                + state.accel_bias_body_mps2
            )
            state = propagate_state(
                state,
                corrected_gyro,
                corrected_force,
                time[index] - time[index - 1],
                noise,
            )
            dynamic_noise = replace(
                noise,
                lateral_velocity=noise.lateral_velocity * measurement_scale[index, 0],
                upward_velocity=noise.upward_velocity * measurement_scale[index, 1],
            )
            state, _ = update_nonholonomic(state, corrected_gyro, dynamic_noise)
            if index % 100 == 0:
                state.rotation_imu_to_enu = normalize_rotation(state.rotation_imu_to_enu)
        prediction, _ = pseudo_measurement_model(state, corrected_gyro)
        positions[index], velocities[index] = state.position_enu_m, state.velocity_enu_mps
        rotations[index] = state.rotation_imu_to_enu
        gyro_biases[index], accel_biases[index] = state.gyro_bias_body_rps, state.accel_bias_body_mps2
        lateral_upward[index], covariance_diagonal[index] = prediction, np.diag(state.covariance)
        car_rotations[index], car_translations[index] = state.rotation_car_to_imu, state.translation_car_to_imu_m
    return IEKFResult(
        time,
        positions,
        velocities,
        rotations,
        gyro_biases,
        accel_biases,
        lateral_upward,
        covariance_diagonal,
        car_rotations,
        car_translations,
    )
