from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.preprocessing.frames import bearing_deg_to_enu_velocity, vehicle_to_enu_rotation


ERROR_DIM = 21
PROCESS_NOISE_DIM = 18


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Closed-form SO(3) exponential (Rodrigues), with a small-angle branch."""
    phi = np.asarray(phi, dtype=float)
    angle = float(np.linalg.norm(phi))
    phi_x = skew(phi)
    if angle < 1e-8:
        return np.eye(3) + phi_x + 0.5 * phi_x @ phi_x
    a = np.sin(angle) / angle
    b = (1.0 - np.cos(angle)) / angle**2
    return np.eye(3) + a * phi_x + b * (phi_x @ phi_x)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    angle = float(np.linalg.norm(phi))
    phi_x = skew(phi)
    if angle < 1e-8:
        return np.eye(3) + 0.5 * phi_x + (phi_x @ phi_x) / 6.0
    return (
        np.eye(3)
        + (1.0 - np.cos(angle)) / angle**2 * phi_x
        + (angle - np.sin(angle)) / angle**3 * (phi_x @ phi_x)
    )


def se23_exp(xi: np.ndarray) -> np.ndarray:
    """Closed-form SE_2(3) exponential from Brossard et al. Appendix B.

    xi is [rotation, velocity, position]. The returned 5x5 matrix embeds
    orientation, velocity and position as one extended pose.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (9,):
        raise ValueError("SE_2(3) tangent vector must have shape (9,)")
    jacobian = so3_left_jacobian(xi[:3])
    result = np.eye(5)
    result[:3, :3] = so3_exp(xi[:3])
    result[:3, 3] = jacobian @ xi[3:6]
    result[:3, 4] = jacobian @ xi[6:9]
    return result


def normalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation, full_matrices=False)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u) * np.linalg.det(vt)
    return u @ correction @ vt


@dataclass(frozen=True)
class IEKFNoise:
    gyro: float = 1e-3
    accelerometer: float = 1e-2
    gyro_bias: float = 6e-9
    accelerometer_bias: float = 2e-4
    car_to_imu_rotation: float = 1e-9
    car_to_imu_translation: float = 1e-9
    lateral_velocity: float = 0.2
    upward_velocity: float = 300.0
    initial_roll_pitch: float = 1e-3
    initial_velocity: float = 1e-1
    initial_gyro_bias: float = 6e-3
    initial_accelerometer_bias: float = 4e-3
    initial_car_to_imu_rotation: float = 1e-6
    initial_car_to_imu_translation: float = 5e-3

    def process_matrix(self) -> np.ndarray:
        return np.diag(
            np.repeat(
                [
                    self.gyro,
                    self.accelerometer,
                    self.gyro_bias,
                    self.accelerometer_bias,
                    self.car_to_imu_rotation,
                    self.car_to_imu_translation,
                ],
                3,
            )
        )

    def measurement_matrix(self) -> np.ndarray:
        return np.diag([self.lateral_velocity, self.upward_velocity])

    def initial_matrix(self) -> np.ndarray:
        covariance = np.zeros((ERROR_DIM, ERROR_DIM))
        covariance[:2, :2] = self.initial_roll_pitch * np.eye(2)
        covariance[3:5, 3:5] = self.initial_velocity * np.eye(2)
        covariance[9:12, 9:12] = self.initial_gyro_bias * np.eye(3)
        covariance[12:15, 12:15] = self.initial_accelerometer_bias * np.eye(3)
        covariance[15:18, 15:18] = self.initial_car_to_imu_rotation * np.eye(3)
        covariance[18:21, 18:21] = self.initial_car_to_imu_translation * np.eye(3)
        return covariance


@dataclass
class IEKFState:
    rotation_imu_to_enu: np.ndarray
    velocity_enu_mps: np.ndarray
    position_enu_m: np.ndarray
    gyro_bias_body_rps: np.ndarray
    accel_bias_body_mps2: np.ndarray
    rotation_car_to_imu: np.ndarray
    translation_car_to_imu_m: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class IEKFInput:
    time_s: np.ndarray
    gyro_body_rps: np.ndarray
    specific_force_body_mps2: np.ndarray
    initial_speed_mps: float
    initial_bearing_deg: float
    initial_gyro_bias_body_rps: np.ndarray
    initial_accel_bias_body_mps2: np.ndarray
    initial_rotation_car_to_imu: np.ndarray = field(default_factory=lambda: np.eye(3))
    initial_translation_car_to_imu_m: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class IEKFResult:
    time_s: np.ndarray
    position_enu_m: np.ndarray
    velocity_enu_mps: np.ndarray
    rotation_imu_to_enu: np.ndarray
    gyro_bias_body_rps: np.ndarray
    accel_bias_body_mps2: np.ndarray
    lateral_upward_velocity_mps: np.ndarray
    covariance_diagonal: np.ndarray
    rotation_car_to_imu: np.ndarray
    translation_car_to_imu_m: np.ndarray


@dataclass(frozen=True)
class GnssUpdateResult:
    state: IEKFState
    accepted: bool
    normalized_innovation_squared: float
    innovation: np.ndarray


def propagation_jacobians(state: IEKFState, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Right-invariant error transition and process Jacobian, paper eqs. 26-27."""
    rotation = state.rotation_imu_to_enu
    velocity = state.velocity_enu_mps
    position = state.position_enu_m
    gravity = np.array([0.0, 0.0, -9.80665])

    continuous_f = np.zeros((ERROR_DIM, ERROR_DIM))
    continuous_g = np.zeros((ERROR_DIM, PROCESS_NOISE_DIM))
    velocity_skew_rotation = skew(velocity) @ rotation
    position_skew_rotation = skew(position) @ rotation

    continuous_f[3:6, :3] = skew(gravity)
    continuous_f[6:9, 3:6] = np.eye(3)
    continuous_f[:3, 9:12] = -rotation
    continuous_f[3:6, 9:12] = -velocity_skew_rotation
    continuous_f[6:9, 9:12] = -position_skew_rotation
    continuous_f[3:6, 12:15] = -rotation

    continuous_g[:3, :3] = rotation
    continuous_g[3:6, :3] = velocity_skew_rotation
    continuous_g[6:9, :3] = position_skew_rotation
    continuous_g[3:6, 3:6] = rotation
    continuous_g[9:15, 6:12] = np.eye(6)
    continuous_g[15:18, 12:15] = np.eye(3)
    continuous_g[18:21, 15:18] = np.eye(3)

    f_dt = continuous_f * dt
    f2 = f_dt @ f_dt
    transition = np.eye(ERROR_DIM) + f_dt + 0.5 * f2 + (f2 @ f_dt) / 6.0
    return transition, continuous_g * dt


def pseudo_measurement_model(
    state: IEKFState, gyro_measurement_body_rps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted [lateral, upward] car velocity and its 2x21 Jacobian."""
    rotation = state.rotation_imu_to_enu
    car_to_imu = state.rotation_car_to_imu
    velocity_imu = rotation.T @ state.velocity_enu_mps
    omega = np.asarray(gyro_measurement_body_rps) - state.gyro_bias_body_rps
    velocity_car = car_to_imu.T @ velocity_imu + skew(state.translation_car_to_imu_m) @ omega

    rotation_body = rotation @ car_to_imu
    h_car_rotation = car_to_imu.T @ skew(velocity_imu)
    h_translation = -skew(state.translation_car_to_imu_m)
    omega_skew = skew(omega)
    jacobian = np.zeros((2, ERROR_DIM))
    jacobian[:, 3:6] = rotation_body.T[1:]
    jacobian[:, 15:18] = h_car_rotation[1:]
    jacobian[:, 9:12] = h_translation[1:]
    jacobian[:, 18:21] = -omega_skew[1:]
    return velocity_car[1:], jacobian


def apply_right_invariant_error(state: IEKFState, error: np.ndarray) -> IEKFState:
    correction = se23_exp(error[:9])
    delta_rotation = correction[:3, :3]
    return IEKFState(
        rotation_imu_to_enu=delta_rotation @ state.rotation_imu_to_enu,
        velocity_enu_mps=delta_rotation @ state.velocity_enu_mps + correction[:3, 3],
        position_enu_m=delta_rotation @ state.position_enu_m + correction[:3, 4],
        gyro_bias_body_rps=state.gyro_bias_body_rps + error[9:12],
        accel_bias_body_mps2=state.accel_bias_body_mps2 + error[12:15],
        rotation_car_to_imu=so3_exp(error[15:18]) @ state.rotation_car_to_imu,
        translation_car_to_imu_m=state.translation_car_to_imu_m + error[18:21],
        covariance=state.covariance,
    )


def propagate_state(
    state: IEKFState,
    gyro_measurement_body_rps: np.ndarray,
    specific_force_body_mps2: np.ndarray,
    dt: float,
    noise: IEKFNoise,
    gravity_mps2: float = 9.80665,
) -> IEKFState:
    transition, process_jacobian = propagation_jacobians(state, dt)
    acceleration_enu = state.rotation_imu_to_enu @ (
        specific_force_body_mps2 - state.accel_bias_body_mps2
    ) + np.array([0.0, 0.0, -gravity_mps2])
    omega = gyro_measurement_body_rps - state.gyro_bias_body_rps
    position = state.position_enu_m + state.velocity_enu_mps * dt + 0.5 * acceleration_enu * dt**2
    velocity = state.velocity_enu_mps + acceleration_enu * dt
    rotation = state.rotation_imu_to_enu @ so3_exp(omega * dt)
    predicted_covariance = transition @ (
        state.covariance + process_jacobian @ noise.process_matrix() @ process_jacobian.T
    ) @ transition.T
    return IEKFState(
        normalize_rotation(rotation),
        velocity,
        position,
        state.gyro_bias_body_rps.copy(),
        state.accel_bias_body_mps2.copy(),
        state.rotation_car_to_imu.copy(),
        state.translation_car_to_imu_m.copy(),
        0.5 * (predicted_covariance + predicted_covariance.T),
    )


def update_nonholonomic(
    state: IEKFState,
    gyro_measurement_body_rps: np.ndarray,
    noise: IEKFNoise,
) -> tuple[IEKFState, np.ndarray]:
    prediction, jacobian = pseudo_measurement_model(state, gyro_measurement_body_rps)
    measurement_covariance = noise.measurement_matrix()
    innovation_covariance = jacobian @ state.covariance @ jacobian.T + measurement_covariance
    gain = np.linalg.solve(
        innovation_covariance, (state.covariance @ jacobian.T).T
    ).T
    error = gain @ (-prediction)
    updated = apply_right_invariant_error(state, error)
    identity_minus_kh = np.eye(ERROR_DIM) - gain @ jacobian
    covariance = (
        identity_minus_kh @ state.covariance @ identity_minus_kh.T
        + gain @ measurement_covariance @ gain.T
    )
    updated.covariance = 0.5 * (covariance + covariance.T)
    return updated, prediction


def gnss_measurement_model(state: IEKFState) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal GNSS position/velocity prediction and right-invariant Jacobian."""
    prediction = np.r_[state.position_enu_m[:2], state.velocity_enu_mps[:2]]
    jacobian = np.zeros((4, ERROR_DIM))
    jacobian[:2, :3] = -skew(state.position_enu_m)[:2]
    jacobian[:2, 6:9] = np.eye(3)[:2]
    jacobian[2:, :3] = -skew(state.velocity_enu_mps)[:2]
    jacobian[2:, 3:6] = np.eye(3)[:2]
    return prediction, jacobian


def update_gnss(
    state: IEKFState,
    position_enu_m: np.ndarray,
    velocity_enu_mps: np.ndarray,
    position_std_m: float,
    velocity_std_mps: float,
    innovation_gate: float = 18.467,
    measurement_weight: float = 1.0,
) -> GnssUpdateResult:
    """Fuse a horizontal GNSS fix, rejecting it when its chi-square NIS is too large.

    ``measurement_weight`` scales the Kalman gain. It can therefore distribute a
    validated correction across several samples during smooth reacquisition.
    """
    position = np.asarray(position_enu_m, dtype=float)
    velocity = np.asarray(velocity_enu_mps, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("GNSS position and velocity must have shape (3,)")
    scalars = (position_std_m, velocity_std_mps, innovation_gate, measurement_weight)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("GNSS update parameters must be finite")
    if position_std_m <= 0 or velocity_std_mps <= 0 or innovation_gate <= 0:
        raise ValueError("GNSS standard deviations and innovation gate must be positive")
    if not 0.0 <= measurement_weight <= 1.0:
        raise ValueError("measurement_weight must lie in [0, 1]")

    prediction, jacobian = gnss_measurement_model(state)
    innovation = np.r_[position[:2], velocity[:2]] - prediction
    base_covariance = np.diag(
        [position_std_m**2, position_std_m**2, velocity_std_mps**2, velocity_std_mps**2]
    )
    innovation_covariance = jacobian @ state.covariance @ jacobian.T + base_covariance
    nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
    if nis > innovation_gate or measurement_weight == 0.0:
        return GnssUpdateResult(state, nis <= innovation_gate, nis, innovation)

    measurement_covariance = base_covariance
    innovation_covariance = jacobian @ state.covariance @ jacobian.T + measurement_covariance
    gain = np.linalg.solve(
        innovation_covariance, (state.covariance @ jacobian.T).T
    ).T * measurement_weight
    updated = apply_right_invariant_error(state, gain @ innovation)
    identity_minus_kh = np.eye(ERROR_DIM) - gain @ jacobian
    covariance = (
        identity_minus_kh @ state.covariance @ identity_minus_kh.T
        + gain @ measurement_covariance @ gain.T
    )
    updated.covariance = 0.5 * (covariance + covariance.T)
    return GnssUpdateResult(updated, True, nis, innovation)


def run_iekf(data: IEKFInput, noise: IEKFNoise | None = None) -> IEKFResult:
    noise = noise or IEKFNoise()
    time = np.asarray(data.time_s, dtype=float)
    gyro = np.asarray(data.gyro_body_rps, dtype=float)
    specific_force = np.asarray(data.specific_force_body_mps2, dtype=float)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("IEKF timestamps must be strictly increasing")
    if gyro.shape != (len(time), 3) or specific_force.shape != (len(time), 3):
        raise ValueError("IEKF IMU arrays must have shape (N, 3)")

    initial_car_to_imu = normalize_rotation(
        np.asarray(data.initial_rotation_car_to_imu, dtype=float)
    )
    if initial_car_to_imu.shape != (3, 3):
        raise ValueError("initial car-to-IMU rotation must have shape (3, 3)")
    initial_translation = np.asarray(data.initial_translation_car_to_imu_m, dtype=float)
    if initial_translation.shape != (3,):
        raise ValueError("initial car-to-IMU translation must have shape (3,)")

    state = IEKFState(
        rotation_imu_to_enu=(
            vehicle_to_enu_rotation(data.initial_bearing_deg) @ initial_car_to_imu.T
        ),
        velocity_enu_mps=bearing_deg_to_enu_velocity(
            data.initial_speed_mps, data.initial_bearing_deg
        ),
        position_enu_m=np.zeros(3),
        gyro_bias_body_rps=np.asarray(data.initial_gyro_bias_body_rps, dtype=float).copy(),
        accel_bias_body_mps2=np.asarray(data.initial_accel_bias_body_mps2, dtype=float).copy(),
        rotation_car_to_imu=initial_car_to_imu,
        translation_car_to_imu_m=initial_translation.copy(),
        covariance=noise.initial_matrix(),
    )
    n = len(time)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    rotations = np.zeros((n, 3, 3))
    gyro_biases = np.zeros((n, 3))
    accel_biases = np.zeros((n, 3))
    lateral_upward = np.zeros((n, 2))
    covariance_diagonal = np.zeros((n, ERROR_DIM))
    car_rotations = np.zeros((n, 3, 3))
    car_translations = np.zeros((n, 3))

    for index in range(n):
        if index:
            state = propagate_state(
                state, gyro[index], specific_force[index], time[index] - time[index - 1], noise
            )
            state, _ = update_nonholonomic(state, gyro[index], noise)
            if index % 100 == 0:
                state.rotation_imu_to_enu = normalize_rotation(state.rotation_imu_to_enu)
        prediction, _ = pseudo_measurement_model(state, gyro[index])
        positions[index] = state.position_enu_m
        velocities[index] = state.velocity_enu_mps
        rotations[index] = state.rotation_imu_to_enu
        gyro_biases[index] = state.gyro_bias_body_rps
        accel_biases[index] = state.accel_bias_body_mps2
        lateral_upward[index] = prediction
        covariance_diagonal[index] = np.diag(state.covariance)
        car_rotations[index] = state.rotation_car_to_imu
        car_translations[index] = state.translation_car_to_imu_m

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
