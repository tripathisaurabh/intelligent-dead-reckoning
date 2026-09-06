from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.adapter_cnn import AdapterOutput


@dataclass
class TorchIEKFState:
    rotation_imu_to_enu: torch.Tensor
    velocity_enu_mps: torch.Tensor
    position_enu_m: torch.Tensor
    gyro_bias_body_rps: torch.Tensor
    accel_bias_body_mps2: torch.Tensor
    rotation_car_to_imu: torch.Tensor
    translation_car_to_imu_m: torch.Tensor
    covariance: torch.Tensor


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    angle2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    angle = torch.sqrt(angle2.clamp_min(1e-12))
    small = angle2 < 1e-8
    a = torch.where(small, 1.0 - angle2 / 6.0, torch.sin(angle) / angle)
    b = torch.where(
        small,
        0.5 - angle2 / 24.0,
        (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    phi_x = skew(phi)
    identity = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(phi_x.shape)
    return identity + a[..., None] * phi_x + b[..., None] * (phi_x @ phi_x)


def so3_left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    angle2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    angle = torch.sqrt(angle2.clamp_min(1e-12))
    small = angle2 < 1e-8
    a = torch.where(
        small,
        0.5 - angle2 / 24.0,
        (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    b = torch.where(
        small,
        1.0 / 6.0 - angle2 / 120.0,
        (angle - torch.sin(angle)) / (angle2 * angle).clamp_min(1e-12),
    )
    phi_x = skew(phi)
    identity = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(phi_x.shape)
    return identity + a[..., None] * phi_x + b[..., None] * (phi_x @ phi_x)


def vehicle_to_enu_rotation(bearing_deg: torch.Tensor) -> torch.Tensor:
    bearing = torch.deg2rad(bearing_deg)
    sine, cosine = torch.sin(bearing), torch.cos(bearing)
    zero, one = torch.zeros_like(sine), torch.ones_like(sine)
    return torch.stack(
        (sine, -cosine, zero, cosine, sine, zero, zero, zero, one), dim=-1
    ).reshape(-1, 3, 3)


def initial_covariance(batch: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    diagonal = torch.zeros(21, dtype=dtype, device=device)
    diagonal[0:2] = 1e-3
    diagonal[3:5] = 1e-1
    diagonal[9:12] = 6e-3
    diagonal[12:15] = 4e-3
    diagonal[15:18] = 1e-6
    diagonal[18:21] = 5e-3
    return torch.diag(diagonal).expand(batch, -1, -1).clone()


def initialize_state(
    initial_speed_mps: torch.Tensor,
    initial_bearing_deg: torch.Tensor,
    gyro_bias_body_rps: torch.Tensor,
    accel_bias_body_mps2: torch.Tensor,
    rotation_car_to_imu: torch.Tensor,
) -> TorchIEKFState:
    batch = initial_speed_mps.shape[0]
    car_to_enu = vehicle_to_enu_rotation(initial_bearing_deg)
    bearing = torch.deg2rad(initial_bearing_deg)
    velocity = torch.stack(
        (
            initial_speed_mps * torch.sin(bearing),
            initial_speed_mps * torch.cos(bearing),
            torch.zeros_like(initial_speed_mps),
        ),
        dim=-1,
    )
    return TorchIEKFState(
        rotation_imu_to_enu=car_to_enu @ rotation_car_to_imu.transpose(-1, -2),
        velocity_enu_mps=velocity,
        position_enu_m=torch.zeros(batch, 3, dtype=velocity.dtype, device=velocity.device),
        gyro_bias_body_rps=gyro_bias_body_rps,
        accel_bias_body_mps2=accel_bias_body_mps2,
        rotation_car_to_imu=rotation_car_to_imu,
        translation_car_to_imu_m=torch.zeros_like(velocity),
        covariance=initial_covariance(batch, dtype=velocity.dtype, device=velocity.device),
    )


def _propagation_matrices(state: TorchIEKFState, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, dtype, device = len(dt), dt.dtype, dt.device
    f = torch.zeros(batch, 21, 21, dtype=dtype, device=device)
    g = torch.zeros(batch, 21, 18, dtype=dtype, device=device)
    gravity = torch.tensor([0.0, 0.0, -9.80665], dtype=dtype, device=device).expand(batch, -1)
    velocity_sr = skew(state.velocity_enu_mps) @ state.rotation_imu_to_enu
    position_sr = skew(state.position_enu_m) @ state.rotation_imu_to_enu
    identity3 = torch.eye(3, dtype=dtype, device=device).expand(batch, -1, -1)
    f[:, 3:6, 0:3] = skew(gravity)
    f[:, 6:9, 3:6] = identity3
    f[:, 0:3, 9:12] = -state.rotation_imu_to_enu
    f[:, 3:6, 9:12] = -velocity_sr
    f[:, 6:9, 9:12] = -position_sr
    f[:, 3:6, 12:15] = -state.rotation_imu_to_enu
    g[:, 0:3, 0:3] = state.rotation_imu_to_enu
    g[:, 3:6, 0:3] = velocity_sr
    g[:, 6:9, 0:3] = position_sr
    g[:, 3:6, 3:6] = state.rotation_imu_to_enu
    g[:, 9:15, 6:12] = torch.eye(6, dtype=dtype, device=device).expand(batch, -1, -1)
    g[:, 15:18, 12:15] = identity3
    g[:, 18:21, 15:18] = identity3
    fdt = f * dt[:, None, None]
    f2 = fdt @ fdt
    identity21 = torch.eye(21, dtype=dtype, device=device).expand(batch, -1, -1)
    transition = identity21 + fdt + 0.5 * f2 + (f2 @ fdt) / 6.0
    return transition, g * dt[:, None, None]


def propagate(
    state: TorchIEKFState,
    gyro_body_rps: torch.Tensor,
    specific_force_body_mps2: torch.Tensor,
    dt: torch.Tensor,
    adapter: AdapterOutput,
) -> TorchIEKFState:
    transition, process = _propagation_matrices(state, dt)
    process_diagonal = torch.tensor(
        [1e-3] * 3 + [1e-2] * 3 + [6e-9] * 3 + [2e-4] * 3 + [1e-9] * 6,
        dtype=dt.dtype,
        device=dt.device,
    )
    process_covariance = torch.diag(process_diagonal).expand(len(dt), -1, -1)
    covariance = transition @ (
        state.covariance + process @ process_covariance @ process.transpose(-1, -2)
    ) @ transition.transpose(-1, -2)

    omega = gyro_body_rps - state.gyro_bias_body_rps - adapter.gyro_bias_delta_rps
    corrected_force = (
        specific_force_body_mps2
        - state.accel_bias_body_mps2
        - adapter.accel_bias_delta_mps2
    ) * adapter.accel_scale
    gravity = torch.tensor([0.0, 0.0, -9.80665], dtype=dt.dtype, device=dt.device)
    acceleration = (state.rotation_imu_to_enu @ corrected_force[..., None]).squeeze(-1) + gravity
    position = (
        state.position_enu_m
        + state.velocity_enu_mps * dt[:, None]
        + 0.5 * acceleration * dt[:, None] ** 2
    )
    velocity = state.velocity_enu_mps + acceleration * dt[:, None]
    rotation = state.rotation_imu_to_enu @ so3_exp(omega * dt[:, None])
    return TorchIEKFState(
        rotation,
        velocity,
        position,
        state.gyro_bias_body_rps,
        state.accel_bias_body_mps2,
        state.rotation_car_to_imu,
        state.translation_car_to_imu_m,
        0.5 * (covariance + covariance.transpose(-1, -2)),
    )


def update_nonholonomic(
    state: TorchIEKFState,
    gyro_body_rps: torch.Tensor,
    measurement_scale: torch.Tensor,
) -> TorchIEKFState:
    batch, dtype, device = len(gyro_body_rps), gyro_body_rps.dtype, gyro_body_rps.device
    velocity_imu = (
        state.rotation_imu_to_enu.transpose(-1, -2) @ state.velocity_enu_mps[..., None]
    ).squeeze(-1)
    omega = gyro_body_rps - state.gyro_bias_body_rps
    velocity_car = (
        state.rotation_car_to_imu.transpose(-1, -2) @ velocity_imu[..., None]
    ).squeeze(-1) + (skew(state.translation_car_to_imu_m) @ omega[..., None]).squeeze(-1)
    prediction = velocity_car[:, 1:]

    h = torch.zeros(batch, 2, 21, dtype=dtype, device=device)
    rotation_body = state.rotation_imu_to_enu @ state.rotation_car_to_imu
    h[:, :, 3:6] = rotation_body.transpose(-1, -2)[:, 1:, :]
    h[:, :, 15:18] = (
        state.rotation_car_to_imu.transpose(-1, -2) @ skew(velocity_imu)
    )[:, 1:, :]
    h[:, :, 9:12] = (-skew(state.translation_car_to_imu_m))[:, 1:, :]
    h[:, :, 18:21] = (-skew(omega))[:, 1:, :]
    measurement_covariance = torch.diag_embed(10.0 * measurement_scale)
    innovation_covariance = h @ state.covariance @ h.transpose(-1, -2) + measurement_covariance
    gain = torch.linalg.solve(
        innovation_covariance,
        (state.covariance @ h.transpose(-1, -2)).transpose(-1, -2),
    ).transpose(-1, -2)
    error = (gain @ (-prediction)[..., None]).squeeze(-1)

    delta_rotation = so3_exp(error[:, 0:3])
    jacobian = so3_left_jacobian(error[:, 0:3])
    velocity = (delta_rotation @ state.velocity_enu_mps[..., None]).squeeze(-1) + (
        jacobian @ error[:, 3:6, None]
    ).squeeze(-1)
    position = (delta_rotation @ state.position_enu_m[..., None]).squeeze(-1) + (
        jacobian @ error[:, 6:9, None]
    ).squeeze(-1)
    car_rotation = so3_exp(error[:, 15:18]) @ state.rotation_car_to_imu
    identity = torch.eye(21, dtype=dtype, device=device).expand(batch, -1, -1)
    ikh = identity - gain @ h
    covariance = (
        ikh @ state.covariance @ ikh.transpose(-1, -2)
        + gain @ measurement_covariance @ gain.transpose(-1, -2)
    )
    return TorchIEKFState(
        delta_rotation @ state.rotation_imu_to_enu,
        velocity,
        position,
        state.gyro_bias_body_rps + error[:, 9:12],
        state.accel_bias_body_mps2 + error[:, 12:15],
        car_rotation,
        state.translation_car_to_imu_m + error[:, 18:21],
        0.5 * (covariance + covariance.transpose(-1, -2)),
    )


def rollout(
    state: TorchIEKFState,
    gyro_body_rps: torch.Tensor,
    specific_force_body_mps2: torch.Tensor,
    dt_s: torch.Tensor,
    adapters: AdapterOutput,
) -> TorchIEKFState:
    """Differentiable propagation/update sequence used by the adapter loss."""
    for index in range(gyro_body_rps.shape[1]):
        step_adapter = AdapterOutput(
            adapters.measurement_scale[:, index],
            adapters.gyro_bias_delta_rps[:, index],
            adapters.accel_bias_delta_mps2[:, index],
            adapters.accel_scale[:, index],
        )
        state = propagate(
            state,
            gyro_body_rps[:, index],
            specific_force_body_mps2[:, index],
            dt_s[:, index],
            step_adapter,
        )
        state = update_nonholonomic(
            state,
            gyro_body_rps[:, index] - step_adapter.gyro_bias_delta_rps,
            step_adapter.measurement_scale,
        )
    return state
