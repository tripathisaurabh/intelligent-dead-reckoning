from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


OUTPUT_DIM = 11


@dataclass(frozen=True)
class AdapterOutput:
    measurement_scale: torch.Tensor
    gyro_bias_delta_rps: torch.Tensor
    accel_bias_delta_mps2: torch.Tensor
    accel_scale: torch.Tensor


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation, bias=False
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = functional.pad(inputs, (self.left_padding, 0))
        return self.dropout(self.activation(self.norm(self.conv(values))))


class DilatedCNNAdapter(nn.Module):
    """Small causal IMU adapter; it never emits navigation states.

    Output channels are limited to two non-holonomic covariance multipliers,
    gyro/accelerometer bias corrections, and accelerometer scale corrections.
    """

    def __init__(self, channels: int = 32, dropout: float = 0.10):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 6
        for dilation in (1, 2, 4, 8):
            layers.append(CausalConvBlock(in_channels, channels, dilation, dropout))
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels, OUTPUT_DIM)
        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std", torch.ones(6))

        # Begin exactly at the fixed Step 3b filter: scale=1 and no IMU change.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=self.input_mean.dtype)
        std_tensor = torch.as_tensor(std, dtype=self.input_std.dtype).clamp_min(1e-6)
        if mean_tensor.shape != (6,) or std_tensor.shape != (6,):
            raise ValueError("adapter normalization must contain six channels")
        self.input_mean.copy_(mean_tensor)
        self.input_std.copy_(std_tensor)

    def forward(self, imu_window: torch.Tensor) -> torch.Tensor:
        if imu_window.ndim != 3 or imu_window.shape[-1] != 6:
            raise ValueError("IMU window must have shape (batch, time, 6)")
        normalized = (imu_window - self.input_mean) / self.input_std
        features = self.features(normalized.transpose(1, 2))[:, :, -1]
        return torch.tanh(self.head(features))

    @staticmethod
    def decode(raw_output: torch.Tensor) -> AdapterOutput:
        if raw_output.shape[-1] != OUTPUT_DIM:
            raise ValueError(f"adapter output must have {OUTPUT_DIM} channels")
        # tanh output gives covariance inflation/deflation within a factor 10.
        measurement_scale = torch.exp(np.log(10.0) * raw_output[..., 0:2])
        return AdapterOutput(
            measurement_scale=measurement_scale,
            gyro_bias_delta_rps=0.02 * raw_output[..., 2:5],
            accel_bias_delta_mps2=0.50 * raw_output[..., 5:8],
            accel_scale=1.0 + 0.10 * raw_output[..., 8:11],
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
