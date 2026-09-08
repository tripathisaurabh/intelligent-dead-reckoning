from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ForwardSpeedGRU(nn.Module):
    """Causal forward-speed estimator with heteroscedastic uncertainty."""

    def __init__(self, input_channels: int = 8, hidden_size: int = 32, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_channels,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor, initial_speed_mps: torch.Tensor):
        hidden, _ = self.gru(features)
        output = self.head(hidden)
        # Predict a correction around the last trusted GNSS speed. Softplus
        # enforces a physical non-negative forward speed.
        mean = F.softplus(initial_speed_mps[:, None] + output[..., 0])
        std = F.softplus(output[..., 1]) + 0.05
        return mean, std
