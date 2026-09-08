import torch

from src.models.forward_speed_gru import ForwardSpeedGRU


def test_forward_speed_model_outputs_positive_mean_and_uncertainty():
    model = ForwardSpeedGRU(input_channels=8, hidden_size=8, layers=1)
    mean, std = model(torch.zeros(2, 20, 8), torch.tensor([0.0, 5.0]))
    assert mean.shape == (2, 20)
    assert std.shape == (2, 20)
    assert torch.all(mean >= 0.0)
    assert torch.all(std >= 0.05)
