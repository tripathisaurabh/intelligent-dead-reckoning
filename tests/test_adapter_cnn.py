import numpy as np
import torch

from src.models.adapter_cnn import DilatedCNNAdapter, OUTPUT_DIM


def test_adapter_is_small_and_outputs_only_bounded_filter_parameters():
    model = DilatedCNNAdapter(channels=32, dropout=0.0).eval()
    output = model(torch.randn(4, 32, 6))
    decoded = model.decode(output)

    assert output.shape == (4, OUTPUT_DIM)
    assert model.parameter_count < 50_000
    assert torch.all(decoded.measurement_scale >= 0.1)
    assert torch.all(decoded.measurement_scale <= 10.0)
    assert torch.all(decoded.accel_scale >= 0.9)
    assert torch.all(decoded.accel_scale <= 1.1)
    assert not hasattr(decoded, "position")
    assert not hasattr(decoded, "velocity")


def test_adapter_zero_initialized_head_reproduces_fixed_filter_settings():
    model = DilatedCNNAdapter().eval()
    decoded = model.decode(model(torch.randn(2, 32, 6)))

    assert torch.allclose(decoded.measurement_scale, torch.ones(2, 2))
    assert torch.allclose(decoded.gyro_bias_delta_rps, torch.zeros(2, 3))
    assert torch.allclose(decoded.accel_bias_delta_mps2, torch.zeros(2, 3))
    assert torch.allclose(decoded.accel_scale, torch.ones(2, 3))


def test_normalization_is_stored_with_model():
    model = DilatedCNNAdapter()
    model.set_normalization(np.arange(6), np.arange(1, 7))
    assert torch.allclose(model.input_mean, torch.arange(6, dtype=torch.float32))
    assert torch.allclose(model.input_std, torch.arange(1, 7, dtype=torch.float32))
