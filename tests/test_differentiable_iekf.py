import torch

from src.ins.differentiable_iekf import initialize_state, rollout, so3_exp
from src.models.adapter_cnn import AdapterOutput


def test_torch_so3_exponential_is_a_rotation():
    rotation = so3_exp(torch.tensor([[0.1, -0.2, 0.3]]))[0]
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)
    assert torch.isclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-5)


def test_rollout_backpropagates_through_measurement_and_bias_adapters():
    batch, steps = 2, 3
    raw = torch.zeros(batch, steps, 11, requires_grad=True)
    measurement_scale = torch.exp(raw[..., :2])
    adapters = AdapterOutput(
        measurement_scale,
        0.02 * raw[..., 2:5],
        0.5 * raw[..., 5:8],
        1.0 + 0.1 * raw[..., 8:11],
    )
    state = initialize_state(
        torch.full((batch,), 5.0),
        torch.zeros(batch),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.eye(3).expand(batch, -1, -1).clone(),
    )
    final = rollout(
        state,
        torch.zeros(batch, steps, 3),
        torch.tensor([0.0, 0.0, 9.80665]).expand(batch, steps, -1).clone(),
        torch.full((batch, steps), 0.1),
        adapters,
    )
    loss = final.position_enu_m.square().mean() + final.velocity_enu_mps.square().mean()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad) > 0
