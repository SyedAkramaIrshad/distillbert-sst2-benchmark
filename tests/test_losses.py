import torch
import pytest

from mini_distill.losses import compute_kd_loss, compute_total_loss


def test_kd_loss_positive():
    s = torch.tensor([[2.0, 0.5], [0.2, 1.3]], dtype=torch.float32)
    t = torch.tensor([[2.2, 0.1], [0.0, 1.8]], dtype=torch.float32)
    loss = compute_kd_loss(s, t, temperature=2.0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_total_loss_shapes_and_values():
    s = torch.tensor([[2.0, 0.5], [0.2, 1.3]], dtype=torch.float32)
    t = torch.tensor([[2.2, 0.1], [0.0, 1.8]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    total, ce, kd = compute_total_loss(s, t, y, alpha=0.7, temperature=2.0)
    assert total.ndim == 0
    assert ce.ndim == 0
    assert kd.ndim == 0
    assert total.item() >= 0.0


def test_invalid_alpha_raises():
    s = torch.randn(2, 2)
    t = torch.randn(2, 2)
    y = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError):
        compute_total_loss(s, t, y, alpha=1.5, temperature=2.0)


def test_invalid_temperature_raises():
    s = torch.randn(2, 2)
    t = torch.randn(2, 2)
    with pytest.raises(ValueError):
        compute_kd_loss(s, t, temperature=0)
