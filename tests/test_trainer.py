"""训练器单元测试。"""

import torch
import pytest
from src.training.losses import L1Loss, SpectralAngleLoss, CombinedLoss


class TestLosses:
    def test_l1(self):
        loss_fn = L1Loss()
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_sam(self):
        loss_fn = SpectralAngleLoss()
        pred = torch.ones(2, 3, 16, 16)
        target = torch.ones(2, 3, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_combined(self):
        cfg = {"l1": 1.0, "sam": 0.1}
        loss_fn = CombinedLoss(cfg)
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() > 0
