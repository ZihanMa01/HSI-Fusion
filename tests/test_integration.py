"""集成测试 — 端到端流程。"""

import torch
import pytest
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestIntegration:
    """端到端训练 + 评估集成测试。"""

    def test_model_registry(self):
        from src.models import ModelRegistry
        models = ModelRegistry.list_models()
        assert len(models) >= 2

    def test_degradation_forward(self):
        from src.data.degradation import DegradationModel
        deg = DegradationModel({
            "psf": {"type": "gaussian", "sigma": 2.0, "kernel_size": 7},
            "srf": {"type": "nikonos", "n_msi_bands": 3},
            "noise": {"type": "none"},
        })
        hr_hsi = torch.rand(1, 31, 64, 64)
        lr_hsi, hr_msi = deg(hr_hsi, spatial_scale=4)
        assert lr_hsi.shape == (1, 31, 16, 16)
        assert hr_msi.shape == (1, 3, 64, 64)

    def test_full_pipeline(self):
        """模拟完整训练 → 评估流程（不实际训练）。"""
        from src.models import ModelRegistry

        config = {
            "model": {"n_hsi_bands": 31, "n_msi_bands": 3, "n_feat": 32},
            "spatial_scale": 4,
        }

        # 构建模型
        model_cls = ModelRegistry.get("baseline_cnn")
        model = model_cls(config)

        # 模拟数据
        lr_hsi = torch.rand(2, 31, 16, 16)
        hr_msi = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 31, 64, 64)

        # 前向
        pred = model(lr_hsi, hr_msi)
        assert pred.shape == target.shape

        # 损失
        losses = model.get_loss(pred, target, lr_hsi, hr_msi)
        assert losses["loss"].item() > 0

        # 评估
        from src.evaluation.metrics import calc_all_metrics
        metrics = calc_all_metrics(pred, target)
        for key in ["PSNR ↑", "SSIM ↑", "SAM ↓"]:
            assert key in metrics
