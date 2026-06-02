"""模型注册与基础功能测试。"""

import torch
import pytest
from src.models import ModelRegistry, BaseFusionModel


class TestModelRegistry:
    def test_list_models(self):
        models = ModelRegistry.list_models()
        assert "baseline_cnn" in models
        assert "template_method" in models

    def test_get_model(self):
        model_cls = ModelRegistry.get("baseline_cnn")
        assert issubclass(model_cls, BaseFusionModel)

    def test_get_unknown(self):
        with pytest.raises(KeyError):
            ModelRegistry.get("unknown_method")


class TestBaselineCNN:
    def setup_method(self):
        config = {
            "model": {"n_hsi_bands": 31, "n_msi_bands": 3, "n_feat": 32}
        }
        model_cls = ModelRegistry.get("baseline_cnn")
        self.model = model_cls(config)

    def test_forward_shape(self):
        lr_hsi = torch.rand(2, 31, 32, 32)
        hr_msi = torch.rand(2, 3, 128, 128)
        out = self.model(lr_hsi, hr_msi)
        assert out.shape == (2, 31, 128, 128)

    def test_get_loss(self):
        pred = torch.rand(2, 31, 128, 128)
        target = torch.rand(2, 31, 128, 128)
        losses = self.model.get_loss(pred, target)
        assert "loss" in losses
        assert "l1" in losses
        assert losses["loss"].item() > 0
