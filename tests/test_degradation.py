"""退化模型单元测试。"""

import torch
import pytest
from src.data.degradation import DegradationModel


class TestDegradation:
    def setup_method(self):
        self.config = {
            "psf": {"type": "gaussian", "sigma": 2.0, "kernel_size": 5},
            "srf": {"type": "nikonos", "n_msi_bands": 3},
            "noise": {"type": "none"},
        }
        self.deg = DegradationModel(self.config)

    def test_spatial_degrade_shape(self):
        hr_hsi = torch.rand(2, 31, 128, 128)
        lr_hsi = self.deg.spatial_degrade(hr_hsi, scale=4)
        assert lr_hsi.shape == (2, 31, 32, 32)

    def test_spectral_degrade_shape(self):
        hr_hsi = torch.rand(2, 31, 64, 64)
        hr_msi = self.deg.spectral_degrade(hr_hsi)
        assert hr_msi.shape == (2, 3, 64, 64)

    def test_full_pipeline_shape(self):
        hr_hsi = torch.rand(2, 31, 128, 128)
        lr_hsi, hr_msi = self.deg(hr_hsi, spatial_scale=4)
        assert lr_hsi.shape == (2, 31, 32, 32)
        assert hr_msi.shape == (2, 3, 128, 128)
