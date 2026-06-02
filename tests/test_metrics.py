"""评估指标单元测试。"""

import torch
import pytest
from src.evaluation.metrics import calc_psnr, calc_sam, calc_ergas


class TestMetrics:
    def test_psnr_identical(self):
        x = torch.ones(1, 3, 16, 16)
        psnr = calc_psnr(x, x)
        assert psnr == float("inf")

    def test_psnr_different(self):
        x = torch.ones(1, 3, 16, 16)
        y = torch.zeros(1, 3, 16, 16)
        psnr = calc_psnr(x, y)
        assert psnr == 0.0

    def test_sam_identical(self):
        x = torch.ones(1, 3, 16, 16)
        sam = calc_sam(x, x)
        assert sam == pytest.approx(0.0, abs=1e-4)

    def test_ergas_identical(self):
        x = torch.ones(1, 3, 16, 16)
        ergas = calc_ergas(x, x)
        assert ergas == pytest.approx(0.0, abs=1e-4)
