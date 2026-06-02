"""退化模型 — 模拟 LR-HSI 和 HR-MSI 的生成过程。

参考文献 [Wald's protocol]:
    - LR-HSI: HR-HSI 经空间 PSF 模糊 + 下采样得到
    - HR-MSI: HR-HSI 经光谱 SRF 响应积分得到

支持的 SRF 类型:
    - "nikonos": 模拟 3 波段可见光
    - "xlsx": 从 Excel 文件读取真实传感器 SRF
    - "random": 随机 SRF
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DegradationModel(nn.Module):
    """
    HSI-MSI 退化模型。

    支持:
        - 空间退化: 高斯 PSF 模糊 + 下采样
        - 光谱退化: 支持模拟 / 真实传感器 SRF
        - 噪声注入: 高斯 / 泊松
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.psf_config = config.get("psf", {"type": "gaussian"})
        self.srf_config = config.get("srf", {"type": "nikonos", "n_msi_bands": 3})
        self.noise_config = config.get("noise", {"type": "none"})
        self._cached_srf = None  # (n_msi, n_hsi)

    def spatial_degrade(self, hr_hsi: Tensor, scale: int = None) -> Tensor:
        """
        空间退化: 高斯模糊 → 下采样
        Wald's protocol: sigma = scale / 2.35482, ksize = scale|1
        Args:
            hr_hsi: (B, C, H, W)
            scale: 下采样倍数，默认取 degradation.psf.spatial_scale
        Returns:
            lr_hsi: (B, C, H//scale, W//scale)
        """
        psf_scale = self.psf_config.get("spatial_scale")
        scale = scale or psf_scale or 4
        sigma = self.psf_config.get("sigma", scale / 2.35482)  # FWHM = scale
        ksize = self.psf_config.get("kernel_size", scale | 1)
        kernel = self._gaussian_kernel(ksize, sigma, hr_hsi.device)
        # kernel: (1, 1, k, k) → (C, 1, k, k) 逐通道卷积
        kernel = kernel.expand(hr_hsi.shape[1], 1, *kernel.shape[-2:])

        pad = kernel.shape[-1] // 2
        blurred = F.conv2d(
            F.pad(hr_hsi, [pad, pad, pad, pad], mode="replicate"),
            kernel,
            groups=hr_hsi.shape[1],
        )
        lr_hsi = blurred[:, :, ::scale, ::scale]
        return lr_hsi

    def spectral_degrade(self, hr_hsi: Tensor) -> Tensor:
        """
        光谱退化: 通过 SRF 积分得到多光谱图像。
        Args:
            hr_hsi: (B, C_hsi, H, W)
        Returns:
            hr_msi: (B, C_msi, H, W)
        """
        srf = self._get_srf(hr_hsi.shape[1], hr_hsi.device)
        # (B, C_hsi, H, W) → (B, C_msi, H, W)
        hr_msi = torch.einsum("bchw,mc->bmhw", hr_hsi, srf)
        return hr_msi

    def add_noise(self, image: Tensor) -> Tensor:
        """添加噪声（SNR dB 控制）。"""
        noise_type = self.noise_config.get("type", "gaussian")
        snr_db = self.noise_config.get("snr_db", 30)

        if noise_type == "none":
            return image

        signal_power = image.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(image) * noise_power.sqrt()
        return image + noise

    def forward(self, hr_hsi: Tensor, spatial_scale: int = 4) -> Tuple[Tensor, Tensor]:
        """生成 LR-HSI 和 HR-MSI。"""
        lr_hsi = self.spatial_degrade(hr_hsi, spatial_scale)
        lr_hsi = self.add_noise(lr_hsi)
        hr_msi = self.spectral_degrade(hr_hsi)
        hr_msi = self.add_noise(hr_msi)
        return lr_hsi, hr_msi

    # ========================
    # 退化核生成
    # ========================

    @staticmethod
    def _gaussian_kernel(ksize: int, sigma: float, device) -> Tensor:
        """生成 2D 高斯核。"""
        coords = torch.arange(ksize, dtype=torch.float32, device=device)
        coords -= ksize // 2
        g = coords ** 2
        g = (-g / (2 * sigma ** 2)).exp()
        g /= g.sum()
        return g.view(1, 1, ksize, 1) @ g.view(1, 1, 1, ksize)

    # ========================
    # SRF 加载
    # ========================

    def _get_srf(self, n_hsi: int, device) -> Tensor:
        """获取 SRF 矩阵，形状 (n_msi, n_hsi)。"""
        srf_type = self.srf_config.get("type", "nikonos")

        if srf_type == "nikonos":
            return self._srf_nikonos(n_hsi, device)
        elif srf_type == "random":
            return self._srf_random(n_hsi, device)
        elif srf_type == "xlsx":
            return self._srf_from_xlsx(n_hsi, device)
        else:
            raise ValueError(f"未知 SRF 类型: {srf_type}，可选: nikonos / random / xlsx")

    def _srf_nikonos(self, n_hsi: int, device) -> Tensor:
        """模拟 Nikon 相机的 R/G/B SRF（高斯型）。"""
        n_msi = self.srf_config.get("n_msi_bands", 3)
        centers = torch.linspace(0.2, 0.8, n_msi)
        sigma = 0.15
        x = torch.linspace(0, 1, n_hsi)
        srf = torch.exp(-((x.view(1, -1) - centers.view(-1, 1)) ** 2) / (2 * sigma ** 2))
        srf = srf / srf.sum(dim=1, keepdim=True)
        return srf.to(device)

    def _srf_random(self, n_hsi: int, device) -> Tensor:
        """随机 SRF。"""
        n_msi = self.srf_config.get("n_msi_bands", 3)
        srf = torch.rand(n_msi, n_hsi, device=device)
        srf = srf / srf.sum(dim=1, keepdim=True)
        return srf

    def _srf_from_xlsx(self, n_hsi: int, device) -> Tensor:
        """从 xlsx 文件读取真实传感器 SRF。"""
        import pandas as pd
        import numpy as np

        path = self.srf_config.get("path")
        if not path:
            raise ValueError("xlsx SRF 需要指定 path 参数")

        sheet = self.srf_config.get("sheet", 0)
        df = pd.read_excel(path, sheet_name=sheet, header=None)

        # 第一列为波长 (nm)，其余列为各波段的 SRF 值
        wavelengths = df.iloc[:, 0].values.astype(np.float64)  # (n_rows,)
        srf_values = df.iloc[:, 1:].values.astype(np.float64)   # (n_rows, n_msi)

        # 归一化：每列和为 1
        col_sums = srf_values.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        srf_values = srf_values / col_sums

        # 插值到 HSI 波段数
        if len(wavelengths) != n_hsi:
            # 线性插值到 [0, 1] 范围
            old_pos = np.linspace(0, 1, len(wavelengths))
            new_pos = np.linspace(0, 1, n_hsi)
            srf_values = np.array([np.interp(new_pos, old_pos, srf_values[:, b])
                                   for b in range(srf_values.shape[1])])
            # srf_values: (n_msi, n_hsi)
        else:
            srf_values = srf_values.T  # (n_msi, n_hsi)

        srf = torch.from_numpy(srf_values).float().to(device)
        srf = srf / srf.sum(dim=1, keepdim=True)
        return srf
