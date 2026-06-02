"""无参考评估指标 — 不依赖 Ground Truth 的图像质量评估。

参考文献:
    - D_lambda / D_s / HQNR: Vivone et al., "A New Benchmark Based on Recent Advances"
    - NQE: Naturalness Quality Evaluator
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def calc_d_lambda(hr_hsi: Tensor, hr_msi: Tensor, srf: Tensor = None) -> float:
    """
    光谱失真指标 D_lambda — 值越小越好。
    计算 MSI 与降采样 HSI 之间的光谱角差异。
    """
    # 简单实现: 将 HR-HSI 按 SRF 降采样到 MSI 光谱空间
    if srf is not None:
        n_msi = srf.shape[0]
        # (B, C_hsi, H, W) → (B, n_msi, H, W)
        hsi_msi = torch.einsum("bchw,mc->bmhw", hr_hsi, srf)
    else:
        # 如果没有 SRF，取前 n_msi 个波段近似
        n_msi = min(hr_msi.shape[1], hr_hsi.shape[1])
        hsi_msi = hr_hsi[:, :n_msi]
        msi = hr_msi[:, :n_msi]

    # 计算逐像素光谱角
    hsi_msi = hsi_msi.reshape(hsi_msi.shape[0], hsi_msi.shape[1], -1)
    msi = hr_msi[:, :n_msi].reshape(hr_msi.shape[0], n_msi, -1)

    cos = F.cosine_similarity(hsi_msi, msi, dim=1)
    d_lambda = (1 - cos).mean()
    return d_lambda.item()


def calc_d_s(hr_hsi: Tensor, lr_hsi: Tensor) -> float:
    """
    空间失真指标 D_s — 值越小越好。
    比较 LR-HSI 上采样后与 HR-HSI 降采样后的空间差异。
    """
    # 将 LR-HSI 上采样到 HR 尺寸
    up_lr = F.interpolate(lr_hsi, size=hr_hsi.shape[-2:], mode="bilinear", align_corners=False)

    # 将 HR-HSI 光谱下采样到 LR-HSI 波段数
    n_lr = lr_hsi.shape[1]
    down_hr = hr_hsi[:, :n_lr]

    # 计算每波段 SSIM 的差异
    d_s = F.l1_loss(up_lr, down_hr)
    return d_s.item()


def calc_hqnr(hr_hsi: Tensor, lr_hsi: Tensor, hr_msi: Tensor,
              srf: Tensor = None, scale: int = 4) -> float:
    """
    HQNR (Hybrid Quality with No Reference) — 值越大越好。
    HQNR = (1 - D_lambda) * (1 - D_s)
    """
    d_lambda = calc_d_lambda(hr_hsi, hr_msi, srf)
    d_s = calc_d_s(hr_hsi, lr_hsi)
    hqnr = (1 - d_lambda) * (1 - d_s)
    return hqnr
