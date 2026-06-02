"""全参考评估指标 — HSI-MSI 融合标准指标集合。

参考文献:
    - PSNR / SSIM: 图像质量基础指标
    - SAM: Spectral Angle Mapper (光谱保真度)
    - ERGAS: Erreur Relative Globale Adimensionnelle de Synthèse
    - UIQI: Universal Image Quality Index
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from skimage.metrics import structural_similarity as ssim_skimage
from typing import Dict, Optional, Union


def calc_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """峰值信噪比 (PSNR)。"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val ** 2 / mse).item()


def calc_ssim(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """结构相似性 (SSIM) — 逐波段平均。"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_vals = []
    for b in range(pred_np.shape[1]):
        ssim_vals.append(ssim_skimage(
            pred_np[0, b], target_np[0, b],
            data_range=max_val, win_size=11
        ))
    return float(np.mean(ssim_vals))


def calc_sam(pred: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    """光谱角 (SAM) — 值越小越好。"""
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target = target.reshape(target.shape[0], target.shape[1], -1)
    cos = F.cosine_similarity(pred, target, dim=1)
    sam = torch.acos(cos.clamp(-1 + eps, 1 - eps))
    return sam.mean().item()


def calc_ergas(pred: Tensor, target: Tensor, scale: int = 4) -> float:
    """ERGAS — 值越小越好。"""
    mse_per_band = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    mean_per_band = target.mean(dim=(0, 2, 3))
    rmse_ratio = torch.sqrt(mse_per_band) / (mean_per_band + 1e-8)
    ergas = 100 * (1 / scale) * torch.sqrt((rmse_ratio ** 2).mean())
    return ergas.item()


def calc_uiqi(pred: Tensor, target: Tensor, block_size: int = 8) -> float:
    """通用图像质量指标 (UIQI) — 逐波段平均。"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    uiqi_vals = []
    for b in range(pred_np.shape[1]):
        uiqi_vals.append(_q_index(pred_np[0, b], target_np[0, b], block_size))
    return float(np.mean(uiqi_vals))


def _q_index(img1: np.ndarray, img2: np.ndarray, block_size: int = 8) -> float:
    """单个波段的 Q 指数。"""
    h, w = img1.shape
    q_total = 0.0
    count = 0
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            b1 = img1[i:i+block_size, j:j+block_size]
            b2 = img2[i:i+block_size, j:j+block_size]
            sigma12 = np.mean((b1 - b1.mean()) * (b2 - b2.mean()))
            sigma1 = np.std(b1)
            sigma2 = np.std(b2)
            mu1 = b1.mean()
            mu2 = b2.mean()
            numerator = 4 * sigma12 * mu1 * mu2
            denominator = (sigma1**2 + sigma2**2) * (mu1**2 + mu2**2) + 1e-8
            q = numerator / denominator
            q_total += q
            count += 1
    return q_total / count


def calc_all_metrics(pred: Tensor, target: Tensor, scale: int = 4,
                     max_val: float = 1.0) -> Dict[str, float]:
    """计算所有全参考指标。"""
    return {
        "PSNR ↑": calc_psnr(pred, target, max_val),
        "SSIM ↑": calc_ssim(pred, target, max_val),
        "SAM ↓": calc_sam(pred, target),
        "ERGAS ↓": calc_ergas(pred, target, scale),
        "UIQI ↑": calc_uiqi(pred, target),
    }
