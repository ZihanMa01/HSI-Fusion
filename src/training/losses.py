"""损失函数。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class L1Loss(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(pred, target)


class SpectralAngleLoss(nn.Module):
    """光谱角损失 (SAM) — 衡量光谱保真度。"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # (B, C, H, W) → (B, C, H*W)
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)
        cos = F.cosine_similarity(pred, target, dim=1)
        return (1 - cos).mean()


class CombinedLoss(nn.Module):
    """组合损失: L1 + λ * SAM + 可扩展。"""

    def __init__(self, config: dict):
        super().__init__()
        self.l1_weight = config.get("l1", 1.0)
        self.sam_weight = config.get("sam", 0.1)
        self.l1 = L1Loss()
        self.sam = SpectralAngleLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.l1_weight * self.l1(pred, target)
        if self.sam_weight > 0:
            loss += self.sam_weight * self.sam(pred, target)
        return loss
