"""Baseline CNN — 简单的 CNN 基线方法，作为实现新方法的参考模板。"""
import torch
import torch.nn as nn
from torch import Tensor

from ...registry import register_model
from ...base_model import BaseFusionModel


@register_model("baseline_cnn")
class BaselineCNN(BaseFusionModel):
    """
    简单 CNN 基线：将 LR-HSI 上采样后与 HR-MSI 拼接，通过若干卷积层融合。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        n_hsi = model_cfg.get("n_hsi_bands", 31)   # LR-HSI 光谱通道数
        n_msi = model_cfg.get("n_msi_bands", 3)     # HR-MSI 光谱通道数
        n_feat = model_cfg.get("n_feat", 64)         # 特征通道数

        # 双三次上采样 LR-HSI → HR 空间尺寸
        self.upsample = nn.Upsample(
            scale_factor=model_cfg.get("spatial_scale", 4),
            mode="bilinear",
            align_corners=False,
        )

        # 特征提取与融合
        self.conv_in = nn.Sequential(
            nn.Conv2d(n_hsi + n_msi, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(n_feat) for _ in range(4)],
        )
        self.conv_out = nn.Conv2d(n_feat, n_hsi, 3, 1, 1)

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        # 上采样 LR-HSI 到 HR 空间
        up_hsi = self.upsample(lr_hsi)                     # (B, C_hsi, H, W)
        # 通道拼接
        fused = torch.cat([up_hsi, hr_msi], dim=1)         # (B, C_hsi+C_msi, H, W)
        # 特征提取与重建
        feat = self.conv_in(fused)
        feat = self.res_blocks(feat)
        hr_hsi = self.conv_out(feat)                       # (B, C_hsi, H, W)
        # 全局残差连接
        return hr_hsi + up_hsi

    def get_loss(self, pred, target, lr_hsi=None, hr_msi=None):
        l1 = nn.functional.l1_loss(pred, target)
        # 光谱角损失 (SAM)
        cos = nn.functional.cosine_similarity(
            pred.reshape(pred.shape[0], pred.shape[1], -1),
            target.reshape(target.shape[0], target.shape[1], -1),
            dim=1,
        )
        sam_loss = (1 - cos).mean()
        return {"loss": l1 + 0.1 * sam_loss, "l1": l1, "sam": sam_loss}


class ResBlock(nn.Module):
    """简单残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.conv(x)
