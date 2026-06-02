"""融合模型基类 — 所有 HSI-MSI 融合方法的统一接口。

设计原则:
    forward(lr_hsi, hr_msi) → hr_hsi  是所有方法的唯一约束。

    训练方式由各方法自行决定:
    - training_required = True  → 走默认训练循环（监督训练）
    - training_required = False → 跳过训练（CNMF 等传统方法）
    - 重写 train()              → 自定义训练逻辑（GAN、多阶段等）

    评估永远走同一套指标，不关心内部怎么训练的。
"""

from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class BaseFusionModel(nn.Module):
    """
    所有 HSI-MSI 融合方法的通用基类。

    子类必须实现:
        forward(lr_hsi, hr_msi) -> hr_hsi

    子类可以声明:
        training_required: bool    — 是否需要训练（默认 True）
        重写 train()               — 自定义训练流程
        重写 get_loss()            — 自定义损失函数
    """

    # ========== 元信息：框架根据这些字段自动调度 ==========

    training_required: bool = True   # False = 零训练方法（如 CNMF）
    training_mode: str = "supervised"  # supervised | unsupervised | multi_stage
    multi_stage: bool = False        # True = 多阶段训练（如 GAN）

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._name = self.__class__.__name__

    # ========== 所有方法必须实现 ==========

    @abstractmethod
    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        """
        融合前向传播。

        Args:
            lr_hsi: (B, C_hsi, H_lr, W_lr)  低分辨率高光谱
            hr_msi: (B, C_msi, H_hr, W_hr)  高分辨率多光谱

        Returns:
            hr_hsi: (B, C_hsi, H_hr, W_hr)  融合结果
        """
        raise NotImplementedError

    # ========== 训练 ==========

    def train_model(self, train_loader, val_loader=None, config: dict = None) -> dict:
        """
        训练入口。框架自动调用此方法。

        默认行为:
            - 如果 training_required=False, 直接返回
            - 否则使用默认 Trainer 做监督训练

        子类可重写以实现:
            - 无监督训练
            - GAN 式多阶段对抗训练
            - 预训练+微调
            - 任何自定义流程
        """
        from ..training.trainer import Trainer
        trainer = Trainer(self, config)
        trainer.train(train_loader, val_loader)
        return {"status": "done"}

    def get_loss(
        self,
        pred: Tensor,
        target: Tensor,
        lr_hsi: Tensor = None,
        hr_msi: Tensor = None,
    ) -> dict:
        """
        默认监督损失。子类可重写。

        Returns:
            {"loss": Tensor, ...}  必须包含 "loss" 键
        """
        l1 = nn.functional.l1_loss(pred, target)
        return {"loss": l1, "l1": l1}

    def get_optimizer_params(self):
        """返回优化器参数，子类可重写以分组不同 learning rate。"""
        return self.parameters()

    # ========== 工具 ==========

    @property
    def name(self) -> str:
        return self._name
