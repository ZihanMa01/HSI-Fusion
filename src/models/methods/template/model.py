"""
新方法模板 — 复制此目录并重命名，按 TODO 指引实现即可。

步骤:
  1. 复制 `template/` → `your_method_name/`
  2. 在 __init__.py 中修改 import
  3. 实现 YourMethod 类并添加 @register_model("your_method_name")
  4. 在 src/models/methods/__init__.py 中 from . import your_method_name
  5. 在 config/your_method.yaml 中配置参数

训练类型（选择一种）:
  A) 监督学习 (默认)
     设置 training_required = True, training_mode = "supervised"
     实现 get_loss() → 框架自动调用 Trainer

  B) 无监督学习
     设置 training_required = True, training_mode = "unsupervised"
     重写 get_loss() 使用内部一致性损失（不需 target）

  C) 多阶段训练 (如 GAN)
     设置 multi_stage = True
     重写 train_model() → 自定义训练流程

  D) 零训练方法 (如 CNMF)
     设置 training_required = False → 框架跳过训练，直接推理
"""
import torch.nn as nn
from torch import Tensor

from ...registry import register_model
from ...base_model import BaseFusionModel


@register_model("template_method")
class YourFusionMethod(BaseFusionModel):
    """
    TODO: 在此编写方法的描述和参考文献。

    训练元信息（根据实际情况修改）:
    """

    # ========== 训练元信息 ==========
    # 默认监督训练，覆盖以下字段声明你的方法类型
    training_required: bool = True
    training_mode: str = "supervised"   # supervised | unsupervised
    multi_stage: bool = False

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: 从 config 读取超参数并构建网络
        pass

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        """
        TODO: 实现融合前向传播。

        Args:
            lr_hsi: (B, C_hsi, H, W)  低分辨率高光谱图像
            hr_msi: (B, C_msi, H, W)  高分辨率多光谱图像

        Returns:
            hr_hsi: (B, C_hsi, H, W)  融合后的高分辨率高光谱图像
        """
        raise NotImplementedError

    def get_loss(self, pred, target, lr_hsi=None, hr_msi=None):
        """TODO: 按需自定义损失函数。"""
        return super().get_loss(pred, target, lr_hsi, hr_msi)

    # def train_model(self, train_loader, val_loader=None, config=None):
    #     """TODO: 仅多阶段/特殊训练需重写此方法。"""
    #     return super().train_model(train_loader, val_loader, config)
