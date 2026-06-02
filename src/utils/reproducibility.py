"""可复现性控制 — 随机种子 & 确定性计算。"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置全局随机种子，确保实验可复现。

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True):
    """
    设置 cuDNN 确定性计算（会降低性能）。

    Args:
        deterministic: 是否开启确定性模式
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
