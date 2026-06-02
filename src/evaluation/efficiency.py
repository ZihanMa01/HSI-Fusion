"""效率评估 — 模型参数量、FLOPs、推理时间。"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def calc_params(model: nn.Module) -> int:
    """计算模型参数量。"""
    return sum(p.numel() for p in model.parameters())


def calc_flops(model: nn.Module, lr_hsi: Tensor, hr_msi: Tensor) -> Optional[int]:
    """计算 FLOPs (需要安装 thop)。"""
    if not THOP_AVAILABLE:
        return None
    try:
        flops, _ = profile(model, inputs=(lr_hsi, hr_msi), verbose=False)
        return flops
    except Exception:
        return None


def measure_inference_time(model: nn.Module, lr_hsi: Tensor, hr_msi: Tensor,
                           num_warmup: int = 10, num_iter: int = 100,
                           device: str = "cuda") -> Dict[str, float]:
    """测量推理时间。"""
    model.eval()
    model.to(device)
    lr_hsi = lr_hsi.to(device)
    hr_msi = hr_msi.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(lr_hsi, hr_msi)
            if device == "cuda":
                torch.cuda.synchronize()

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(num_iter):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(lr_hsi, hr_msi)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }
