"""Checkpoint 保存与加载。"""

from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict,
    path: Union[str, Path],
    is_best: bool = False,
):
    """
    保存 checkpoint。

    Args:
        state: 包含 model_state_dict / optimizer_state_dict / epoch / metric 等
        path: 保存路径
        is_best: 是否为最佳模型
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save(state, best_path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    map_location: str = None,
) -> Dict:
    """
    加载 checkpoint。

    Args:
        path: checkpoint 路径
        model: 若提供则加载模型权重
        optimizer: 若提供则加载优化器状态
        map_location: 设备映射

    Returns:
        state dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")

    map_location = map_location or ("cpu" if not torch.cuda.is_available() else None)
    state = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif model is not None and "state_dict" in state:
        model.load_state_dict(state["state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state
