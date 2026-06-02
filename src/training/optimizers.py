"""优化器 & 学习率调度器工厂。"""

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """按配置构建优化器。"""
    opt_type = config.get("type", "adam").lower()
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 0)

    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            betas=config.get("betas", (0.9, 0.999))
        )
    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=config.get("momentum", 0.9)
        )
    else:
        raise ValueError(f"未知优化器类型: {opt_type}")


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """按配置构建学习率调度器。"""
    sched_type = config.get("type", "none").lower()
    epochs = config.get("epochs", 200)

    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=config.get("eta_min", 1e-6)
        )
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get("step_size", 50),
            gamma=config.get("gamma", 0.5)
        )
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.get("factor", 0.5),
            patience=config.get("patience", 10)
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"未知调度器类型: {sched_type}")
