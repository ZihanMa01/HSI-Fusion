"""配置管理 — YAML 配置加载、合并。"""

import os
from pathlib import Path
from typing import Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(path: str) -> DictConfig:
    """
    加载 YAML 配置文件。

    Args:
        path: 配置文件路径

    Returns:
        OmegaConf DictConfig 配置对象
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = OmegaConf.load(f)

    return config


def merge_configs(*configs) -> DictConfig:
    """合并多个配置（后面的覆盖前面的）。"""
    base = OmegaConf.create()
    for cfg in configs:
        if cfg is not None:
            base = OmegaConf.merge(base, cfg)
    return base


def save_config(config, path: str):
    """保存配置到 YAML 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def config_to_dict(config) -> dict:
    """将 OmegaConf 配置转为普通 dict。"""
    return OmegaConf.to_container(config, resolve=True)
