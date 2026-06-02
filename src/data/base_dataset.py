"""数据集基类 — 统一各数据集的加载接口。

支持两种模式:
  1. 预退化: _load_data() 返回 [(hr_hsi, hr_msi, lr_hsi), ...]
  2. 即时退化 (大图): _load_data() 返回 [hr_hsi, ...]，__getitem__ 先裁 patch 再退化
"""

import os
import random
from abc import abstractmethod
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor

from .degradation import DegradationModel


class BaseHSIDataset(Dataset):

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, config: dict = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.config = config or {}

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"数据集目录不存在: {root_dir}")

        # 退化模型（即时退化模式使用）
        deg_cfg = self.config.get("degradation", {})
        self.degradation_model = DegradationModel(deg_cfg) if deg_cfg else None

        # 大图 patch 参数
        self.full_image = self.config.get("full_image", False)
        self.hr_size = self.config.get("hr_crop_size", 128)
        deg_cfg = self.config.get("degradation", {})
        psf_cfg = deg_cfg.get("psf", {})
        self.spatial_scale = psf_cfg.get("spatial_scale", 4)
        self.lr_size = self.hr_size // self.spatial_scale

        self.data = self._load_data()
        if len(self.data) == 0:
            raise ValueError(f"数据集为空: {root_dir}")

        # 检测模式:
        #   tuple(3) = (hr_hsi, hr_msi, lr_hsi) → 预退化
        #   否则 = hr_hsi → 即时退化 (需 degradation_model)
        first = self.data[0]
        self._lazy_degrade = isinstance(first, (Tensor, np.ndarray)) and not isinstance(first, tuple)
        if self._lazy_degrade and self.degradation_model is None:
            raise ValueError(
                "数据集返回了 HR-HSI（即时退化模式），但未配置 degradation 参数。\n"
                "请在 config 中添加:\n"
                "  degradation:\n"
                "    psf: {type: gaussian, sigma: 2.0, kernel_size: 11}\n"
                "    srf: {type: nikonos, n_msi_bands: 3}\n"
            )

    @abstractmethod
    def _load_data(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def get_band_info(self) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        if self._lazy_degrade:
            # === 即时退化模式 ===
            hr_hsi = item
            if not isinstance(hr_hsi, Tensor):
                hr_hsi = torch.from_numpy(hr_hsi).float()

            if self.full_image:
                # 全图模式：裁到 scale 倍数 → 退化整张图
                h, w = hr_hsi.shape[-2:]
                h = h - h % self.spatial_scale
                w = w - w % self.spatial_scale
                hr_patch = hr_hsi[..., :h, :w]
            else:
                # 随机裁剪 patch
                h, w = hr_hsi.shape[-2:]
                if h >= self.hr_size and w >= self.hr_size:
                    i = random.randint(0, h - self.hr_size)
                    j = random.randint(0, w - self.hr_size)
                    hr_patch = hr_hsi[..., i:i+self.hr_size, j:j+self.hr_size]
                else:
                    hr_patch = hr_hsi

            lr_hsi, hr_msi = self.degradation_model(
                hr_patch.unsqueeze(0), self.spatial_scale
            )
            hr_hsi = hr_patch
            lr_hsi = lr_hsi.squeeze(0)
            hr_msi = hr_msi.squeeze(0)
        else:
            # === 预退化模式 ===
            hr_hsi, hr_msi, lr_hsi = item
            if not isinstance(hr_hsi, Tensor):
                hr_hsi = torch.from_numpy(hr_hsi).float()
                hr_msi = torch.from_numpy(hr_msi).float()
                lr_hsi = torch.from_numpy(lr_hsi).float()

        sample = {"hr_hsi": hr_hsi, "hr_msi": hr_msi, "lr_hsi": lr_hsi, "idx": idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
