"""数据预处理与增强。"""

import random
import torch
from torch import Tensor
from torchvision import transforms as T


class RandomCrop:
    """随机裁剪 HR-HSI / HR-MSI / LR-HSI 对。"""

    def __init__(self, hr_size: int = 128, lr_scale: int = 4):
        self.hr_size = hr_size
        self.lr_size = hr_size // lr_scale

    def __call__(self, sample: dict) -> dict:
        h, w = sample["hr_hsi"].shape[-2:]
        if h == self.hr_size and w == self.hr_size:
            return sample

        i = random.randint(0, h - self.hr_size)
        j = random.randint(0, w - self.hr_size)

        sample["hr_hsi"] = sample["hr_hsi"][..., i:i+self.hr_size, j:j+self.hr_size]
        sample["hr_msi"] = sample["hr_msi"][..., i:i+self.hr_size, j:j+self.hr_size]
        li, lj = i // 4, j // 4
        sample["lr_hsi"] = sample["lr_hsi"][..., li:li+self.lr_size, lj:lj+self.lr_size]
        return sample


class RandomFlip:
    """随机水平/垂直翻转。"""

    def __call__(self, sample: dict) -> dict:
        if random.random() < 0.5:
            for k in ["hr_hsi", "hr_msi", "lr_hsi"]:
                sample[k] = sample[k].flip(-1)
        if random.random() < 0.5:
            for k in ["hr_hsi", "hr_msi", "lr_hsi"]:
                sample[k] = sample[k].flip(-2)
        return sample


class ToTensor:
    """确保数据为 Tensor 类型。"""

    def __call__(self, sample: dict) -> dict:
        for k in ["hr_hsi", "hr_msi", "lr_hsi"]:
            if not isinstance(sample[k], Tensor):
                sample[k] = torch.from_numpy(sample[k]).float()
        return sample


class NormalizeTo01:
    """将数据线性归一化到 [0, 1]。"""

    def __call__(self, sample: dict) -> dict:
        for k in ["hr_hsi", "hr_msi", "lr_hsi"]:
            x = sample[k]
            min_v, max_v = x.min(), x.max()
            if max_v > min_v:
                sample[k] = (x - min_v) / (max_v - min_v)
        return sample


def get_transform_pipeline(split: str = "train", config: dict = None) -> object:
    """根据 split 返回变换流水线。"""
    cfg = config or {}
    hr_size = cfg.get("hr_crop_size", 128)
    lr_scale = cfg.get("spatial_scale", 4)

    transforms = [ToTensor(), NormalizeTo01()]
    if split == "train":
        transforms += [RandomCrop(hr_size, lr_scale), RandomFlip()]

    return T.Compose(transforms)
