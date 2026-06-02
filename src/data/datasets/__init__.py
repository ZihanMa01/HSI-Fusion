"""统一 HSI 数据集加载器。

所有常见数据集（CAVE, Harvard, Pavia, Chikusei 等）
底层都是 .mat 文件，一个 HSIDataset 类全通用。
"""

from .hsi_dataset import HSIDataset

__all__ = ["HSIDataset"]
