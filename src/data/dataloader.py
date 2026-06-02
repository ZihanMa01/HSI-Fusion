"""DataLoader 工厂 — 按配置构建 DataLoader。"""

from torch.utils.data import DataLoader, Dataset


def build_dataloader(dataset: Dataset, config: dict, split: str = "train") -> DataLoader:
    """
    根据配置构建 DataLoader。

    Args:
        dataset: 数据集实例
        config: 配置字典，至少包含 batch_size / num_workers
        split: "train" | "val" | "test"
    """
    loader_cfg = config.get("dataloader", {})

    batch_size = loader_cfg.get("batch_size", config.get("batch_size", 16))
    num_workers = loader_cfg.get("num_workers", 4)
    shuffle = split == "train"
    pin_memory = loader_cfg.get("pin_memory", True)
    drop_last = loader_cfg.get("drop_last", split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
