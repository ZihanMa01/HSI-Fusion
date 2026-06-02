"""数据集注册表。"""

from typing import Dict, Type
from torch.utils.data import Dataset


class DatasetRegistry:
    _datasets: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(ds_cls):
            cls._datasets[name] = ds_cls
            return ds_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[Dataset]:
        if name not in cls._datasets:
            raise KeyError(f"未知数据集 '{name}'，可用: {list(cls._datasets.keys())}")
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._datasets.keys())


register_dataset = DatasetRegistry.register
