"""模型注册表 — 通过装饰器注册 / 按名称获取模型类。"""

from typing import Dict, Type

from .base_model import BaseFusionModel


class ModelRegistry:
    """全局模型注册表。"""

    _models: Dict[str, Type[BaseFusionModel]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册一个模型类。"""
        def wrapper(model_cls: Type[BaseFusionModel]):
            if name in cls._models:
                raise KeyError(f"模型 '{name}' 已注册: {cls._models[name]}")
            cls._models[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[BaseFusionModel]:
        """按名称获取模型类。"""
        if name not in cls._models:
            available = ", ".join(cls.list_models())
            raise KeyError(
                f"未知模型 '{name}'。可用模型: [{available}]"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """列出所有已注册的模型名称。"""
        return list(cls._models.keys())


# 便捷别名
register_model = ModelRegistry.register
