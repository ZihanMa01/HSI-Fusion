from .registry import ModelRegistry, register_model
from .base_model import BaseFusionModel

# 自动发现所有已注册的方法
from . import methods

__all__ = ["ModelRegistry", "register_model", "BaseFusionModel"]
