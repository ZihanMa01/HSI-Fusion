"""模板方法默认配置。"""
from yacs.config import CfgNode as CN

_CONFIG = CN()

_CONFIG.model = CN()
_CONFIG.model.n_hsi_bands = 31
_CONFIG.model.n_msi_bands = 3

_CONFIG.training = CN()
_CONFIG.training.lr = 1e-4
_CONFIG.training.batch_size = 16
_CONFIG.training.epochs = 200
