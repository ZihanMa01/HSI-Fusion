"""Baseline CNN 默认配置。"""
from yacs.config import CfgNode as CN

_CONFIG = CN()

_CONFIG.model = CN()
_CONFIG.model.n_hsi_bands = 31       # LR-HSI 光谱通道数
_CONFIG.model.n_msi_bands = 3        # HR-MSI 光谱通道数
_CONFIG.model.n_feat = 64            # 特征通道数
_CONFIG.model.spatial_scale = 4      # 空间上采样倍数

_CONFIG.training = CN()
_CONFIG.training.lr = 1e-4
_CONFIG.training.weight_decay = 0
_CONFIG.training.batch_size = 16
_CONFIG.training.epochs = 200
_CONFIG.training.lr_scheduler = "cosine"
