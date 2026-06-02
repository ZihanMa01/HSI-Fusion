"""文件读写工具 — .mat / .npy / 图像格式。"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def save_mat(data: np.ndarray, path: Union[str, Path], key: str = "hsi"):
    """保存为 .mat 文件。"""
    if not SCIPY_AVAILABLE:
        raise ImportError("需要安装 scipy: pip install scipy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), {key: data})


def load_mat(path: Union[str, Path], key: str = None) -> np.ndarray:
    """
    加载 .mat 文件。

    Args:
        path: 文件路径
        key: 变量名，若为 None 则返回第一个数组变量
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("需要安装 scipy: pip install scipy")
    path = Path(path)
    data = sio.loadmat(str(path))

    if key is not None:
        return data[key]

    # 自动找到第一个数组变量（排除 MATLAB 元数据）
    for k, v in data.items():
        if isinstance(v, np.ndarray) and not k.startswith("__"):
            return v
    raise ValueError(f"未找到数组变量: {path}")


def save_npy(data: np.ndarray, path: Union[str, Path]):
    """保存为 .npy 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), data)


def load_npy(path: Union[str, Path]) -> np.ndarray:
    """加载 .npy 文件。"""
    return np.load(str(path))
