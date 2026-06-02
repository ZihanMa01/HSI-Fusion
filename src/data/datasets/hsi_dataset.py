"""
统一 HSI 数据集加载器。

所有 .mat 格式的数据集通用的一个类，参数全由 YAML 配置驱动。
不内嵌任何数据集硬编码，新增数据集只需写 YAML，不改 Python。

Config 参数:
    file:           glob 匹配模式，默认 "*.mat"
    shape_order:    "HWC" (默认) 或 "CHW"
    single_scene:   True=整张图当一样本, False=每个 .mat 当一样本
    crop_black:     True=自动裁黑边
    tag:            日志用名称 (可选)

用法:
    HSIDataset("data/pavia", config={
        "file": "PaviaU.mat", "shape_order": "HWC", "single_scene": True,
    })
"""

import os
import glob
import numpy as np
import scipy.io as sio

from ..base_dataset import BaseHSIDataset


def _load_mat(path: str):
    """加载 .mat 文件，自动兼容 v5-v7 (scipy) 和 v7.3 (h5py)。"""
    try:
        data = sio.loadmat(path)
        # scipy 返回 dict，变量为 numpy 数组
        return data, "scipy"
    except NotImplementedError:
        # v7.3 格式，用 h5py
        import h5py
        f = h5py.File(path, "r")
        # 转成 dict of numpy arrays
        data = {}
        for k in f.keys():
            data[k] = np.array(f[k])
        f.close()
        return data, "h5py"


def _transpose_to_chw(arr: np.ndarray, shape_order: str, backend: str) -> np.ndarray:
    """统一转成 (C, H, W) 格式。

    scipy:  (H, W, C) → transpose(2,0,1) → (C, H, W)
    h5py:   (C, W, H) → transpose(0,2,1) → (C, H, W)  (原 MATLAB 为 HWC)
            (W, H, C) → transpose(2,1,0) → (C, H, W)  (原 MATLAB 为 CHW)
    """
    if shape_order == "CHW":
        if backend == "scipy":
            # 已经是 (C, H, W)
            return arr
        else:
            # h5py 读 CHW 原文件 → (W, H, C)
            return np.transpose(arr, (2, 1, 0))
    else:  # "HWC"
        if backend == "scipy":
            # (H, W, C) → (C, H, W)
            return np.transpose(arr, (2, 0, 1))
        else:
            # h5py 读 HWC 原文件 → (C, W, H)
            return np.transpose(arr, (0, 2, 1))


def _find_hsi_var(data: dict, backend: str) -> str:
    """从 .mat 中找到 3D 数组变量。"""
    cand = []
    for k, v in data.items():
        if k.startswith("__") or not isinstance(v, np.ndarray):
            continue
        if v.ndim == 3:
            cand.append((k, v.size))
    if not cand:
        raise KeyError(f"未找到 3D 数组: {[k for k in data if not k.startswith('__')]}")
    cand.sort(key=lambda x: -x[1])
    return cand[0][0]


def _crop_black_border(hsi: np.ndarray) -> np.ndarray:
    """去除全黑边框（Pavia Centre 用）。"""
    s = hsi.sum(axis=0)
    rows = np.where(s.sum(axis=1) > 0)[0]
    cols = np.where(s.sum(axis=0) > 0)[0]
    if rows.size and cols.size:
        hsi = hsi[:, rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return hsi


class HSIDataset(BaseHSIDataset):
    """统一 HSI 数据集加载器。"""

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, config: dict = None):
        config = config or {}
        self.ds_config = config
        super().__init__(root_dir, split, transform, config)

    def _load_data(self) -> list:
        cfg = self.ds_config
        file_pat = cfg.get("file", "*.mat")
        shape_order = cfg.get("shape_order", "HWC").upper()
        single_scene = cfg.get("single_scene", True)
        crop_black = cfg.get("crop_black", False)

        paths = sorted(glob.glob(os.path.join(self.root_dir, file_pat)))
        paths = [p for p in paths if not p.endswith("_gt.mat")
                 and not p.endswith("_gtGT.mat")]
        if not paths:
            raise FileNotFoundError(
                f"{self.root_dir} 下未找到匹配 {file_pat} 的 .mat 文件"
            )

        scenes = []
        for fp in paths:
            raw, backend = _load_mat(fp)
            key = _find_hsi_var(raw, backend)
            hsi = raw[key].astype(np.float32)
            hsi = _transpose_to_chw(hsi, shape_order, backend)
            scenes.append(hsi)

        if single_scene and scenes:
            hsi = scenes[0]
            if crop_black:
                hsi = _crop_black_border(hsi)
            return [hsi]

        return scenes

    def get_band_info(self) -> dict:
        n_hsi = self.data[0].shape[0] if self.data else 0
        return {
            "n_hsi": n_hsi,
            "n_msi": self.config.get("n_msi_bands", 3),
            "name": self.ds_config.get("tag", "HSI Dataset"),
        }
