"""空间可视化 — RGB 伪彩色图对比。"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional


def _to_rgb(img: np.ndarray, bands: List[int] = None) -> np.ndarray:
    """从多波段图像提取 RGB 波段，并归一化到 [0, 255]。"""
    if bands is None:
        # 默认取中间三个波段
        c = img.shape[0]
        bands = [c // 4, c // 2, 3 * c // 4]
    rgb = np.stack([img[b] for b in bands], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)


def plot_rgb(
    img: np.ndarray,
    bands: List[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
    ax=None,
):
    """绘制 RGB 伪彩色图。"""
    rgb = _to_rgb(img, bands)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_spatial_comparison(
    results: List[np.ndarray],
    target: np.ndarray,
    method_names: List[str],
    bands: List[int] = None,
    save_path: Optional[str] = None,
):
    """多方法空间对比图。"""
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

    plot_rgb(target, bands, "Ground Truth", ax=axes[0])

    for i, (pred, name) in enumerate(zip(results, method_names)):
        plot_rgb(pred, bands, name, ax=axes[i + 1])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"空间对比图已保存: {save_path}")
    plt.show()
