"""光谱曲线对比可视化。"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional


def plot_spectral_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    pixel_coords: List[tuple] = [(16, 16)],
    labels: List[str] = None,
    method_names: List[str] = None,
    wavelengths: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Spectral Signature Comparison",
):
    """
    绘制指定像素位置的光谱曲线对比图。

    Args:
        pred: (C, H, W) 融合结果
        target: (C, H, W) 真实值
        pixel_coords: 需要对比的像素坐标列表
        labels: GT 和预测的标签
        method_names: 各方法名称
        wavelengths: 波长数组
    """
    n_pixels = len(pixel_coords)
    fig, axes = plt.subplots(1, n_pixels, figsize=(5 * n_pixels, 4))
    if n_pixels == 1:
        axes = [axes]

    for ax, (x, y) in zip(axes, pixel_coords):
        x_axis = wavelengths if wavelengths is not None else np.arange(pred.shape[0])

        ax.plot(x_axis, target[:, x, y], "k-", linewidth=2, label="Ground Truth")
        ax.plot(x_axis, pred[:, x, y], "r--", linewidth=1.5, label="Predicted")

        ax.set_xlabel("Wavelength" if wavelengths is not None else "Band")
        ax.set_ylabel("Reflectance")
        ax.set_title(f"Pixel ({x}, {y})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"光谱对比图已保存: {save_path}")

    plt.show()
