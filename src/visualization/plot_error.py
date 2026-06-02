"""误差可视化 — 差值图、误差直方图。"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional


def plot_error_map(
    pred: np.ndarray,
    target: np.ndarray,
    band: int = None,
    save_path: Optional[str] = None,
    cmap: str = "RdBu_r",
):
    """绘制误差图 (pred - target)。"""
    if band is not None:
        error = pred[band] - target[band]
        title = f"Error Map (Band {band})"
    else:
        error = np.mean(pred - target, axis=0)
        title = "Mean Error Map (All Bands)"

    vmax = max(abs(error.min()), abs(error.max()))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(error, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_error_histogram(
    pred: np.ndarray,
    target: np.ndarray,
    method_names: List[str] = None,
    band: int = None,
    save_path: Optional[str] = None,
    bins: int = 50,
):
    """绘制误差直方图 (多方法对比)。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(len(pred))]

    if band is not None:
        errors = [p[band] - target[band] for p in pred]
        title = f"Error Distribution (Band {band})"
    else:
        errors = [np.mean(p - target, axis=0).ravel() for p in pred]
        title = "Mean Error Distribution"

    for err, name in zip(errors, method_names):
        ax.hist(err.ravel(), bins=bins, alpha=0.5, label=name)

    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"误差直方图已保存: {save_path}")
    plt.show()
