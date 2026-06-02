"""指标可视化 — 柱状图、雷达图。"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def plot_metric_bar(
    results: Dict[str, Dict[str, float]],
    metrics: list = None,
    save_path: Optional[str] = None,
    figsize=(10, 6),
):
    """
   绘制多方法多指标柱状图对比。

   Args:
        results: {"method_name": {"PSNR ↑": 35.2, "SSIM ↑": 0.98, ...}}
        metrics: 要显示的指标列表，默认全部
    """
    methods = list(results.keys())
    all_metrics = list(results[methods[0]].keys())
    if metrics is None:
        metrics = [m for m in all_metrics if "↑" in m or "↓" in m]

    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(methods):
        vals = [results[method].get(m, 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"指标对比图已保存: {save_path}")
    plt.show()


def plot_radar_chart(
    results: Dict[str, Dict[str, float]],
    metrics: list = None,
    save_path: Optional[str] = None,
):
    """雷达图 — 综合能力对比。"""
    methods = list(results.keys())
    all_metrics = list(results[methods[0]].keys())
    if metrics is None:
        # 自动选择方向一致的指标（"↑" 越多越好）
        metrics = [m for m in all_metrics if "↑" in m]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method in methods:
        vals = [results[method].get(m, 0) for m in metrics]
        # 归一化到 [0, 1]
        vmin, vmax = min(vals), max(vals)
        norm = [(v - vmin) / (vmax - vmin + 1e-8) for v in vals]
        norm += norm[:1]
        ax.plot(angles, norm, "o-", linewidth=2, label=method)
        ax.fill(angles, norm, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"雷达图已保存: {save_path}")
    plt.show()
