"""消融实验可视化。"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def plot_ablation_study(
    results: Dict[str, Dict[str, float]],
    metric: str = "PSNR ↑",
    save_path: Optional[str] = None,
):
    """
    绘制消融实验对比图。

    Args:
        results: {"Full Model": {"PSNR ↑": 35.0, ...},
                  "w/o Module A": {"PSNR ↑": 34.2, ...}, ...}
        metric: 用于对比的指标名称
    """
    methods = list(results.keys())
    values = [results[m].get(metric, 0) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71"] + ["#e74c3c"] * (len(methods) - 1)

    bars = ax.barh(methods, values, color=colors)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01 * max(values),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)

    ax.set_xlabel(metric)
    ax.set_title(f"Ablation Study — {metric}")
    ax.grid(axis="x", alpha=0.3)

    # 标注性能下降
    full_val = values[0]
    for i, v in enumerate(values[1:], 1):
        drop = full_val - v
        ax.annotate(f"↓ {drop:.4f}", xy=(v, i-1),
                    xytext=(v + 0.05 * max(values), i-1),
                    fontsize=9, color="red", va="center")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"消融实验图已保存: {save_path}")
    plt.show()
