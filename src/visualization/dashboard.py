"""综合结果看板 — 在一个页面中整合所有对比图表。"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from .plot_metrics import plot_metric_bar, plot_radar_chart


def create_comparison_dashboard(
    results: Dict[str, Dict[str, float]],
    save_dir: str = "experiments/dashboard",
):
    """生成综合对比看板。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 柱状图
    plot_metric_bar(
        results,
        save_path=str(save_dir / "metric_comparison_bar.png"),
    )

    # 2. 雷达图
    plot_radar_chart(
        results,
        save_path=str(save_dir / "radar_comparison.png"),
    )

    print(f"综合看板已生成: {save_dir}/")
