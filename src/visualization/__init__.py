"""可视化模块。"""
from .plot_spectral import plot_spectral_comparison
from .plot_spatial import plot_spatial_comparison, plot_rgb
from .plot_metrics import plot_metric_bar, plot_radar_chart
from .plot_error import plot_error_map, plot_error_histogram
from .plot_ablation import plot_ablation_study
from .dashboard import create_comparison_dashboard

__all__ = [
    "plot_spectral_comparison", "plot_spatial_comparison", "plot_rgb",
    "plot_metric_bar", "plot_radar_chart",
    "plot_error_map", "plot_error_histogram",
    "plot_ablation_study", "create_comparison_dashboard",
]
