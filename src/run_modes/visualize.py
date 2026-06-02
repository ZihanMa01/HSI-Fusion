"""可视化模式。"""

from omegaconf import DictConfig
from src.utils.logger import setup_logger


def run_visualize(config: DictConfig, method_name: str, checkpoint_path: str = None):
    """运行可视化。"""
    logger = setup_logger("HSI-Fusion")
    logger.info(f"可视化: {method_name}")

    # TODO: 加载模型 → 推理 → 可视化
    logger.info("可视化完成")
