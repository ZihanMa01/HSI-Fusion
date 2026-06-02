"""日志记录工具。"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "HSI-Fusion", log_file: str = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    配置日志记录器：输出到控制台和文件。

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 控制台输出
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger
