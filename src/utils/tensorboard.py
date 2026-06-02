"""TensorBoard 日志封装。"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TensorboardWriter:
    """TensorBoard 写入器封装。"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def add_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, tag: str, values: dict, step: int):
        self.writer.add_scalars(tag, values, step)

    def add_image(self, tag: str, img_tensor, step: int):
        self.writer.add_image(tag, img_tensor, step)

    def add_histogram(self, tag: str, values, step: int):
        self.writer.add_histogram(tag, values, step)

    def close(self):
        self.writer.close()
