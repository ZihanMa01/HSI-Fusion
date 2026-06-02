"""训练回调 — checkpoint、早停、日志等。"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class Callback:
    """回调基类。"""
    def step(self, model: nn.Module, metric: float, epoch: int, experiment_dir: Path):
        pass


class ModelCheckpoint(Callback):
    """保存最佳模型权重。"""

    def __init__(self, save_dir: Path, mode: str = "min", keep_top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.keep_top_k = keep_top_k
        self.saved_checkpoints = []

    def step(self, model: nn.Module, metric: float, epoch: int, experiment_dir: Path):
        improved = (self.mode == "min" and metric < self.best_metric) or \
                   (self.mode == "max" and metric > self.best_metric)
        if improved:
            self.best_metric = metric
            path = self.save_dir / f"epoch_{epoch+1:04d}_metric_{metric:.4f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metric": metric,
            }, path)
            self.saved_checkpoints.append(path)
            # 清理旧的 checkpoint
            while len(self.saved_checkpoints) > self.keep_top_k:
                old = self.saved_checkpoints.pop(0)
                if old.exists():
                    old.unlink()


class EarlyStopping(Callback):
    """早停。"""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float("inf")
        self.early_stop = False

    def step(self, model: nn.Module, metric: float, epoch: int, experiment_dir: Path):
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class TensorBoardLogger(Callback):
    """TensorBoard 日志回调。"""
    def __init__(self, writer):
        self.writer = writer

    def step(self, model: nn.Module, metric: float, epoch: int, experiment_dir: Path):
        self.writer.add_scalar("val/metric", metric, epoch)
