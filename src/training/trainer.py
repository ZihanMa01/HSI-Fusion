"""训练器 — 管理训练循环、验证、日志与 checkpoint。"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .losses import CombinedLoss
from .optimizers import build_optimizer, build_lr_scheduler
from .callbacks import Callback, ModelCheckpoint


class Trainer:
    """
    通用训练器。

    用法:
        trainer = Trainer(model, config)
        trainer.train(train_loader, val_loader)
    """

    def __init__(self, model: nn.Module, config: dict,
                 experiment_dir: str = "experiments"):
        self.model = model
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.model.to(self.device)

        # 损失函数
        loss_cfg = config.get("loss", {"l1": 1.0, "sam": 0.1})
        self.criterion = CombinedLoss(loss_cfg)

        # 优化器
        opt_cfg = config.get("optimizer", {"type": "adam", "lr": 1e-4})
        self.optimizer = build_optimizer(model, opt_cfg)
        self.scheduler = build_lr_scheduler(
            self.optimizer, config.get("scheduler", {"type": "cosine", "epochs": 200})
        )

        # 实验目录
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.experiment_dir / "tensorboard"))

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.callbacks = [ModelCheckpoint(self.experiment_dir / "checkpoints")]

        # 混合精度
        self.amp = config.get("amp", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: Optional[int] = None):
        """运行训练循环。"""
        max_epochs = epochs or self.config.get("epochs", 200)
        log_interval = self.config.get("log_interval", 50)

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_loss = 0.0
            start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                lr_hsi = batch["lr_hsi"].to(self.device)
                hr_msi = batch["hr_msi"].to(self.device)
                target = batch["hr_hsi"].to(self.device)

                self.optimizer.zero_grad()

                if self.amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(lr_hsi, hr_msi)
                        losses = self.model.get_loss(pred, target, lr_hsi, hr_msi)
                    self.scaler.scale(losses["loss"]).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(lr_hsi, hr_msi)
                    losses = self.model.get_loss(pred, target, lr_hsi, hr_msi)
                    losses["loss"].backward()
                    self.optimizer.step()

                epoch_loss += losses["loss"].item()
                self.global_step += 1

                # 日志
                if batch_idx % log_interval == 0:
                    self.writer.add_scalar("train/loss", losses["loss"].item(), self.global_step)
                    if "l1" in losses:
                        self.writer.add_scalar("train/l1", losses["l1"].item(), self.global_step)

            # Epoch 结束
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / len(train_loader)
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)

            # 验证
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader)

            # 调度器
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", avg_loss))
                else:
                    self.scheduler.step()

            # 保存最佳模型
            current_metric = val_metrics.get("val_loss", avg_loss)
            for cb in self.callbacks:
                cb.step(self.model, current_metric, epoch, self.experiment_dir)

            # 打印进度
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s | LR: {lr:.2e}")

        self.writer.close()

    def _validate(self, val_loader: DataLoader) -> dict:
        """验证循环。"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr_hsi = batch["lr_hsi"].to(self.device)
                hr_msi = batch["hr_msi"].to(self.device)
                target = batch["hr_hsi"].to(self.device)
                pred = self.model(lr_hsi, hr_msi)
                loss = self.criterion(pred, target)
                total_loss += loss.item()
        return {"val_loss": total_loss / len(val_loader)}
