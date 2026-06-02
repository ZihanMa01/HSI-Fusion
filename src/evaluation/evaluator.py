"""评估器 & 批量对比运行器。"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
import numpy as np

from .metrics import calc_all_metrics
from .nr_metrics import calc_hqnr
from .efficiency import calc_params, calc_flops, measure_inference_time


class Evaluator:
    """模型评估器 — 在测试集上计算全面指标。"""

    def __init__(self, model: torch.nn.Module, config: dict, device: str = None):
        self.model = model.to(device).eval() if device else model.eval()
        self.config = config
        self.device = device or config.get("device", "cpu")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """在测试集上评估，返回平均指标。"""
        all_metrics = []
        deg_cfg = self.config.get("degradation", {})
        psf_cfg = deg_cfg.get("psf", {})
        spatial_scale = psf_cfg.get("spatial_scale", 4)

        with torch.no_grad():
            for batch in test_loader:
                lr_hsi = batch["lr_hsi"].to(self.device)
                hr_msi = batch["hr_msi"].to(self.device)
                target = batch["hr_hsi"].to(self.device)

                pred = self.model(lr_hsi, hr_msi)
                metrics = calc_all_metrics(pred, target, scale=spatial_scale)
                all_metrics.append(metrics)

        # 平均
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

        # 效率指标
        sample = next(iter(test_loader))
        lr_hsi = sample["lr_hsi"][:1].to(self.device)
        hr_msi = sample["hr_msi"][:1].to(self.device)

        avg_metrics["Params (M)"] = calc_params(self.model) / 1e6
        flops = calc_flops(self.model, lr_hsi, hr_msi)
        if flops:
            avg_metrics["FLOPs (G)"] = flops / 1e9

        return avg_metrics

    def save_results(self, metrics: Dict[str, float], save_dir: Union[str, Path]):
        """保存评估结果到 JSON。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


class BenchmarkRunner:
    """批量对比运行器 — 在相同测试集上评估多个方法并汇总结果。"""

    def __init__(self, test_loader: DataLoader, config: dict):
        self.test_loader = test_loader
        self.config = config

    def run(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """运行所有模型评估并返回汇总表格。"""
        results = {}
        for name, model in models.items():
            print(f"评估: {name}")
            evaluator = Evaluator(model, self.config)
            metrics = evaluator.evaluate(self.test_loader)
            results[name] = metrics
        return results

    @staticmethod
    def print_comparison_table(results: Dict[str, Dict[str, float]]):
        """打印对比表格。"""
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys()
                       ) if methods else []

        header = f"{'Method':<20}" + "".join(f"{m:<14}" for m in metrics)
        sep = "-" * len(header)
        print("\n" + sep)
        print(header)
        print(sep)
        for method in methods:
            row = f"{method:<20}"
            for m in metrics:
                val = results[method].get(m, float("nan"))
                row += f"{val:<14.4f}"
            print(row)
        print(sep + "\n")

    @staticmethod
    def save_comparison_csv(results: Dict[str, Dict[str, float]], path: str):
        """保存对比结果为 CSV。"""
        import csv
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys()) if methods else []
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Method"] + metrics)
            for method in methods:
                writer.writerow([method] + [results[method].get(m, "") for m in metrics])
        print(f"对比结果已保存: {path}")
